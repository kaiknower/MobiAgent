"""Native Gemini endpoint client (via google-genai SDK).

Used when the proxy's OpenAI-compatible path strips video uploads. The native
endpoint at https://generativelanguage.googleapis.com accepts inline video bytes.
"""
from __future__ import annotations

import base64
import os
from typing import Any


# Empty string ⇒ use google-genai SDK's built-in default endpoint
# (https://generativelanguage.googleapis.com). Override via GEMINI_NATIVE_BASE_URL.
DEFAULT_GEMINI_NATIVE_BASE_URL = ""
DEFAULT_GEMINI_NATIVE_MODEL = "gemini-2.5-flash"


def _decode_data_url(video_data_url: str) -> tuple[bytes, str]:
    if not video_data_url.startswith("data:"):
        raise ValueError("Expected a data: URL for video_data_url")
    header, _, b64 = video_data_url.partition(",")
    mime = header[len("data:") :].split(";", 1)[0] or "video/mp4"
    return base64.b64decode(b64), mime


def build_native_gemini_client(*, api_key: str | None = None) -> Any:
    from google import genai

    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY must be set to use the native Gemini client")
    base_url = os.getenv("GEMINI_NATIVE_BASE_URL", DEFAULT_GEMINI_NATIVE_BASE_URL)
    extra_headers: dict[str, str] = {}
    cf_auth = os.getenv("CF_AIG_AUTHORIZATION", "")
    if cf_auth:
        if not cf_auth.lower().startswith("bearer "):
            cf_auth = f"Bearer {cf_auth}"
        extra_headers["cf-aig-authorization"] = cf_auth
    cf_timeout_ms = os.getenv("CF_AIG_REQUEST_TIMEOUT_MS", "300000")
    if cf_timeout_ms:
        extra_headers["cf-aig-request-timeout"] = cf_timeout_ms
    http_options: dict[str, Any] = {}
    if base_url:
        http_options["base_url"] = base_url
    if extra_headers:
        http_options["headers"] = extra_headers
    if http_options:
        return genai.Client(api_key=key, http_options=http_options)
    return genai.Client(api_key=key)


def run_native_gemini_inference(
    *,
    messages: list[dict[str, Any]],
    model: str,
    video_data_url: str | None = None,
    frame_data_urls: list[str] | None = None,
    max_output_tokens: int = 32768,
    enable_thinking: bool = True,
    response_mime_type: str = "application/json",
    max_retries: int = 6,
    retry_base_delay_sec: float = 8.0,
) -> str:
    """Call the native Gemini endpoint with the same messages/video payload and return raw text."""
    import time

    from google.genai import types

    client = build_native_gemini_client()

    # Allow disabling retry via env var (set to 0 to let errors surface immediately)
    env_max = os.getenv("GEMINI_MAX_RETRIES")
    if env_max is not None:
        try:
            max_retries = max(0, int(env_max))
        except ValueError:
            pass
    # If caller set retries to 0 or 1, just run once without wrapping
    if max_retries <= 1:
        max_retries = 1

    # Flatten system + user messages into a single Gemini `contents` list.
    # google-genai treats strings as user parts; we prepend system instruction separately via config.
    system_text = ""
    user_parts: list[Any] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            if isinstance(content, str):
                system_text = content
            continue
        if isinstance(content, str):
            user_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                itype = item.get("type")
                if itype == "text" and isinstance(item.get("text"), str):
                    user_parts.append(item["text"])

    fps_hint_env = os.getenv("CLAW_VIDEO_FPS_HINT", "").strip()
    video_metadata = None
    if fps_hint_env:
        try:
            video_metadata = types.VideoMetadata(fps=float(fps_hint_env))
        except (ValueError, TypeError):
            video_metadata = None

    if video_data_url:
        data, mime = _decode_data_url(video_data_url)
        if video_metadata is not None:
            user_parts.append(
                types.Part(
                    inline_data=types.Blob(data=data, mime_type=mime),
                    video_metadata=video_metadata,
                )
            )
        else:
            user_parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    if frame_data_urls:
        for frame_url in frame_data_urls:
            data, mime = _decode_data_url(frame_url)
            user_parts.append(types.Part.from_bytes(data=data, mime_type=mime))

    if enable_thinking:
        # budget = -1 → dynamic thinking (model decides). Larger = more thinking.
        thinking_config = types.ThinkingConfig(thinking_budget=-1)
    else:
        # explicit 0 to disable thinking (None defaults to dynamic on some models)
        thinking_config = types.ThinkingConfig(thinking_budget=0)
    budget_override = os.getenv("CLAW_GEMINI_THINKING_BUDGET", "").strip()
    if budget_override:
        try:
            thinking_config = types.ThinkingConfig(thinking_budget=int(budget_override))
        except ValueError:
            pass

    config = types.GenerateContentConfig(
        system_instruction=system_text or None,
        response_mime_type=response_mime_type,
        max_output_tokens=max_output_tokens,
        thinking_config=thinking_config,
        temperature=0.0,
    )

    response = None
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_parts,
                config=config,
            )
            break
        except Exception as exc:
            message = str(exc)
            is_overloaded = (
                "503" in message
                or "UNAVAILABLE" in message
                or "high demand" in message
                or "429" in message
                or "RESOURCE_EXHAUSTED" in message
                or "Server disconnected" in message
                or "RemoteProtocolError" in type(exc).__name__
                or "500" in message and "INTERNAL" in message
                or "524" in message  # Cloudflare origin timeout
                or "502" in message  # bad gateway
                or "504" in message  # gateway timeout
                or "Read timed out" in message
                or "ReadTimeout" in type(exc).__name__
            )
            last_exc = exc
            if not is_overloaded or attempt == max_retries - 1:
                raise
            delay = retry_base_delay_sec * (2 ** attempt)
            time.sleep(min(delay, 120.0))
    if response is None:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Gemini generate_content returned no response")
    text = response.text
    if not text:
        # fall back to first candidate text part
        try:
            parts = response.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") or "" for p in parts)
        except Exception:
            text = ""
    text = text or ""
    debug_dir = os.getenv("CLAW_GEMINI_DEBUG_DIR", "").strip()
    if debug_dir:
        try:
            finish_reason = "?"
            usage = {}
            try:
                finish_reason = str(response.candidates[0].finish_reason)
            except Exception:
                pass
            try:
                um = response.usage_metadata
                usage = {
                    "prompt_tokens": getattr(um, "prompt_token_count", None),
                    "candidates_tokens": getattr(um, "candidates_token_count", None),
                    "total_tokens": getattr(um, "total_token_count", None),
                    "thoughts_tokens": getattr(um, "thoughts_token_count", None),
                }
            except Exception:
                pass
            ts = time.strftime("%Y%m%d_%H%M%S")
            ddir = __import__("pathlib").Path(debug_dir)
            ddir.mkdir(parents=True, exist_ok=True)
            (ddir / f"resp_{ts}_{model}.txt").write_text(text, encoding="utf-8")
            (ddir / f"resp_{ts}_{model}.meta.json").write_text(
                __import__("json").dumps({
                    "finish_reason": finish_reason,
                    "text_chars": len(text),
                    "max_output_tokens": max_output_tokens,
                    "usage": usage,
                }, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
    return text


__all__ = [
    "DEFAULT_GEMINI_NATIVE_BASE_URL",
    "DEFAULT_GEMINI_NATIVE_MODEL",
    "build_native_gemini_client",
    "run_native_gemini_inference",
]
