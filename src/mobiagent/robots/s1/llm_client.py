"""Azure OpenAI client used by both the planner and the judge.

Required env (set in your shell or via launcher):
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT          # e.g. https://YOUR-RESOURCE.openai.azure.com/
    AZURE_OPENAI_API_VERSION       # default: 2024-12-01-preview
    AZURE_OPENAI_DEPLOYMENT        # default: gpt-5.4

A singleton AzureOpenAI client is reused process-wide with a wide
keepalive pool so the LLM round-trip avoids fresh TCP/TLS handshakes.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any

DEFAULT_API_VERSION = "2024-12-01-preview"
DEFAULT_DEPLOYMENT = "gpt-5.4"


_CLIENT_SINGLETON: Any = None
_TEMPERATURE_SUPPORTED: bool = True


def build_azure_client() -> Any:
    global _CLIENT_SINGLETON
    if _CLIENT_SINGLETON is not None:
        return _CLIENT_SINGLETON

    from openai import AzureOpenAI
    import httpx
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY must be set")
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT must be set")

    http_client = httpx.Client(
        limits=httpx.Limits(
            max_keepalive_connections=20,
            max_connections=40,
            keepalive_expiry=300.0,
        ),
        timeout=httpx.Timeout(connect=15.0, read=180.0, write=15.0, pool=15.0),
        transport=httpx.HTTPTransport(retries=0),
    )

    _CLIENT_SINGLETON = AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
        azure_endpoint=endpoint,
        api_key=api_key,
        http_client=http_client,
        max_retries=0,
    )
    return _CLIENT_SINGLETON


def _encode_image_data_url(arr: Any) -> str | None:
    """numpy HWC uint8 -> data:image/jpeg;base64 URL."""
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return None
    if arr is None:
        return None
    try:
        img = Image.fromarray(np.asarray(arr).astype("uint8"))
    except Exception:
        return None
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def make_user_content(*, text: str, images: list[Any] | None = None) -> list[dict[str, Any]]:
    """Build an OpenAI multimodal user-content array."""
    content: list[dict[str, Any]] = []
    if images:
        for arr in images:
            url = _encode_image_data_url(arr)
            if url is not None:
                content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": text})
    return content


def _is_transient_exc(exc: Exception) -> bool:
    try:
        from openai import (
            APIConnectionError, APITimeoutError, RateLimitError,
            InternalServerError, APIStatusError,
        )
        if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)):
            return True
        if isinstance(exc, APIStatusError):
            status = getattr(exc, "status_code", None)
            if status is not None and 500 <= int(status) < 600:
                return True
            if status == 429:
                return True
    except Exception:
        pass

    if isinstance(exc, (OSError, TimeoutError)):
        return True

    msg = str(exc).lower()
    keywords = (
        "connection error", "connection reset", "connection refused",
        "connection aborted", "remote disconnected",
        "503", "502", "500", "429", "504",
        "unavailable", "overload", "overloaded",
        "rate limit", "timeout", "timed out",
        "name or service not known", "temporary failure",
    )
    return any(k in msg for k in keywords)


def chat_completion_json(
    *,
    system_text: str,
    user_content: list[dict[str, Any]],
    deployment: str | None = None,
    max_completion_tokens: int = 16384,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Call Azure OpenAI chat completion with JSON response_format.

    One fast retry on transient errors. The singleton client is invalidated
    between retries so the next call rebuilds the httpx pool.
    """
    deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", DEFAULT_DEPLOYMENT)

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_content},
    ]

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            client = build_azure_client()
            create_kwargs: dict[str, Any] = dict(
                model=deployment,
                messages=messages,
                response_format={"type": "json_object"},
                max_completion_tokens=max_completion_tokens,
            )
            global _TEMPERATURE_SUPPORTED
            if _TEMPERATURE_SUPPORTED:
                create_kwargs["temperature"] = 0
            try:
                resp = client.chat.completions.create(**create_kwargs)
            except Exception as exc_t:
                if _TEMPERATURE_SUPPORTED and "temperature" in str(exc_t).lower():
                    _TEMPERATURE_SUPPORTED = False
                    create_kwargs.pop("temperature", None)
                    print("  llm: deployment rejects `temperature`; "
                          "dropping it for the rest of the session", flush=True)
                    resp = client.chat.completions.create(**create_kwargs)
                else:
                    raise
            text = resp.choices[0].message.content or ""
            return json.loads(text)
        except Exception as exc:
            last_exc = exc
            transient = _is_transient_exc(exc)
            if not transient or attempt == max_attempts - 1:
                raise
            global _CLIENT_SINGLETON
            try:
                if _CLIENT_SINGLETON is not None:
                    _CLIENT_SINGLETON._client.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            _CLIENT_SINGLETON = None
            print(f"  llm attempt {attempt+1}/{max_attempts} got {type(exc).__name__} "
                  f"({str(exc)[:80]!r}); rebuilding client + retrying once", flush=True)
            time.sleep(2.0)
    raise last_exc  # type: ignore[misc]


__all__ = [
    "build_azure_client",
    "chat_completion_json",
    "make_user_content",
]
