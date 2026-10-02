"""Azure OpenAI client for MobiAgent planning and visual reflection.

Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT
in the environment. AZURE_OPENAI_API_VERSION selects the Azure API version.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any

DEFAULT_API_VERSION = "2024-12-01-preview"
DEFAULT_ENDPOINT = ""
DEFAULT_DEPLOYMENT = ""


_CLIENT_SINGLETON: Any = None
# Set False once a deployment is observed to reject the `temperature` arg.
_TEMPERATURE_SUPPORTED: bool = True


def build_azure_client() -> Any:
    """Reuse an Azure client and connection pool across planner and critic calls."""
    global _CLIENT_SINGLETON
    if _CLIENT_SINGLETON is not None:
        return _CLIENT_SINGLETON

    from openai import AzureOpenAI
    import httpx
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY must be set")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", DEFAULT_ENDPOINT).strip()
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT must be set; see README.md#api-configuration")

    http_client = httpx.Client(
        # Wide keepalive pool — many idle connections kept warm so a chunk
        # request never has to handshake from cold.
        limits=httpx.Limits(
            max_keepalive_connections=20,
            max_connections=40,
            keepalive_expiry=300.0,
        ),
        # Generous read timeout — planner/judge calls on GPT-5.4 can take
        # 10-30 s legitimately. We don't want short read timeouts to surface
        # as "Connection error".
        timeout=httpx.Timeout(connect=15.0, read=180.0, write=15.0, pool=15.0),
        # Disable httpx-level retries; we control retry policy explicitly.
        transport=httpx.HTTPTransport(retries=0),
    )

    _CLIENT_SINGLETON = AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
        azure_endpoint=endpoint,
        api_key=api_key,
        http_client=http_client,
        max_retries=0,  # SDK-level retries off; we use the loop below.
    )
    return _CLIENT_SINGLETON


def _encode_image_data_url(arr: Any) -> str | None:
    """numpy HWC uint8 image -> data:image/jpeg;base64 URL string for OpenAI."""
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
    """Build an OpenAI multimodal user 'content' array.

    images: list of numpy HWC uint8 arrays (or None entries to skip).
    Returns the content list compatible with chat.completions.create messages[user].content.
    """
    content: list[dict[str, Any]] = []
    if images:
        for arr in images:
            url = _encode_image_data_url(arr)
            if url is not None:
                content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": text})
    return content


def _is_transient_exc(exc: Exception) -> bool:
    """Decide if an Azure OpenAI exception is worth retrying.

    Uses BOTH exception-class detection (preferred — covers exceptions whose
    `str()` doesn't include the HTTP code) AND string heuristics (fallback for
    older SDK versions / wrapped exceptions).

    Transient cases worth retrying:
      - APIConnectionError       — TCP / TLS connection blip (DNS, reset, etc.)
      - APITimeoutError          — server didn't respond in time
      - RateLimitError           — 429
      - InternalServerError      — 500/502/503 from upstream
      - Any APIStatusError with 5xx status
      - Generic OSError / TimeoutError from the transport layer
    """
    # Exception-class detection (most reliable on openai>=1.x).
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

    # Transport-level fallbacks (httpx / urllib3 / socket).
    if isinstance(exc, (OSError, TimeoutError)):
        return True

    # String heuristics — last-resort match for older SDKs and stringified errors.
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

    Connection stability is achieved primarily via the singleton client +
    keepalive pool in `build_azure_client()`. The retry loop is a thin
    safety net: ONE fast retry on transient errors (connection blip, 5xx,
    429, timeout). No exponential backoff — if the singleton is healthy a
    second attempt usually succeeds within ~2 s; if not, the underlying
    issue is bigger than retries can paper over.

    On a transient retry, the singleton client is INVALIDATED so the next
    call rebuilds the httpx pool — useful when an Azure LB instance dropped
    the keepalive connections behind it.

    Returns the parsed JSON object from the model's reply.
    """
    deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", DEFAULT_DEPLOYMENT)
    if not deployment.strip():
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT must be set; use the deployment name from your Azure resource")

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_content},
    ]

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            client = build_azure_client()
            # temperature=0 → greedy/deterministic: the SAME frames must give the
            # SAME verdict (no sampling jitter). Some reasoning-style deployments
            # reject `temperature`; if so, drop it for the rest of the session.
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
            # Invalidate the singleton — a wedged keepalive pool stays wedged
            # if we keep reusing it. Next build_azure_client() creates a fresh
            # httpx Client.
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
