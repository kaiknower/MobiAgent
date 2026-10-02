import json
import os
import time
from typing import Any
import urllib.error
import urllib.request


from mobiagent.api import build_openai_client, get_openai_model, get_openai_settings


DEFAULT_DASHSCOPE_BASE_URL = "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DASHSCOPE_MODEL = "qwen3.6-plus-2026-04-02"
_LEGACY_QWEN_MODEL = "qwen3-vl-plus"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_HTTP_RETRY_DELAYS_SEC = (1.0, 2.0, 4.0)


def _get_dashscope_settings() -> tuple[str, str, str]:
    api_key = os.getenv("DASHSCOPE_API_KEY", "") or os.getenv("ALIBABA_API_KEY", "")
    if api_key == "":
        raise ValueError("DASHSCOPE_API_KEY or ALIBABA_API_KEY must be set to a non-empty value")

    base_url = (
        os.getenv("DASHSCOPE_BASE_URL", "")
        or os.getenv("ALIBABA_BASE_URL", "")
        or DEFAULT_DASHSCOPE_BASE_URL
    ).rstrip("/")
    model_name = os.getenv("DASHSCOPE_MODEL", "") or os.getenv("ALIBABA_MODEL", "") or DEFAULT_DASHSCOPE_MODEL
    if model_name == _LEGACY_QWEN_MODEL:
        model_name = DEFAULT_DASHSCOPE_MODEL
    return base_url, model_name, api_key


def _execute_chat_completion_via_rest(request: dict[str, Any]) -> dict[str, Any]:
    api_key, base_url = get_openai_settings()
    request = {**request, "model": get_openai_model(request.get("model"))}
    url = f"{base_url}/chat/completions"
    payload = json.dumps(request).encode("utf-8")
    last_error: Exception | None = None
    for attempt, delay_sec in enumerate((0.0, *_HTTP_RETRY_DELAYS_SEC)):
        if attempt > 0:
            time.sleep(delay_sec)
        http_request = urllib.request.Request(
            url=url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8")
            except Exception:
                error_body = ""
            detail = f"{exc.code} {exc.reason}"
            if error_body:
                detail = f"{detail} body={error_body}"
            raise RuntimeError(f"GPT API request failed: {detail}") from exc
        except urllib.error.URLError as exc:
            last_error = exc
            continue
    assert last_error is not None
    raise last_error


def _execute_chat_completion_via_dashscope_rest(request: dict[str, Any]) -> dict[str, Any]:
    base_url, model_name, api_key = _get_dashscope_settings()
    payload_request = dict(request)
    payload_request["model"] = model_name
    url = f"{base_url}/chat/completions"
    payload = json.dumps(payload_request).encode("utf-8")
    last_error: Exception | None = None
    for attempt, delay_sec in enumerate((0.0, *_HTTP_RETRY_DELAYS_SEC)):
        if attempt > 0:
            time.sleep(delay_sec)
        http_request = urllib.request.Request(
            url=url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8")
            except Exception:
                error_body = ""
            detail = f"{exc.code} {exc.reason}"
            if error_body:
                detail = f"{detail} body={error_body}"
            raise RuntimeError(f"DashScope REST request failed: {detail}") from exc
        except urllib.error.URLError as exc:
            last_error = exc
            continue
    assert last_error is not None
    raise last_error


def _get_gemini_settings() -> tuple[str, str, str]:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key == "":
        raise ValueError("GEMINI_API_KEY must be set to a non-empty value")
    base_url = os.getenv("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL).rstrip("/")
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    return base_url, model_name, api_key


def _execute_chat_completion_via_gemini_rest(request: dict[str, Any]) -> dict[str, Any]:
    base_url, model_name, api_key = _get_gemini_settings()
    payload_request = dict(request)
    # Respect an explicit gemini-* model in the request; otherwise use env default.
    if not str(payload_request.get("model", "")).startswith("gemini"):
        payload_request["model"] = model_name
    url = f"{base_url}/chat/completions"
    payload = json.dumps(payload_request).encode("utf-8")
    last_error: Exception | None = None
    for attempt, delay_sec in enumerate((0.0, *_HTTP_RETRY_DELAYS_SEC)):
        if attempt > 0:
            time.sleep(delay_sec)
        http_request = urllib.request.Request(
            url=url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8")
            except Exception:
                error_body = ""
            detail = f"{exc.code} {exc.reason}"
            if error_body:
                detail = f"{detail} body={error_body}"
            raise RuntimeError(f"Gemini REST request failed: {detail}") from exc
        except urllib.error.URLError as exc:
            last_error = exc
            continue
    assert last_error is not None
    raise last_error


def build_chat_completion_request(
    model: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int,
    **extra_fields: Any,
) -> dict[str, Any]:
    request = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    request.update(extra_fields)
    return request


def extract_first_message_text(response: dict[str, Any]) -> str:
    return response["choices"][0]["message"]["content"]


def execute_chat_completion(request: dict[str, Any], client: Any | None = None) -> dict[str, Any]:
    return execute_chat_completion_with_provider(request, client=client, provider="auto")


def execute_chat_completion_with_provider(
    request: dict[str, Any],
    client: Any | None = None,
    provider: str = "auto",
) -> dict[str, Any]:
    if provider not in {"auto", "openai", "gemini", "dashscope"}:
        raise ValueError(f"Unsupported API provider: {provider!r}")
    if client is None:
        if provider == "gemini":
            return _execute_chat_completion_via_gemini_rest(request)
        if provider == "dashscope":
            return _execute_chat_completion_via_dashscope_rest(request)
        request_model = str(request.get("model", ""))
        if provider == "auto":
            if request_model.startswith("gemini") and os.getenv("GEMINI_API_KEY", ""):
                return _execute_chat_completion_via_gemini_rest(request)
            if request_model.startswith("qwen") and (
                os.getenv("DASHSCOPE_API_KEY", "") or os.getenv("ALIBABA_API_KEY", "")
            ):
                return _execute_chat_completion_via_dashscope_rest(request)
            if not request_model.startswith("gpt") and not os.getenv("OPENAI_API_KEY", ""):
                if os.getenv("DASHSCOPE_API_KEY", "") or os.getenv("ALIBABA_API_KEY", ""):
                    return _execute_chat_completion_via_dashscope_rest(request)
        request = {**request, "model": get_openai_model(request.get("model"))}
        try:
            chat_client = build_openai_client()
        except ImportError:
            return _execute_chat_completion_via_rest(request)
    else:
        chat_client = client

    response = chat_client.chat.completions.create(**request)
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response


def parse_chat_completion_response(response: dict[str, Any]) -> str:
    return extract_first_message_text(response)


__all__ = [
    "DEFAULT_GEMINI_BASE_URL",
    "DEFAULT_GEMINI_MODEL",
    "build_chat_completion_request",
    "build_openai_client",
    "execute_chat_completion",
    "execute_chat_completion_with_provider",
    "extract_first_message_text",
    "parse_chat_completion_response",
]
