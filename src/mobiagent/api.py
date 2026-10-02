"""User-configured GPT API connection shared by execution and skill discovery."""
from __future__ import annotations

import os
from typing import Any


def get_openai_settings() -> tuple[str, str]:
    """Read the API key and API base URL supplied by the user."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set; see README.md#api-configuration")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL must be set; see README.md#api-configuration")
    return api_key, base_url


def get_openai_model(model: str | None = None) -> str:
    """Resolve an explicit model or the user's OPENAI_MODEL setting."""
    name = (model or os.getenv("OPENAI_MODEL", "")).strip()
    if not name:
        raise ValueError("OPENAI_MODEL must be set; see README.md#api-configuration")
    return name


def build_openai_client(**options: Any) -> Any:
    """Construct the GPT SDK client at the API integration point."""
    api_key, base_url = get_openai_settings()
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url, **options)
