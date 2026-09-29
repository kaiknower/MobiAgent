"""Single point of env construction.

`make_env('mock')`  → in-process MockEnv (cheap, dev-time)
`make_env('omni')`  → real OmniGibsonEnv (heavy; lazy-loads on first reset)
"""
from __future__ import annotations

from typing import Any

from .env_protocol import EnvProtocol


def make_env(kind: str, **kwargs: Any) -> EnvProtocol:
    if kind == "mock":
        from .env_mock import MockEnv
        return MockEnv(**kwargs)  # type: ignore[return-value]
    if kind == "omni":
        from .env_omni import OmniGibsonEnv
        return OmniGibsonEnv(**kwargs)  # type: ignore[return-value]
    if kind == "robocasa":
        from mobiagent.environments.robocasa import RoboCasaEnv
        return RoboCasaEnv(**kwargs)
    raise ValueError(f"unknown env kind: {kind!r}; valid: mock, omni")


__all__ = ["make_env"]
