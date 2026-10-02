"""Shared simulator interface for mock, BEHAVIOR and RoboCasa environments.

The orchestrator uses this protocol through env_factory.make_env.
"""
from __future__ import annotations

from typing import Any, Protocol


class EnvProtocol(Protocol):
    def reset(self, task: str) -> dict[str, Any]:
        """Reset env to a clean starting state for `task` (e.g. 'task-0001')."""
        ...

    def step(self, action: Any) -> dict[str, Any]:
        """Apply action chunk; return new observation dict."""
        ...

    def get_camera_frame(self, camera: str = "head") -> Any:
        """Return current frame from a named camera (head/left_wrist/right_wrist)."""
        ...

    def is_success(self) -> bool:
        """True iff the task's BDDL goal is satisfied."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...
