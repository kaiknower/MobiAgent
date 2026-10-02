"""Pure-Python simulation environment for offline usage examples.

Returns synthetic observations and can mark a task successful after a
configurable number of steps."""
from __future__ import annotations

from typing import Any

import numpy as np


class MockEnv:
    """Minimal EnvProtocol implementation.

    - reset(task): returns synthetic obs, resets step counter
    - step(action): increments step counter, returns synthetic obs
    - get_camera_frame: returns 224x224x3 black image (or color-coded by camera)
    - is_success: True once `success_after_steps` reached
    """

    CAMERA_TINT = {
        "head":        (10, 10, 50),    # navy-ish so it's distinguishable in dumps
        "left_wrist":  (10, 50, 10),
        "right_wrist": (50, 10, 10),
    }

    def __init__(self, *, success_after_steps: int = 30, image_hw: tuple[int, int] = (224, 224)) -> None:
        self.success_after_steps = success_after_steps
        self.image_hw = image_hw
        self.steps = 0
        self._task: str | None = None

    def reset(self, task: str) -> dict[str, Any]:
        self.steps = 0
        self._task = task
        return self._observation()

    def step(self, action: Any) -> dict[str, Any]:
        self.steps += 1
        return self._observation()

    def get_camera_frame(self, camera: str = "head") -> np.ndarray:
        h, w = self.image_hw
        tint = self.CAMERA_TINT.get(camera, (32, 32, 32))
        img = np.full((h, w, 3), tint, dtype=np.uint8)
        return img

    def is_success(self) -> bool:
        return self.steps >= self.success_after_steps

    def close(self) -> None:
        pass

    def _observation(self) -> dict[str, Any]:
        # Deployment-schema keys so MockEnv obs flows through the policy_client
        # without any extra remap. Also keep the legacy unprefixed names as
        # aliases for backwards-compat with older test harnesses.
        head  = self.get_camera_frame("head")
        left  = self.get_camera_frame("left_wrist")
        right = self.get_camera_frame("right_wrist")
        state = np.zeros(256, dtype=np.float32)
        return {
            "task": self._task,
            "step": self.steps,
            "observation/head_image":        head,
            "observation/left_wrist_image":  left,
            "observation/right_wrist_image": right,
            "observation/state":             state,
            # legacy aliases
            "head_image":        head,
            "left_wrist_image":  left,
            "right_wrist_image": right,
            "robot_state":       {"base_pose": [0.0, 0.0, 0.0]},
        }
