"""State summarization for s1_mobile cartesian SO3 vectors (length 34).

The 34-dim cartesian SO3 command/state vector is organized as:
    torso           [0 : 9]   — torso SO3 pose
    left_arm        [9 :18]   — left arm SO3 pose
    left_gripper    [18:19]   — scalar
    right_arm       [19:28]   — right arm SO3 pose
    right_gripper   [28:29]   — scalar
    head            [29:31]   — yaw, pitch
    chassis         [31:34]   — base motion command/state

This is aligned with the `names` list in meta/info.json under
`cartesian_so3_dict.cartesian_pose_command`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .config import ACTION_DIM, ACTION_GROUP_SLICES


STATE_SCHEMA: dict[str, tuple[int, int]] = {
    name: (slc.start, slc.stop) for name, slc in ACTION_GROUP_SLICES
}

STATE_MIN_LENGTH = ACTION_DIM

_STATE_SCHEMA_DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("torso", "torso SO3 pose (9 values: 3D position + 3x3 rotation row-major or similar)"),
    ("left_arm", "left arm SO3 pose (9 values)"),
    ("left_gripper", "left gripper scalar command/state"),
    ("right_arm", "right arm SO3 pose (9 values)"),
    ("right_gripper", "right gripper scalar command/state"),
    ("head", "head yaw, pitch"),
    ("chassis", "base chassis motion (x, y, yaw or similar)"),
)


def _to_list(values: Sequence[Any] | np.ndarray) -> list[Any]:
    return np.asarray(values).tolist()


def _validate_1d_length(name: str, values: Sequence[Any] | np.ndarray, min_length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence; got shape {array.shape}")
    if array.shape[0] < min_length:
        raise ValueError(f"{name} must have at least {min_length} elements; got {array.shape[0]}")
    return array


def summarize_state_row(state: Sequence[Any] | np.ndarray) -> dict[str, Any]:
    """Split a single 34-dim cartesian SO3 vector into named semantic slices.

    `state` may be either the command (action) or the observed state — both
    share the same layout. The optional `task_info` argument that the original
    behavior_224_rgb skill required does not apply here; s1_mobile's per-frame
    task tag is a small int (task_index) carried in the parquet directly.
    """
    state_array = _validate_1d_length("state", state, STATE_MIN_LENGTH)

    summary: dict[str, Any] = {}
    for field_name, (start, end) in STATE_SCHEMA.items():
        summary[field_name] = _to_list(state_array[start:end])

    summary["left_gripper_scalar"] = float(state_array[STATE_SCHEMA["left_gripper"][0]])
    summary["right_gripper_scalar"] = float(state_array[STATE_SCHEMA["right_gripper"][0]])
    return summary


def build_state_schema_text() -> str:
    lines = [
        "Preserve these semantic state slices exactly as provided"
        " (cartesian SO3 representation, length 34):",
    ]
    for field_name, description in _STATE_SCHEMA_DESCRIPTIONS:
        start, end = STATE_SCHEMA[field_name]
        lines.append(f"- {field_name} [{start}:{end}]: {description}")
    lines.append("- left_gripper_scalar / right_gripper_scalar: scalar gripper values")
    return "\n".join(lines)
