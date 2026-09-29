from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


STATE_SCHEMA: dict[str, tuple[int, int]] = {
    "robot_pos": (140, 143),
    "robot_ori_cos": (143, 146),
    "robot_ori_sin": (146, 149),
    "arm_left_qpos": (158, 165),
    "eef_left_pos": (186, 189),
    "eef_left_quat": (189, 193),
    "gripper_left_qpos": (193, 195),
    "arm_right_qpos": (197, 204),
    "eef_right_pos": (225, 228),
    "eef_right_quat": (228, 232),
    "gripper_right_qpos": (232, 234),
    "trunk_qpos": (236, 240),
    "base_vel": (253, 256),
}

STATE_MIN_LENGTH = max(end for _, end in STATE_SCHEMA.values())
TASK_INFO_MIN_LENGTH = 82

_STATE_SCHEMA_DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("robot_pos", "robot base position"),
    ("robot_ori_cos", "robot orientation encoded as cosine components"),
    ("robot_ori_sin", "robot orientation encoded as sine components"),
    ("arm_left_qpos", "left arm joint positions"),
    ("eef_left_pos", "left end-effector position"),
    ("eef_left_quat", "left end-effector orientation quaternion"),
    ("gripper_left_qpos", "left gripper joint positions"),
    ("arm_right_qpos", "right arm joint positions"),
    ("eef_right_pos", "right end-effector position"),
    ("eef_right_quat", "right end-effector orientation quaternion"),
    ("gripper_right_qpos", "right gripper joint positions"),
    ("trunk_qpos", "trunk joint positions"),
    ("base_vel", "robot base velocity"),
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


def summarize_state_row(state: Sequence[Any] | np.ndarray, task_info: Sequence[Any] | np.ndarray) -> dict[str, Any]:
    state_array = _validate_1d_length("state", state, STATE_MIN_LENGTH)
    task_info_array = _validate_1d_length("task_info", task_info, TASK_INFO_MIN_LENGTH)

    summary: dict[str, Any] = {
        "task_info_full": task_info_array.tolist(),
    }

    for field_name, (start, end) in STATE_SCHEMA.items():
        summary[field_name] = _to_list(state_array[start:end])

    summary["gripper_left"] = float(np.sum(state_array[slice(*STATE_SCHEMA["gripper_left_qpos"])], dtype=np.float32))
    summary["gripper_right"] = float(np.sum(state_array[slice(*STATE_SCHEMA["gripper_right_qpos"])], dtype=np.float32))
    return summary


def build_state_schema_text() -> str:
    lines = [
        "Preserve these semantic state slices exactly as provided:",
    ]
    for field_name, description in _STATE_SCHEMA_DESCRIPTIONS:
        start, end = STATE_SCHEMA[field_name]
        lines.append(f"- {field_name} [{start}:{end}]: {description}")
    lines.append("- gripper_left: scalar sum of gripper_left_qpos")
    lines.append("- gripper_right: scalar sum of gripper_right_qpos")
    lines.append("- task_info_full: preserve the full task_info payload without truncation or normalization")
    return "\n".join(lines)
