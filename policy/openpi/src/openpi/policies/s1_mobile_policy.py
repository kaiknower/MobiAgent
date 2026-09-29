"""Data transforms for the s1_mobile (LeRobot v2.0) skill_segments_v2 plan.

Source schema (per the s1_mobile lerobot_so3_data_30hz parquets):
- ``cartesian_so3_dict.cartesian_pose_state``   : (T, 34) EE-pose state
- ``cartesian_so3_dict.cartesian_pose_command`` : (T, 34) EE-pose target

34-dim cartesian layout (verified from info.json `names`):
    idx 0-8    torso        (9)
    idx 9-17   left_arm     (9)
    idx 18     left_gripper (1)
    idx 19-27  right_arm    (9)
    idx 28     right_gripper(1)
    idx 29     head_yaw      ← REMOVED
    idx 30     head_pitch    ← REMOVED
    idx 31-33  chassis      (3) — base; delta processing handled in
                                  DeltaActions outside this file; state is
                                  zeroed here so the model cannot rely on
                                  absolute base position.

After head removal the 32-dim layout is:
    idx 0-8 torso, 9-17 left_arm, 18 left_grip, 19-27 right_arm,
    idx 28 right_grip, 29-31 chassis  (CHASSIS_DIMS_POST below).
"""
from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# Indices on the 34-dim cartesian_pose_(state|command).
KEEP_DIMS: list[int] = list(range(0, 29)) + list(range(31, 34))   # 32 dims; skips head 29:31
CHASSIS_DIMS_POST: tuple[int, int, int] = (29, 30, 31)            # chassis indices AFTER head removal


def extract_s1_mobile_state(proprio: np.ndarray) -> np.ndarray:
    """Project the 34-dim cartesian_pose_state to the 32-dim s1 state.

    - Removes the head dims (29 yaw, 30 pitch).
    - Does NOT zero chassis here — DeltaActionsSO3 needs the raw chassis to
      compute action chassis delta correctly. Chassis is zeroed in a separate
      transform AFTER DeltaActionsSO3 (see _ZeroChassisInState below).
    """
    return np.asarray(proprio, dtype=np.float32)[..., KEEP_DIMS]


def extract_s1_mobile_action(action: np.ndarray) -> np.ndarray:
    """Project the 34-dim cartesian_pose_command to the 32-dim s1 action.

    Same head-removal slice as state; no chassis zeroing (action chassis is
    used by DeltaActions to learn relative base motion).
    """
    return np.asarray(action, dtype=np.float32)[..., KEEP_DIMS]


def make_s1_mobile_example() -> dict:
    """Minimal stub example for policy unit tests / smoke runs."""
    return {
        "observation/head_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(34).astype(np.float32),
        "prompt": "move to the bottle",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _get_any(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing keys: {keys}")


@dataclasses.dataclass(frozen=True)
class S1MobileInputs(transforms.DataTransformFn):
    """Per-sample input transform for s1_mobile cartesian-pose training.

    Reads the 34-dim ``cartesian_pose_(state|command)`` from the dataset row,
    removes the two head dims (29 yaw, 30 pitch) to produce 32-dim state and
    action, and zeros chassis in state. Image keys mirror the SkillSegmentDataset
    output (``observation/{head,left_wrist,right_wrist}_image``) with fallbacks
    to the raw lerobot keys for serving-time compatibility.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = extract_s1_mobile_state(_get_any(data, "observation/state", "observation.state"))

        base_image = _parse_image(
            _get_any(
                data,
                "observation/head_image",
                "observation/cam_high",
                "observation.images.rgb.head",
                "observation.images.head",
            )
        )
        left_wrist_image = _parse_image(
            _get_any(
                data,
                "observation/left_wrist_image",
                "observation/cam_left_wrist",
                "observation.images.rgb.left_wrist",
                "observation.images.left_wrist",
            )
        )
        right_wrist_image = _parse_image(
            _get_any(
                data,
                "observation/right_wrist_image",
                "observation/cam_right_wrist",
                "observation.images.rgb.right_wrist",
                "observation.images.right_wrist",
            )
        )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = extract_s1_mobile_action(data["actions"])  # (H, 32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        # Forward the head-id so the Pi0SixHead router dispatches to the right
        # expert (must end up as observation.skill_canonical_ids on the model).
        if "skill_canonical_id" in data:
            inputs["skill_canonical_ids"] = np.asarray(data["skill_canonical_id"], dtype=np.int32)

        return inputs


@dataclasses.dataclass(frozen=True)
class S1MobileOutputs(transforms.DataTransformFn):
    """No-op for now — the model already outputs 32-dim, matching the target.

    Kept as a dataclass for parity with other policy modules and so the
    SkillSegmentsDataConfig output chain can drop in a serving-side bridge
    later without changing the factory plumbing.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])}
