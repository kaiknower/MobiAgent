import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


BASE_QVEL = slice(253, 256)
TRUNK_QPOS = slice(236, 240)
LEFT_ARM_QPOS = slice(158, 165)
LEFT_GRIPPER_QPOS = slice(193, 195)
RIGHT_ARM_QPOS = slice(197, 204)
RIGHT_GRIPPER_QPOS = slice(232, 234)
MAX_GRIPPER_WIDTH = 0.1


def make_behavior_example() -> dict:
    return {
        "observation/head_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(256),
        "prompt": "do something",
    }


def extract_behavior_state(proprio_data: np.ndarray) -> np.ndarray:
    proprio = np.asarray(proprio_data, dtype=np.float32)
    left_gripper = 2.0 * (proprio[..., LEFT_GRIPPER_QPOS].sum(axis=-1, keepdims=True) / MAX_GRIPPER_WIDTH) - 1.0
    right_gripper = 2.0 * (proprio[..., RIGHT_GRIPPER_QPOS].sum(axis=-1, keepdims=True) / MAX_GRIPPER_WIDTH) - 1.0
    return np.concatenate(
        [
            proprio[..., BASE_QVEL],
            proprio[..., TRUNK_QPOS],
            proprio[..., LEFT_ARM_QPOS],
            left_gripper,
            proprio[..., RIGHT_ARM_QPOS],
            right_gripper,
        ],
        axis=-1,
    )


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
class BehaviorInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = extract_behavior_state(_get_any(data, "observation/state", "observation.state"))
        base_image = _parse_image(
            _get_any(data, "observation/head_image", "observation/egocentric_camera", "observation.images.rgb.head")
        )
        left_wrist_image = _parse_image(
            _get_any(
                data,
                "observation/left_wrist_image",
                "observation/wrist_image_left",
                "observation.images.rgb.left_wrist",
            )
        )
        right_wrist_image = _parse_image(
            _get_any(
                data,
                "observation/right_wrist_image",
                "observation/wrist_image_right",
                "observation.images.rgb.right_wrist",
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
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BehaviorOutputs(transforms.DataTransformFn):
    action_dim: int = 23

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
