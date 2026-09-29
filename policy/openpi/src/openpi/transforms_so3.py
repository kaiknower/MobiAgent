"""SO(3)-aware delta/absolute action transforms for s1_mobile.

These mirror the reference implementation in ``S1/transforms.py``. The s1_mobile
post-head-removal 32-dim action layout has 6D-rotation blocks inside torso/arm
groups; naive elementwise subtraction is mathematically wrong on those dims.

The block ``structure`` argument encodes alternating groups:

  positive N (e.g. +9): a 9-dim torso/arm block laid out as
      ``[xyz (3), so3_6d (6)]``. xyz uses direct subtraction; the 6D rotation
      uses matrix-domain delta ``delta_mat = mat_act @ mat_state^T`` and
      then projects back to 6D via the first two rows of the rotation
      matrix.

  negative N (e.g. -1, -3): a per-dim block where each entry is treated
      independently — subtract iff ``mask[idx]`` is True, otherwise leave
      the value as absolute. Use this for grippers / chassis dims.

Only ``DeltaActionsSO3``, ``AbsoluteActionsSO3``, ``so3_6d_to_matrix`` and
``matrix_to_so3_6d`` live here — by design, the rest of openpi.transforms is
left untouched. The implementations are copied verbatim from the S1 reference
(``the original S1 transforms module``).
"""

from collections.abc import Sequence
import dataclasses

import numpy as np

from openpi.shared import array_typing as at
from openpi import transforms as _transforms

DataDict = at.PyTree
DataTransformFn = _transforms.DataTransformFn


def so3_6d_to_matrix(x: np.ndarray) -> np.ndarray:
    """Convert 6D SO3 representation to 3x3 rotation matrix.

    The 6D representation stores the first two rows of the rotation matrix.
    Args:
        x: (..., 6) array — first two rows of the rotation matrix.
    Returns:
        (..., 3, 3) rotation matrix.
    """
    a1 = x[..., :3]
    a2 = x[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - (np.sum(b1 * a2, axis=-1, keepdims=True)) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)  # (..., 3, 3), rows = [b1, b2, b3]


def matrix_to_so3_6d(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D SO3 representation.

    Args:
        R: (..., 3, 3) rotation matrix.
    Returns:
        (..., 6) array — first two rows of R flattened.
    """
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


@dataclasses.dataclass(frozen=True)
class DeltaActionsSO3(DataTransformFn):
    """SO(3)-aware delta action transform for block-structured layouts.

    For block structure ``[9, 9, -1, 9, -1, -3]`` the 32-dim s1_mobile layout
    is interpreted as:
      idx 0-8   torso       (+9 → xyz delta + 6D-rotation delta)
      idx 9-17  left_arm    (+9 → xyz delta + 6D-rotation delta)
      idx 18    left_grip   (-1 → per-dim mask)
      idx 19-27 right_arm   (+9 → xyz delta + 6D-rotation delta)
      idx 28    right_grip  (-1 → per-dim mask)
      idx 29-31 chassis     (-3 → per-dim mask)
    """

    mask: Sequence[bool] | None
    structure: Sequence[int] = dataclasses.field(default_factory=lambda: [9, 9, -1, 9, -1, -3])
    state_key: str = "state"

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state = np.asarray(data.get(self.state_key, data["state"]))
        actions = np.asarray(data["actions"])
        mask = np.asarray(self.mask)
        assert actions.shape[-1] == mask.shape[0], "mask length must equal action dim"

        # msgpack-numpy returns read-only arrays; copy to allow in-place writes.
        new_actions = np.array(actions, copy=True)
        idx = 0
        for block in self.structure:
            if block > 0:
                # 9-dim block: 3 xyz (direct subtract) + 6 SO(3) (matrix delta).
                new_actions[..., idx:idx+3] -= state[..., idx:idx+3]
                so3_act = actions[..., idx+3:idx+9]
                so3_state = state[..., idx+3:idx+9]
                mat_act = so3_6d_to_matrix(so3_act)
                mat_state = so3_6d_to_matrix(so3_state)
                delta_mat = np.matmul(mat_act, np.swapaxes(mat_state, -2, -1))
                delta_6d = matrix_to_so3_6d(delta_mat)
                new_actions[..., idx+3:idx+9] = delta_6d
                idx += block
            elif block < 0:
                # Negative: per-dim mask delta (subtract only where mask True).
                for j in range(idx, idx + abs(block)):
                    if mask[j]:
                        new_actions[..., j] -= state[..., j]
                idx += abs(block)

        data["actions"] = new_actions
        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActionsSO3(DataTransformFn):
    """Inverse of :class:`DeltaActionsSO3`.

    xyz: ``abs = delta + state`` (direct add).
    SO(3) 6D: ``abs_mat = delta_mat @ state_mat`` (note left multiplication —
    inverse of ``delta_mat = abs_mat @ state_mat^T``).
    Negative blocks: per-dim mask add (only where mask True).
    """

    mask: Sequence[bool] | None
    structure: Sequence[int] = dataclasses.field(default_factory=lambda: [9, 9, -1, 9, -1, -3])
    state_key: str = "state"

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data
        state = np.asarray(data.get(self.state_key, data["state"]))
        actions = np.asarray(data["actions"])
        mask = np.asarray(self.mask)
        assert actions.shape[-1] == mask.shape[0], "mask length must equal action dim"

        new_actions = np.array(actions, copy=True)

        idx = 0
        for block in self.structure:
            if block > 0:
                new_actions[..., idx:idx+3] += state[..., idx:idx+3]
                delta_6d = actions[..., idx+3:idx+9]
                state_6d = state[..., idx+3:idx+9]
                delta_mat = so3_6d_to_matrix(delta_6d)
                state_mat = so3_6d_to_matrix(state_6d)
                abs_mat = np.matmul(delta_mat, state_mat)  # left multiplication
                abs_6d = matrix_to_so3_6d(abs_mat)
                new_actions[..., idx+3:idx+9] = abs_6d
                idx += block
            elif block < 0:
                for j in range(idx, idx + abs(block)):
                    if mask[j]:
                        new_actions[..., j] += state[..., j]
                idx += abs(block)

        data["actions"] = new_actions
        return data
