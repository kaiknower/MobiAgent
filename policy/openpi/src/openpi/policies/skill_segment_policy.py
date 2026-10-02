"""Data transforms for routed six-expert π0.5 policies.

Wraps :class:`openpi.policies.behavior_policy.BehaviorInputs` (which already
handles BEHAVIOR-1K's 256-dim full proprio → 23-dim canonical state slice and
the head/left_wrist/right_wrist image remapping) and adds the
``skill_canonical_ids`` field that ``Pi0SixHead`` needs for per-sample expert
dispatch.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import behavior_policy


@dataclasses.dataclass(frozen=True)
class SkillSegmentInputs(transforms.DataTransformFn):
    """Per-sample input transform for the 6-head pipeline.

    Drops ``skill_canonical`` (string label, kept by the dataset for debugging)
    and renames the integer id to plural form so it lines up with the
    ``Observation.skill_canonical_ids`` batched field after DataLoader collation.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        out = behavior_policy.BehaviorInputs(model_type=self.model_type)(data)
        if "skill_canonical_id" in data:
            sid = np.asarray(data["skill_canonical_id"], dtype=np.int32)
            out["skill_canonical_ids"] = sid  # singular per-sample → torch collation gives (B,)
        if "action_delta_state" in data:
            out["action_delta_state"] = behavior_policy.extract_behavior_state(
                np.asarray(data["action_delta_state"], dtype=np.float32)
            )
        return out


@dataclasses.dataclass(frozen=True)
class SkillSegmentOutputs(transforms.DataTransformFn):
    """Inverse transform for serving (slice padded actions back to 23-dim)."""

    action_dim: int = 23

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class StageHintToSkillCanonicalId(transforms.DataTransformFn):
    """Serving-side bridge: client sends ``stage_hint`` (int 0..5), training-side
    ``SkillSegmentInputs`` reads ``skill_canonical_id`` — copy across so the
    expert dispatcher routes correctly at inference time.

    No-op at training (the dataset already sets ``skill_canonical_id``).
    """

    def __call__(self, data: dict) -> dict:
        if "skill_canonical_id" not in data and "stage_hint" in data:
            data = {**data, "skill_canonical_id": int(np.asarray(data["stage_hint"]).item())}
        return data
