from __future__ import annotations

import math
import os
from dataclasses import dataclass, field


SKILLS = ("close", "open", "switch", "manipulate", "navigate", "pnp")
STAGE_SKILLS = {
    "move_to": "navigate",
    "pick_up_from": "pnp",
    "pick": "pnp",
    "place_in": "pnp",
    "place_on": "pnp",
    "place": "pnp",
    "action": "manipulate",
    "open_close": "switch",
}
WINDOWS = {
    "pick": 50,
    "place": 43,
    "navigate": 27,
    "open": 83,
    "close": 61,
    "switch": 72,
    "action": 98,
}


def action_kind(stage: str, prompt: str = "") -> str:
    if stage in ("pick_up_from", "pick"):
        return "pick"
    if stage in ("place_in", "place_on", "place"):
        return "place"
    if stage == "pnp":
        return "place" if prompt.lower().startswith("place") else "pick"
    if stage == "open_close":
        return (
            "action"
            if prompt.lower().startswith(("move ", "stir ", "scrub ", "wash "))
            else "switch"
        )
    return {"move_to": "navigate", "manipulate": "action"}.get(stage, stage)


def _counts(name: str, defaults: dict[str, int]) -> dict[str, int]:
    result = dict(defaults)
    for item in os.getenv(name, "").split(","):
        if not item.strip():
            continue
        kind, count = item.split("=", 1)
        if kind.strip() not in WINDOWS or int(count) < 1:
            raise ValueError(f"Invalid {name}: {item}")
        result[kind.strip()] = int(count)
    return result


@dataclass(frozen=True)
class ExecutionConfig:
    horizon_scale: float = 2.0
    execute_steps: int = 5
    windows: dict[str, int] = field(default_factory=lambda: dict(WINDOWS))
    retries: dict[str, int] = field(
        default_factory=lambda: {
            "navigate": 1,
            "switch": 2,
            "action": 1,
        }
    )
    retry_cap: int = 2
    visual_cadence: int = 12
    freeze_confirmations: int = 5
    pick_contact_min_mm: float = 5.0
    pick_contact_max_mm: float = 62.0
    pick_open_min_mm: float = 70.0
    pick_close_delta_mm: float = 8.0
    pick_stable_delta_mm: float = 2.5

    def __post_init__(self):
        if not math.isfinite(self.horizon_scale) or self.horizon_scale <= 0:
            raise ValueError("horizon_scale must be finite and positive")
        if not 1 <= self.execute_steps <= 50:
            raise ValueError("execute_steps must be between 1 and 50")

    @classmethod
    def from_environment(cls):
        return cls(
            horizon_scale=float(os.getenv("CLAW_HORIZON_SCALE", "2")),
            execute_steps=int(os.getenv("MOBIAGENT_EXECUTE_STEPS", "5")),
            windows=_counts("CLAW_CHUNKS_BY_KIND", WINDOWS),
            retries=_counts(
                "CLAW_RETRIES_BY_KIND",
                {
                    "navigate": 1,
                    "switch": 2,
                    "action": 1,
                },
            ),
        )

    def window(self, stage: str, prompt: str) -> int:
        count = self.windows[action_kind(stage, prompt)]
        if prompt.lower() == "pick up the bread from the toaster.":
            count = max(count, 65)
        if prompt.lower() == "wait for the toaster to finish and the lever to pop up":
            count = max(count, 120)
        return max(1, math.ceil(count * 5 / self.execute_steps))

    def effective_steps(self, task: str, stage: str, obs) -> int:
        if task == "KettleBoiling" and action_kind(stage) == "place":
            from .robocasa_control import AttemptControl

            gap = AttemptControl.gap(obs)
            if gap is not None and gap >= 25:
                return min(self.execute_steps, 3)
        return self.execute_steps

    def retry_budget(self, stage: str, prompt: str, requested: int) -> int:
        cap = self.retries.get(action_kind(stage, prompt), self.retry_cap)
        return min(requested, cap) if requested > 0 else cap
