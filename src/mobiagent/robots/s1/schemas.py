"""Schemas for the S1-mobile 3-head skill stack.

  - Subtask           : one step the planner emits (`prompt` + `stage_hint`)
  - DynamicPlan       : growing list of Subtasks across a single episode
  - PlannerDecision   : per-chunk metadata the orchestrator emits
  - JudgeDecision     : verdict the VLM judge returns per attempt
  - RunMemory         : mutable per-episode state

Three policy heads, fixed routing order:
    move_to     -> head 0
    pick_up     -> head 1
    place       -> head 2

`pick_up_from` and `place_in / place_on / pour_into` are accepted as
planner-facing synonyms but route to the same three heads (1 and 2).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal


ActionType = Literal["navigation", "manipulation"]

# Canonical heads the policy server exposes (3-head).
CANONICAL_STAGE_HINTS: tuple[str, ...] = ("move_to", "pick_up", "place")

# Planner may emit any of these strings — they collapse to the three heads.
PLANNER_STAGE_HINTS: tuple[str, ...] = (
    "move_to",
    "pick_up",
    "pick_up_from",
    "place",
    "place_in",
    "place_on",
    "pour_into",
)

StageHint = Literal[
    "move_to", "pick_up", "pick_up_from",
    "place", "place_in", "place_on", "pour_into",
]
JudgeVerdict = Literal["complete", "incomplete", "error"]
JudgeFollowup = Literal["next", "retry", "replan_plan_deviated", ""]

STAGE_HINT_TO_INT: dict[str, int] = {
    "move_to":      0,
    "pick_up":      1,
    "pick_up_from": 1,
    "place":        2,
    "place_in":     2,
    "place_on":     2,
    "pour_into":    2,
}
STAGE_INT_TO_HINT: dict[int, str] = {0: "move_to", 1: "pick_up", 2: "place"}


# GLOBAL GOAL strings the orchestrator hands to the planner per task id.
# These are the natural-language descriptions of each task — the planner
# treats them as the source of truth for object/surface noun phrasing
# (verbatim copy into per-step skill prompts).
TASK_GOALS: dict[str, str] = {
    "trash-general": "Pick up the garbage and place it in the trash can.",
    "trash-bottle":  "Pick up the bottle and place it in the trash can.",
    "trash-can":     "Pick up the can and place it in the trash can.",
    "pour-blue":     "Pick up the bottle containing blue particles and pour the blue particles into the cup.",
}


def build_full_prompt(global_goal: str, skill_text: str) -> str:
    """Assemble the runtime prompt sent to the policy server.

    Two styles, selected by env var `CLAW_PROMPT_STYLE`:
      - `"skill_only"` (default for S1 training): wire prompt is JUST the
        bare skill string, e.g. ``"move to the garbage"``.
      - `"task_then_now"`: ``"<task>. Now: <skill>."`` — task prefix.
    Trailing periods / whitespace are stripped on each part.
    """
    skill = (skill_text or "").rstrip(". ")
    style = os.environ.get("CLAW_PROMPT_STYLE", "skill_only").strip().lower()
    if style == "skill_only":
        return skill
    task = (global_goal or "").rstrip(". ")
    return f"{task}. Now: {skill}."


@dataclass(slots=True)
class Subtask:
    id: str                         # e.g. "subtask-001"
    prompt: str                     # natural-language instruction sent to the policy
    success_check: str              # text rule for the VLM judge OR literal "vlm_judge"
    stage_hint: StageHint
    max_retries: int = 50
    target_object_name: str | None = None
    failure_cues: list[str] = field(default_factory=list)
    rationale: str = ""
    plan_sketch: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.stage_hint not in PLANNER_STAGE_HINTS:
            raise ValueError(
                f"invalid stage_hint {self.stage_hint!r}; must be one of {PLANNER_STAGE_HINTS}"
            )
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")


@dataclass(slots=True)
class DynamicPlan:
    global_goal: str
    task_name: str                  # short id, e.g. "trash-general"
    subtasks: list[Subtask]
    plan_revision: int = 0
    rationale: str = ""
    scene_summary: str = ""

    def __post_init__(self) -> None:
        ids = [s.id for s in self.subtasks]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate subtask ids: {ids}")


@dataclass(slots=True)
class PlannerDecision:
    action_type: ActionType
    subtask_id: str
    target_object_name: str | None
    instruction: str
    completion_criteria: list[str] = field(default_factory=list)
    failure_cues: list[str] = field(default_factory=list)
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.action_type not in ("navigation", "manipulation"):
            raise ValueError(f"invalid action_type: {self.action_type!r}")


@dataclass(slots=True)
class JudgeDecision:
    verdict: JudgeVerdict
    reason: str
    evidence: list[str] = field(default_factory=list)
    recommended_followup: JudgeFollowup = ""

    def __post_init__(self) -> None:
        if self.verdict not in ("complete", "incomplete", "error"):
            raise ValueError(f"invalid verdict: {self.verdict!r}")
        if self.recommended_followup not in ("", "retry", "replan_plan_deviated", "next"):
            raise ValueError(
                f"invalid recommended_followup: {self.recommended_followup!r}"
            )


@dataclass(slots=True)
class Attempt:
    subtask_id: str
    attempt_number: int
    plan_revision: int
    started_at: str
    finished_at: str = ""
    judge: JudgeDecision | None = None
    chunk_size: int = 0
    notes: str = ""


@dataclass(slots=True)
class RunMemory:
    current_plan: DynamicPlan
    current_idx: int = 0
    attempts_per_subtask: dict[str, int] = field(default_factory=dict)
    history: list[Attempt] = field(default_factory=list)
    replans_done: int = 0
    total_ticks: int = 0

    @property
    def current_subtask(self) -> Subtask | None:
        if 0 <= self.current_idx < len(self.current_plan.subtasks):
            return self.current_plan.subtasks[self.current_idx]
        return None

    @property
    def is_finished(self) -> bool:
        return self.current_idx >= len(self.current_plan.subtasks)


__all__ = [
    "ActionType",
    "Attempt",
    "CANONICAL_STAGE_HINTS",
    "DynamicPlan",
    "JudgeDecision",
    "JudgeFollowup",
    "JudgeVerdict",
    "PLANNER_STAGE_HINTS",
    "PlannerDecision",
    "RunMemory",
    "STAGE_HINT_TO_INT",
    "STAGE_INT_TO_HINT",
    "StageHint",
    "Subtask",
    "TASK_GOALS",
    "build_full_prompt",
]
