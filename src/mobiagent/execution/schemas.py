"""Typed subtask plans, runtime memory and planner/critic decisions.

Subtasks carry a stage_hint for expert routing. Plans track plan_revision
to distinguish updates after replanning.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal


ActionType = Literal["navigation", "manipulation"]
StageHint = Literal[
    "move_to", "pick_up_from",
    "place_in", "place_on", "open", "close",     # Six-expert names
    "place", "open_close",                         # Four-expert merged names
]
JudgeVerdict = Literal["complete", "incomplete", "error"]
JudgeFollowup = Literal["next", "retry", "replan_plan_deviated", "replan_keep_pose", ""]

CANONICAL_STAGE_HINTS: tuple[str, ...] = (
    # Six-expert canonical stages
    "move_to",
    "pick_up_from",
    "place_in",
    "place_on",
    "open",
    "close",
    # Four-expert merged stages: place_in+place_on → place ; open+close → open_close
    "place",
    "open_close",
    # Single-expert configuration: all skills route to action (id 0)
    "action",
)

# `explore` is a planner-emittable stage that is NOT a policy head — it routes
# to the orchestrator's scripted base-spin scan instead of a ckpt server.
EXPLORE_STAGE = "explore"
PLANNER_STAGE_HINTS: tuple[str, ...] = CANONICAL_STAGE_HINTS + ("navigate", "pnp", "switch", "manipulate")

# Integer routing must match the policy server's expert order.
# Six-expert layout: move_to, pick_up_from, place_in, place_on, open, close.
# Four-expert layout: move_to, pick_up_from, place, open_close.
# Single-expert layout: action. Shared integer slots have different meanings
# between layouts; callers must use names matching their checkpoint.
STAGE_HINT_TO_INT: dict[str, int] = {
    "move_to":      0,
    "pick_up_from": 1,
    "place_in":     2,
    "place_on":     3,
    "open":         4,
    "close":        5,
    "place":        2,  # Merged place expert uses slot 2.
    "open_close":   3,  # Merged open/close expert uses slot 3.
    "action":       0,  # Single action expert uses slot 0.
}
# Display-only reverse map: later aliases take precedence for shared slots.
STAGE_INT_TO_HINT: dict[int, str] = {v: k for k, v in STAGE_HINT_TO_INT.items()}


def build_full_prompt(global_goal: str, skill_text: str) -> str:
    """Assemble a policy prompt using CLAW_PROMPT_STYLE.

    ``skill_only`` returns the skill description. ``task_then_now`` combines
    the task instruction and skill as "<task>. Now: <skill>.". Trailing
    whitespace and periods are stripped from each input."""
    skill = (skill_text or "").rstrip(". ")
    style = os.environ.get("CLAW_PROMPT_STYLE", "task_then_now").strip().lower()
    if style == "skill_only":
        return skill
    task = (global_goal or "").rstrip(". ")
    return f"{task}. Now: {skill}."


@dataclass(slots=True)
class Subtask:
    id: str                         # e.g. "subtask-001"
    prompt: str                     # natural-language instruction for the policy
    success_check: str              # text rule for VLM judge OR the literal "vlm_judge"
    stage_hint: StageHint           # routes to the correct ckpt server
    max_retries: int = 2
    target_object_name: str | None = None
    failure_cues: list[str] = field(default_factory=list)
    rationale: str = ""
    plan_sketch: list[str] = field(default_factory=list)  # planner's CoT remaining-plan outline

    def __post_init__(self) -> None:
        if self.stage_hint not in PLANNER_STAGE_HINTS:
            raise ValueError(
                f"invalid stage_hint: {self.stage_hint!r}. Must be one of {PLANNER_STAGE_HINTS}"
            )
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")


@dataclass(slots=True)
class DynamicPlan:
    global_goal: str                # one-sentence task description
    sim_task_name: str              # e.g. "task-0001" / "turning_on_radio"
    subtasks: list[Subtask]
    plan_revision: int = 0          # bumps on each replan
    rationale: str = ""             # planner's reasoning
    scene_summary: str = ""         # planner's note on starting scene state

    def __post_init__(self) -> None:
        # In reactive mode, the plan starts empty and grows as the planner
        # emits each next subtask. No min-length check.
        ids = [s.id for s in self.subtasks]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate subtask ids: {ids}")


@dataclass(slots=True)
class PlannerDecision:
    """Per-subtask decision the orchestrator emits before each chunk request."""
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
    """VLM judge's verdict after a chunk completes."""
    verdict: JudgeVerdict
    reason: str
    evidence: list[str] = field(default_factory=list)
    recommended_followup: JudgeFollowup = ""

    def __post_init__(self) -> None:
        if self.verdict not in ("complete", "incomplete", "error"):
            raise ValueError(f"invalid verdict: {self.verdict!r}")
        if self.recommended_followup not in ("next", "retry", "replan_plan_deviated", "replan_keep_pose", ""):
            raise ValueError(
                f"invalid recommended_followup: {self.recommended_followup!r}"
            )


@dataclass(slots=True)
class Attempt:
    subtask_id: str
    attempt_number: int                 # 1-based within this plan revision
    plan_revision: int
    started_at: str                     # ISO-8601
    finished_at: str = ""
    judge: JudgeDecision | None = None
    chunk_size: int = 0
    notes: str = ""


@dataclass(slots=True)
class RunMemory:
    """Mutable state for an ongoing run."""
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
    "EXPLORE_STAGE",
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
    "build_full_prompt",
]
