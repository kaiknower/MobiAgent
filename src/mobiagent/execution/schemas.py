"""Schemas for the DIMOS-free behavior-1k eval runner.

Subtask -> DynamicPlan -> RunMemory + PlannerDecision/JudgeDecision are the
five types that flow between the orchestrator, VLM planner, VLM judge, and the
6-ckpt-server routing layer.

Compared to dimos_pi0_5GT/schemas.py:
  - PlannerDecision now references subtask_id (plan-list index) instead of monotonic stage
  - JudgeDecision adds 'replan_plan_deviated' as a recommended_followup
  - Subtask carries stage_hint (one of 6 canonicals) for ckpt routing
  - DynamicPlan carries plan_revision; bumps each replan
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal


ActionType = Literal["navigation", "manipulation"]
StageHint = Literal[
    "move_to", "pick_up_from",
    "place_in", "place_on", "open", "close",     # v13 6-head names
    "place", "open_close",                         # v15 4-head merged names
]
JudgeVerdict = Literal["complete", "incomplete", "error"]
JudgeFollowup = Literal["next", "retry", "replan_plan_deviated", "replan_keep_pose", ""]

CANONICAL_STAGE_HINTS: tuple[str, ...] = (
    # v13 6-head canonical stages
    "move_to",
    "pick_up_from",
    "place_in",
    "place_on",
    "open",
    "close",
    # v15 4-head merged stages: place_in+place_on → place ; open+close → open_close
    "place",
    "open_close",
    # v15 1-head single expert ("action"): all skills route to this head (id 0)
    "action",
)

# `explore` is a planner-emittable stage that is NOT a policy head — it routes
# to the orchestrator's scripted base-spin scan instead of a ckpt server.
EXPLORE_STAGE = "explore"
PLANNER_STAGE_HINTS: tuple[str, ...] = CANONICAL_STAGE_HINTS + ("navigate", "pnp", "switch", "manipulate")

# Server-side multi-head model routes by integer stage_hint via
# StageHintToSkillCanonicalId in skill_segment_policy.py. The order MUST match
# the server's expert array order.
#
# Dual layout: v13 6-head and v15 4-head share the same int slots 0..3 by
# design (move_to=0, pick_up_from=1, "place-like"=2, "open/close-like"=3),
# and the v13-only stages keep their original 4..5 slots. So:
#   - v15 4-head server: send "place" (2) and "open_close" (3). The v13 names
#     `place_in/place_on/open/close` will MISROUTE on a v15 server
#     (`place_on`→3=open_close, `open`/`close` exceed the 4-expert range).
#   - v13 6-head server: send `place_in/place_on/open/close`. Sending v15
#     names also works because `place`→2 still hits `place_in`, but
#     `open_close`→3 will hit `place_on` (wrong).
# Picking the right name is the caller's responsibility.
STAGE_HINT_TO_INT: dict[str, int] = {
    "move_to":      0,
    "pick_up_from": 1,
    "place_in":     2,
    "place_on":     3,
    "open":         4,
    "close":        5,
    "place":        2,  # v15 4-head: merged place head shares slot 2 with v13's place_in
    "open_close":   3,  # v15 4-head: merged open_close head shares slot 3 with v13's place_on
    "action":       0,  # v15 1-head single expert: server expert_names=("action",), id 0
}
# Reverse map: dict-comprehension keeps the LAST insertion per int, so v15 names
# win for slots 2/3. That's fine — reverse map is only used for display/logging
# (no live routing reads it), and v15 is the current model generation.
STAGE_INT_TO_HINT: dict[int, str] = {v: k for k, v in STAGE_HINT_TO_INT.items()}


def build_full_prompt(global_goal: str, skill_text: str) -> str:
    """Assemble the runtime prompt the policy server expects.

    Two styles, selected by env var CLAW_PROMPT_STYLE:
      - "skill_only" (v17 and later): the wire prompt is JUST the bare skill
        description, e.g. ``move to radio``. v17 training data is skill-only
        (the JSONL `task_instruction` field already holds only the skill text).
      - "task_then_now" (v15/v16, default): ``"<task>. Now: <skill>."`` —
        the task instruction followed by the skill.
    Trailing whitespace + periods are stripped from each part.
    """
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
