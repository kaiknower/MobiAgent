"""Reactive VLM planner — emits ONE subtask at a time.

Provider: Azure OpenAI GPT-5.4 (default). Override via CLAW_PLANNER_MODEL or
AZURE_OPENAI_DEPLOYMENT.

Contract:

    next_subtask(
        global_goal, sim_task_name,
        completed_subtasks=[],
        last_failed=None,         # (stage_hint, prompt, last_reason) | None
        head_image=None,
        deployment=None,
    ) -> Subtask

ALWAYS returns a Subtask. The planner does NOT decide task termination —
the orchestrator owns that decision via env.is_success() (BDDL predicate)
and per-subtask judge verdicts. A retry of the SAME subtask does NOT call
the planner again — that's an internal orchestrator decision.

Prompt body lives in PROMPTS.md `## Next-Subtask Planner` section.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from .llm_client import chat_completion_json, make_user_content
from .timing import timed
from .schemas import (
    Attempt,
    CANONICAL_STAGE_HINTS,
    PLANNER_STAGE_HINTS,
    STAGE_HINT_TO_INT,
    Subtask,
)


PROMPTS_PATH = Path(__file__).parent / "PROMPTS.md"

_PROMPT_CACHE: str | None = None


def _load_prompt() -> str:
    """Read the canonical planner prompt from PROMPTS.md (## Next-Subtask Planner)."""
    global _PROMPT_CACHE
    if _PROMPT_CACHE is not None:
        return _PROMPT_CACHE
    text = PROMPTS_PATH.read_text(encoding="utf-8")
    marker = "## Next-Subtask Planner"
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"Prompt section {marker!r} missing from {PROMPTS_PATH}")
    end = text.find("\n## ", start + len(marker))
    block = text[start:end if end > 0 else len(text)]
    # The `explore` subtask is opt-in: strip the EXPLORE_CLAUSE block from the
    # planner prompt unless CLAW_ENABLE_EXPLORE=1, so the planner never emits it.
    if os.environ.get("CLAW_ENABLE_EXPLORE", "0").strip() != "1":
        block = re.sub(
            r"<!-- EXPLORE_CLAUSE_START -->.*?<!-- EXPLORE_CLAUSE_END -->\n?",
            "", block, flags=re.DOTALL,
        )
    else:
        block = block.replace("<!-- EXPLORE_CLAUSE_START -->\n", "").replace("<!-- EXPLORE_CLAUSE_END -->\n", "")
    _PROMPT_CACHE = block.strip().replace("{available_skills}", os.environ.get("MOBIAGENT_SKILLS", "move_to, pick_up_from, place_in, place_on, open, close"))
    return _PROMPT_CACHE


def _format_history(history: list[dict] | None) -> str:
    """Render the unified history as a numbered list for the planner prompt.

    Each entry is a dict shaped like
        {"subtask": Subtask, "outcome": "complete"|"failed",
         "judge_reason": str, "n_attempts": int}
    """
    if not history:
        return "  (none — this is the very first step.)"
    rows: list[str] = []
    for i, item in enumerate(history, 1):
        st: Subtask = item["subtask"]
        outcome = item.get("outcome", "unknown")
        n = item.get("n_attempts", "?")
        reason = (item.get("judge_reason") or "").strip()
        if len(reason) > 220:
            reason = reason[:217] + "..."
        verb = "completed" if outcome == "complete" else "failed"
        rows.append(
            f"  {i:>2}. Subtask {st.stage_hint} \"{st.prompt}\" — {verb} after "
            f"{n} attempt{'s' if n != 1 else ''}.\n"
            f"        Judge reason: \"{reason}\""
        )
    return "\n".join(rows)


def _parse_subtask_json(obj: dict[str, Any], *, fallback_id: str) -> Subtask:
    """Parse the planner's JSON into a Subtask. Always returns one."""
    # Planner is supposed to emit a flat Subtask dict. Some legacy models may
    # still wrap in `{"subtask": {...}}` — accept both shapes.
    inner = obj.get("subtask") if isinstance(obj.get("subtask"), dict) else obj

    stage_hint = str(inner.get("stage_hint", "")).lower().strip()
    if stage_hint == "explore" and os.environ.get("CLAW_ENABLE_EXPLORE", "0").strip() != "1":
        raise ValueError("'explore' subtask is disabled (set CLAW_ENABLE_EXPLORE=1 to allow it); emit a normal subtask instead")
    if stage_hint not in PLANNER_STAGE_HINTS:
        raise ValueError(
            f"invalid stage_hint {stage_hint!r}; must be one of {PLANNER_STAGE_HINTS}"
        )
    prompt = str(inner.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("subtask.prompt is empty")

    # max_retries is fixed at 50 chunks (≈ 50 s @ 30 Hz) on the orchestrator
    # side — the planner is not asked to estimate it.
    raw_sketch = inner.get("plan_sketch") or []
    plan_sketch: list[str] = []
    if isinstance(raw_sketch, list):
        plan_sketch = [str(x).strip() for x in raw_sketch if str(x).strip()]

    return Subtask(
        id=str(inner.get("id") or fallback_id),
        prompt=prompt,
        success_check="vlm_judge",
        stage_hint=stage_hint,  # type: ignore[arg-type]
        max_retries=50,
        target_object_name=inner.get("target_object_name"),
        failure_cues=[],
        rationale=str(inner.get("rationale", "")),
        plan_sketch=plan_sketch,
    )


_ORDINAL_RE = re.compile(r"\s*\(\s*\d+\s*/\s*\d+\s*\)\s*$")


def _strip_ordinal(p: str) -> str:
    """Remove a trailing `(i/N)` suffix if present."""
    return _ORDINAL_RE.sub("", p or "").strip()


def _apply_ordinals(st: Subtask, history: list[dict]) -> Subtask:
    """Re-write `st.prompt` and `st.plan_sketch` with deterministic `(i/N)`
    suffixes based on temporal position across history + plan_sketch.

    Convention: `plan_sketch[0]` IS the current step (== `st.prompt`). When
    the LLM follows this convention, `plan_sketch` already encodes the full
    remaining plan including the current step; we count over `plan_sketch`
    alone (not `[current] + plan_sketch`, which would double-count). When
    the LLM diverges (plan_sketch is empty or its first item differs from
    prompt), fall back to inserting the current step at the head.
    """
    from collections import Counter
    import dataclasses

    # 1. Pull stripped prompts from history
    history_prompts: list[str] = []
    for h in history or []:
        sub = h.get("subtask") if isinstance(h, dict) else None
        if sub is None:
            continue
        p = getattr(sub, "prompt", None)
        if p is None and isinstance(sub, dict):
            p = sub.get("prompt")
        if p:
            history_prompts.append(_strip_ordinal(str(p)))

    # 2. Strip current + plan_sketch
    cur = _strip_ordinal(st.prompt)
    plan_stripped = [_strip_ordinal(p) for p in (st.plan_sketch or [])]

    # 3. Decide whether plan_sketch[0] already represents the current step.
    #    If yes, use plan_stripped as-is; if no, prepend current.
    if plan_stripped and plan_stripped[0] == cur:
        future_plan = plan_stripped
    else:
        future_plan = [cur] + plan_stripped

    # 4. Build full plan, count totals, walk in temporal order assigning indices
    full_plan = history_prompts + future_plan
    totals = Counter(full_plan)
    seen: Counter = Counter()
    full_suffixed: list[str] = []
    for p in full_plan:
        if totals[p] >= 2:
            seen[p] += 1
            full_suffixed.append(f"{p} ({seen[p]}/{totals[p]})")
        else:
            full_suffixed.append(p)

    # 5. Slice — the current emission is at index len(history_prompts),
    #    plan_sketch starts at the same index (since plan_sketch[0] is
    #    current) OR at +1 (if we prepended current).
    idx_current = len(history_prompts)
    new_prompt = full_suffixed[idx_current]
    plan_start = idx_current if (plan_stripped and plan_stripped[0] == cur) else idx_current + 1
    new_plan_sketch = full_suffixed[plan_start:]
    return dataclasses.replace(st, prompt=new_prompt, plan_sketch=new_plan_sketch)


@timed("planner", "n_planner")
def next_subtask(
    *,
    global_goal: str,
    sim_task_name: str,
    history: list[dict] | None = None,
    head_image: Any | None = None,
    deployment: str | None = None,
) -> Subtask:
    """Ask the VLM for the NEXT subtask. ALWAYS returns a Subtask.

    `history` is a unified, ordered list of every subtask that has already
    ended (either via judge=complete OR via retry-budget exhausted). Each
    entry is a dict:
        {"subtask": Subtask, "outcome": "complete"|"failed",
         "judge_reason": str, "n_attempts": int}

    Termination is the orchestrator's decision (env.is_success() / max_ticks),
    not the planner's. Even if the goal looks satisfied in the head image,
    the planner is expected to emit a sensible next step (which the judge
    will then mark complete or incomplete).
    """
    history = history or []
    deployment = deployment or os.getenv("CLAW_PLANNER_MODEL") or os.getenv("AZURE_OPENAI_DEPLOYMENT")

    fallback_id = f"subtask-{len(history) + 1:03d}"
    # `sim_task_name` is kept as a function parameter for vocab/log lookup but
    # NOT sent to the LLM — `global_goal` carries all the semantics it needs.
    user_text = (
        f"GLOBAL GOAL: {global_goal}\n\n"
        f"HISTORY ({len(history)} ended subtask{'s' if len(history) != 1 else ''}):\n"
        f"{_format_history(history)}\n\n"
        + _load_prompt()
    )
    images = [head_image] if head_image is not None else None

    obj = chat_completion_json(
        system_text=(
            "You are a behavior-1k mobile-manipulation reactive planner. "
            "On each call, return one JSON object describing the next subtask. "
            "No markdown, no commentary outside the JSON."
        ),
        user_content=make_user_content(text=user_text, images=images),
        deployment=deployment,
    )

    try:
        return _parse_subtask_json(obj, fallback_id=fallback_id)
    except Exception as exc:
        raise RuntimeError(f"planner returned invalid JSON: {exc!s}; raw={obj!r}") from exc

    # NOTE: deterministic `_apply_ordinals` post-processor was REMOVED. v12
    # training distribution is multi-modal (e.g. task-0022's `move to
    # hallstand` count distributes as 2/3/4 across episodes at 87/10/3%) —
    # forcing a count from history+plan_sketch like our previous post-
    # processor did was producing OOD strings such as `move to hallstand
    # (6/7)` that the policy was never trained on. The planner is asked
    # via PROMPTS.md to emit `(i/N)` itself; trust it.


# Backwards-compat: `build_dynamic_plan` is no longer the primary path. Kept as
# a thin wrapper that calls `next_subtask` once for callers who still want a
# single-step "warmup" plan (used by run.py --dry-run).
def build_dynamic_plan(
    *,
    global_goal: str,
    sim_task_name: str,
    scene_obs: dict[str, Any] | None = None,
    head_image: Any | None = None,
    deployment: str | None = None,
    cache_path: Path | None = None,
) -> "DynamicPlan":  # noqa: F821  (forward ref)
    """Compatibility shim: emit ONE subtask via `next_subtask` and wrap as a
    DynamicPlan. New code should call `next_subtask` directly.
    """
    from .schemas import DynamicPlan

    if cache_path is not None and cache_path.exists():
        try:
            obj = json.loads(cache_path.read_text())
            return _plan_from_dict(obj, global_goal=global_goal, sim_task_name=sim_task_name)
        except Exception:
            pass

    st = next_subtask(
        global_goal=global_goal, sim_task_name=sim_task_name,
        head_image=head_image, deployment=deployment,
    )
    plan = DynamicPlan(
        global_goal=global_goal,
        sim_task_name=sim_task_name,
        subtasks=[st],
        plan_revision=0,
        rationale="single-shot warmup via next_subtask()",
        scene_summary="",
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_plan_to_json(plan), encoding="utf-8")
    return plan


def _plan_from_dict(obj: dict[str, Any], *, global_goal: str, sim_task_name: str) -> "DynamicPlan":  # noqa: F821
    from .schemas import DynamicPlan
    subs: list[Subtask] = []
    for i, raw in enumerate(obj.get("subtasks", []), 1):
        stage = str(raw.get("stage_hint", "")).lower().strip()
        if stage not in CANONICAL_STAGE_HINTS:
            continue
        subs.append(Subtask(
            id=str(raw.get("id") or f"subtask-{i:03d}"),
            prompt=str(raw["prompt"]),
            success_check=str(raw.get("success_check", "vlm_judge")),
            stage_hint=stage,  # type: ignore[arg-type]
            max_retries=int(raw.get("max_retries", 2)),
            target_object_name=raw.get("target_object_name"),
            failure_cues=list(raw.get("failure_cues", []) or []),
            rationale=str(raw.get("rationale", "")),
        ))
    return DynamicPlan(
        global_goal=global_goal,
        sim_task_name=sim_task_name,
        subtasks=subs,
        plan_revision=int(obj.get("plan_revision", 0)),
        rationale=str(obj.get("rationale", "")),
        scene_summary=str(obj.get("scene_summary", "")),
    )


def _plan_to_json(plan) -> str:  # noqa: ANN001
    return json.dumps({
        "global_goal": plan.global_goal,
        "sim_task_name": plan.sim_task_name,
        "plan_revision": plan.plan_revision,
        "rationale": plan.rationale,
        "scene_summary": plan.scene_summary,
        "subtasks": [
            {
                "id": s.id, "prompt": s.prompt,
                "success_check": s.success_check,
                "stage_hint": s.stage_hint,
                "max_retries": s.max_retries,
                "target_object_name": s.target_object_name,
                "failure_cues": s.failure_cues,
                "rationale": s.rationale,
            }
            for s in plan.subtasks
        ],
    }, indent=2)


__all__ = [
    "next_subtask",
    "build_dynamic_plan",
]
