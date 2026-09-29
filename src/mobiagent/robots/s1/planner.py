"""Reactive VLM planner — emits ONE subtask at a time.

Provider: Azure OpenAI (default deployment GPT-5.4 — override with
`S1_PLANNER_MODEL` or `AZURE_OPENAI_DEPLOYMENT`).

Contract:

    next_subtask(
        global_goal=..., task_name=...,
        history=[],                # list of {"subtask": Subtask, "outcome": "complete|failed",
                                   #          "judge_reason": str, "n_attempts": int}
        head_image=None,           # HWC uint8 numpy array, optional
        deployment=None,
    ) -> Subtask

Termination is the orchestrator's call (env success check + max steps);
the planner just emits the next single step every turn.

Prompt body lives in `prompts/PROMPTS.md` under `## Next-Subtask Planner`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .llm_client import chat_completion_json, make_user_content
from .schemas import (
    CANONICAL_STAGE_HINTS,
    PLANNER_STAGE_HINTS,
    Subtask,
)


PROMPTS_PATH = Path(__file__).parent / "prompts" / "PROMPTS.md"

_PROMPT_CACHE: str | None = None


def _load_prompt() -> str:
    """Read the canonical planner prompt from PROMPTS.md (## Next-Subtask Planner)."""
    global _PROMPT_CACHE
    if _PROMPT_CACHE is not None:
        return _PROMPT_CACHE
    text = PROMPTS_PATH.read_text(encoding="utf-8")
    marker = "\n## Next-Subtask Planner"
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"Prompt section {marker.strip()!r} missing from {PROMPTS_PATH}")
    start += 1  # drop the leading newline so the block starts at "## ..."
    end = text.find("\n## ", start + len(marker))
    block = text[start:end if end > 0 else len(text)]
    _PROMPT_CACHE = block.strip()
    return _PROMPT_CACHE


def _format_history(history: list[dict] | None) -> str:
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
    # Planner is supposed to emit a flat Subtask dict. Accept `{"subtask": {...}}` as
    # a legacy alternative.
    inner = obj.get("subtask") if isinstance(obj.get("subtask"), dict) else obj

    stage_hint = str(inner.get("stage_hint", "")).lower().strip()
    if stage_hint not in PLANNER_STAGE_HINTS:
        raise ValueError(
            f"invalid stage_hint {stage_hint!r}; must be one of {PLANNER_STAGE_HINTS}"
        )
    prompt = str(inner.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("subtask.prompt is empty")

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


def next_subtask(
    *,
    global_goal: str,
    task_name: str = "",
    history: list[dict] | None = None,
    head_image: Any | None = None,
    deployment: str | None = None,
) -> Subtask:
    """Ask the VLM for the NEXT subtask. ALWAYS returns a Subtask.

    `history` is a unified, ordered list of every subtask that has already
    ended (either via judge=complete OR via retry-budget exhausted). Each
    entry is a dict:
        {"subtask": Subtask, "outcome": "complete|failed",
         "judge_reason": str, "n_attempts": int}

    Termination is the orchestrator's decision (env success / max_ticks),
    not the planner's. Even if the goal looks satisfied in the head image,
    the planner is expected to emit a sensible next step (which the judge
    will then mark complete or incomplete).
    """
    history = history or []
    deployment = (
        deployment
        or os.getenv("S1_PLANNER_MODEL")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )

    fallback_id = f"subtask-{len(history) + 1:03d}"
    user_text = (
        f"GLOBAL GOAL: {global_goal}\n\n"
        f"HISTORY ({len(history)} ended subtask{'s' if len(history) != 1 else ''}):\n"
        f"{_format_history(history)}\n\n"
        + _load_prompt()
    )
    images = [head_image] if head_image is not None else None

    obj = chat_completion_json(
        system_text=(
            "You are an S1-mobile humanoid reactive planner. "
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


__all__ = ["next_subtask"]
