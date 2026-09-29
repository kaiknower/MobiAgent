"""VLM-driven per-attempt judge for the S1-mobile skill stack.

Provider: Azure OpenAI (default GPT-5.4 — override via `S1_JUDGE_MODEL` or
`AZURE_OPENAI_DEPLOYMENT`).

Given the current Subtask + post-chunk observation + history of prior
attempts, returns a JudgeDecision (verdict / reason / evidence /
recommended_followup).

Prompt body lives in `prompts/PROMPTS.md` under `## Judge`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .llm_client import chat_completion_json, make_user_content
from .schemas import Attempt, DynamicPlan, JudgeDecision, Subtask


PROMPTS_PATH = Path(__file__).parent / "prompts" / "PROMPTS.md"

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _label_image(arr: Any, label: str) -> Any:
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return arr
    try:
        base = Image.fromarray(np.asarray(arr).astype("uint8"))
    except Exception:
        return arr
    W, H = base.size
    strip_h = 24
    out = Image.new("RGB", (W, H + strip_h), color=(0, 0, 0))
    out.paste(base, (0, strip_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(_FONT_PATH, 14)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((W - tw) // 2, (strip_h - th) // 2 - 2), label, fill=(255, 255, 255), font=font)
    import numpy as np  # noqa: F811
    return np.array(out)


# Real-machine S1-mobile camera native resolutions (W × H).
# Probed from the S1 dataset camera frames:
#   head : 1280 × 720
#   left : 640 × 360
#   right: 640 × 360
HEAD_W, HEAD_H = 1280, 720
WRIST_W, WRIST_H = 640, 360


def _composite_three_views(head_img: Any, lw_img: Any, rw_img: Any, ts_label: str) -> Any:
    """Combine head + left_wrist + right_wrist into ONE image with a timestamp
    header. Layout matches the real-machine S1-mobile native resolutions:

        ┌────┬─────────────┐
        │ LW │             │
        │640 │             │
        │×360│   HEAD      │
        ├────┤  1280×720   │
        │ RW │             │
        │640 │             │
        │×360│             │
        └────┴─────────────┘

    Returns ndarray. If any view is missing or PIL is unavailable, falls back
    to a single-camera labeled image.
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        for arr in (head_img, lw_img, rw_img):
            if arr is not None:
                return _label_image(arr, ts_label)
        return None
    try:
        font_lg = ImageFont.truetype(_FONT_PATH, 18)
        font_sm = ImageFont.truetype(_FONT_PATH, 14)
    except Exception:
        font_lg = ImageFont.load_default()
        font_sm = font_lg

    def _to_pil(arr):
        if arr is None:
            return None
        try:
            return Image.fromarray(np.asarray(arr).astype("uint8"))
        except Exception:
            return None

    head = _to_pil(head_img)
    lw = _to_pil(lw_img)
    rw = _to_pil(rw_img)

    if head is None:
        for arr, name in ((lw_img, "LEFT WRIST"), (rw_img, "RIGHT WRIST")):
            if arr is not None:
                return _label_image(arr, f"{ts_label} · {name}")
        return None

    if head.size != (HEAD_W, HEAD_H):
        head = head.resize((HEAD_W, HEAD_H), Image.BILINEAR)
    if lw is not None and lw.size != (WRIST_W, WRIST_H):
        lw = lw.resize((WRIST_W, WRIST_H), Image.BILINEAR)
    if rw is not None and rw.size != (WRIST_W, WRIST_H):
        rw = rw.resize((WRIST_W, WRIST_H), Image.BILINEAR)

    HEADER_H = 28
    LEFT_COL_W = WRIST_W                 # 640
    W = LEFT_COL_W + HEAD_W              # 640 + 1280 = 1920
    H = HEADER_H + HEAD_H                # 28 + 720    = 748
    canvas = Image.new("RGB", (W, H), color=(0, 0, 0))

    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), ts_label, font=font_lg)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((W - tw) // 2, (HEADER_H - th) // 2 - 2), ts_label, fill=(255, 255, 255), font=font_lg)

    # left column: LW top, RW bottom (each WRIST_H tall; 2*360 = 720 = HEAD_H)
    if lw is not None:
        canvas.paste(lw, (0, HEADER_H))
    if rw is not None:
        canvas.paste(rw, (0, HEADER_H + WRIST_H))

    # right column: HEAD at native size, top-aligned with the left column
    canvas.paste(head, (LEFT_COL_W, HEADER_H))

    def _caption(x, y, text):
        bb = draw.textbbox((0, 0), text, font=font_sm)
        cw, ch = bb[2] - bb[0], bb[3] - bb[1]
        pad = 4
        draw.rectangle((x, y, x + cw + 2 * pad, y + ch + 2 * pad), fill=(0, 0, 0))
        draw.text((x + pad, y + pad - 2), text, fill=(255, 255, 0), font=font_sm)

    _caption(2, HEADER_H + 2, "[2] LEFT WRIST")
    _caption(2, HEADER_H + WRIST_H + 2, "[3] RIGHT WRIST")
    _caption(LEFT_COL_W + 2, HEADER_H + 2, "[1] HEAD")

    return np.array(canvas)


_PROMPT_CACHE: str | None = None


def _load_prompt() -> str:
    global _PROMPT_CACHE
    if _PROMPT_CACHE is not None:
        return _PROMPT_CACHE
    text = PROMPTS_PATH.read_text(encoding="utf-8")
    marker = "\n## Judge"
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"Prompt section {marker.strip()!r} missing from {PROMPTS_PATH}")
    start += 1  # drop the leading newline so the block starts at "## ..."
    end = text.find("\n## ", start + len(marker))
    block = text[start:end if end > 0 else len(text)]
    _PROMPT_CACHE = block.strip()
    return _PROMPT_CACHE


def _format_subtask(st: Subtask) -> str:
    return json.dumps({
        "id": st.id,
        "prompt": st.prompt,
        "stage_hint": st.stage_hint,
        "success_check": st.success_check,
        "target_object_name": st.target_object_name,
        "failure_cues": st.failure_cues,
    }, indent=2)


def judge(
    *,
    subtask: Subtask,
    obs_after: dict[str, Any],
    obs_prev: dict[str, Any] | None = None,
    obs_mid: dict[str, Any] | None = None,
    robot_info: dict[str, Any] | None = None,
    history: list[Attempt],
    deployment: str | None = None,
) -> JudgeDecision:
    """Call the VLM judge and return a typed JudgeDecision.

    Up to three ATTEMPT-resolution frames are passed (an "attempt" = one burst
    of N policy chunks executed back-to-back by the orchestrator):
      - `obs_after` = this attempt's final frame (t, now) — the one being graded
      - `obs_prev`  = previous attempt's final frame (t-1)
      - `obs_mid`   = the attempt-two-back's final frame (t-2)

    The earlier frames let the judge see progress/motion across attempts;
    the verdict is graded on `t`.

    `robot_info` is an optional dict the orchestrator passes:
        gripper_now  : proprioceptive gripper state at t   ("OPEN"|"CLOSED"|"PARTLY-OPEN"|"?")
        gripper_prev : same at t-1
    """
    deployment = (
        deployment
        or os.getenv("S1_JUDGE_MODEL")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )

    has_prev = obs_prev is not None
    has_mid = obs_mid is not None
    n_composites = 1 + (1 if has_prev else 0) + (1 if has_mid else 0)
    layout_desc = (
        "You are shown 1 composite image with all three cameras (HEAD on the right, "
        "LEFT WRIST top-left, RIGHT WRIST bottom-left). No prior attempts yet (start of subtask)."
        if not has_prev and not has_mid
        else (
            f"You are shown {n_composites} COMPOSITE images, one per attempt, oldest first. "
            "Each composite has the SAME LAYOUT: HEAD on the right (native 1280×720, 16:9), "
            "LEFT WRIST top-left (native 640×360, 16:9), RIGHT WRIST bottom-left (native 640×360, "
            "16:9). The timestamp label is at the TOP of each composite. Tags: "
            + ("`t-2` = end of the attempt TWO back; " if has_mid else "")
            + ("`t-1` = end of the PREVIOUS attempt; " if has_prev else "")
            + "`t` = CURRENT (latest) frame — this is the one you grade. "
            "Use the earlier frames only to gauge motion/progress (is the robot getting closer, "
            "the object rising, the substance pouring — or is it stuck / oscillating?). "
            "Grade the success condition on `t`. (Soft exception: if the condition is unmistakably "
            "satisfied at `t-1` AND the robot has not visibly moved away from that pose by `t`, "
            "you may still call `complete`.) "
            "When citing evidence, refer to the camera + tag explicitly "
            "(e.g. `[1] HEAD @ t shows the target in the lower foreground`)."
        )
    )
    _prior_on_this = sum(
        1 for h in history if getattr(h, "subtask_id", None) == getattr(subtask, "id", None)
    )
    _max_retries = getattr(subtask, "max_retries", None)
    _attempt_line = (
        f"THIS SUBTASK PROGRESS — this is attempt #{_prior_on_this + 1}"
        + (f" of up to {_max_retries}" if isinstance(_max_retries, int) and _max_retries >= 0 else "")
        + f"; {_prior_on_this} earlier attempt(s) on this same subtask were all judged not-yet-done. "
        + "(Do NOT lower the bar for `complete` just because many attempts have passed.)"
    )
    _ri = robot_info or {}
    _gripper_block = (
        "GRIPPER STATE (proprioception — ground truth for whether each gripper is open or closed; "
        "you do NOT need to read gripper fingers in the wrist images):\n"
        f"  - now (t):    {_ri.get('gripper_now', '?')}\n"
        f"  - prev (t-1): {_ri.get('gripper_prev', '?')}\n"
        "For `pick_up <X>`: a CLOSED gripper at t (or open → closed across t-1 → t) is the grasp "
        "signal. For a TABLE-TOP target, CLOSED alone is sufficient → `complete`. For a FLOOR "
        "target, also require that the object has been lifted AND the torso has raised back up "
        "and come to rest (see the per-verb rule). Both grippers OPEN at t → `incomplete`."
    )
    user_text = (
        f"CURRENT SUBTASK:\n{_format_subtask(subtask)}\n\n"
        f"{_attempt_line}\n\n"
        f"{_gripper_block}\n\n"
        f"FRAME LAYOUT: {layout_desc}\n\n"
        + _load_prompt()
    )

    def _get_img(d, long_key, short_key):
        if d is None:
            return None
        v = d.get(long_key)
        if v is None:
            v = d.get(short_key)
        return v

    def _composite_for(obs, ts_label):
        if obs is None:
            return None
        h = _get_img(obs, "observation/head_image_orig",        "head_image")
        if h is None:
            h = _get_img(obs, "observation/head_image",         "head_image")
        l = _get_img(obs, "observation/left_wrist_image_orig",  "left_wrist_image")
        if l is None:
            l = _get_img(obs, "observation/left_wrist_image",   "left_wrist_image")
        r = _get_img(obs, "observation/right_wrist_image_orig", "right_wrist_image")
        if r is None:
            r = _get_img(obs, "observation/right_wrist_image",  "right_wrist_image")
        return _composite_three_views(h, l, r, ts_label)

    images = []
    _tags = []
    if has_mid:
        c = _composite_for(obs_mid, "FRAME @ t-2 (end of the attempt two back)")
        if c is not None:
            images.append(c); _tags.append("t-2")
    if has_prev:
        c = _composite_for(obs_prev, "FRAME @ t-1 (end of the previous attempt)")
        if c is not None:
            images.append(c); _tags.append("t-1")
    c = _composite_for(obs_after, "FRAME @ t (this attempt's final frame — NOW, grade this one)")
    if c is not None:
        images.append(c); _tags.append("t")

    _save_dir = os.getenv("S1_JUDGE_SAVE_DIR")
    if _save_dir and images:
        try:
            import time as _t
            import numpy as _np_sv
            from PIL import Image as _Img_sv
            Path(_save_dir).mkdir(parents=True, exist_ok=True)
            _sid = str(getattr(subtask, "id", None) or "subtask")
            _attempt_n = _prior_on_this + 1
            _stamp = _t.strftime("%H%M%S")
            for _img, _tag in zip(images, _tags):
                _fn = f"{_sid}_att{_attempt_n:02d}_{_stamp}_{_tag}.png"
                _Img_sv.fromarray(_np_sv.asarray(_img).astype("uint8")).save(str(Path(_save_dir) / _fn))
        except Exception:
            pass

    system_text = (
        "You are a strict observable-cue judge for an S1-mobile robot evaluation. "
        "Return a single JSON object with keys verdict, reason, evidence, recommended_followup."
    )
    user_content = make_user_content(text=user_text, images=images if images else None)

    obj = chat_completion_json(
        system_text=system_text,
        user_content=user_content,
        deployment=deployment,
        max_completion_tokens=4096,
    )

    verdict = str(obj.get("verdict", "")).lower().strip()
    if verdict not in ("complete", "incomplete", "error"):
        verdict = "incomplete"
    followup = str(obj.get("recommended_followup", "")).lower().strip()
    if followup not in ("", "retry", "replan_plan_deviated", "next"):
        followup = ""
    return JudgeDecision(
        verdict=verdict,  # type: ignore[arg-type]
        reason=str(obj.get("reason", "")),
        evidence=list(obj.get("evidence", []) or []),
        recommended_followup=followup,  # type: ignore[arg-type]
    )


__all__ = ["judge"]
