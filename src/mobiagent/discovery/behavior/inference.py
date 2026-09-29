"""Minimal inference scaffolding for offline skill discovery."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from mobiagent.discovery.behavior.azure_client import build_chat_completion_request
from mobiagent.discovery.behavior.azure_client import execute_chat_completion
from mobiagent.discovery.behavior.azure_client import extract_first_message_text


def validate_timeline_prediction(payload: dict) -> dict:
    if "skill_timeline" not in payload:
        raise ValueError("skill_timeline is required")
    if not isinstance(payload["skill_timeline"], list):
        raise ValueError("skill_timeline must be a list")
    if "activity_timeline" in payload and not isinstance(payload["activity_timeline"], list):
        raise ValueError("activity_timeline must be a list")
    for item in payload["skill_timeline"]:
        if not isinstance(item, dict):
            continue
        if "start_time_sec" not in item or "end_time_sec" not in item:
            raise ValueError("start_time_sec and end_time_sec are required for each segment")
    return payload


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("No complete JSON object found")


def parse_timeline_response_text(text: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as direct_error:
        try:
            payload = json.loads(_extract_first_json_object(text))
        except Exception as wrapped_error:
            preview = text.strip().replace("\n", "\\n")
            if len(preview) > 500:
                preview = preview[:500] + "..."
            raise ValueError(
                f"Unable to parse model response into JSON. Preview: {preview!r}"
            ) from wrapped_error
    return validate_timeline_prediction(payload)


def _detect_paired_transitions(transitions: list, max_gap_sec: float = 2.0) -> dict[int, int]:
    valid: list[tuple[int, float, str, str]] = []
    for idx, event in enumerate(transitions):
        if not isinstance(event, dict):
            continue
        try:
            t_val = float(event["t"])
        except (KeyError, TypeError, ValueError):
            continue
        side = event.get("side")
        direction = event.get("direction")
        if isinstance(side, str) and isinstance(direction, str):
            valid.append((idx, t_val, side, direction))

    pairs: dict[int, int] = {}
    already_paired: set[int] = set()
    for vi, (i, t_i, side_i, dir_i) in enumerate(valid):
        if dir_i != "open_to_close" or i in already_paired:
            continue
        for vj in range(vi + 1, len(valid)):
            j, t_j, side_j, dir_j = valid[vj]
            if side_j != side_i:
                continue
            if (
                dir_j == "close_to_open"
                and j not in already_paired
                and t_j - t_i <= max_gap_sec
            ):
                pairs[i] = j
                pairs[j] = i
                already_paired.add(i)
                already_paired.add(j)
            break
    return pairs


def _silent_actuation_lines(payload: dict) -> list[str]:
    """Highlight arm-active spans with NO gripper event. These are the highest-
    signal candidates for silent body/forearm/arm-sweep actuation on an
    articulated target — the exact situations where the model tends to miss a
    state change because the gripper doesn't fire."""
    intervals = payload.get("silent_actuation_intervals")
    if not isinstance(intervals, list) or not intervals:
        return []
    lines = [
        "## Silent-Actuation Candidates (DERIVED)",
        "- Arm-active span with NO gripper event inside — usually arm/forearm/body silently pushing/sweeping/closing an articulated target (pushing a door wider after release, sweeping another panel during rotation, closing a flap with the forearm). Treat as manipulation, not navigation.",
        "- `side` matters: SAME-hand entry following an `open/close <X>` continues that segment — extend `open/close <X>` end_time_sec to this entry's END (brief base nonzero inside the entry stays in). OPPOSITE-hand entry is that hand reaching for its upcoming grasp/release — absorb into the UPCOMING acquire/release segment.",
        "- Silent-actuation alone NEVER creates a new open/close; a PAIRED contact alone is not enough — the panel must visibly transition in the video.",
    ]
    for iv in intervals:
        if not isinstance(iv, dict):
            continue
        try:
            s = float(iv["start_time_sec"])
            e = float(iv["end_time_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        side = iv.get("side", "?")
        lines.append(f"- {side}: [{s:.2f}s - {e:.2f}s]")
    return lines


def _arm_active_lines(payload: dict) -> list[str]:
    """Inject arm-active intervals as a compact list. Gives the model a second
    physical signal alongside base[0:3] so it can distinguish walking-while-
    actuating from pure navigation."""
    intervals = payload.get("arm_active_intervals")
    if not isinstance(intervals, list) or not intervals:
        return []
    lines = [
        "## Arm-Active Intervals",
        "- Spans where the named arm's joint velocity is above the session idle level. Combined with base[0:3]:",
        "  • base-zero + arm-active = stationary manipulation",
        "  • base-nonzero + arm-active = walking-while-actuating (manipulation, NOT navigation)",
        "  • base-nonzero + arm-idle = pure navigation",
        "  • base-zero + arm-idle = waiting / settling",
    ]
    for iv in intervals:
        if not isinstance(iv, dict):
            continue
        try:
            s = float(iv["start_time_sec"])
            e = float(iv["end_time_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        side = iv.get("side", "?")
        lines.append(f"- {side}: [{s:.2f}s - {e:.2f}s]")
    return lines


def _minimal_base_zero_lines(payload: dict) -> list[str]:
    """Inject a compact base-zero interval list (compressed seconds) so the
    model can spot hidden manipulations that leave no gripper event but show
    as a sustained stop."""
    intervals = payload.get("base_zero_intervals")
    if not isinstance(intervals, list) or not intervals:
        return []
    lines = [
        "## Base-Zero Intervals",
        "- Spans where base[0:3] is near zero — robot is stationary. NO `move to ...` may be placed inside a base-zero interval.",
        "- Base-zero is always manipulation or waiting. Arm motion / head rotation / approach during base-zero is absorbed into the adjacent manipulation segment (post-release completion or approach for the next acquire). Do NOT hallucinate an open/close/pick/place to fill an idle stop.",
    ]
    for iv in intervals:
        if not isinstance(iv, dict):
            continue
        try:
            s = float(iv["start_time_sec"])
            e = float(iv["end_time_sec"])
            d = float(iv.get("duration_sec", e - s))
        except (KeyError, TypeError, ValueError):
            continue
        if d < 0.3:
            continue
        cid = iv.get("checkpoint_id") or "base_zero"
        lines.append(f"- {cid}: [{s:.2f}s - {e:.2f}s] dur={d:.2f}s")
    return lines


def _gripper_transition_data_only_lines(payload: dict) -> list[str]:
    """Minimal gripper-event data injection — timestamps, PAIRED labels, grip durations.

    Each event is listed with its timestamp, side, direction, and (when it is
    part of a sustained grip cycle) the grip duration. Short PAIRED pairs are
    flagged to identify brief-contact events like door handles; long non-PAIRED
    grips are annotated with their hold duration so the model can tell a
    handle-touch apart from a task-object carry. All labels are computed
    purely from the event signals; no vocabulary rules are added here.
    """
    transitions = payload.get("gripper_transitions")
    if not isinstance(transitions, list) or not transitions:
        return []
    pairs = _detect_paired_transitions(transitions)

    # Precompute sustained grip matches: for each non-PAIRED open_to_close,
    # find the next close_to_open on the same side that is itself non-PAIRED.
    grip_match_fwd: dict[int, int] = {}
    grip_match_back: dict[int, int] = {}
    for idx, event in enumerate(transitions):
        if idx in pairs:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("direction") != "open_to_close":
            continue
        side = event.get("side")
        for jdx in range(idx + 1, len(transitions)):
            if jdx in pairs:
                continue
            partner = transitions[jdx]
            if not isinstance(partner, dict):
                continue
            if partner.get("side") != side:
                continue
            if partner.get("direction") != "close_to_open":
                continue
            grip_match_fwd[idx] = jdx
            grip_match_back[jdx] = idx
            break

    lines = [
        "## Gripper Transition Events",
        "- Ground-truth gripper state changes. `open_to_close` = closing on something; `close_to_open` = releasing.",
        "- PAIRED (open_to_close → close_to_open on same side within ~2s): ONE segment only if nothing was carried (handle/button/latch touch). If an object was visibly carried during the PAIRED window, it is a quick pick+place → TWO segments (rule 2 applies).",
        "- `grip duration Xs` flags a sustained grip — the gripper carried a task object for X seconds.",
        "- One-segment-per-event: every non-PAIRED open_to_close gets its own acquire segment, every non-PAIRED close_to_open gets its own release segment. Never merge events from different objects or different hands into one segment — including when one hand is sustained-carrying while the other hand does a separate pick+place cycle (each cycle must keep its own pick + place segments).",
        "- Acquire→travel→release: when an open_to_close acquires X and a later close_to_open releases X with sustained nonzero base[0:3] in between, emit THREE segments (acquire, `move to ... while carrying X`, release). Never combine into one `grasp X and place on Y`.",
        "- Use timestamps as evidence to anchor segments — boundaries come from visible `t=...s` and base[0:3].",
    ]
    for idx, event in enumerate(transitions):
        if not isinstance(event, dict):
            continue
        try:
            event_time = float(event["t"])
        except (KeyError, TypeError, ValueError):
            continue
        side = event.get("side")
        direction = event.get("direction")
        if not isinstance(side, str) or not isinstance(direction, str):
            continue
        suffixes: list[str] = []
        if idx in pairs:
            partner_idx = pairs[idx]
            partner = transitions[partner_idx]
            try:
                partner_time = float(partner["t"])
                suffixes.append(f"PAIRED with t={partner_time:.2f}s")
            except (KeyError, TypeError, ValueError):
                suffixes.append("PAIRED")
        elif idx in grip_match_fwd:
            release_idx = grip_match_fwd[idx]
            try:
                release_time = float(transitions[release_idx]["t"])
                duration = release_time - event_time
                suffixes.append(
                    f"held until t={release_time:.2f}s, grip duration {duration:.1f}s"
                )
            except (KeyError, TypeError, ValueError):
                pass
        elif idx in grip_match_back:
            acquire_idx = grip_match_back[idx]
            try:
                acquire_time = float(transitions[acquire_idx]["t"])
                duration = event_time - acquire_time
                suffixes.append(
                    f"matches grip that started at t={acquire_time:.2f}s, grip duration {duration:.1f}s"
                )
            except (KeyError, TypeError, ValueError):
                pass
        suffix = f"  ({'; '.join(suffixes)})" if suffixes else ""
        lines.append(f"- t={event_time:.2f}s  {side} gripper {direction}{suffix}")
    return lines


def build_minimal_video_request(
    payload: dict,
    model: str,
    max_completion_tokens: int,
    frame_data_urls: list[str] | None = None,
    video_data_url: str | None = None,
) -> dict:
    """A drastically stripped prompt used to test whether the model follows
    the same structural rules with only the essential constraints. No redundant
    rephrasings, no phase theory, no hardcoded keyword lists. Kept as a separate
    function so the full prompt remains untouched."""
    if frame_data_urls and video_data_url:
        raise ValueError("Provide either frame_data_urls or video_data_url, not both")

    task_instruction = payload.get("task") if isinstance(payload, dict) else None
    task_context_lines: list[str] = []
    if isinstance(task_instruction, str) and task_instruction.strip():
        task_context_lines = [
            "## Task Context",
            f"- Overall task (for object/destination naming only): {task_instruction.strip()}",
            "- Use the task description ONLY to make skill_description object names consistent with the stated objects/destinations. Do NOT use it to plan, reorder, or infer actions that are not visible in the video. The skill_timeline must reflect what actually happens on screen.",
            "",
        ]

    prompt_lines = [
        "Analyze the full demo video and return only JSON describing what the robot does in chronological order.",
        "",
        "## Input",
        "- Composite video: head-camera (left), wrist cameras (right column). The current playback time is shown as a `t=X.XXs` label in the BOTTOM-RIGHT corner of every frame.",
        "- Copy the visible `t=X.XXs` directly into start_time_sec / end_time_sec — it is the only valid timing source.",
        "- Segment boundaries mark INTENT changes (travel ↔ manipulation), not every base[0:3] dip or rise.",
        "",
        *task_context_lines,
        *_gripper_transition_data_only_lines(payload),
        "",
        *_minimal_base_zero_lines(payload),
        "",
        *_arm_active_lines(payload),
        "",
        *_silent_actuation_lines(payload),
        "",
        "## Segments",
        "- A segment is one contiguous window of one atomic intent: either a TRAVEL (`move to <destination>`) or a STATE CHANGE (acquire / release / open / close / push / pour / wipe / etc.). The two types are mutually exclusive within a segment.",
        "- Navigation = long-distance travel between distinct work areas. One navigation segment absorbs brief mid-travel stops/rotations and any initial startup waiting (do NOT split one transit by base zero/non-zero dips).",
        "- At a workspace, head/torso rotation, arm reach, brief base nudges (≲2s, no gripper event, no opposite-hand activity) count as MANIPULATION — never `move to ...`. A brief base shift (≲2s) immediately before OR after a state-change segment (open/close/pick/place) is absorbed INTO that state-change segment (e.g. nudging forward to reach a handle is part of the following `close`; a small retraction after placing is part of the preceding `place`). BUT if the base translates between two DIFFERENT target objects (e.g. one plywood sheet to the next, one sandal to the next), emit a short `move to <next target>` between the two pick/place segments.",
        "- Walking-while-actuating: if arm/body is visibly engaging a panel while base moves, the interval stays MANIPULATION (not navigation).",
        "",
        "## Rules (all MANDATORY)",
        "1. COVERAGE. Start 0.0, end at last visible `t=...s`, consecutive segments touch exactly (`next.start == prev.end`), every segment `end > start + 0.1s`. First segment starts at 0.0 (absorb initial idle waiting). Pre-arrival travel to ANY workspace must be its OWN `move to <workspace>` segment; never fold base-nonzero travel into the state-change segment that follows (including `open <X>`, `close <X>`, `pick up <X>`, `place <X>`). This applies both to the very first segment and to any state change later in the timeline that follows base-nonzero travel. Last segment must be a manipulation (absorb end-of-demo retreat).",
        "2. ACQUIRE/RELEASE SPLIT. For every same-hand (open_to_close t_a, close_to_open t_r) with Δt > 0.5s and a visibly carried object, emit TWO segments: acquire (ends ~t_a) + release (ends ~t_r); insert a navigation between them if the base translates to a different area during the carry. Always split at every magnitude. Only Δt ≤ 0.5s with no object carried stays one segment. Every full gripper cycle (open_to_close then close_to_open on the same hand) that carries a task object is ALWAYS a pick+place pair — use `pick up <X>` + `place <X> in/on <Y>`. NEVER use compound descriptions like `pick up and place X`. PAIRED label, base-zero, short Δt do NOT exempt this split. When multiple gripper cycles happen in quick succession (e.g. picking several small items from a basket one-by-one), EACH cycle gets its own pick and place segment. The idle time between one release and the next acquire on the same hand (arm returning from drop back to source) is absorbed into the NEXT `pick up <X>` segment as its approach — do NOT emit a filler segment for it. EVERY skill_description names an environment/object state change or travel intent (e.g. `pick up <X>`, `place <X>`, `open/close <X>`, `push/pour/wipe <X>`, `move to <Y>`). Descriptions that only describe the robot's state or inaction (e.g. `adjust X`, `reposition X`, `handle X`, `stationary manipulation`, `manipulation`, `robot is stationary`, `wait`, `idle`, `pause`, `observe`) are FORBIDDEN — absorb such intervals into the adjacent state-change or navigation segment.",
        "3. BIMANUAL TIMING. For two same-direction gripper events on different hands with Δt = |t_R − t_L| > 0.5s, split into two segments naming each instance by location/color/order/contents. Only Δt ≤ 0.5s stays one segment. Exceptions (stay one regardless of Δt): (a) mechanically-linked parts of ONE receptacle (two doors of one cabinet); (b) both hands lifting the SAME single physical object (large basket/crate/tray needing two hands).",
        "4. DESCRIPTION STYLE. `skill_description` names what is acted on and where, using natural everyday verbs: `pick up <X> from <source>`, `place <X> in/on <Y>`, `open/close <X>`, `move to <destination>` (never `move away` / `back away` / `leave`). NEVER mention hand/gripper side.",
        "5. ARM-MOTION BOUNDARIES.",
        "   - A state-change segment covers approach → change → completion. It ends only when the arm is VISUALLY clear of the target (not holding/touching/pushing/inside). A gripper transition is one signal, not the end — after a release the arm often keeps pushing the target.",
        "   - Whole-target completion: `open/close <X>` ends when the WHOLE X has settled. If X has multiple parts and only one has a gripper event, other parts are often silently actuated (body/forearm push, arm-sweep) — their transitions belong to the same segment. Extend until visible motion on every part has stopped.",
        "   - `open <X>` / `close <X>` is decided by CONTINUOUSLY watching the specific panel itself across consecutive frames: is THAT panel moving closed→open (label `open`) or open→closed (label `close`)? A demo may involve several distinct panels / doors / drawers; each is independent — a later cabinet interaction may be opening a different door that happens to be closed, not closing the one opened earlier. Do NOT infer direction from sequence order, pairing, or the fact that an earlier open on ANY structure happened. If the specific panel ends MORE open than it started → `open`; if MORE closed → `close`. If you cannot see the panel clearly enough to tell the direction, do NOT emit open/close — absorb the interval into the surrounding segment. Consistency check: if the robot's NEXT segment reaches INTO X (picks from X / places into X / interacts with contents), X must be in the OPEN state at that moment; therefore the preceding state-change on X must be an `open X`, never a `close X` (no re-opening happens before the pick). If you are about to emit `close X` followed soon by `pick <from X>` / `place <into X>` with no intervening `open X`, the label is wrong — change it to `open X`.",
        "   - Prep-only segments forbidden. Approach / align / re-grip / orient (no state change) is absorbed into the adjacent state-change segment.",
        "   - If an object is acquired at A and released at a visibly different location B, emit a navigation segment between them; never let acquire or release absorb the transport.",
        "6. NO HALLUCINATION. A state change requires direct evidence: a non-PAIRED gripper transition in the window, a PAIRED contact on an articulated panel that visibly actuates, or a visibly observed scene change. When a non-PAIRED gripper event is present, the description must name the specific object and outcome; no generic phrasing. A PAIRED brief contact DOES NOT by itself justify a `close <X>` or `open <X>` segment — the panel must visibly transition in the video; if you cannot see the panel, do NOT emit open/close for that PAIRED event (absorb it into the surrounding carry/place/pick instead). A `place <X>` segment should end within ~3s of the release event; do NOT stretch it backward to swallow travel / base-zero stops that are not at the final placement workspace (those belong to the preceding `move to ... while carrying <X>`).",
        "7. CHRONOLOGICAL ORDER. Sort by visible playback time; never reorder based on an inferred task plan.",
        "",
        "## Output JSON shape",
        '{"skill_timeline":[{"segment_id":"segment-001","start_time_sec":0.0,"end_time_sec":0.0,"skill_description":"concise natural-language instruction for this one observed skill"}]}',
    ]
    prompt_text = "\n".join(prompt_lines)
    user_content: str | list[dict[str, object]] = prompt_text
    if frame_data_urls or video_data_url:
        user_content = [{"type": "text", "text": prompt_text}]
        if video_data_url:
            user_content.append({"type": "video_url", "video_url": {"url": video_data_url}})
        else:
            user_content.extend(
                {"type": "image_url", "image_url": {"url": frame_data_url}}
                for frame_data_url in frame_data_urls or []
            )
    messages = [
        {
            "role": "system",
            "content": (
                "Return valid JSON only with a top-level `skill_timeline` list. "
                "No markdown, no commentary."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    extra_request_fields: dict[str, object] = {}
    if video_data_url and model.startswith("qwen"):
        extra_request_fields["response_format"] = {"type": "json_object"}
        extra_request_fields["enable_thinking"] = True
    elif model.startswith("gemini"):
        extra_request_fields["response_format"] = {"type": "json_object"}
        extra_request_fields["reasoning_effort"] = "high"
    return build_chat_completion_request(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        **extra_request_fields,
    )


def build_full_video_request(
    payload: dict,
    model: str,
    max_completion_tokens: int,
    frame_data_urls: list[str] | None = None,
    video_data_url: str | None = None,
    visual_mode: str = "head_only",
) -> dict:
    if frame_data_urls and video_data_url:
        raise ValueError("Provide either frame_data_urls or video_data_url, not both")

    if video_data_url:
        opening_line = "Analyze the full demo video, then return only JSON."
        input_mode_lines = [
            "## Input Mode",
            "- The input is one full chronological demo video, not sparse sampled frames.",
            "- The video covers the full observed demo from start to finish.",
            "- Segment strictly in the chronological order shown in the video.",
            "- Do not reorder actions based on an inferred goal or what would be a more logical plan.",
        ]
    else:
        opening_line = "Analyze the sampled demo frames, then return only JSON."
        input_mode_lines = [
            "## Input Mode",
            "- The sampled images are ordered from earliest to latest in time.",
            "- The current pipeline uses uniform sampling across the whole demo video.",
            "- Infer a best-effort time-ordered skill timeline from the sampled visual evidence.",
            "- Segment strictly in the chronological order shown by the input frames.",
            "- Do not reorder actions based on an inferred goal or what would be a more logical plan.",
        ]

    prompt_lines = [
        opening_line,
        "## Required Two-Level Segmentation",
        "- First create activity_timeline: a coarse chronological sequence where each activity is either navigation or manipulation.",
        "- activity_timeline is not a task-stage summary.",
        "- Each activity must be a contiguous single-mode interval: either navigation or manipulation.",
        "- It must not contain both travel and object/environment state change.",
        "- If an activity description would contain multiple intents joined by \"and\", split it into multiple activities.",
        "- An activity description must describe only what happens inside that activity time range.",
        "- Do not output two consecutive navigation activities; merge adjacent navigation intervals into one unless a state-changing manipulation occurs between them.",
        "- Sustained near-zero base[0:3] means the robot has arrived and may be performing manipulation. Do not skip manipulation that happens while base[0:3] is near zero, and do not classify a near-zero-base interval as navigation.",
        "- Navigation is long-distance movement to another work area, object, room, receptacle, surface, or operation location. Changing rooms, crossing thresholds, or moving through scene transitions is navigation when no object/environment state changes.",
        "- A navigation activity ends as soon as the top-bar base[0:3] values become all near zero after travel. Do not extend navigation through later stationary body, torso, arm, wrist, or gripper motion.",
        "- Manipulation is local operation after the robot has reached the work area. A manipulation activity may include small local repositioning during the operation, but it must not start with travel to the work area.",
        "- Then create skill_timeline: navigation activities become move to skills that preserve the parent activity's visually observed destination/context, and manipulation activities are split into the visible manipulation sub-actions.",
        "- Every skill_timeline item should reference its parent activity with parent_activity_id.",
        "- The activity_timeline and skill_timeline must both cover the entire uploaded video from 0.0 playback seconds to the last visible `t=...s` label.",
        "## Input Visual Layout",
        "- The video is a composite robot-observation view.",
        "- The large left panel is the primary head-camera view.",
        "- A small right-side column contains auxiliary wrist-camera views: left wrist on top and right wrist on bottom.",
        "- Use the wrist views only to clarify manipulation details such as contact, grasping, opening, closing, and placement.",
        "- All panels are synchronized at the same compressed playback time shown in the top bar.",
        "- There is no full action panel; only the top bar shows base[0:3] and compressed playback time.",
        "- A top bar shows three base action values for the current frame.",
        "- The three top-bar values are base[0:3], the robot base velocity controls.",
        "- The base action values can help distinguish navigation from manipulation: sustained nonzero base values usually indicate navigation only when the robot is traveling between work areas or toward the next operation location.",
        "- Base movement that happens inside an ongoing manipulation around the same object or workspace should be treated as local adjustment within that manipulation, not as a separate navigation skill.",
        "- The compressed playback time label in the top-right corner (rendered as text like `t=12.34s`) is authoritative for segment boundaries. Copy those visible `t=...s` values directly into `start_time_sec` and `end_time_sec` (so 12.34, not 12.0 or 12.5); never use source-video duration or hidden metadata. If you cannot read an exact label, estimate from the nearest visible ones; never infer seconds from frame numbers. The only valid timeline length is the visible `t=...s` playback scale.",
        *(_gripper_transition_data_only_lines(payload)),
        *input_mode_lines,
        "## Segmentation Rules",
        "- Return start_time_sec and end_time_sec only.",
        "- Do not return start_frame or end_frame.",
        "- Use one navigation skill whenever the primary intent is traveling between work areas or reaching the next operation location.",
        "- Navigation skill descriptions must start with move to, but should keep the visually observed destination/context from the parent activity.",
        "- Do not rename navigation based on how the robot travels or whether it is carrying something while traveling.",
        "- Treat travel to the next work area or operation location as a separate navigation segment, even if that travel enables a later manipulation.",
        "- End navigation when the main base movement to the next work area or target location is complete.",
        "- End navigation at the first sustained near-zero base[0:3] moment after travel, before any stationary body, torso, arm, wrist, or gripper motion.",
        "- Stationary body, torso, arm, or gripper adjustments after arrival belong to the following manipulation segment when they prepare for object interaction.",
        "- Do not create a separate skill for preparatory body alignment, torso alignment, arm alignment, aiming, repositioning, or local adjustment.",
        "- Do not output adjust position, reposition, align, aim, prepare, or similar adjustment-only descriptions as skill segments.",
        "- A manipulation segment must not begin with substantial travel; substantial travel at the beginning belongs to a preceding navigation segment.",
        "- A manipulation segment may include only local repositioning that happens after the robot has already reached the current work area and still serves the same manipulation intent.",
        "- Do not create a separate movement skill when small adjustments remain part of the same manipulation intent.",
        "- Do not output two consecutive navigation skill segments.",
        "- If adjacent skill segments are both navigation, merge them into one move-to skill unless a state-changing manipulation occurs between them.",
        "- If the robot stops moving its base and the body, arm, wrist, gripper, object, door, container, surface, or receptacle changes state, create a manipulation segment for that interval.",
        "- A manipulation segment must include a visible state change of an object, carried object, articulated structure, container, surface, or receptacle.",
        "- If no such state change occurs, classify the interval as navigation or merge it into adjacent navigation.",
        "- Do not split navigation only because the robot turns, passes through a doorway, enters or exits a room, crosses a threshold, changes scene, or carries an object.",
        "- Track carried objects continuously across segments.",
        "- Do not ignore objects held in the robot's hands.",
        "- When the robot is carrying an object, changing rooms, crossing a threshold, or moving through a scene transition is still navigation unless the robot changes the object/environment state.",
        "- If the robot sets down, releases, picks up, re-grasps, inserts, removes, opens, closes, or otherwise changes the state of a carried object or nearby object, create a manipulation segment for that state change.",
        "- Paired open/close rule: emit a dedicated closing-action segment ONLY when ALL of the following hold: (a) earlier in the video the robot performed a sustained open action on an articulated structure (door, cabinet, drawer, hatch, lid, refrigerator, oven, panel, flap, or similar) and the structure stayed visibly open for a meaningful duration afterwards, (b) later in the video you can directly see the same structure change from open to closed, AND (c) the closing is clearly driven by the robot's arm or body (a hand reaching for the handle, a push with a closed gripper, or a deliberate body contact against the panel). If the earlier interaction with the structure was itself a brief PAIRED open-close pair (a short touch that opens and immediately releases the handle within ~2s), treat the PAIRED pair as one complete door-operation action and do NOT hallucinate a separate later close segment — the PAIRED event already accounts for both the open and release. Never emit a close segment based only on base-movement inference or because the prompt mentions closing; the closing must be visually verified by the panel actually moving from open to closed. When a close segment is valid, it covers reaching for the handle/edge, pushing or pulling the panel shut, and releasing contact — it is never navigation and must not be merged into an adjacent place/release or move-away segment.",
        "- Do not let scene transitions, camera viewpoint changes, or room changes override object-state tracking.",
        "- Keep different manipulation intents as different skills whenever the primary manipulation intent changes.",
        "- Do not describe any segment with mixed navigation/manipulation wording.",
        "- If a description would combine movement to a target with object interaction, split it into a move to segment followed by the manipulation segment.",
        "- If the primary intent is grasping or pickup after local repositioning, describe it as the manipulation skill rather than as navigation.",
        "- For each manipulation, distinguish these phases:",
        "- preparation/contact phase: reaching, aligning, touching, securing, or otherwise preparing contact.",
        "- state-changing phase: acquiring an object, placing an object, opening/closing something, inserting/removing something, or otherwise changing the object/environment state.",
        "- completion/recovery phase: stabilizing, releasing, withdrawing, returning the hand/body to neutral, or otherwise finishing after the state change.",
        "- Only create a skill for the state-changing intent.",
        "- Preparation/contact and completion/recovery phases should be included in the same skill when they are contiguous and serve that same state-changing intent.",
        "- If a preparation/contact or completion/recovery phase does not change the task state and is not needed to define the state-changing intent, omit it rather than making a separate skill.",
        "- Do not create standalone skills for motion phases that only prepare for or recover from the same state-changing manipulation.",
        "- Do not create segments for bookkeeping, observation, checking progress, or other non-actions.",
        "- Do not force manipulation into a fixed skill vocabulary.",
        "- Let later clustering and naming merge related manipulation descriptions into canonical skills.",
        "- Focus on segmenting what the robot is actually doing into coherent skill units, then describe each unit clearly for later clustering.",
        "- `skill_description` is a downstream training prompt for the policy model, not a canonical skill name.",
        "- It must be a complete but single-intent natural-language training instruction for this observed skill segment.",
        "- Include key sub-steps needed to accomplish the same state-changing or navigation intent, such as relevant approach, contact, grasp, release, placement, or carried-object context.",
        "- Do not make it so terse that necessary action context appears only in evidence.",
        "- Evidence is only for justification; do not put essential action steps only in evidence.",
        "- Keep necessary visible context such as destination, object, receptacle, support surface, or spatial relation.",
        "- Do not normalize skill_description into generic canonical labels.",
        "- Do not include irrelevant scene details.",
        "- Do not combine multiple state-changing intents.",
        "- Later clustering will map these training prompts to canonical names.",
        "## Critical: Never Do These",
        "**HARD RULE — 3-SEGMENT SPLIT FOR PICK+TRAVEL+PLACE**: before emitting JSON, for EVERY manipulation segment you are about to emit, scan whether its `[start_time_sec, end_time_sec]` window contains (a) a non-PAIRED `open_to_close` on some hand AND (b) a non-PAIRED `close_to_open` on the same hand, with sustained non-zero base[0:3] travel between them. If yes, this segment is STRUCTURALLY A COMPOUND and must be REJECTED. Replace it with three consecutive segments: first, a manipulation segment ending at the arrival at the acquire location that covers only the `open_to_close` (describe as `grasp/pick X …`); second, a navigation segment covering only the sustained travel (describe as `move to <destination> while carrying X`); third, a manipulation segment starting at the arrival at the release location that covers only the `close_to_open` (describe as `place X …` or `release X …`). The three segments together must cover the same total time window as the compound would have. Descriptions that join acquire and release with `and place`, `and release`, `then place`, `, place`, `carry it to Y and place it`, `carry X and place X on Y`, or any similar phrasing combining a pick verb with a place/release verb in the same segment are ALWAYS wrong and must be split.",
        "- Never reorder actions according to an inferred goal or a more logical plan; follow the video chronology only.",
        "- Never output carry or carry to as a skill description; if the robot moves while holding an object, describe the movement as move to.",
        "- Never absorb a VISIBLY OCCURRING closing of an articulated structure (door, cabinet, drawer, hatch, lid, refrigerator, oven, etc.) into an adjacent place/release segment or into a move-away navigation segment; it must be its own manipulation segment. But symmetrically, never hallucinate a close segment that is not visually observable — if the video does not actually show the panel changing from open to closed, do not emit a close segment even if you expect one (no close rule compels you to invent one).",
        "- Never emit a manipulation skill whose time window contains no state-change evidence. A manipulation skill is only valid when its window either (a) contains at least one non-PAIRED gripper transition event from the list above, or (b) contains a visibly produced state change on an object, container, articulated structure, receptacle, or surface. Motion-only intervals — alignment, aiming, examination, arrangement, repositioning, or any preparation/recovery without a state change — must be absorbed into the adjacent state-changing skill or merged into neighboring navigation; do not emit them as their own skills.",
        "The first segment must start at 0.0 playback seconds, even if the first visible time label is slightly after 0.0; attach any startup, waiting, weak motion, or partially visible beginning to the first meaningful skill segment.",
        "**COVERAGE IS MANDATORY**: the skill_timeline must cover the entire uploaded video from 0.0 playback seconds to the last visible `t=...s` label, and for every consecutive pair of segments, the later segment's `start_time_sec` MUST equal the earlier segment's `end_time_sec`. Any non-zero gap is a structural error. Before emitting JSON, scan your skill_timeline: if any pair has `next.start_time_sec > prev.end_time_sec`, extend the earlier segment's `end_time_sec` (or the later segment's `start_time_sec`) so the gap is exactly zero.",
        "**NO ZERO-DURATION SEGMENTS**: every segment must satisfy `end_time_sec > start_time_sec + 0.1`. Never collapse a real action (grasp, release, open, close, place) into a single instant just to make the timeline touch cleanly. If two actions happen close in time, give each its real visible duration from the video (even 0.3s is fine) and let the neighbors' boundaries be their actual endpoints; do not force both endpoints onto the same timestamp.",
        "Do not omit sub-manipulation actions that are visibly part of completing a manipulation intent, such as contact, grasping, opening, closing, placement, release, or containment.",
        "The output skill_timeline must be sorted by increasing visible playback time and must describe what happens in that order; start a new segment whenever the primary action intent changes, and split segments when they cross from navigation into manipulation or when one completed manipulation intent transitions into a different one.",
        "Do not return an empty skill_timeline unless there is truly no visible robot activity.",
        "Use this exact top-level JSON shape:",
        "Field values in this JSON shape are type placeholders only; do not copy placeholder descriptions or placeholder times.",
        (
            '{"activity_timeline":['
            '{"activity_id":"activity-001","activity_type":"navigation | manipulation","start_time_sec":0.0,"end_time_sec":0.0,'
            '"description":"concise natural-language summary of the coarse observed activity","evidence":"brief visual evidence"}],'
            '"skill_timeline":['
            '{"segment_id":"segment-001","parent_activity_id":"activity-001","start_time_sec":0.0,"end_time_sec":0.0,'
            '"skill_description":"concise downstream training prompt for this one observed skill","evidence":"brief visual evidence","confidence":0.5}'
            ']}'
        ),
        "Each skill_timeline item should describe one contiguous time span.",
        "The segment boundaries must be defined by the visible compressed playback time labels in the video.",
        "Use the visible `t=...s` text in the video as the only timing source.",
        "Source-video duration is irrelevant to the model output timeline; source time can be reconstructed later from video_context.time_scale.",
        "Keep skill_description natural, concrete, and usable as a downstream training prompt; do not normalize it into a canonical name.",
    ]

    prompt_text = "\n".join(prompt_lines)
    user_content: str | list[dict[str, object]] = prompt_text
    if frame_data_urls or video_data_url:
        user_content = [{"type": "text", "text": prompt_text}]
        if video_data_url:
            user_content.append({"type": "video_url", "video_url": {"url": video_data_url}})
        else:
            user_content.extend(
                {"type": "image_url", "image_url": {"url": frame_data_url}}
                for frame_data_url in frame_data_urls or []
            )
    messages = [
        {
            "role": "system",
            "content": (
                "You are Claw's offline demo skill discovery assistant. "
                "Return valid JSON only. "
                "Do not ask follow-up questions. "
                "Do not add markdown fences. "
                "The top-level JSON object must contain: skill_timeline. "
                "skill_timeline must be a list."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    extra_request_fields: dict[str, object] = {}
    if video_data_url and model.startswith("qwen"):
        extra_request_fields["response_format"] = {"type": "json_object"}
        extra_request_fields["enable_thinking"] = True
    elif model.startswith("gemini"):
        extra_request_fields["response_format"] = {"type": "json_object"}
        extra_request_fields["reasoning_effort"] = "high"
    return build_chat_completion_request(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        **extra_request_fields,
    )

def build_frame_fallback_request(
    deployment: str,
    payload: dict,
    frame_summaries: list[str],
    max_completion_tokens: int = 4096,
) -> dict:
    validated_payload = validate_timeline_prediction(payload)
    user_content = json.dumps(
        {"payload": validated_payload, "frame_summaries": frame_summaries},
        sort_keys=True,
    )
    return build_chat_completion_request(
        model=deployment,
        messages=[{"role": "user", "content": user_content}],
        max_completion_tokens=max_completion_tokens,
    )


def _redact_data_url(url: str) -> str:
    if not url.startswith("data:"):
        return url
    prefix = url.split(",", 1)[0]
    return f"{prefix},<omitted data URL; chars={len(url)}>"


def _sanitize_request_for_artifact(request: dict) -> dict:
    sanitized = deepcopy(request)
    messages = sanitized.get("messages")
    if not isinstance(messages, list):
        return sanitized
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            for key in ("video_url", "image_url"):
                value = item.get(key)
                if not isinstance(value, dict):
                    continue
                url = value.get("url")
                if isinstance(url, str):
                    value["url"] = _redact_data_url(url)
    return sanitized


def _content_text_for_artifact(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            text_parts.append(item["text"])
    return "\n\n".join(text_parts)


def _write_prompt_artifacts(
    request: dict,
    *,
    prompt_artifact_dir: Path,
    prompt_artifact_stem: str,
    attempt: int,
) -> None:
    prompt_artifact_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{prompt_artifact_stem}_attempt_{attempt:02d}"
    role_texts: dict[str, list[str]] = {}
    for message in request.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if not isinstance(role, str):
            continue
        text = _content_text_for_artifact(message.get("content"))
        if text:
            role_texts.setdefault(role, []).append(text)
    for role, texts in role_texts.items():
        (prompt_artifact_dir / f"{prefix}_{role}.txt").write_text(
            "\n\n--- message ---\n\n".join(texts),
            encoding="utf-8",
        )
    (prompt_artifact_dir / f"{prefix}_request.json").write_text(
        json.dumps(_sanitize_request_for_artifact(request), indent=2),
        encoding="utf-8",
    )


def run_full_video_inference(
    payload: dict,
    model: str,
    max_completion_tokens: int,
    client: object | None = None,
    frame_data_urls: list[str] | None = None,
    video_data_url: str | None = None,
    visual_mode: str = "head_only",
    prompt_artifact_dir: Path | None = None,
    prompt_artifact_stem: str = "inference",
) -> dict:
    import os as _os

    prompt_mode = _os.getenv("CLAW_PROMPT_MODE", "minimal").lower()
    if prompt_mode == "minimal":
        request = build_minimal_video_request(
            payload=payload,
            model=model,
            max_completion_tokens=max_completion_tokens,
            frame_data_urls=frame_data_urls,
            video_data_url=video_data_url,
        )
    else:
        request = build_full_video_request(
            payload=payload,
            model=model,
            max_completion_tokens=max_completion_tokens,
            frame_data_urls=frame_data_urls,
            video_data_url=video_data_url,
            visual_mode=visual_mode,
        )
    if prompt_artifact_dir is not None:
        _write_prompt_artifacts(
            request,
            prompt_artifact_dir=prompt_artifact_dir,
            prompt_artifact_stem=prompt_artifact_stem,
            attempt=0,
        )
    if model.startswith("gemini") and (video_data_url or frame_data_urls):
        from mobiagent.discovery.behavior.gemini_client import run_native_gemini_inference

        raw_text = run_native_gemini_inference(
            messages=request["messages"],
            model=model,
            video_data_url=video_data_url,
            frame_data_urls=frame_data_urls,
            max_output_tokens=max_completion_tokens,
            enable_thinking=False,
            response_mime_type="application/json",
        )
        return parse_timeline_response_text(raw_text)
    response = execute_chat_completion(request, client=client)
    raw_text = extract_first_message_text(response)
    return parse_timeline_response_text(raw_text)


__all__ = [
    "build_frame_fallback_request",
    "build_full_video_request",
    "parse_timeline_response_text",
    "run_full_video_inference",
    "validate_timeline_prediction",
]
