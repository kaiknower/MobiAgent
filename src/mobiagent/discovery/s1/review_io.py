from __future__ import annotations

import json
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def build_review_rows(predictions: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for prediction in predictions:
        if "skill_timeline" not in prediction:
            raise ValueError("prediction['skill_timeline'] is required")
        row = {
            "episode_id": prediction["episode_id"],
            "auto_skill_timeline": prediction["skill_timeline"],
            "final_skill_timeline": None,
        }
        if prediction.get("task_id") is not None:
            row["task_id"] = prediction["task_id"]
        for key in ("base_zero_intervals", "gripper_transitions", "bimanual_events"):
            if prediction.get(key) is not None:
                row[key] = prediction[key]
        rows.append(row)
    return rows


def finalize_timelines(review_path: Path) -> list[dict]:
    rows: list[dict] = []
    with review_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            final_skill_timeline = row.get("final_skill_timeline")
            if final_skill_timeline is None:
                final_skill_timeline = row.get("auto_skill_timeline", [])
            finalized_row = {
                "episode_id": row["episode_id"],
                "final_skill_timeline": final_skill_timeline,
            }
            if row.get("task_id") is not None:
                finalized_row["task_id"] = row["task_id"]
            rows.append(finalized_row)
    return rows


def write_finalized_timelines(review_path: Path, output_path: Path) -> list[dict]:
    rows = finalize_timelines(review_path)
    write_jsonl(output_path, rows)
    return rows


def _format_base_zero_interval_lines(row: dict) -> list[str]:
    intervals = row.get("base_zero_intervals") or []
    if not intervals:
        return []
    out = ["  base_zero_intervals:"]
    for interval in intervals:
        try:
            start = float(interval["start_time_sec"])
            end = float(interval["end_time_sec"])
            duration = float(interval.get("duration_sec", end - start))
        except (KeyError, TypeError, ValueError):
            continue
        checkpoint_id = interval.get("checkpoint_id", "")
        prefix = f"    - {checkpoint_id}:" if checkpoint_id else "    -"
        out.append(f"{prefix} [{start:.2f}s - {end:.2f}s] dur={duration:.2f}s")
    return out


def _format_gripper_transition_lines(row: dict) -> list[str]:
    transitions = row.get("gripper_transitions") or []
    bimanual = row.get("bimanual_events") or []
    if not transitions and not bimanual:
        return []
    out: list[str] = []
    if transitions:
        bimanual_times = {
            round(float(event["t"]), 2)
            for event in bimanual
            if isinstance(event, dict) and "t" in event
        }
        out.append("  gripper_transitions:")
        for event in transitions:
            try:
                event_time = float(event["t"])
            except (KeyError, TypeError, ValueError):
                continue
            side = event.get("side", "?")
            direction = event.get("direction", "?")
            suffix = "  (bimanual)" if round(event_time, 2) in bimanual_times else ""
            out.append(f"    - t={event_time:.2f}s  {side} {direction}{suffix}")
    if bimanual:
        out.append("  bimanual_events:")
        for event in bimanual:
            try:
                event_time = float(event["t"])
            except (KeyError, TypeError, ValueError):
                continue
            sides = ",".join(event.get("sides") or [])
            direction = event.get("direction", "?")
            out.append(f"    - t={event_time:.2f}s  sides=[{sides}]  {direction}")
    return out


def build_timeline_review_text(rows: list[dict]) -> str:
    lines: list[str] = []
    for row in rows:
        lines.append(f"{row.get('task_id')} / {row.get('episode_id')}")
        timeline = row.get("final_skill_timeline") or row.get("auto_skill_timeline") or []
        for segment in timeline:
            description = segment.get("skill_description")
            start_time = float(segment.get("start_time_sec", 0.0))
            end_time = float(segment.get("end_time_sec", 0.0))
            lines.append(f"- {description} [{start_time:.2f}s - {end_time:.2f}s]")
        lines.extend(_format_base_zero_interval_lines(row))
        lines.extend(_format_gripper_transition_lines(row))
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def write_timeline_review(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_timeline_review_text(rows), encoding="utf-8")


def build_readable_predictions_text(
    *,
    run_dir: Path,
    predictions: list[dict],
    semantic_skill_items: list[dict] | None = None,
    semantic_skill_groups: list[dict] | None = None,
) -> str:
    items = semantic_skill_items or []
    groups = semantic_skill_groups or []
    canonical_by_item = {
        item_id: group.get("canonical_name", "UNMAPPED")
        for group in groups
        for item_id in group.get("item_ids", [])
    }
    item_by_key = {
        (item.get("task_id"), item.get("episode_id"), item.get("segment_id")): item
        for item in items
    }
    lines: list[str] = ["# Readable Predictions", "", f"Run: `{run_dir}`", ""]
    for prediction in predictions:
        task_id = prediction.get("task_id", "unknown_task")
        episode_id = prediction.get("episode_id", "unknown_episode")
        lines.extend([f"## {task_id} / {episode_id}", ""])

        video_context = prediction.get("video_context") or {}
        if video_context:
            lines.extend(
                [
                    "### Video Context",
                    f"- source_duration_sec: `{video_context.get('source_duration_sec')}`",
                    f"- compressed_duration_sec: `{video_context.get('compressed_duration_sec')}`",
                    f"- time_scale: `{video_context.get('time_scale')}`",
                    f"- navigation_boundary_snaps: `{len(prediction.get('navigation_boundary_snaps', []))}`",
                    "",
                ]
            )

        snaps = prediction.get("navigation_boundary_snaps") or []
        if snaps:
            lines.append("### Navigation Boundary Snaps")
            for snap in snaps:
                lines.append(
                    f"- `{snap.get('segment_id')}` {snap.get('old_end_time_sec')}s -> "
                    f"{snap.get('new_end_time_sec')}s; next=`{snap.get('next_segment_id')}`; "
                    f"source_stop_frame=`{snap.get('source_stop_frame')}`; "
                    f"base_epsilon=`{snap.get('base_epsilon')}`; stable_frames=`{snap.get('stable_frames')}`"
                )
            lines.append("")

        base_zero_intervals = prediction.get("base_zero_intervals") or []
        if base_zero_intervals:
            lines.append("### Base-Zero Intervals")
            for interval in base_zero_intervals:
                try:
                    start = float(interval["start_time_sec"])
                    end = float(interval["end_time_sec"])
                    duration = float(interval.get("duration_sec", end - start))
                except (KeyError, TypeError, ValueError):
                    continue
                checkpoint_id = interval.get("checkpoint_id", "")
                lines.append(
                    f"- `{checkpoint_id}` [{start:.2f}s - {end:.2f}s] dur={duration:.2f}s"
                )
            lines.append("")

        gripper_transitions = prediction.get("gripper_transitions") or []
        bimanual_events = prediction.get("bimanual_events") or []
        if gripper_transitions or bimanual_events:
            bimanual_times = {
                round(float(event["t"]), 2)
                for event in bimanual_events
                if isinstance(event, dict) and "t" in event
            }
            lines.append("### Gripper Transitions")
            for event in gripper_transitions:
                try:
                    event_time = float(event["t"])
                except (KeyError, TypeError, ValueError):
                    continue
                side = event.get("side", "?")
                direction = event.get("direction", "?")
                suffix = "  (bimanual)" if round(event_time, 2) in bimanual_times else ""
                lines.append(f"- t={event_time:.2f}s  `{side}` `{direction}`{suffix}")
            if bimanual_events:
                lines.append("")
                lines.append("### Bimanual Coincident Events")
                for event in bimanual_events:
                    try:
                        event_time = float(event["t"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    sides = ",".join(event.get("sides") or [])
                    direction = event.get("direction", "?")
                    lines.append(f"- t={event_time:.2f}s  sides=[{sides}]  `{direction}`")
            lines.append("")

        lines.append("### Activity Timeline")
        activities = prediction.get("activity_timeline") or []
        if not activities:
            lines.append("- none")
        for activity in activities:
            lines.append(
                f"- `{activity.get('activity_id')}` `{activity.get('activity_type')}` "
                f"[{float(activity.get('start_time_sec', 0.0)):.2f}s - "
                f"{float(activity.get('end_time_sec', 0.0)):.2f}s] "
                f"{activity.get('description', '')}"
            )
            if activity.get("evidence"):
                lines.append(f"  - evidence: {activity.get('evidence')}")
        lines.append("")

        lines.append("### Skill Timeline")
        segments = prediction.get("skill_timeline") or []
        if not segments:
            lines.append("- none")
        for segment in segments:
            segment_id = segment.get("segment_id")
            item = item_by_key.get((task_id, episode_id, segment_id), {})
            canonical_name = canonical_by_item.get(item.get("item_id"), "UNMAPPED")
            lines.append(
                f"- `{segment_id}` -> `{canonical_name}` "
                f"[{float(segment.get('start_time_sec', 0.0)):.2f}s - "
                f"{float(segment.get('end_time_sec', 0.0)):.2f}s] "
                f"{segment.get('skill_description', '')}"
            )
            details: list[str] = []
            if item.get("activity_type"):
                details.append(f"activity_type: `{item.get('activity_type')}`")
            if segment.get("parent_activity_id"):
                details.append(f"parent: `{segment.get('parent_activity_id')}`")
            if item.get("item_id"):
                details.append(f"item_id: `{item.get('item_id')}`")
            if segment.get("confidence") is not None:
                details.append(f"confidence: `{segment.get('confidence')}`")
            if details:
                lines.append("  - " + ", ".join(details))
            if segment.get("evidence"):
                lines.append(f"  - evidence: {segment.get('evidence')}")
        lines.append("")

    if groups:
        lines.append("## Semantic Groups")
        for group in groups:
            lines.append(
                f"- `{group.get('canonical_name')}` `{group.get('activity_type')}`: "
                f"{len(group.get('item_ids', []))} item(s)"
            )
            for description in group.get("descriptions", []):
                lines.append(f"  - {description}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_readable_predictions(
    output_path: Path,
    *,
    run_dir: Path,
    predictions: list[dict],
    semantic_skill_items: list[dict] | None = None,
    semantic_skill_groups: list[dict] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_readable_predictions_text(
            run_dir=run_dir,
            predictions=predictions,
            semantic_skill_items=semantic_skill_items,
            semantic_skill_groups=semantic_skill_groups,
        ),
        encoding="utf-8",
    )


__all__ = [
    "build_readable_predictions_text",
    "build_review_rows",
    "build_timeline_review_text",
    "finalize_timelines",
    "write_finalized_timelines",
    "write_jsonl",
    "write_readable_predictions",
    "write_timeline_review",
]
