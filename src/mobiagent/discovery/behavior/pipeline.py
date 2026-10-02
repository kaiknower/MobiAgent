import base64
import cv2
import json
import os
import subprocess
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .clustering import build_skill_description_items
from .clustering import run_cluster_naming
from .clustering import summarize_timeline_skill_descriptions
from .clustering import summarize_value_frequencies
from .config import DEFAULT_TASK_IDS
from .dataset_selection import select_first_demo_per_task
from .head_action_video import build_head_with_wrist_overlay_video
from .head_action_video import _load_actions
from .inference import run_full_video_inference
from .review_io import build_review_rows
from .review_io import write_timeline_review
from .review_io import write_finalized_timelines
from .review_io import write_jsonl
from .review_io import write_readable_predictions
from .task_context import load_task_instruction


DEFAULT_OUTPUT_ROOT = Path("outputs/discovery/behavior")
MERGED_VIDEO_SAMPLE_COUNT = 10
CLUSTER_NAMING_MODEL = os.getenv("MOBIAGENT_NAMING_MODEL", "").strip() or os.getenv("OPENAI_MODEL", "").strip()
CLUSTER_NAMING_MAX_COMPLETION_TOKENS = 8192
QWEN_FULL_VIDEO_MODEL = "qwen3.6-plus"
DEFAULT_GEMINI_FULL_VIDEO_MODEL = "gemini-3.1-pro-preview"
GPT_FRAME_MODEL = os.getenv("MOBIAGENT_FRAME_MODEL", "").strip() or os.getenv("OPENAI_MODEL", "").strip()
MAX_DASHSCOPE_DATA_URI_BYTES = 10 * 1024 * 1024
DATA_VIDEO_MP4_BASE64_PREFIX_BYTES = len("data:video/mp4;base64,")
VIDEO_PLAYBACK_TIME_SCALE = 5.0


def _create_run_dir(output_root: Path) -> Path:
    base_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    suffix = 0

    while True:
        run_dir = output_root / base_name if suffix == 0 else output_root / f"{base_name}_{suffix:02d}"
        try:
            (run_dir / "manifests").mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            suffix += 1
            continue
        return run_dir


def _append_pipeline_log(run_dir: Path, message: str) -> str:
    log_path = run_dir / "logs" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    if hasattr(now, "isoformat"):
        timestamp = now.isoformat(timespec="seconds")
    else:
        timestamp = str(now)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
    return str(log_path)


def _write_failure_artifacts(run_dir: Path, exc: Exception) -> dict[str, object]:
    error_type = type(exc).__name__
    error_message = str(exc)
    trace_text = traceback.format_exc()
    _append_pipeline_log(run_dir, f"FAILED {error_type}: {error_message}")
    trace_path = run_dir / "logs" / "failure_traceback.txt"
    trace_path.write_text(trace_text, encoding="utf-8")
    reason_path = run_dir / "logs" / "failure_reason.txt"
    reason_path.write_text(f"{error_type}: {error_message}\n", encoding="utf-8")
    return {
        "failure_reason": f"{error_type}: {error_message}",
        "failure_traceback_path": str(trace_path),
        "failure_reason_path": str(reason_path),
    }


def _has_dashscope_api_key() -> bool:
    return bool(os.getenv("DASHSCOPE_API_KEY", "") or os.getenv("ALIBABA_API_KEY", ""))


def _has_gemini_api_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", ""))


def _has_openai_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def _has_video_inference_api_key() -> bool:
    """Any provider that can ingest a video data URL (DashScope qwen or Gemini)."""
    return _has_dashscope_api_key() or _has_gemini_api_key()


def _has_any_model_api_key() -> bool:
    return _has_dashscope_api_key() or _has_gemini_api_key() or _has_openai_api_key()


def choose_inference_mode(
    video_path: Path,
    max_video_bytes: int,
    duration_sec: float,
    max_duration_sec: float,
) -> str:
    video_size_bytes = video_path.stat().st_size
    if video_size_bytes > max_video_bytes or duration_sec > max_duration_sec:
        return "frames"
    return "full_video"


def sample_video_frames_as_data_urls(video_path: Path, sample_count: int) -> list[str]:
    from .video_merge import sample_video_frames_as_data_urls as _sample_video_frames_as_data_urls

    return _sample_video_frames_as_data_urls(video_path=video_path, sample_count=sample_count)


def build_inference_video(*, head_video: Path, parquet_path: Path, output_video: Path) -> None:
    task_id = head_video.parents[1].name
    episode_id = head_video.stem
    video_root = head_video.parents[2]
    build_head_with_wrist_overlay_video(
        head_video=head_video,
        left_wrist_video=video_root / task_id / "observation.images.rgb.left_wrist" / f"{episode_id}.mp4",
        right_wrist_video=video_root / task_id / "observation.images.rgb.right_wrist" / f"{episode_id}.mp4",
        parquet_path=parquet_path,
        output_video=output_video,
        playback_time_scale=VIDEO_PLAYBACK_TIME_SCALE,
    )


def compress_video_for_upload(*, input_video: Path, output_video: Path) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    profiles = [
        ("setpts=(PTS-STARTPTS)/5,fps=fps=30:start_time=0", ["-crf", "23"]),
        ("setpts=(PTS-STARTPTS)/5,fps=fps=30:start_time=0", ["-crf", "26"]),
        ("setpts=(PTS-STARTPTS)/5,fps=fps=30:start_time=0", ["-crf", "28"]),
        ("setpts=(PTS-STARTPTS)/5,fps=fps=30:start_time=0,scale=trunc(iw*0.9/2)*2:trunc(ih*0.9/2)*2", ["-crf", "23"]),
        ("setpts=(PTS-STARTPTS)/5,fps=fps=30:start_time=0,scale=trunc(iw*0.8/2)*2:trunc(ih*0.8/2)*2", ["-crf", "23"]),
        ("setpts=(PTS-STARTPTS)/5,fps=fps=30:start_time=0,scale=trunc(iw*0.7/2)*2:trunc(ih*0.7/2)*2", ["-crf", "23"]),
    ]
    for video_filter, quality_args in profiles:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_video),
                "-vf",
                video_filter,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                *quality_args,
                "-an",
                str(output_video),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        estimated_data_uri_bytes = DATA_VIDEO_MP4_BASE64_PREFIX_BYTES + ((output_video.stat().st_size + 2) // 3) * 4
        if estimated_data_uri_bytes <= MAX_DASHSCOPE_DATA_URI_BYTES:
            return
    raise RuntimeError(
        f"compressed upload video still exceeds {MAX_DASHSCOPE_DATA_URI_BYTES} bytes: {output_video.stat().st_size}"
    )


def video_file_as_data_url(*, video_path: Path) -> str:
    return "data:video/mp4;base64," + base64.b64encode(video_path.read_bytes()).decode("ascii")


def _read_video_duration_sec(video_path: Path) -> float | None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps <= 0.0 or frame_count <= 0.0:
            return None
        return float(frame_count / fps)
    finally:
        capture.release()


def _build_video_context(
    *,
    task_id: str,
    task_average_source_duration_sec: float | None,
    source_video: Path,
    compressed_video: Path | None = None,
) -> dict[str, object]:
    context: dict[str, object] = {
        "task_id": task_id,
    }
    if task_average_source_duration_sec is not None:
        context["task_average_source_duration_sec"] = round(float(task_average_source_duration_sec), 2)
    source_duration = _read_video_duration_sec(source_video)
    if source_duration is not None:
        context["source_duration_sec"] = round(source_duration, 2)
    if compressed_video is not None:
        compressed_duration = _read_video_duration_sec(compressed_video)
        if compressed_duration is not None:
            context["compressed_duration_sec"] = round(compressed_duration, 2)
            if source_duration is not None and compressed_duration > 0:
                context["time_scale"] = round(source_duration / compressed_duration, 2)
    return context


def _build_task_average_source_durations(selected_demos: list[object]) -> dict[str, float]:
    durations_by_task: dict[str, list[float]] = {}
    for selected_demo in selected_demos:
        duration = _read_video_duration_sec(selected_demo.head_video_path)
        if duration is None:
            continue
        durations_by_task.setdefault(selected_demo.task_id, []).append(duration)
    return {
        task_id: sum(durations) / len(durations)
        for task_id, durations in durations_by_task.items()
        if durations
    }


def _backfill_segment_frames_from_times(*, prediction: dict, source_video: Path) -> dict:
    capture = cv2.VideoCapture(str(source_video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    for segment in prediction.get("skill_timeline", []):
        if not isinstance(segment, dict):
            continue
        if "start_time_sec" not in segment or "end_time_sec" not in segment:
            continue
        segment["start_frame"] = int(round(float(segment["start_time_sec"]) * fps))
        segment["end_frame"] = int(round(float(segment["end_time_sec"]) * fps))
    return prediction


def _activity_type_by_id(prediction: dict) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for activity in prediction.get("activity_timeline", []):
        if not isinstance(activity, dict):
            continue
        activity_id = activity.get("activity_id")
        activity_type = activity.get("activity_type")
        if isinstance(activity_id, str) and isinstance(activity_type, str):
            lookup[activity_id] = activity_type
    return lookup


def _is_navigation_segment(segment: dict, activity_types: dict[str, str]) -> bool:
    parent_activity_id = segment.get("parent_activity_id")
    if isinstance(parent_activity_id, str) and activity_types.get(parent_activity_id) == "navigation":
        return True
    description = segment.get("skill_description")
    return isinstance(description, str) and description.strip().lower().startswith("move to")


def _read_video_fps(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    try:
        return float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    finally:
        capture.release()


def _first_sustained_base_stop_frame(
    *,
    actions: list[list[float]],
    start_frame: int,
    end_frame: int,
    search_window_frames: int,
    base_epsilon: float,
    stable_frames: int,
) -> int | None:
    if not actions:
        return None
    start = max(0, min(start_frame, len(actions) - 1))
    stop = min(len(actions), max(end_frame + search_window_frames, start + stable_frames))
    seen_motion = False
    latest_start = max(start, stop - stable_frames)
    for frame_index in range(start, latest_start + 1):
        base = actions[frame_index][:3]
        is_zero = all(abs(float(value)) <= base_epsilon for value in base)
        if not is_zero:
            seen_motion = True
            continue
        if not seen_motion:
            continue
        if all(
            all(abs(float(value)) <= base_epsilon for value in actions[next_frame][:3])
            for next_frame in range(frame_index, min(frame_index + stable_frames, len(actions)))
        ):
            return frame_index
    return None


def snap_navigation_boundaries_to_base_stops(
    *,
    prediction: dict,
    parquet_path: Path,
    source_video: Path,
    playback_time_scale: float = VIDEO_PLAYBACK_TIME_SCALE,
    base_epsilon: float = 0.03,
    stable_frames: int | None = None,
    min_sustained_base_stop_sec: float = 1.0,
    search_window_sec: float = 2.0,
    min_navigation_duration_sec: float = 1.0,
    max_snap_back_sec: float = 4.0,
) -> dict:
    actions = _load_actions(parquet_path)
    source_fps = _read_video_fps(source_video)
    if source_fps <= 0.0 or playback_time_scale <= 0.0:
        return prediction

    # Require a meaningful sustained stop — at least `min_sustained_base_stop_sec`
    # seconds of continuous base-zero frames — so that brief posture-adjustment
    # pauses during travel don't get misread as arrival.
    if stable_frames is None:
        stable_frames = max(1, int(round(min_sustained_base_stop_sec * source_fps)))

    activity_types = _activity_type_by_id(prediction)
    segments = prediction.get("skill_timeline", [])
    if not isinstance(segments, list):
        return prediction

    search_window_frames = int(round(search_window_sec * source_fps))
    snap_records: list[dict[str, object]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or not _is_navigation_segment(segment, activity_types):
            continue
        if "start_time_sec" not in segment or "end_time_sec" not in segment:
            continue
        start_time = float(segment["start_time_sec"])
        end_time = float(segment["end_time_sec"])
        start_frame = int(round(start_time * playback_time_scale * source_fps))
        end_frame = int(round(end_time * playback_time_scale * source_fps))
        stop_frame = _first_sustained_base_stop_frame(
            actions=actions,
            start_frame=start_frame,
            end_frame=end_frame,
            search_window_frames=search_window_frames,
            base_epsilon=base_epsilon,
            stable_frames=stable_frames,
        )
        if stop_frame is None:
            continue
        snapped_time = round(stop_frame / source_fps / playback_time_scale, 2)
        if snapped_time <= start_time or snapped_time >= end_time:
            continue
        if snapped_time - start_time < min_navigation_duration_sec:
            continue
        if end_time - snapped_time > max_snap_back_sec:
            continue
        next_segment_id = None
        if index + 1 < len(segments) and isinstance(segments[index + 1], dict):
            next_segment_id = segments[index + 1].get("segment_id")
        snap_records.append(
            {
                "segment_id": segment.get("segment_id"),
                "next_segment_id": next_segment_id,
                "old_end_time_sec": end_time,
                "new_end_time_sec": snapped_time,
                "source_stop_frame": stop_frame,
                "base_epsilon": base_epsilon,
                "stable_frames": stable_frames,
            }
        )
        segment["end_time_sec"] = snapped_time
        if index + 1 < len(segments) and isinstance(segments[index + 1], dict):
            segments[index + 1]["start_time_sec"] = snapped_time

        parent_activity_id = segment.get("parent_activity_id")
        activities = prediction.get("activity_timeline", [])
        if isinstance(parent_activity_id, str) and isinstance(activities, list):
            for activity_index, activity in enumerate(activities):
                if not isinstance(activity, dict) or activity.get("activity_id") != parent_activity_id:
                    continue
                activity["end_time_sec"] = snapped_time
                if activity_index + 1 < len(activities) and isinstance(activities[activity_index + 1], dict):
                    activities[activity_index + 1]["start_time_sec"] = snapped_time
                break
    if snap_records:
        prediction["navigation_boundary_snaps"] = snap_records
    return prediction


def export_timeline_segment_clips(*, source_video: Path, prediction: dict, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    exported: list[dict] = []
    for index, segment in enumerate(prediction.get("skill_timeline", []), start=1):
        segment_id = str(segment.get("segment_id") or f"segment-{index:03d}")
        start_frame = int(segment.get("start_frame", 0))
        end_frame = int(segment.get("end_frame", start_frame))
        if end_frame <= start_frame:
            continue
        start_time = start_frame / fps
        end_time = end_frame / fps
        clip_path = output_dir / f"{segment_id}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source_video),
                "-ss",
                f"{start_time:.3f}",
                "-t",
                f"{end_time - start_time:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-an",
                str(clip_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        exported.append(
            {
                "task_id": prediction.get("task_id"),
                "episode_id": prediction.get("episode_id"),
                "segment_id": segment_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_sec": start_time,
                "end_time_sec": end_time,
                "skill_description": segment.get("skill_description"),
                "clip_path": str(clip_path),
            }
        )
    return exported


def _skill_timeline_gap_details(
    prediction: dict, tolerance_sec: float = 0.05
) -> list[tuple[float, float]]:
    """Return a list of (gap_start, gap_end) tuples inside the skill_timeline.

    A gap is any span within the covered time range where no skill segment is
    emitted. The first-segment-before-0 gap and any trailing gap after the last
    skill are intentionally ignored here because total video duration is not
    always known; this function focuses on inter-skill gaps.
    """
    skills = prediction.get("skill_timeline")
    if not isinstance(skills, list) or len(skills) < 2:
        return []
    gaps: list[tuple[float, float]] = []
    for a, b in zip(skills, skills[1:]):
        try:
            a_end = float(a["end_time_sec"])
            b_start = float(b["start_time_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if b_start - a_end > tolerance_sec:
            gaps.append((a_end, b_start))
    return gaps


def _skill_timeline_zero_duration_segments(
    prediction: dict, min_duration_sec: float = 0.1
) -> list[dict]:
    """Return segments whose end_time_sec <= start_time_sec + min_duration_sec.

    A zero- or near-zero-duration segment is a structural defect: the model
    collapsed a real action onto a single instant to satisfy coverage, losing
    all information about that action's extent.
    """
    skills = prediction.get("skill_timeline")
    if not isinstance(skills, list):
        return []
    bad: list[dict] = []
    for seg in skills:
        if not isinstance(seg, dict):
            continue
        try:
            start = float(seg["start_time_sec"])
            end = float(seg["end_time_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if end - start < min_duration_sec:
            bad.append(seg)
    return bad


BASE_ZERO_THRESHOLD = 0.03
BASE_ZERO_MERGE_GAP_SEC = 1.0
BASE_ZERO_STABLE_FRAMES = 5


def _events_in_window(
    events: list[dict],
    start_sec: float,
    end_sec: float,
    bimanual_event_times: set[float],
) -> list[dict]:
    out: list[dict] = []
    seen_bimanual: set[float] = set()
    for event in events:
        t = float(event.get("t", -1.0))
        if not (start_sec <= t <= end_sec):
            continue
        if t in bimanual_event_times:
            if t in seen_bimanual:
                continue
            seen_bimanual.add(t)
        out.append(event)
    return out


def _build_segment_resplit_prompt(
    segment: dict, events: list[dict], video_duration_sec: float | None
) -> str:
    start = float(segment["start_time_sec"])
    end = float(segment["end_time_sec"])
    original_desc = segment.get("skill_description", "")
    event_lines = []
    for event in events:
        event_lines.append(
            f"- t={float(event['t']):.2f}s  {event.get('side','?')} {event.get('direction','?')}"
        )
    n = len(events)
    lines = [
        "You are splitting ONE already-detected compound skill segment into its atomic sub-skills.",
        "",
        "## Original Segment",
        f"- time range: [{start:.2f}s - {end:.2f}s]",
        f"- current description (may merge multiple actions): {original_desc!r}",
        "",
        "## Gripper Transition Events Inside This Range",
        *event_lines,
        "",
        "## Task",
        f"Produce EXACTLY {n} sub-skill segments covering [{start:.2f}s - {end:.2f}s] with no gaps and no overlaps.",
        "Each sub-skill contains exactly one transition event from the list above and its preparation/settle phases.",
        "Order sub-skills strictly by the chronological order of their transition events.",
        "",
        "## Rules for Each sub-skill",
        "- skill_description uses AT MOST ONE action verb. Never join two action verbs with `and`, `then`, `by <gerund>`, etc.",
        "- Do not mention `left hand`, `right hand`, `left gripper`, `right gripper`, `left arm`, `right arm`, `second hand`, `other hand`, `bimanually`, `with each hand`, or any side-of-body phrasing.",
        "- Describe only the atomic state change (what happened to the object/environment), with necessary context like object identity, source, destination, or carried-object.",
        "- Use the nearest visible `t=...s` playback label for each boundary; the first sub-skill starts at the original segment's start, the last sub-skill ends at the original segment's end.",
        "- For the split boundaries between consecutive sub-skills, choose the nearest visible `t=...s` label between the two transition events (roughly at their midpoint).",
        "",
        "## Output JSON (no markdown fences, no extra text)",
        '{"sub_skills":[{"start_time_sec":0.0,"end_time_sec":0.0,"skill_description":"..."},{"start_time_sec":0.0,"end_time_sec":0.0,"skill_description":"..."}]}',
    ]
    return "\n".join(lines)


def _parse_sub_skills_json(text: str) -> list[dict] | None:
    from .inference import _extract_first_json_object
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = json.loads(_extract_first_json_object(text))
        except Exception:
            return None
    subs = payload.get("sub_skills") if isinstance(payload, dict) else None
    if not isinstance(subs, list):
        return None
    cleaned: list[dict] = []
    for item in subs:
        if not isinstance(item, dict):
            return None
        try:
            start = float(item["start_time_sec"])
            end = float(item["end_time_sec"])
        except (KeyError, ValueError, TypeError):
            return None
        desc = str(item.get("skill_description", "")).strip()
        if end <= start or not desc:
            return None
        cleaned.append({"start_time_sec": start, "end_time_sec": end, "skill_description": desc})
    if not cleaned:
        return None
    return cleaned


def _call_qwen_segment_resplit(
    *,
    segment: dict,
    events: list[dict],
    video_data_url: str,
    video_duration_sec: float | None,
    model: str,
    max_completion_tokens: int,
) -> list[dict] | None:
    from .api_client import build_chat_completion_request, execute_chat_completion, extract_first_message_text
    prompt_text = _build_segment_resplit_prompt(segment, events, video_duration_sec)
    user_content = [
        {"type": "text", "text": prompt_text},
        {"type": "video_url", "video_url": {"url": video_data_url}},
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You split one compound skill segment into atomic sub-skills. "
                "Return valid JSON only. No markdown fences. Use the exact shape requested."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    request = build_chat_completion_request(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        response_format={"type": "json_object"},
        enable_thinking=True,
    )
    try:
        response = execute_chat_completion(request)
        raw = extract_first_message_text(response)
    except Exception:
        return None
    return _parse_sub_skills_json(raw)


def split_compound_skill_segments(
    *,
    prediction: dict,
    video_data_url: str,
    video_duration_sec: float | None,
    model: str = QWEN_FULL_VIDEO_MODEL,
    max_completion_tokens: int = 4096,
) -> dict:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    skills = prediction.get("skill_timeline")
    if not isinstance(skills, list) or not skills:
        return prediction
    transitions = prediction.get("gripper_transitions") or []
    bimanual_event_times = {
        float(e["t"]) for e in (prediction.get("bimanual_events") or [])
        if isinstance(e, dict) and "t" in e
    }

    violation_indices: list[int] = []
    per_index_events: dict[int, list[dict]] = {}
    for idx, segment in enumerate(skills):
        if not isinstance(segment, dict):
            continue
        try:
            start = float(segment["start_time_sec"])
            end = float(segment["end_time_sec"])
        except (KeyError, ValueError, TypeError):
            continue
        events = _events_in_window(transitions, start, end, bimanual_event_times)
        if len(events) >= 2:
            violation_indices.append(idx)
            per_index_events[idx] = events
    if not violation_indices:
        return prediction

    resplit_log: list[dict] = []
    new_subs_by_index: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(violation_indices))) as pool:
        future_to_idx = {
            pool.submit(
                _call_qwen_segment_resplit,
                segment=skills[idx],
                events=per_index_events[idx],
                video_data_url=video_data_url,
                video_duration_sec=video_duration_sec,
                model=model,
                max_completion_tokens=max_completion_tokens,
            ): idx
            for idx in violation_indices
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            subs = fut.result()
            events = per_index_events[idx]
            original = skills[idx]
            if not subs or len(subs) != len(events):
                resplit_log.append({
                    "segment_id": original.get("segment_id"),
                    "status": "failed_fallback_to_original",
                    "event_count": len(events),
                    "returned_count": len(subs) if subs else 0,
                })
                continue
            # Enforce boundary alignment with original
            subs_sorted = sorted(subs, key=lambda s: s["start_time_sec"])
            subs_sorted[0]["start_time_sec"] = float(original["start_time_sec"])
            subs_sorted[-1]["end_time_sec"] = float(original["end_time_sec"])
            # Inherit parent_activity_id and carry-over fields
            parent_id = original.get("parent_activity_id")
            base_seg_id = original.get("segment_id", f"segment-{idx+1:03d}")
            enriched: list[dict] = []
            for sub_idx, sub in enumerate(subs_sorted):
                new_item = {
                    "segment_id": f"{base_seg_id}-{chr(ord('a')+sub_idx)}" if sub_idx > 0 or len(subs_sorted) > 1 else base_seg_id,
                    "parent_activity_id": parent_id,
                    "start_time_sec": sub["start_time_sec"],
                    "end_time_sec": sub["end_time_sec"],
                    "skill_description": sub["skill_description"],
                    "evidence": original.get("evidence", ""),
                    "confidence": original.get("confidence", 0.5),
                }
                enriched.append(new_item)
            new_subs_by_index[idx] = enriched
            resplit_log.append({
                "segment_id": original.get("segment_id"),
                "status": "resplit",
                "event_count": len(events),
                "resplit_into": len(enriched),
                "sub_segment_ids": [s["segment_id"] for s in enriched],
            })

    # Rebuild skill_timeline: replace violations with their sub-skills
    rebuilt: list[dict] = []
    for idx, segment in enumerate(skills):
        if idx in new_subs_by_index:
            rebuilt.extend(new_subs_by_index[idx])
        else:
            rebuilt.append(segment)
    prediction["skill_timeline"] = rebuilt
    prediction["compound_resplits"] = resplit_log
    return prediction


def _compute_base_zero_intervals_and_gripper_transitions(
    parquet_path: Path,
    *,
    time_scale: float = VIDEO_PLAYBACK_TIME_SCALE,
    base_threshold: float = BASE_ZERO_THRESHOLD,
    merge_gap_sec: float = BASE_ZERO_MERGE_GAP_SEC,
    stable_frames: int = BASE_ZERO_STABLE_FRAMES,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Compute gap-merged base-zero intervals, gripper state-change events, and arm-active intervals."""

    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("numpy and pyarrow are required to compute base-zero intervals") from exc

    table = pq.read_table(parquet_path, columns=["action", "timestamp"])
    action = np.stack(table["action"].to_pylist()).astype(float)
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=float)
    if action.ndim != 2 or action.shape[0] != timestamps.shape[0]:
        raise RuntimeError("action/timestamp shape mismatch in parquet")

    base_mag = np.max(np.abs(action[:, 0:3]), axis=1)
    is_zero = base_mag < base_threshold

    raw_intervals: list[tuple[float, float]] = []
    i = 0
    n = len(is_zero)
    while i < n:
        if not is_zero[i]:
            i += 1
            continue
        j = i
        while j < n and is_zero[j]:
            j += 1
        if j - i < stable_frames:
            i = j
            continue
        start_sec = float(timestamps[i]) / time_scale
        end_sec = float(timestamps[j - 1]) / time_scale
        if end_sec > start_sec:
            raw_intervals.append((start_sec, end_sec))
        i = j

    merged: list[list[float]] = []
    for start_sec, end_sec in raw_intervals:
        if merged and start_sec - merged[-1][1] <= merge_gap_sec:
            merged[-1][1] = end_sec
        else:
            merged.append([start_sec, end_sec])

    base_zero_intervals = [
        {
            "checkpoint_id": f"merged_base_zero_{index + 1:03d}",
            "start_time_sec": round(start_sec, 2),
            "end_time_sec": round(end_sec, 2),
            "duration_sec": round(end_sec - start_sec, 2),
        }
        for index, (start_sec, end_sec) in enumerate(merged)
    ]

    gripper_transitions: list[dict] = []
    for dim, side in [(14, "left"), (22, "right")]:
        signal = (action[:, dim] > 0).astype(int)
        change_idx = np.where(np.diff(signal) != 0)[0]
        for idx in change_idx:
            transition_time = float(timestamps[idx + 1]) / time_scale
            direction = "open_to_close" if signal[idx] == 1 else "close_to_open"
            gripper_transitions.append(
                {
                    "t": round(transition_time, 2),
                    "side": side,
                    "direction": direction,
                }
            )
    gripper_transitions.sort(key=lambda event: event["t"])

    arm_active_intervals = _compute_arm_active_intervals(
        action=action,
        timestamps=timestamps,
        time_scale=time_scale,
    )

    silent_actuation_intervals = _compute_silent_actuation_intervals(
        arm_active_intervals=arm_active_intervals,
        gripper_transitions=gripper_transitions,
    )

    return (
        base_zero_intervals,
        gripper_transitions,
        arm_active_intervals,
        silent_actuation_intervals,
    )


def _compute_silent_actuation_intervals(
    *,
    arm_active_intervals: list[dict],
    gripper_transitions: list[dict],
    min_duration_sec: float = 0.5,
    final_min_duration_sec: float = 0.5,
    gripper_guard_sec: float = 0.3,
    same_side_merge_gap_sec: float = 1.7,
    trim_when_other_hand_starts: bool = True,
    extend_back_to_prev_gripper_max_gap_sec: float = 2.5,
) -> list[dict]:
    """Return arm-active spans that contain NO gripper transition — strong
    candidates for silent body/forearm/arm-sweep actuation on an articulated
    target. Only arm-active intervals individually >= min_duration_sec are
    considered; these surviving spans are then merged on the same side when a
    short arm-idle gap (<= same_side_merge_gap_sec) separates them and no
    gripper event falls in the gap, since brief arm pauses inside a continued
    actuation (final nudge on a swinging panel, settling push) should not
    split the signal."""
    grip_times = [float(ev["t"]) for ev in gripper_transitions if isinstance(ev, dict) and "t" in ev]

    def _has_grip(s: float, e: float) -> bool:
        return any(s - gripper_guard_sec <= t <= e + gripper_guard_sec for t in grip_times)

    candidates: list[tuple[str, float, float]] = []
    for arm in arm_active_intervals:
        a_s = float(arm["start_time_sec"])
        a_e = float(arm["end_time_sec"])
        if a_e - a_s < min_duration_sec:
            continue
        side = arm["side"]
        if _has_grip(a_s, a_e):
            continue
        candidates.append((side, a_s, a_e))

    candidates.sort(key=lambda item: (item[0], item[1]))

    merged: list[tuple[str, float, float]] = []
    for side, s, e in candidates:
        if merged and merged[-1][0] == side and s - merged[-1][2] <= same_side_merge_gap_sec:
            gap_s, gap_e = merged[-1][2], s
            if not any(gap_s <= t <= gap_e for t in grip_times):
                merged[-1] = (side, merged[-1][1], e)
                continue
        merged.append((side, s, e))

    # Trim each silent span to end when the OTHER hand begins its own activity
    # (arm-active start or gripper event). Rationale: once the opposite hand
    # starts reaching / grasping / releasing, the `open/close` actuation done
    # by THIS hand is effectively concluding — the span beyond that point
    # belongs to the next segment (approach/acquire/release by the other hand),
    # not to the continued open/close by this hand.
    def _other_side_starts(side: str) -> list[float]:
        starts: list[float] = []
        for ev in gripper_transitions:
            if not isinstance(ev, dict):
                continue
            other = ev.get("side")
            if not isinstance(other, str) or other == side:
                continue
            try:
                starts.append(float(ev["t"]))
            except (KeyError, TypeError, ValueError):
                continue
        for arm in arm_active_intervals:
            if not isinstance(arm, dict):
                continue
            other = arm.get("side")
            if not isinstance(other, str) or other == side:
                continue
            try:
                starts.append(float(arm["start_time_sec"]))
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(starts)

    if trim_when_other_hand_starts:
        trimmed: list[tuple[str, float, float]] = []
        for side, s, e in merged:
            other_starts = _other_side_starts(side)
            new_e = e
            for t in other_starts:
                if s < t < new_e:
                    new_e = t
                    break
            trimmed.append((side, s, new_e))
        merged = trimmed

    # Extend each silent span's START backwards to a recent SAME-side
    # RELEASE event (`close_to_open`) when the gap is short. Motivation: a
    # brief base shift right after a handle is released still belongs to the
    # same continued open/close, and the silent signal should reach back
    # through that shift. We intentionally do NOT back-extend to an
    # `open_to_close` (acquire) event, because that marks the start of a
    # sustained grip, not the end of an open/close push.
    if extend_back_to_prev_gripper_max_gap_sec > 0:
        extended: list[tuple[str, float, float]] = []
        for side, s, e in merged:
            candidates_back = [
                float(ev["t"])
                for ev in gripper_transitions
                if isinstance(ev, dict)
                and ev.get("side") == side
                and ev.get("direction") == "close_to_open"
                and "t" in ev
                and float(ev["t"]) <= s
            ]
            if candidates_back:
                prev_t = max(candidates_back)
                if 0 <= s - prev_t <= extend_back_to_prev_gripper_max_gap_sec:
                    s = prev_t
            extended.append((side, s, e))
        merged = extended

    # Keep only silent spans that immediately follow a SAME-side
    # `close_to_open` (gripper release) within a short window — those are the
    # "continue-open after release" signals we actually care about. All other
    # silent spans (ambient arm motion, reaches, adjustments far from any
    # release) are dropped to keep the prompt short.
    follow_release_max_gap_sec = 3.0
    filtered: list[tuple[str, float, float]] = []
    for side, s, e in merged:
        same_side_releases = [
            float(ev["t"])
            for ev in gripper_transitions
            if isinstance(ev, dict)
            and ev.get("side") == side
            and ev.get("direction") == "close_to_open"
            and "t" in ev
            and float(ev["t"]) <= s
        ]
        if not same_side_releases:
            continue
        gap = s - max(same_side_releases)
        if gap <= follow_release_max_gap_sec:
            filtered.append((side, s, e))
    merged = filtered

    flagged: list[dict] = []
    for side, s, e in merged:
        if e - s < final_min_duration_sec:
            continue
        flagged.append(
            {
                "side": side,
                "start_time_sec": round(s, 2),
                "end_time_sec": round(e, 2),
                "duration_sec": round(e - s, 2),
            }
        )
    flagged.sort(key=lambda iv: (iv["start_time_sec"], iv["side"]))
    return flagged


def _compute_arm_active_intervals(
    *,
    action,
    timestamps,
    time_scale: float,
    min_duration_sec: float = 0.3,
    merge_gap_sec: float = 0.4,
) -> list[dict]:
    """Return intervals (compressed seconds) where each arm's joint command was moving.

    Data layout per row in `action`: base velocity at [0:3], left arm joints at
    [3:14] (gripper at 14), right arm joints at [15:22] (gripper at 22). We
    compute finite-difference velocity magnitude across the joint dims per side
    and flag stretches where magnitude exceeds a session-adaptive threshold.
    """
    import numpy as np

    ranges = {"left": (3, 14), "right": (15, 22)}
    comp_t = timestamps / time_scale
    intervals: list[dict] = []

    for side, (lo, hi) in ranges.items():
        joint_action = action[:, lo:hi]
        vel = np.diff(joint_action, axis=0)
        mag = np.linalg.norm(vel, axis=1)
        mag = np.concatenate([[0.0], mag])

        nonzero = mag[mag > 1e-6]
        if len(nonzero) == 0:
            continue
        threshold = max(float(np.percentile(nonzero, 30)), 1e-4)

        active = mag > threshold
        raw: list[tuple[float, float]] = []
        i = 0
        n = len(active)
        while i < n:
            if not active[i]:
                i += 1
                continue
            j = i
            while j < n and active[j]:
                j += 1
            s = float(comp_t[i])
            e = float(comp_t[j - 1])
            if e - s >= min_duration_sec:
                raw.append((s, e))
            i = j

        merged: list[list[float]] = []
        for s, e in raw:
            if merged and s - merged[-1][1] <= merge_gap_sec:
                merged[-1][1] = e
            else:
                merged.append([s, e])

        for s, e in merged:
            intervals.append(
                {
                    "side": side,
                    "start_time_sec": round(s, 2),
                    "end_time_sec": round(e, 2),
                    "duration_sec": round(e - s, 2),
                }
            )

    intervals.sort(key=lambda iv: (iv["start_time_sec"], iv["side"]))
    return intervals


def _group_bimanual_coincident_events(
    transitions: list[dict],
    *,
    time_tolerance_sec: float = 0.3,
) -> list[dict]:
    """Group gripper transitions that occur near-simultaneously on different hands."""

    bimanual_events: list[dict] = []
    sorted_transitions = sorted(transitions, key=lambda e: (float(e["t"]), e.get("direction", "")))
    i = 0
    while i < len(sorted_transitions):
        current = sorted_transitions[i]
        current_time = float(current["t"])
        sides = {current["side"]}
        direction = current["direction"]
        j = i + 1
        while j < len(sorted_transitions):
            other = sorted_transitions[j]
            if other["direction"] != direction:
                break
            if float(other["t"]) - current_time > time_tolerance_sec:
                break
            sides.add(other["side"])
            j += 1
        if len(sides) >= 2:
            bimanual_events.append(
                {
                    "t": round(current_time, 2),
                    "direction": direction,
                    "sides": sorted(sides),
                }
            )
        i = j if j > i + 1 and len(sides) >= 2 else i + 1
    return bimanual_events


def _run_single_demo_prediction(
    *,
    selected_demo,
    run_dir: Path,
    task_average_source_duration_sec: float | None = None,
    task_instruction: str | None = None,
) -> tuple[dict, Path]:
    preprocessed_root_env = os.getenv("CLAW_PREPROCESSED_VIDEO_ROOT", "").strip()
    preprocessed_candidate = None
    if preprocessed_root_env:
        cand = (
            Path(preprocessed_root_env)
            / selected_demo.task_id
            / f"{selected_demo.episode_id}.mp4"
        )
        if cand.exists():
            preprocessed_candidate = cand

    inference_video_path = (
        run_dir
        / "prepared"
        / "inference_videos"
        / selected_demo.task_id
        / f"{selected_demo.episode_id}.mp4"
    )
    if preprocessed_candidate is not None:
        _append_pipeline_log(
            run_dir,
            f"skip build_inference_video for {selected_demo.task_id}/{selected_demo.episode_id} "
            f"(preprocessed available, using head_video_path as source reference)",
        )
        # Downstream `_build_video_context`, `snap_navigation_boundaries_to_base_stops`,
        # and `_backfill_segment_frames_from_times` only need duration/fps from a
        # source-speed video — the raw head video from the dataset works.
        inference_video_path = selected_demo.head_video_path
    else:
        _append_pipeline_log(
            run_dir,
            f"building inference video for {selected_demo.task_id}/{selected_demo.episode_id}",
        )
        build_inference_video(
            head_video=selected_demo.head_video_path,
            parquet_path=selected_demo.parquet_path,
            output_video=inference_video_path,
        )
    (
        base_zero_intervals,
        gripper_transitions,
        arm_active_intervals,
        silent_actuation_intervals,
    ) = _compute_base_zero_intervals_and_gripper_transitions(
        parquet_path=selected_demo.parquet_path,
    )
    bimanual_events = _group_bimanual_coincident_events(gripper_transitions)
    payload: dict[str, object] = {
        "base_zero_intervals": base_zero_intervals,
        "gripper_transitions": gripper_transitions,
        "bimanual_events": bimanual_events,
        "arm_active_intervals": arm_active_intervals,
        "silent_actuation_intervals": silent_actuation_intervals,
    }
    if task_instruction:
        payload["task"] = task_instruction
    if _has_video_inference_api_key():
        compressed_video_path = (
            run_dir
            / "prepared"
            / "compressed_videos"
            / selected_demo.task_id
            / f"{selected_demo.episode_id}.mp4"
        )
        preprocessed_root = os.getenv("CLAW_PREPROCESSED_VIDEO_ROOT", "").strip()
        preprocessed_video = None
        if preprocessed_root:
            candidate = (
                Path(preprocessed_root)
                / selected_demo.task_id
                / f"{selected_demo.episode_id}.mp4"
            )
            if candidate.exists():
                preprocessed_video = candidate
        if preprocessed_video is not None:
            _append_pipeline_log(
                run_dir,
                f"using preprocessed upload video for {selected_demo.task_id}/{selected_demo.episode_id} from {preprocessed_video}",
            )
            compressed_video_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil as _shutil
            _shutil.copyfile(preprocessed_video, compressed_video_path)
        else:
            _append_pipeline_log(
                run_dir,
                f"compressing upload video for {selected_demo.task_id}/{selected_demo.episode_id}",
            )
            compress_video_for_upload(
                input_video=inference_video_path,
                output_video=compressed_video_path,
            )
        inference_model = os.getenv("CLAW_INFERENCE_MODEL", QWEN_FULL_VIDEO_MODEL)
        inference_max_tokens = 524288 if inference_model.startswith("gemini") else 8192
        _override = os.getenv("CLAW_INFERENCE_MAX_TOKENS", "").strip()
        if _override:
            try:
                inference_max_tokens = int(_override)
            except ValueError:
                pass
        _append_pipeline_log(
            run_dir,
            f"{inference_model} inference request start for {selected_demo.task_id}/{selected_demo.episode_id}",
        )
        video_data_url_inline = video_file_as_data_url(video_path=compressed_video_path)
        prediction = None
        max_coverage_retries = 2
        for attempt in range(max_coverage_retries + 1):
            try:
                prediction = run_full_video_inference(
                    payload=payload,
                    model=inference_model,
                    max_completion_tokens=inference_max_tokens,
                    video_data_url=video_data_url_inline,
                    prompt_artifact_dir=run_dir / "manifests" / "prompts" / selected_demo.task_id,
                    prompt_artifact_stem=f"{selected_demo.task_id}_{selected_demo.episode_id}_inference_attempt_{attempt:02d}",
                )
            except ValueError as exc:
                msg = str(exc)
                if attempt < max_coverage_retries and (
                    "Unable to parse model response" in msg
                    or "skill_timeline is required" in msg
                ):
                    _append_pipeline_log(
                        run_dir,
                        f"JSON parse failure for {selected_demo.task_id}/{selected_demo.episode_id} "
                        f"attempt={attempt} (likely output truncation) — retrying",
                    )
                    continue
                raise
            gaps = _skill_timeline_gap_details(prediction)
            zero_segs = _skill_timeline_zero_duration_segments(prediction)
            if not gaps and not zero_segs:
                if attempt > 0:
                    _append_pipeline_log(
                        run_dir,
                        f"validation retry succeeded for {selected_demo.task_id}/{selected_demo.episode_id} "
                        f"on attempt {attempt}",
                    )
                break
            if attempt >= max_coverage_retries:
                _append_pipeline_log(
                    run_dir,
                    f"validation retry exhausted for {selected_demo.task_id}/{selected_demo.episode_id} "
                    f"remaining_gaps={len(gaps)} total_gap_sec={sum(g[1]-g[0] for g in gaps):.2f} "
                    f"zero_duration_segments={len(zero_segs)}",
                )
                break
            reason_parts: list[str] = []
            if gaps:
                reason_parts.append(
                    f"gaps={len(gaps)} total_gap_sec={sum(g[1]-g[0] for g in gaps):.2f}"
                )
            if zero_segs:
                reason_parts.append(
                    f"zero_duration_segments={len(zero_segs)} ids=[{','.join(str(s.get('segment_id', '?')) for s in zero_segs)}]"
                )
            _append_pipeline_log(
                run_dir,
                f"validation failure for {selected_demo.task_id}/{selected_demo.episode_id} "
                f"attempt={attempt} {' '.join(reason_parts)} — retrying",
            )
        prediction["video_context"] = _build_video_context(
            task_id=selected_demo.task_id,
            task_average_source_duration_sec=task_average_source_duration_sec,
            source_video=inference_video_path,
            compressed_video=compressed_video_path,
        )
        prediction = snap_navigation_boundaries_to_base_stops(
            prediction=prediction,
            parquet_path=selected_demo.parquet_path,
            source_video=inference_video_path,
        )
        prediction = _backfill_segment_frames_from_times(
            prediction=prediction,
            source_video=inference_video_path,
        )
        prediction.setdefault("task_id", selected_demo.task_id)
        prediction.setdefault("episode_id", selected_demo.episode_id)
        prediction["base_zero_intervals"] = base_zero_intervals
        prediction["gripper_transitions"] = gripper_transitions
        prediction["bimanual_events"] = bimanual_events
        prediction["arm_active_intervals"] = arm_active_intervals
        prediction["silent_actuation_intervals"] = silent_actuation_intervals
        _append_pipeline_log(
            run_dir,
            f"{inference_model} inference request finish for {selected_demo.task_id}/{selected_demo.episode_id}",
        )
        return prediction, inference_video_path

    _append_pipeline_log(
        run_dir,
        f"sampling fallback frames for {selected_demo.task_id}/{selected_demo.episode_id}",
    )
    frame_data_urls = sample_video_frames_as_data_urls(
        video_path=inference_video_path,
        sample_count=MERGED_VIDEO_SAMPLE_COUNT,
    )
    _append_pipeline_log(
        run_dir,
        f"GPT frame inference request start for {selected_demo.task_id}/{selected_demo.episode_id}",
    )
    prediction = run_full_video_inference(
        payload=payload,
        model=GPT_FRAME_MODEL,
        max_completion_tokens=4096,
        frame_data_urls=frame_data_urls,
        prompt_artifact_dir=run_dir / "manifests" / "prompts" / selected_demo.task_id,
        prompt_artifact_stem=f"{selected_demo.task_id}_{selected_demo.episode_id}_inference",
    )
    prediction["video_context"] = _build_video_context(
        task_id=selected_demo.task_id,
        task_average_source_duration_sec=task_average_source_duration_sec,
        source_video=inference_video_path,
    )
    prediction = snap_navigation_boundaries_to_base_stops(
        prediction=prediction,
        parquet_path=selected_demo.parquet_path,
        source_video=inference_video_path,
    )
    prediction = _backfill_segment_frames_from_times(
        prediction=prediction,
        source_video=inference_video_path,
    )
    _append_pipeline_log(
        run_dir,
        f"GPT frame inference request finish for {selected_demo.task_id}/{selected_demo.episode_id}",
    )
    prediction.setdefault("task_id", selected_demo.task_id)
    prediction.setdefault("episode_id", selected_demo.episode_id)
    return prediction, inference_video_path


def _write_live_timeline_review(*, run_dir: Path, predictions: list[dict]) -> None:
    write_timeline_review(
        run_dir / "review" / "timeline_review.txt",
        build_review_rows(predictions),
    )

def run_demo_skill_discovery(
    dataset_root: Path,
    task_ids: list[str] | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    export_segment_clips: bool = False,
) -> dict[str, object]:
    resolved_task_ids = DEFAULT_TASK_IDS if task_ids is None else task_ids
    selected_demos = select_first_demo_per_task(
        dataset_root=dataset_root,
        task_ids=resolved_task_ids,
    )
    task_average_source_duration_by_task_id = _build_task_average_source_durations(selected_demos)
    task_instruction_by_task_id: dict[str, str] = {}
    for task_id in resolved_task_ids:
        try:
            task_instruction_by_task_id[task_id] = load_task_instruction(dataset_root=dataset_root, task_id=task_id)
        except Exception:
            continue

    run_dir = _create_run_dir(output_root)
    manifests_dir = run_dir / "manifests"

    selected_demos_payload = [
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(selected_demo).items()
        }
        for selected_demo in selected_demos
    ]
    (manifests_dir / "selected_demos.json").write_text(
        json.dumps(selected_demos_payload, indent=2),
        encoding="utf-8",
    )

    manifest_path = manifests_dir / "selected_demos.json"
    log_path = _append_pipeline_log(
        run_dir,
        f"run started selected_demo_count={len(selected_demos)} task_ids={resolved_task_ids}",
    )
    inference_status = "not_started"
    artifact_paths: dict[str, object] = {}
    if not _has_any_model_api_key():
        inference_status = "skipped_no_api_key"
        log_path = _append_pipeline_log(run_dir, "skipping inference because no model API key is configured")
        artifact_paths = write_demo_skill_outputs(run_dir=run_dir, predictions=[])
    elif not selected_demos:
        inference_status = "completed_no_selected_demos"
        log_path = _append_pipeline_log(run_dir, "no demos selected; writing empty artifacts")
        artifact_paths = write_demo_skill_outputs(run_dir=run_dir, predictions=[])
    else:
        try:
            predictions = []
            selected_demo_by_episode_id = {}
            exported_segment_clips: list[dict] = []
            for selected_demo in selected_demos:
                print(f"[demo-skill-discovery] preparing {selected_demo.task_id}/{selected_demo.episode_id}")
                log_path = _append_pipeline_log(
                    run_dir,
                    f"preparing {selected_demo.task_id}/{selected_demo.episode_id}",
                )
                selected_demo_by_episode_id[selected_demo.episode_id] = selected_demo
                prediction, merged_video_path = _run_single_demo_prediction(
                    selected_demo=selected_demo,
                    run_dir=run_dir,
                    task_average_source_duration_sec=task_average_source_duration_by_task_id.get(selected_demo.task_id),
                    task_instruction=task_instruction_by_task_id.get(selected_demo.task_id),
                )
                print(f"[demo-skill-discovery] inference complete for {selected_demo.task_id}/{selected_demo.episode_id}")
                log_path = _append_pipeline_log(
                    run_dir,
                    f"inference complete for {selected_demo.task_id}/{selected_demo.episode_id}",
                )
                predictions.append(prediction)
                _write_live_timeline_review(run_dir=run_dir, predictions=predictions)
                if export_segment_clips:
                    exported_segment_clips.extend(
                        export_timeline_segment_clips(
                            source_video=merged_video_path,
                            prediction=prediction,
                            output_dir=run_dir / "segments" / selected_demo.task_id / selected_demo.episode_id,
                        )
                    )
            inference_status = "completed"
            print("[demo-skill-discovery] writing prediction/review/final artifacts")
            log_path = _append_pipeline_log(run_dir, "writing prediction/review/final artifacts")
            artifact_paths = write_demo_skill_outputs(
                run_dir=run_dir,
                predictions=predictions,
            )
            if export_segment_clips:
                segment_manifest_path = run_dir / "segments" / "segment_clips_manifest.json"
                segment_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                segment_manifest_path.write_text(json.dumps(exported_segment_clips, indent=2), encoding="utf-8")
                artifact_paths["segment_clip_count"] = len(exported_segment_clips)
                artifact_paths["segment_clips_manifest_path"] = str(segment_manifest_path)
        except Exception as exc:
            inference_status = "failed"
            artifact_paths.update(_write_failure_artifacts(run_dir, exc))
    log_path = _append_pipeline_log(run_dir, f"run finished inference_status={inference_status}")

    result: dict[str, object] = {
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "log_path": str(log_path),
        "inference_status": inference_status,
        "selected_demo_count": len(selected_demos),
        "selected_task_frequencies": summarize_value_frequencies(demo.task_id for demo in selected_demos),
        **artifact_paths,
    }
    return result


def write_demo_skill_outputs(
    run_dir: Path,
    predictions: list[dict],
) -> dict[str, object]:
    prediction_path = run_dir / "predictions" / "demo_skill_predictions.jsonl"
    review_path = run_dir / "review" / "demo_skill_review.jsonl"
    timeline_review_path = run_dir / "review" / "timeline_review.txt"
    readable_predictions_path = run_dir / "review" / "readable_predictions.md"
    final_path = run_dir / "final" / "demo_skills.jsonl"
    cluster_summary_path = run_dir / "clusters" / "skill_description_frequencies.json"
    semantic_skill_items_path = run_dir / "clusters" / "semantic_skill_items.json"
    semantic_skill_groups_path = run_dir / "clusters" / "semantic_skill_groups.json"

    write_jsonl(prediction_path, predictions)
    write_jsonl(review_path, build_review_rows(predictions))
    finalized_rows = write_finalized_timelines(review_path, final_path)
    write_timeline_review(timeline_review_path, finalized_rows)
    cluster_summary_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_summary_path.write_text(
        json.dumps(summarize_timeline_skill_descriptions(predictions), indent=2),
        encoding="utf-8",
    )
    semantic_skill_items = build_skill_description_items(predictions)
    semantic_skill_items_path.write_text(
        json.dumps(semantic_skill_items, indent=2),
        encoding="utf-8",
    )
    write_readable_predictions(
        readable_predictions_path,
        run_dir=run_dir,
        predictions=predictions,
        semantic_skill_items=semantic_skill_items,
    )

    outputs: dict[str, object] = {
        "prediction_count": len(predictions),
        "predictions_path": str(prediction_path),
        "review_path": str(review_path),
        "timeline_review_path": str(timeline_review_path),
        "readable_predictions_path": str(readable_predictions_path),
        "final_path": str(final_path),
        "cluster_summary_path": str(cluster_summary_path),
        "semantic_skill_items_path": str(semantic_skill_items_path),
    }
    if not _has_openai_api_key():
        outputs["cluster_naming_status"] = "skipped_no_api_key"
        return outputs

    if not semantic_skill_items:
        outputs["cluster_naming_status"] = "skipped_no_clusters"
        return outputs

    try:
        named_clusters = run_cluster_naming(
            items=semantic_skill_items,
            model=CLUSTER_NAMING_MODEL,
            max_completion_tokens=CLUSTER_NAMING_MAX_COMPLETION_TOKENS,
        )
    except Exception as exc:
        error_path = run_dir / "clusters" / "semantic_skill_groups.error.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_text = f"{type(exc).__name__}: {exc}\n"
        error_path.write_text(error_text, encoding="utf-8")
        outputs["cluster_naming_status"] = "failed"
        outputs["cluster_naming_error"] = error_text.strip()
        outputs["cluster_naming_error_path"] = str(error_path)
        return outputs

    semantic_skill_groups_path.write_text(
        json.dumps(named_clusters, indent=2),
        encoding="utf-8",
    )
    write_readable_predictions(
        readable_predictions_path,
        run_dir=run_dir,
        predictions=predictions,
        semantic_skill_items=semantic_skill_items,
        semantic_skill_groups=named_clusters,
    )
    outputs["semantic_skill_groups_path"] = str(semantic_skill_groups_path)
    outputs["cluster_naming_status"] = "written"
    return outputs


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "choose_inference_mode",
    "run_demo_skill_discovery",
    "write_demo_skill_outputs",
]
