from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import ACTION_COLUMN, ACTION_GROUP_SLICES, CHASSIS_SLICE

ACTION_GROUPS: tuple[tuple[str, slice], ...] = ACTION_GROUP_SLICES

ACTION_GROUP_DESCRIPTIONS: tuple[str, ...] = (
    "torso: indices 0:9, torso SO3 pose command",
    "left_arm: indices 9:18, left arm SO3 pose command",
    "left_gripper: index 18, left gripper scalar command",
    "right_arm: indices 19:28, right arm SO3 pose command",
    "right_gripper: index 28, right gripper scalar command",
    "head: indices 29:31, head yaw and pitch command",
    "chassis: indices 31:34, mobile base motion command (x, y, yaw)",
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WORST_CASE_VALUE_LINE = "-1.00, -1.00, -1.00, -1.00"
_FIXED_PANEL_WIDTH = 220


def _load_actions(parquet_path: Path) -> list[list[float]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to build head/action overlay videos") from exc

    table = pq.read_table(parquet_path, columns=[ACTION_COLUMN])
    return table[ACTION_COLUMN].to_pylist()


def build_action_overlay_schema_text() -> str:
    lines = [
        "The video uses only the head camera. A right-side panel shows action groups for the current frame.",
        "- Each action group label appears on its own line, with the corresponding numeric values shown below it.",
        "- The elapsed time appears at the bottom-right corner of the right-side panel.",
        *[f"- {line}" for line in ACTION_GROUP_DESCRIPTIONS],
    ]
    return "\n".join(lines)


def build_head_only_schema_text() -> str:
    return "\n".join(
        [
            "The video uses only the head camera.",
            "- No action panel is shown.",
            "- A top bar shows three base action values for the current frame.",
            "- The three top-bar values are chassis[31:34] of the cartesian SO3 action, the mobile base motion command.",
            "- The base action values can help distinguish navigation from manipulation: sustained nonzero base values usually indicate navigation only when the robot is traveling between work areas or toward the next operation location.",
            "- When all three top-bar base action values are near zero, that moment is not navigation.",
            "- Base movement that happens inside an ongoing manipulation around the same object or workspace should be treated as local adjustment within that manipulation, not as a separate navigation skill.",
            "- The elapsed time appears in the top-right corner of the video.",
        ]
    )


def build_head_with_wrist_schema_text() -> str:
    return "\n".join(
        [
            "The video uses the head camera as the primary view.",
            "- A small right-side column shows left wrist on top and right wrist on bottom.",
            "- The wrist panels are only auxiliary views for manipulation details; do not let them override the head-camera chronology.",
            "- A top bar shows three base action values for the current frame.",
            "- The three top-bar values are chassis[31:34] of the cartesian SO3 action, the mobile base motion command.",
            "- When all three top-bar base action values are near zero, that moment is not navigation.",
            "- Base movement that happens inside an ongoing manipulation around the same object or workspace should be treated as local adjustment within that manipulation, not as a separate navigation skill.",
            "- The elapsed time appears in the top-right corner of the video.",
        ]
    )


def _put_text(image: np.ndarray, text: str, x: int, y: int, *, color: tuple[int, int, int], scale: float, thickness: int = 1) -> None:
    cv2.putText(image, text, (x, y), _FONT, scale, color, thickness, lineType=cv2.LINE_AA)


def _split_value_lines(values: list[float]) -> list[str]:
    if len(values) > 4:
        return [
            ", ".join(f"{value:.2f}" for value in values[:4]),
            ", ".join(f"{value:.2f}" for value in values[4:]),
        ]
    return [", ".join(f"{value:.2f}" for value in values)]


def _compose_overlay_frame(
    frame: np.ndarray,
    action: list[float],
    frame_index: int,
    fps: float,
    panel_width: int = _FIXED_PANEL_WIDTH,
) -> np.ndarray:
    header_scale = 0.54
    body_scale = 0.50
    white = (255, 255, 255)
    sub = (205, 205, 205)
    label = (140, 230, 255)

    left_padding = 0
    top_y = 16
    label_value_gap = 16
    intra_group_gap = 15
    block_gap = 14

    row_specs: list[tuple[str, list[str]]] = []
    value_widths: list[int] = []
    for name, slice_ in ACTION_GROUPS:
        value_lines = _split_value_lines(action[slice_])
        row_specs.append((name, value_lines))
        longest_value_line = max(value_lines, key=len)
        (value_width, _), _ = cv2.getTextSize(longest_value_line, _FONT, body_scale, 1)
        value_widths.append(value_width)

    max_value_width = max(value_widths, default=0)
    time_text = f"t={frame_index / fps:.2f}s"
    (frame_text_width, _), _ = cv2.getTextSize(time_text, _FONT, header_scale, 1)
    canvas = np.zeros((frame.shape[0], frame.shape[1] + panel_width, 3), dtype=np.uint8)
    canvas[:, : frame.shape[1], :] = frame
    canvas[:, frame.shape[1] :, :] = (18, 18, 18)

    content_x = frame.shape[1] + left_padding
    content_right = content_x + max_value_width
    frame_x = frame.shape[1] + panel_width - frame_text_width
    _put_text(canvas, time_text, frame_x, frame.shape[0] - 12, color=white, scale=header_scale)

    y = top_y
    for name, value_lines in row_specs:
        _put_text(canvas, name, content_x, y, color=label, scale=body_scale)
        y += label_value_gap
        _put_text(canvas, value_lines[0], content_x, y, color=sub, scale=body_scale)
        if len(value_lines) > 1:
            y += intra_group_gap
            _put_text(canvas, value_lines[1], content_x, y, color=sub, scale=body_scale)
        y += block_gap
    return canvas


def build_head_action_overlay_video(
    *,
    head_video: Path,
    parquet_path: Path,
    output_video: Path,
) -> None:
    actions = _load_actions(parquet_path)
    capture = cv2.VideoCapture(str(head_video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open head input video: {head_video}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Unable to read first frame from head input video: {head_video}")
        first_frame = _compose_overlay_frame(frame, actions[0], frame_index=0, fps=fps)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (first_frame.shape[1], first_frame.shape[0]),
        )
        if not writer.isOpened():
            writer.release()
            raise ValueError(f"Unable to open output video writer: {output_video}")
        try:
            writer.write(first_frame)
            frame_index = 1
            while True:
                ok, frame = capture.read()
                if not ok or frame_index >= len(actions):
                    break
                writer.write(_compose_overlay_frame(frame, actions[frame_index], frame_index=frame_index, fps=fps))
                frame_index += 1
        finally:
            writer.release()
    finally:
        capture.release()


def _display_time(frame_index: int, fps: float, playback_time_scale: float) -> float:
    return frame_index / fps / playback_time_scale


def build_head_only_time_overlay_video(
    *,
    head_video: Path,
    output_video: Path,
    playback_time_scale: float = 1.0,
) -> None:
    capture = cv2.VideoCapture(str(head_video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open head input video: {head_video}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Unable to read first frame from head input video: {head_video}")

        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame.shape[1], frame.shape[0]),
        )
        if not writer.isOpened():
            writer.release()
            raise ValueError(f"Unable to open output video writer: {output_video}")
        try:
            frame_index = 0
            while True:
                time_text = f"t={_display_time(frame_index, fps, playback_time_scale):.2f}s"
                (time_width, _), _ = cv2.getTextSize(time_text, _FONT, 0.54, 1)
                rendered = frame.copy()
                _put_text(
                    rendered,
                    time_text,
                    rendered.shape[1] - time_width - 4,
                    rendered.shape[0] - 12,
                    color=(255, 255, 255),
                    scale=0.54,
                )
                writer.write(rendered)
                frame_index += 1
                ok, frame = capture.read()
                if not ok:
                    break
        finally:
            writer.release()
    finally:
        capture.release()


def build_head_only_base_overlay_video(
    *,
    head_video: Path,
    parquet_path: Path,
    output_video: Path,
    playback_time_scale: float = 1.0,
) -> None:
    actions = _load_actions(parquet_path)
    capture = cv2.VideoCapture(str(head_video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open head input video: {head_video}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Unable to read first frame from head input video: {head_video}")

        bar_height = 24
        output_size = (frame.shape[1], frame.shape[0] + bar_height)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            output_size,
        )
        if not writer.isOpened():
            writer.release()
            raise ValueError(f"Unable to open output video writer: {output_video}")
        try:
            frame_index = 0
            while True:
                canvas = np.zeros((frame.shape[0] + bar_height, frame.shape[1], 3), dtype=np.uint8)
                canvas[:bar_height, :, :] = (18, 18, 18)
                canvas[bar_height:, :, :] = frame
                action = actions[frame_index] if frame_index < len(actions) else [0.0] * 34
                base_values = ", ".join(f"{float(value):.2f}" for value in action[CHASSIS_SLICE])
                scale = 0.45
                _put_text(canvas, base_values, 2, 17, color=(255, 255, 255), scale=scale)
                time_text = f"t={_display_time(frame_index, fps, playback_time_scale):.2f}s"
                (time_width, _), _ = cv2.getTextSize(time_text, _FONT, scale, 1)
                _put_text(
                    canvas,
                    time_text,
                    canvas.shape[1] - time_width - 2,
                    17,
                    color=(255, 255, 255),
                    scale=scale,
                )
                writer.write(canvas)
                frame_index += 1
                ok, frame = capture.read()
                if not ok:
                    break
        finally:
            writer.release()
    finally:
        capture.release()


def build_head_with_wrist_overlay_video(
    *,
    head_video: Path,
    left_wrist_video: Path,
    right_wrist_video: Path,
    parquet_path: Path,
    output_video: Path,
    playback_time_scale: float = 1.0,
) -> None:
    actions = _load_actions(parquet_path)
    captures = [cv2.VideoCapture(str(path)) for path in (head_video, left_wrist_video, right_wrist_video)]
    try:
        if not all(capture.isOpened() for capture in captures):
            raise ValueError("Unable to open one or more input videos for head/wrist overlay")
        fps = captures[0].get(cv2.CAP_PROP_FPS) or 30.0
        ok_head, head_frame = captures[0].read()
        ok_left, left_frame = captures[1].read()
        ok_right, right_frame = captures[2].read()
        if not (ok_head and ok_left and ok_right):
            raise ValueError("Unable to read first frame from one or more input videos")

        bar_height = 56
        head_height, head_width = head_frame.shape[:2]
        side_width = max(1, head_width // 2)
        output_size = (head_width + side_width, head_height + bar_height)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            output_size,
        )
        if not writer.isOpened():
            writer.release()
            raise ValueError(f"Unable to open output video writer: {output_video}")
        try:
            frame_index = 0
            while True:
                canvas = np.zeros((head_height + bar_height, head_width + side_width, 3), dtype=np.uint8)
                canvas[:bar_height, :, :] = (18, 18, 18)
                canvas[bar_height:, :head_width, :] = head_frame
                left_small = cv2.resize(left_frame, (side_width, head_height // 2), interpolation=cv2.INTER_AREA)
                right_small = cv2.resize(
                    right_frame,
                    (side_width, head_height - head_height // 2),
                    interpolation=cv2.INTER_AREA,
                )
                canvas[bar_height : bar_height + head_height // 2, head_width:, :] = left_small
                canvas[bar_height + head_height // 2 :, head_width:, :] = right_small
                canvas[bar_height:, head_width - 1 : head_width + 1, :] = (40, 40, 40)
                canvas[bar_height + head_height // 2 - 1 : bar_height + head_height // 2 + 1, head_width:, :] = (
                    40,
                    40,
                    40,
                )

                action = actions[frame_index] if frame_index < len(actions) else [0.0] * 34
                chassis_text = "chassis: " + ", ".join(f"{float(value):.2f}" for value in action[CHASSIS_SLICE])
                bar_scale = 1.3
                bar_thickness = 2
                bar_y = bar_height - 18  # baseline near bottom of bar so glyphs sit centered
                _put_text(canvas, chassis_text, 10, bar_y, color=(255, 255, 255), scale=bar_scale, thickness=bar_thickness)
                time_text = f"t={_display_time(frame_index, fps, playback_time_scale):.2f}s"
                (time_width, _), _ = cv2.getTextSize(time_text, _FONT, bar_scale, bar_thickness)
                _put_text(canvas, time_text, canvas.shape[1] - time_width - 12, bar_y, color=(255, 255, 255), scale=bar_scale, thickness=bar_thickness)
                label_scale = 1.4
                label_thickness = 3
                _put_text(canvas, "L", head_width + 12, bar_height + 36, color=(0, 255, 255), scale=label_scale, thickness=label_thickness)
                _put_text(
                    canvas,
                    "R",
                    head_width + 12,
                    bar_height + head_height // 2 + 36,
                    color=(0, 255, 255),
                    scale=label_scale,
                    thickness=label_thickness,
                )
                writer.write(canvas)

                frame_index += 1
                ok_head, head_frame = captures[0].read()
                ok_left, left_frame = captures[1].read()
                ok_right, right_frame = captures[2].read()
                if not (ok_head and ok_left and ok_right):
                    break
        finally:
            writer.release()
    finally:
        for capture in captures:
            capture.release()


def _sample_frame_indexes(total_frames: int, sample_count: int) -> list[int]:
    if total_frames <= 0 or sample_count <= 0:
        return []
    if sample_count == 1:
        return [0]
    if total_frames == 1:
        return [0]
    return [
        min(total_frames - 1, round(index * (total_frames - 1) / (sample_count - 1)))
        for index in range(sample_count)
    ]


def build_video_contact_sheet(
    *,
    video_path: Path,
    output_image: Path,
    sample_count: int = 8,
    columns: int = 4,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open input video: {video_path}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indexes = _sample_frame_indexes(total_frames, sample_count)
        frames: list[np.ndarray] = []
        for frame_index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
        if not frames:
            raise ValueError(f"Unable to sample frames from input video: {video_path}")
    finally:
        capture.release()

    columns = max(1, columns)
    rows = int(np.ceil(len(frames) / columns))
    height, width = frames[0].shape[:2]
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        row = index // columns
        column = index % columns
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame

    output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image), canvas)


__all__ = [
    "ACTION_GROUP_DESCRIPTIONS",
    "build_action_overlay_schema_text",
    "build_head_only_schema_text",
    "build_head_with_wrist_schema_text",
    "build_head_action_overlay_video",
    "build_head_only_base_overlay_video",
    "build_head_with_wrist_overlay_video",
    "build_head_only_time_overlay_video",
    "build_video_contact_sheet",
]
