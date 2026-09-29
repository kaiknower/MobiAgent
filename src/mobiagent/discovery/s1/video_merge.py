from __future__ import annotations

import base64
from pathlib import Path

import cv2

_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.45
_LABEL_THICKNESS = 1
_LABEL_COLOR = (255, 255, 255)
_LABEL_BACKGROUND = (0, 0, 0)
_MAX_OVERLAY_BAND_HEIGHT = 24


def _resize_to_height(frame, target_height: int):
    height, width = frame.shape[:2]
    if height == target_height:
        return frame

    target_width = max(1, round(width * target_height / height))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _put_text(frame, text: str, origin: tuple[int, int], *, anchor_right: bool = False) -> None:
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        _LABEL_FONT,
        _LABEL_SCALE,
        _LABEL_THICKNESS,
    )
    x, y = origin
    if anchor_right:
        x = max(2, x - text_width)
    cv2.putText(
        frame,
        text,
        (x, y),
        _LABEL_FONT,
        _LABEL_SCALE,
        _LABEL_COLOR,
        _LABEL_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def _annotate_merged_frame(merged_frame, segment_widths: list[int], frame_index: int, fps: float) -> None:
    elapsed_time_sec = frame_index / fps if fps > 0 else 0.0
    frame_height = merged_frame.shape[0]
    if frame_height >= 48:
        band_height = min(_MAX_OVERLAY_BAND_HEIGHT, 26)
    elif frame_height >= 24:
        band_height = 18
    else:
        band_height = max(8, frame_height // 2)
    cv2.rectangle(
        merged_frame,
        (0, 0),
        (merged_frame.shape[1] - 1, min(band_height, merged_frame.shape[0] - 1)),
        _LABEL_BACKGROUND,
        thickness=-1,
    )
    meta_y = 10 if band_height >= 14 else max(7, band_height - 5)
    label_y = min(max(meta_y + 7, band_height - 4), frame_height - 2)
    labels = ["left wrist", "head", "right wrist"]
    x_offset = 0
    left_label_width = segment_widths[0]
    _put_text(merged_frame, f"frame: {frame_index}", (left_label_width // 2, meta_y))
    _put_text(
        merged_frame,
        f"t={elapsed_time_sec:.2f}s",
        (merged_frame.shape[1] - 6, meta_y),
        anchor_right=True,
    )
    for label, width in zip(labels, segment_widths, strict=True):
        _put_text(merged_frame, label, (x_offset + 6, label_y))
        x_offset += width


def sample_video_frames_as_data_urls(video_path: Path, sample_count: int) -> list[str]:
    if sample_count <= 0:
        return []

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if total_frames <= 0:
            return []

        frame_indexes = _sample_frame_indexes(total_frames=total_frames, sample_count=sample_count)
        data_urls = []
        for frame_index in frame_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Unable to read sampled frame {frame_index} from video: {video_path}")

            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError(f"Unable to encode sampled frame {frame_index} from video: {video_path}")

            base64_jpeg = base64.b64encode(encoded.tobytes()).decode("ascii")
            data_urls.append(f"data:image/jpeg;base64,{base64_jpeg}")

        return data_urls
    finally:
        capture.release()


def _sample_frame_indexes(total_frames: int, sample_count: int) -> list[int]:
    actual_count = min(total_frames, sample_count)
    if actual_count == 1:
        return [0]

    return [round(index * (total_frames - 1) / (actual_count - 1)) for index in range(actual_count)]


def merge_triplet_videos(left_video: Path, head_video: Path, right_video: Path, output_video: Path) -> None:
    left_capture = cv2.VideoCapture(str(left_video))
    head_capture = cv2.VideoCapture(str(head_video))
    right_capture = cv2.VideoCapture(str(right_video))

    try:
        captures = (
            ("left", left_video, left_capture),
            ("head", head_video, head_capture),
            ("right", right_video, right_capture),
        )
        for name, path, capture in captures:
            if not capture.isOpened():
                raise ValueError(f"Unable to open {name} input video: {path}")

        fps_values = [capture.get(cv2.CAP_PROP_FPS) for _, _, capture in captures]
        fps = min((value for value in fps_values if value and value > 0), default=5.0)
        first_frames = []
        for name, path, capture in captures:
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Unable to read the first frame from {name} input video: {path}")
            first_frames.append(frame)

        target_height = min(frame.shape[0] for frame in first_frames)
        resized_first_frames = [_resize_to_height(frame, target_height) for frame in first_frames]
        segment_widths = [frame.shape[1] for frame in resized_first_frames]
        merged_first_frame = cv2.hconcat(resized_first_frames)
        _annotate_merged_frame(merged_first_frame, segment_widths, frame_index=0, fps=fps)
        output_video.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video),
            fourcc,
            fps,
            (merged_first_frame.shape[1], merged_first_frame.shape[0]),
        )
        if not writer.isOpened():
            writer.release()
            raise ValueError(f"Unable to open output video writer: {output_video}")

        try:
            writer.write(merged_first_frame)
            frame_index = 1
            while True:
                frames = []
                for _, _, capture in captures:
                    ok, frame = capture.read()
                    if not ok:
                        return
                    frames.append(frame)

                resized_frames = [_resize_to_height(frame, target_height) for frame in frames]
                merged_frame = cv2.hconcat(resized_frames)
                _annotate_merged_frame(merged_frame, segment_widths, frame_index=frame_index, fps=fps)
                writer.write(merged_frame)
                frame_index += 1
        finally:
            writer.release()
    finally:
        for capture in (left_capture, head_capture, right_capture):
            capture.release()
