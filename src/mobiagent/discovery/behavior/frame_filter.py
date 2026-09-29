from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def compute_kept_frame_indices(
    frame_deltas: np.ndarray,
    state_deltas: np.ndarray,
    frame_threshold: float,
    state_threshold: float,
) -> tuple[list[int], list[int]]:
    frame_values = np.asarray(frame_deltas)
    state_values = np.asarray(state_deltas)
    if frame_values.ndim != 1 or state_values.ndim != 1:
        raise ValueError("frame_deltas and state_deltas must be 1-D arrays")
    if frame_values.shape != state_values.shape:
        raise ValueError("frame_deltas and state_deltas must have the same length")

    kept: list[int] = []
    dropped: list[int] = []
    for index, (frame_delta, state_delta) in enumerate(zip(frame_values, state_values)):
        if frame_delta <= frame_threshold and state_delta <= state_threshold:
            dropped.append(index)
        else:
            kept.append(index)
    return kept, dropped


def write_frame_filter_manifest(output_path: Path, kept: list[int], dropped: list[int], total_frames: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "total_frames": total_frames,
        "kept_frame_indices": kept,
        "dropped_frame_indices": dropped,
    }
    output_path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n", encoding="utf-8")


def export_filtered_video_frames(source_video: Path, output_video: Path, kept_indices: list[int]) -> None:
    capture = cv2.VideoCapture(str(source_video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open source video: {source_video}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            raise ValueError(f"Source video does not provide a usable FPS: {source_video}")
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_width <= 0 or frame_height <= 0:
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Unable to read the first frame from source video: {source_video}")
            frame_height, frame_width = frame.shape[:2]
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"Source video does not provide a usable frame count: {source_video}")

        if any(isinstance(index, bool) or not isinstance(index, int) for index in kept_indices):
            raise ValueError(f"kept_indices must contain integers only: {kept_indices}")
        if any(index < 0 for index in kept_indices):
            raise ValueError(f"kept_indices must not contain negative values: {kept_indices}")
        if len(set(kept_indices)) != len(kept_indices):
            raise ValueError(f"kept_indices must not contain duplicates: {kept_indices}")
        if any(index >= frame_count for index in kept_indices):
            raise ValueError(f"kept_indices contains indices out of range for source video: {kept_indices}")

        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            writer.release()
            raise ValueError(f"Unable to open output video writer: {output_video}")

        kept_set = set(kept_indices)
        try:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    return
                if frame_index in kept_set:
                    writer.write(frame)
                frame_index += 1
        finally:
            writer.release()
    finally:
        capture.release()


def export_comparison_video(source_video: Path, filtered_video: Path, output_video: Path) -> None:
    source_capture = cv2.VideoCapture(str(source_video))
    filtered_capture = cv2.VideoCapture(str(filtered_video))

    try:
        if not source_capture.isOpened():
            raise ValueError(f"Unable to open source video: {source_video}")
        if not filtered_capture.isOpened():
            raise ValueError(f"Unable to open filtered video: {filtered_video}")

        fps = source_capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = filtered_capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 5.0

        ok, source_frame = source_capture.read()
        if not ok:
            raise ValueError(f"Unable to read the first frame from source video: {source_video}")

        ok, filtered_frame = filtered_capture.read()
        if not ok:
            raise ValueError(f"Unable to read the first frame from filtered video: {filtered_video}")

        first_frame = cv2.hconcat([source_frame, filtered_frame])
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
            while True:
                ok, source_frame = source_capture.read()
                if not ok:
                    return
                ok, filtered_frame = filtered_capture.read()
                if not ok:
                    return
                writer.write(cv2.hconcat([source_frame, filtered_frame]))
        finally:
            writer.release()
    finally:
        source_capture.release()
        filtered_capture.release()
