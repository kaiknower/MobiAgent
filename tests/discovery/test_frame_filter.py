from pathlib import Path

import cv2
import numpy as np

from mobiagent.discovery.behavior.frame_filter import (
    compute_kept_frame_indices,
    export_comparison_video,
    export_filtered_video_frames,
    write_frame_filter_manifest,
)


def test_compute_kept_frame_indices_drops_static_frames_only_when_video_and_state_are_static() -> None:
    frame_deltas = np.array([0.0, 0.1, 0.8, 0.9], dtype=np.float32)
    state_deltas = np.array([0.0, 0.1, 0.9, 1.2], dtype=np.float32)
    kept, dropped = compute_kept_frame_indices(frame_deltas, state_deltas, 0.1, 0.1)
    assert kept == [2, 3]
    assert dropped == [0, 1]


def test_compute_kept_frame_indices_rejects_mismatched_length() -> None:
    frame_deltas = np.array([0.0, 0.1], dtype=np.float32)
    state_deltas = np.array([0.0], dtype=np.float32)

    try:
        compute_kept_frame_indices(frame_deltas, state_deltas, 0.1, 0.1)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched lengths")


def test_compute_kept_frame_indices_rejects_2d_input() -> None:
    frame_deltas = np.array([[0.0, 0.1]], dtype=np.float32)
    state_deltas = np.array([0.0, 0.1], dtype=np.float32)

    try:
        compute_kept_frame_indices(frame_deltas, state_deltas, 0.1, 0.1)
    except ValueError as exc:
        assert "1-D arrays" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-1-D input")


def test_write_frame_filter_manifest_writes_json_manifest(tmp_path: Path) -> None:
    output_path = tmp_path / "frame_filter_manifest.json"

    write_frame_filter_manifest(output_path, [2, 3], [0, 1], 4)

    assert output_path.read_text(encoding="utf-8") == (
        '{"total_frames":4,"kept_frame_indices":[2,3],"dropped_frame_indices":[0,1]}\n'
    )


def test_export_filtered_video_frames_writes_only_kept_frames(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (16, 16))
    assert writer.isOpened()
    source_colors = (16, 128, 240)
    for color in source_colors:
        writer.write(np.full((16, 16, 3), color, dtype=np.uint8))
    writer.release()

    output = tmp_path / "filtered.mp4"
    export_filtered_video_frames(source, output, kept_indices=[0, 2])

    source_capture = cv2.VideoCapture(str(source))
    source_fps = source_capture.get(cv2.CAP_PROP_FPS)
    source_capture.release()

    capture = cv2.VideoCapture(str(output))
    output_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    assert (frame_width, frame_height) == (16, 16)
    assert frame_count == 2
    assert np.isclose(output_fps, source_fps)

    assert len(frames) == 2
    first_mean = float(frames[0].mean())
    second_mean = float(frames[1].mean())
    assert first_mean < second_mean
    assert abs(first_mean - source_colors[0]) < abs(first_mean - source_colors[2])
    assert abs(second_mean - source_colors[2]) < abs(second_mean - source_colors[0])


def test_export_filtered_video_frames_rejects_invalid_kept_indices(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (16, 16))
    assert writer.isOpened()
    for color in (16, 128, 240):
        writer.write(np.full((16, 16, 3), color, dtype=np.uint8))
    writer.release()

    output = tmp_path / "filtered.mp4"

    for kept_indices, expected_message in (
        ([-1], "negative"),
        ([0, 0], "duplicate"),
        ([3], "out of range"),
    ):
        try:
            export_filtered_video_frames(source, output, kept_indices=kept_indices)
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"Expected ValueError for kept_indices={kept_indices}")


def test_export_filtered_video_frames_rejects_non_integer_kept_indices(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (16, 16))
    assert writer.isOpened()
    for color in (16, 128, 240):
        writer.write(np.full((16, 16, 3), color, dtype=np.uint8))
    writer.release()

    output = tmp_path / "filtered.mp4"

    for kept_indices in ([1.5], [True]):
        try:
            export_filtered_video_frames(source, output, kept_indices=kept_indices)
        except ValueError as exc:
            assert "integers only" in str(exc)
        else:
            raise AssertionError(f"Expected ValueError for kept_indices={kept_indices}")


def test_export_comparison_video_writes_side_by_side_output(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (16, 16))
    assert writer.isOpened()
    for value in (10, 50, 90):
        writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()

    filtered = tmp_path / "filtered.mp4"
    writer = cv2.VideoWriter(str(filtered), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (16, 16))
    assert writer.isOpened()
    for value in (10, 90, 130, 170):
        writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()

    output = tmp_path / "comparison.mp4"
    export_comparison_video(source, filtered, output)

    capture = cv2.VideoCapture(str(output))
    ok, frame = capture.read()
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert ok is True
    assert frame.shape[1] == 32
    assert frame_count == 3
