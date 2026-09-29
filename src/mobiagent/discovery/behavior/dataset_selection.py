from pathlib import Path

from .config import (
    DATA_DIRNAME,
    HEAD_CAMERA_DIRNAME,
    LEFT_CAMERA_DIRNAME,
    META_DIRNAME,
    RIGHT_CAMERA_DIRNAME,
    VIDEOS_DIRNAME,
)
from .models import SelectedDemo


def select_first_demo_per_task(dataset_root: Path, task_ids: list[str]) -> list[SelectedDemo]:
    selected: list[SelectedDemo] = []

    for task_id in task_ids:
        head_dir = dataset_root / VIDEOS_DIRNAME / task_id / HEAD_CAMERA_DIRNAME
        head_candidates = sorted(head_dir.glob("episode_*.mp4"), key=lambda path: path.name)
        for head_video_path in head_candidates:
            episode_id = head_video_path.stem
            left_video_path = dataset_root / VIDEOS_DIRNAME / task_id / LEFT_CAMERA_DIRNAME / f"{episode_id}.mp4"
            right_video_path = dataset_root / VIDEOS_DIRNAME / task_id / RIGHT_CAMERA_DIRNAME / f"{episode_id}.mp4"
            meta_path = dataset_root / META_DIRNAME / "episodes" / task_id / f"{episode_id}.json"
            parquet_path = dataset_root / DATA_DIRNAME / task_id / f"{episode_id}.parquet"

            if not all(
                path.exists()
                for path in (head_video_path, left_video_path, right_video_path, meta_path, parquet_path)
            ):
                continue

            selected.append(
                SelectedDemo(
                    task_id=task_id,
                    episode_id=episode_id,
                    head_video_path=head_video_path,
                    left_video_path=left_video_path,
                    right_video_path=right_video_path,
                    meta_path=meta_path,
                    parquet_path=parquet_path,
                )
            )
            break

    return selected


def select_all_demos_per_task(dataset_root: Path, task_ids: list[str]) -> list[SelectedDemo]:
    """Return every episode for each given task that has all required assets
    (head/left/right videos + meta + parquet). Used for full-dataset runs."""
    selected: list[SelectedDemo] = []

    for task_id in task_ids:
        head_dir = dataset_root / VIDEOS_DIRNAME / task_id / HEAD_CAMERA_DIRNAME
        head_candidates = sorted(head_dir.glob("episode_*.mp4"), key=lambda path: path.name)
        for head_video_path in head_candidates:
            episode_id = head_video_path.stem
            left_video_path = dataset_root / VIDEOS_DIRNAME / task_id / LEFT_CAMERA_DIRNAME / f"{episode_id}.mp4"
            right_video_path = dataset_root / VIDEOS_DIRNAME / task_id / RIGHT_CAMERA_DIRNAME / f"{episode_id}.mp4"
            meta_path = dataset_root / META_DIRNAME / "episodes" / task_id / f"{episode_id}.json"
            parquet_path = dataset_root / DATA_DIRNAME / task_id / f"{episode_id}.parquet"

            if not all(
                path.exists()
                for path in (head_video_path, left_video_path, right_video_path, meta_path, parquet_path)
            ):
                continue

            selected.append(
                SelectedDemo(
                    task_id=task_id,
                    episode_id=episode_id,
                    head_video_path=head_video_path,
                    left_video_path=left_video_path,
                    right_video_path=right_video_path,
                    meta_path=meta_path,
                    parquet_path=parquet_path,
                )
            )

    return selected
