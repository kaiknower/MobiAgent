"""Episode selection for s1_mobile.

A task_id (e.g. "trash-bottle") maps to one or more dataset_ids
(per config.TASK_TO_DATASET_IDS). Each dataset_id corresponds to a
separate recording batch. Episodes are uniquely identified as
"<dataset_id>/<episode_stem>" to avoid cross-batch collisions, but
the on-disk file names (episode_NNNNNN.*) reset per dataset_id.

On-disk layout per dataset_id:
    videos/{head_rgb,left_rgb,right_rgb}/<dataset_id>/episode_NNNNNN.mp4
    moving_trash_disposal/<dataset_id>/lerobot_so3_data_30hz/
        data/chunk-000/episode_NNNNNN.parquet
        meta/episodes.jsonl  (per-dataset_id; one row per episode)
"""

from pathlib import Path

from .config import (
    CHUNK_DIRNAME,
    DATA_DIRNAME,
    DEFAULT_DATASET_FAMILY_DIRNAME,
    HEAD_CAMERA_DIRNAME,
    LEFT_CAMERA_DIRNAME,
    LEROBOT_SUBDIR,
    META_DIRNAME,
    RIGHT_CAMERA_DIRNAME,
    TASK_TO_DATASET_IDS,
    TASK_TO_FAMILY,
    VIDEOS_DIRNAME,
)
from .models import SelectedDemo


def _episode_assets(
    dataset_root: Path, dataset_id: str, episode_stem: str
) -> tuple[Path, Path, Path, Path, Path]:
    from .config import DEFAULT_DATASET_FAMILY_DIRNAME, TASK_TO_FAMILY
    # We don't know task_id here; resolve family by searching configured families.
    family = None
    for cand in (DEFAULT_DATASET_FAMILY_DIRNAME, *set(TASK_TO_FAMILY.values())):
        if (dataset_root / cand / dataset_id).exists():
            family = cand
            break
    if family is None:
        family = DEFAULT_DATASET_FAMILY_DIRNAME
    lerobot_root = dataset_root / family / dataset_id / LEROBOT_SUBDIR
    videos_root = lerobot_root / VIDEOS_DIRNAME / CHUNK_DIRNAME
    head = videos_root / HEAD_CAMERA_DIRNAME / f"{episode_stem}.mp4"
    left = videos_root / LEFT_CAMERA_DIRNAME / f"{episode_stem}.mp4"
    right = videos_root / RIGHT_CAMERA_DIRNAME / f"{episode_stem}.mp4"
    parquet = lerobot_root / DATA_DIRNAME / CHUNK_DIRNAME / f"{episode_stem}.parquet"
    # meta_path is informational (per-dataset_id, not per-episode); episodes.jsonl
    # holds one row per episode and is the closest analogue to the old
    # per-episode meta JSON.
    meta = lerobot_root / META_DIRNAME / "episodes.jsonl"
    return head, left, right, meta, parquet


def _iter_dataset_ids(task_id: str) -> list[str]:
    try:
        return TASK_TO_DATASET_IDS[task_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown task_id {task_id!r}; expected one of {sorted(TASK_TO_DATASET_IDS)}"
        ) from exc


def select_first_demo_per_task(dataset_root: Path, task_ids: list[str]) -> list[SelectedDemo]:
    """Return the first complete episode for each task, picking from the first
    dataset_id that has at least one complete episode (sort order: name)."""
    selected: list[SelectedDemo] = []

    for task_id in task_ids:
        picked = False
        for dataset_id in _iter_dataset_ids(task_id):
            head_dir = (
                dataset_root
                / TASK_TO_FAMILY.get(task_id, DEFAULT_DATASET_FAMILY_DIRNAME)
                / dataset_id
                / LEROBOT_SUBDIR
                / VIDEOS_DIRNAME
                / CHUNK_DIRNAME
                / HEAD_CAMERA_DIRNAME
            )
            if not head_dir.exists():
                continue
            head_candidates = sorted(head_dir.glob("episode_*.mp4"), key=lambda path: path.name)
            for head_video_path in head_candidates:
                episode_stem = head_video_path.stem
                head, left, right, meta, parquet = _episode_assets(
                    dataset_root, dataset_id, episode_stem
                )
                if not all(p.exists() for p in (head, left, right, meta, parquet)):
                    continue
                selected.append(
                    SelectedDemo(
                        task_id=task_id,
                        episode_id=f"{dataset_id}/{episode_stem}",
                        head_video_path=head,
                        left_video_path=left,
                        right_video_path=right,
                        meta_path=meta,
                        parquet_path=parquet,
                        dataset_id=dataset_id,
                    )
                )
                picked = True
                break
            if picked:
                break

    return selected


def select_all_demos_per_task(dataset_root: Path, task_ids: list[str]) -> list[SelectedDemo]:
    """Return every complete episode (head + left + right + parquet present)
    across all dataset_ids that back each task."""
    selected: list[SelectedDemo] = []

    for task_id in task_ids:
        for dataset_id in _iter_dataset_ids(task_id):
            head_dir = (
                dataset_root
                / TASK_TO_FAMILY.get(task_id, DEFAULT_DATASET_FAMILY_DIRNAME)
                / dataset_id
                / LEROBOT_SUBDIR
                / VIDEOS_DIRNAME
                / CHUNK_DIRNAME
                / HEAD_CAMERA_DIRNAME
            )
            if not head_dir.exists():
                continue
            head_candidates = sorted(head_dir.glob("episode_*.mp4"), key=lambda path: path.name)
            for head_video_path in head_candidates:
                episode_stem = head_video_path.stem
                head, left, right, meta, parquet = _episode_assets(
                    dataset_root, dataset_id, episode_stem
                )
                if not all(p.exists() for p in (head, left, right, meta, parquet)):
                    continue
                selected.append(
                    SelectedDemo(
                        task_id=task_id,
                        episode_id=f"{dataset_id}/{episode_stem}",
                        head_video_path=head,
                        left_video_path=left,
                        right_video_path=right,
                        meta_path=meta,
                        parquet_path=parquet,
                        dataset_id=dataset_id,
                    )
                )

    return selected
