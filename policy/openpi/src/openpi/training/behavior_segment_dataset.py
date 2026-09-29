from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import json
import logging
from pathlib import Path

import datasets
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchvision.io import VideoReader

logger = logging.getLogger(__name__)

HEAD_VIDEO_KEY = "observation.images.rgb.head"
LEFT_WRIST_VIDEO_KEY = "observation.images.rgb.left_wrist"
RIGHT_WRIST_VIDEO_KEY = "observation.images.rgb.right_wrist"
FRAME_CACHE_VIEW_NAMES = {
    HEAD_VIDEO_KEY: "head",
    LEFT_WRIST_VIDEO_KEY: "left_wrist",
    RIGHT_WRIST_VIDEO_KEY: "right_wrist",
}


@dataclasses.dataclass(frozen=True)
class ManifestRow:
    policy_type: str
    task_index: int
    task_name: str
    task_instruction: str
    episode_index: int
    dataset_index: int
    segment_id: int
    policy_prompt: str
    stage_name: str


@dataclasses.dataclass(frozen=True)
class EpisodeIndexRange:
    start: int
    end: int


PromptResolver = Callable[[ManifestRow], str]


def normalize_prompt_text(prompt: str) -> str:
    return " ".join(prompt.split())


def _load_manifest_rows(manifest_path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(ManifestRow(**json.loads(line)))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def filter_manifest_rows(rows: Sequence[ManifestRow], allowed_stage_names: Sequence[str]) -> list[ManifestRow]:
    allowed = set(allowed_stage_names)
    return [row for row in rows if row.stage_name in allowed]


def classify_task0_row(row: ManifestRow) -> int:
    from openpi.training import task0_stage_training

    return task0_stage_training.classify_task0_row(row)


def _filter_task0_rows(rows: Sequence[ManifestRow], task_index_filter: int | None) -> list[ManifestRow]:
    if task_index_filter is None:
        return list(rows)
    if task_index_filter != 0:
        raise ValueError("task_index_filter is only supported for task0 unified datasets")
    return [row for row in rows if row.task_index == task_index_filter]


def select_canonical_prompt(rows: Sequence[ManifestRow]) -> str:
    if not rows:
        raise ValueError("Cannot select a canonical prompt from an empty row set.")

    counts: dict[str, int] = {}
    for row in rows:
        normalized = normalize_prompt_text(row.policy_prompt)
        counts[normalized] = counts.get(normalized, 0) + 1

    canonical_prompt, _ = max(counts.items(), key=lambda item: (item[1], item[0]))
    return canonical_prompt


def _episode_file_sort_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _read_frame(video_path: Path, timestamp: float, tolerance_s: float) -> np.ndarray:
    reader = VideoReader(str(video_path), "video")
    reader.seek(max(timestamp - 1.0, 0.0), keyframes_only=True)

    best_frame: torch.Tensor | None = None
    best_distance = float("inf")
    for frame in reader:
        pts = float(frame["pts"])
        distance = abs(pts - timestamp)
        if distance < best_distance:
            best_distance = distance
            best_frame = frame["data"]
        if pts > timestamp and distance > best_distance:
            break
        if pts > timestamp + tolerance_s and best_frame is not None:
            break

    if best_frame is None or best_distance > tolerance_s:
        raise ValueError(
            f"Could not decode frame near timestamp={timestamp} from {video_path}; best_distance={best_distance}"
        )

    return best_frame.permute(1, 2, 0).numpy()


def get_frame_cache_path(
    cache_root: Path,
    task_index: int,
    episode_index: int,
    dataset_index: int,
    video_key: str,
) -> Path:
    view_name = FRAME_CACHE_VIEW_NAMES[video_key]
    return (
        cache_root
        / f"task-{task_index:04d}"
        / f"episode_{episode_index:08d}"
        / view_name
        / f"dataset_{dataset_index:08d}.jpg"
    )


class BehaviorSegmentDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_root: str,
        manifest_path: str,
        action_horizon: int,
        frame_cache_root: str | None = None,
        video_tolerance_s: float = 0.2,
        runtime_stage_id: int | None = None,
        allowed_stage_names: Sequence[str] | None = None,
        canonical_prompt: str | None = None,
        prompt_resolver: PromptResolver | None = None,
        task_index_filter: int | None = None,
    ) -> None:
        self._root = Path(dataset_root).expanduser().resolve()
        self._manifest_path = Path(manifest_path).expanduser().resolve()
        self._action_horizon = action_horizon
        self._frame_cache_root = Path(frame_cache_root).expanduser().resolve() if frame_cache_root else None
        self._video_tolerance_s = video_tolerance_s
        rows = _load_manifest_rows(self._manifest_path)
        rows = _filter_task0_rows(rows, task_index_filter)
        if runtime_stage_id is not None:
            rows = [row for row in rows if classify_task0_row(row) == runtime_stage_id]
        if allowed_stage_names is not None:
            rows = filter_manifest_rows(rows, allowed_stage_names)
        if not rows:
            raise ValueError(f"No manifest rows remain after filtering: {self._manifest_path}")

        self._rows = rows
        self._canonical_prompt = normalize_prompt_text(canonical_prompt) if canonical_prompt else select_canonical_prompt(rows)
        self._prompt_resolver = prompt_resolver
        self._runtime_stage_ids = (
            tuple(classify_task0_row(row) for row in rows) if all(row.task_index == 0 for row in rows) else ()
        )

        task_indices = sorted({row.task_index for row in self._rows})
        parquet_files: list[str] = []
        self._episode_ranges: dict[int, EpisodeIndexRange] = {}
        global_offset = 0
        for task_index in task_indices:
            task_dir = self._root / "data" / f"task-{task_index:04d}"
            files = sorted(task_dir.glob("episode_*.parquet"), key=_episode_file_sort_key)
            if not files:
                raise FileNotFoundError(f"No parquet files found under {task_dir}")
            for parquet_path in files:
                episode_index = _episode_file_sort_key(parquet_path)
                num_rows = pq.ParquetFile(parquet_path).metadata.num_rows
                self._episode_ranges[episode_index] = EpisodeIndexRange(
                    start=global_offset,
                    end=global_offset + num_rows - 1,
                )
                global_offset += num_rows
                parquet_files.append(str(parquet_path))

        self._dataset = datasets.load_dataset("parquet", data_files=parquet_files, split="train")
        max_index = max(row.dataset_index for row in self._rows)
        if max_index >= len(self._dataset):
            raise ValueError(
                f"Manifest references dataset_index={max_index}, but dataset length is only {len(self._dataset)}"
            )

    @classmethod
    def from_manifest_rows(
        cls,
        *,
        rows: Sequence[ManifestRow],
        dataset,
        episode_ranges: dict[int, EpisodeIndexRange],
        action_horizon: int,
        canonical_prompt: str,
        frame_cache_root: str | None = None,
        video_tolerance_s: float = 0.2,
        prompt_resolver: PromptResolver | None = None,
        task_index_filter: int | None = None,
    ) -> BehaviorSegmentDataset:
        instance = cls.__new__(cls)
        row_list = _filter_task0_rows(rows, task_index_filter)
        instance.__dict__.update(
            {
                "_root": Path.cwd(),
                "_manifest_path": Path("<in-memory-manifest>"),
                "_action_horizon": action_horizon,
                "_frame_cache_root": Path(frame_cache_root).expanduser().resolve() if frame_cache_root else None,
                "_video_tolerance_s": video_tolerance_s,
                "_rows": row_list,
                "_canonical_prompt": normalize_prompt_text(canonical_prompt),
                "_dataset": dataset,
                "_episode_ranges": episode_ranges,
                "_prompt_resolver": prompt_resolver,
                "_runtime_stage_ids": tuple(classify_task0_row(row) for row in row_list)
                if all(row.task_index == 0 for row in row_list)
                else (),
            }
        )
        return instance

    @property
    def rows(self) -> tuple[ManifestRow, ...]:
        return tuple(self._rows)

    @property
    def canonical_prompt(self) -> str:
        return self._canonical_prompt

    @property
    def runtime_stage_ids(self) -> tuple[int, ...]:
        return self._runtime_stage_ids

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict:
        row = self._rows[index]
        sample = self._dataset[row.dataset_index]
        episode_index = int(sample["episode_index"])
        task_index = int(sample["task_index"])
        if episode_index != row.episode_index:
            raise ValueError(
                f"Manifest/data mismatch at index={index}: manifest episode={row.episode_index}, data episode={episode_index}"
            )
        if task_index != row.task_index:
            raise ValueError(
                f"Manifest/data mismatch at index={index}: manifest task={row.task_index}, data task={task_index}"
            )

        timestamp = float(sample["timestamp"])
        head_image = self._load_frame(task_index, episode_index, row.dataset_index, timestamp, HEAD_VIDEO_KEY)
        left_wrist_image = self._load_frame(task_index, episode_index, row.dataset_index, timestamp, LEFT_WRIST_VIDEO_KEY)
        right_wrist_image = self._load_frame(
            task_index,
            episode_index,
            row.dataset_index,
            timestamp,
            RIGHT_WRIST_VIDEO_KEY,
        )

        return {
            "observation/state": np.asarray(sample["observation.state"], dtype=np.float32),
            "observation/head_image": head_image,
            "observation/left_wrist_image": left_wrist_image,
            "observation/right_wrist_image": right_wrist_image,
            "actions": self._action_chunk(row.dataset_index, episode_index),
            "prompt": self._prompt_resolver(row) if self._prompt_resolver is not None else self._canonical_prompt,
            "task_index": np.asarray(task_index, dtype=np.int64),
            "episode_index": np.asarray(episode_index, dtype=np.int64),
            "dataset_index": np.asarray(row.dataset_index, dtype=np.int64),
        }

    def _action_chunk(self, dataset_index: int, episode_index: int) -> np.ndarray:
        episode_range = self._episode_ranges[episode_index]
        indices = [min(dataset_index + offset, episode_range.end) for offset in range(self._action_horizon)]
        batch = self._dataset[indices]
        return np.asarray(batch["action"], dtype=np.float32)

    def _load_frame(
        self,
        task_index: int,
        episode_index: int,
        dataset_index: int,
        timestamp: float,
        video_key: str,
    ) -> np.ndarray:
        if self._frame_cache_root is not None:
            cache_path = get_frame_cache_path(self._frame_cache_root, task_index, episode_index, dataset_index, video_key)
            if cache_path.exists():
                with Image.open(cache_path) as image:
                    return np.asarray(image.convert("RGB"))

        return _read_frame(self._video_path(task_index, episode_index, video_key), timestamp, self._video_tolerance_s)

    def _video_path(self, task_index: int, episode_index: int, video_key: str) -> Path:
        video_path = self._root / "videos" / f"task-{task_index:04d}" / video_key / f"episode_{episode_index:08d}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video file: {video_path}")
        return video_path
