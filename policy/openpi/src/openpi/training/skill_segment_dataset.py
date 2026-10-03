"""Skill-segment dataset for the 6-head π0.5 training plan.

Reads `head__*.jsonl` shards produced by `scripts/data/split_per_head.py` and yields
per-segment training samples in the same dict shape consumed by
``openpi.training.config.DataConfig`` -> the model's ``Observation``.

Schema notes
------------
- Each jsonl row points at one segment in the BEHAVIOR-1K 30 Hz dataset.
- `__getitem__` picks one frame inside the segment (random start when
  `n_frames > action_horizon`, deterministic start otherwise) and returns:
    - observation/state        : (256,) float32   parquet 'observation.state'[idx]
    - observation/head_image   : (224, 224, 3) uint8
    - observation/left_wrist_image  : (224, 224, 3) uint8
    - observation/right_wrist_image : (224, 224, 3) uint8
    - actions                  : (H, 23) float32  parquet 'action'[idx : idx+H]
    - prompt                   : f"{task_instruction}. Now: {skill_description}."
    - skill_canonical_id       : int64  index into CANONICAL_HEADS
    - skill_canonical          : str    one of CANONICAL_HEADS (debug-friendly)
    - sample_id                : str    "task-XXXX/episode_XXXXXXXX/segment-XXX"
    - task_index               : int64  parsed from task_id
    - episode_index            : int64  parsed from episode_id

The dataset is intentionally jsonl-driven so it does not need to know about
the 4-stage manifest pipeline in ``behavior_segment_dataset.py``.
"""
from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
import logging
import os
import threading as _threading
import random as _stdlib_random
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _read_parquet_state_action(
    parquet_path: str,
    state_column: str = "observation.state",
    action_column: str = "action",
) -> tuple[str, np.ndarray, np.ndarray]:
    """Worker (top-level for picklability): read one parquet's (state, action) columns.

    Converts Arrow list columns with pa.compute.list_flatten and to_numpy.

    Columns default to BEHAVIOR-1K's ``observation.state`` (256-dim) / ``action`` (23-dim);
    pass alternative column names for other datasets. Default columns retain the explicit shapes
    exactly; for non-default columns the inner dimension is inferred from the parquet
    list values (``value_length``) so this helper supports arbitrary dims.
    """
    tbl = pq.read_table(parquet_path, columns=[state_column, action_column])
    if state_column == "observation.state" and action_column == "action":
        # Legacy path — keep the explicit reshapes so we crash loudly on any
        # schema drift in the BEHAVIOR-1K parquets.
        state_arr = (
            pa.compute.list_flatten(tbl[state_column])
            .to_numpy(zero_copy_only=False)
            .reshape(-1, 256)
            .astype(np.float32, copy=False)
        )
        action_arr = (
            pa.compute.list_flatten(tbl[action_column])
            .to_numpy(zero_copy_only=False)
            .reshape(-1, 23)
            .astype(np.float32, copy=False)
        )
    else:
        def _to_2d(col: "pa.ChunkedArray", name: str) -> np.ndarray:
            n_rows = len(col)
            flat = pa.compute.list_flatten(col).to_numpy(zero_copy_only=False)
            if n_rows == 0:
                return flat.astype(np.float32, copy=False).reshape(0, 0)
            if flat.size % n_rows != 0:
                raise ValueError(
                    f"Cannot reshape parquet column {name!r} from {parquet_path}: "
                    f"flat size {flat.size} not divisible by n_rows {n_rows} "
                    "(variable-length rows are not supported)."
                )
            inner = flat.size // n_rows
            return flat.reshape(n_rows, inner).astype(np.float32, copy=False)

        state_arr = _to_2d(tbl[state_column], state_column)
        action_arr = _to_2d(tbl[action_column], action_column)
    return parquet_path, state_arr, action_arr

CANONICAL_HEADS: tuple[str, ...] = (
    "move_to",
    "pick_up_from",
    "place_in",
    "place_on",
    "open",
    "close",
)
CANONICAL_HEAD_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CANONICAL_HEADS)}

DEFAULT_FPS = 30
DEFAULT_VIDEO_TOLERANCE_S = 0.5 / DEFAULT_FPS  # half a frame


@dataclasses.dataclass(frozen=True)
class SkillSegmentRow:
    """One line of a head__*.jsonl file."""

    sample_id: str
    task_id: str
    episode_id: str
    segment_id: str
    task_instruction: str
    skill_canonical: str
    skill_description: str
    start_idx_30hz: int
    end_idx_30hz: int
    n_frames: int
    head_video: str
    left_video: str
    right_video: str
    parquet: str
    meta: str

    @classmethod
    def from_dict(cls, d: dict) -> "SkillSegmentRow":
        # Tolerate extra fields (start_time_sec_source / end_time_sec_source / etc.)
        start, end, length = (d[key] for key in ("start_idx_30hz", "end_idx_30hz", "n_frames"))
        if any(type(value) is not int for value in (start, end, length)) or start < 0 or end <= start or length != end - start:
            raise ValueError(f"Invalid frame interval for {d.get('sample_id')}")
        return cls(
            sample_id=d["sample_id"],
            task_id=d["task_id"],
            episode_id=d["episode_id"],
            segment_id=d["segment_id"],
            task_instruction=d["task_instruction"],
            skill_canonical=d["skill_canonical"],
            skill_description=d["skill_description"],
            start_idx_30hz=int(d["start_idx_30hz"]),
            end_idx_30hz=int(d["end_idx_30hz"]),
            n_frames=int(d["n_frames"]),
            head_video=d["head_video"],
            left_video=d["left_video"],
            right_video=d["right_video"],
            parquet=d["parquet"],
            meta=d["meta"],
        )

    @property
    def task_index(self) -> int:
        # "task-0020" -> 20 (behavior); for non-numeric task_id (e.g. "pour-blue",
        # "trash-bottle-1" etc.) fall back to a stable deterministic int from
        # the full task_id so downstream code that treats it as a category still
        # works (different task_ids → different ints, same task_id → same int).
        last = self.task_id.split("-")[-1]
        if last.isdigit():
            return int(last)
        return abs(hash(self.task_id)) % (2**31)

    @property
    def episode_index(self) -> int:
        # "episode_00200360" -> 200360 (behavior); "episode_000007" also works.
        last = self.episode_id.split("_")[-1]
        if last.isdigit():
            return int(last)
        return abs(hash(self.episode_id)) % (2**31)


def load_head_jsonl(path: str | Path) -> list[SkillSegmentRow]:
    rows: list[SkillSegmentRow] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(SkillSegmentRow.from_dict(json.loads(line)))
    return rows


def load_all_head_jsonls(
    skill_segments_dir: str | Path,
    canonical_heads: Sequence[str] = CANONICAL_HEADS,
) -> list[SkillSegmentRow]:
    """Load all 6 head__*.jsonl files in ``skill_segments_dir`` in canonical order."""
    skill_segments_dir = Path(skill_segments_dir)
    rows: list[SkillSegmentRow] = []
    for name in canonical_heads:
        shard = skill_segments_dir / f"head__{name}.jsonl"
        if not shard.exists():
            raise FileNotFoundError(f"Missing head shard: {shard}")
        rows.extend(load_head_jsonl(shard))
    return rows


def build_prompt(row: SkillSegmentRow, style: str = "task_then_now") -> str:
    """Build the VLM prompt.

    ``task_then_now`` combines the task instruction and skill description.
    ``skill_only`` returns the skill description verbatim."""
    if style == "skill_only":
        return row.skill_description
    if style != "task_then_now":
        raise ValueError(f"Unknown prompt style: {style!r}")
    instr = row.task_instruction.rstrip(". ")
    desc = row.skill_description.rstrip(". ")
    return f"{instr}. Now: {desc}."


import io
import struct

# Packed JPEG cache: one indexed file per source video.
SKILL_SEGMENT_PACKED_CACHE_ROOT = Path(
    os.environ.get("OPENPI_SKILL_SEGMENT_PACKED_CACHE_ROOT")
    or str(Path(__file__).resolve().parents[5] / "data/behavior/frame_cache")
)
SKILL_SEGMENT_VIDEO_ROOT = Path(
    os.environ.get("OPENPI_SKILL_SEGMENT_VIDEO_ROOT")
    or str(Path(__file__).resolve().parents[5] / "datasets/behavior/videos")
)
_PACK_HEADER = struct.Struct("<Q")
_PACK_ENTRY = struct.Struct("<qqq")


def _packed_path_for(video_path: Path) -> Path:
    rel = Path(video_path).resolve().relative_to(SKILL_SEGMENT_VIDEO_ROOT.resolve()).with_suffix("")
    return SKILL_SEGMENT_PACKED_CACHE_ROOT / rel.parent / f"{rel.name}.pak"


class _PackedJpegReader:
    """Random-access, mmap-backed reader for one packed JPEG file.

    The frame index is loaded once per reader. Memory-mapped reads share the
    operating system page cache."""

    __slots__ = ("path", "_fd", "_mm", "_offsets", "_lengths", "_data_start")

    def __init__(self, path: str) -> None:
        import mmap  # noqa: PLC0415 (stdlib, lazy to keep top-level imports tidy)

        self.path = path
        self._fd = open(path, "rb")  # noqa: SIM115 (handle owned by reader)
        self._mm = mmap.mmap(self._fd.fileno(), 0, prot=mmap.PROT_READ)
        n = _PACK_HEADER.unpack_from(self._mm, 0)[0]
        idx_start = _PACK_HEADER.size
        offsets: dict[int, int] = {}
        lengths: dict[int, int] = {}
        for i in range(n):
            idx, off, length = _PACK_ENTRY.unpack_from(self._mm, idx_start + i * _PACK_ENTRY.size)
            offsets[idx] = off
            lengths[idx] = length
        self._offsets = offsets
        self._lengths = lengths
        self._data_start = idx_start + n * _PACK_ENTRY.size

    def has(self, frame_idx: int) -> bool:
        return frame_idx in self._offsets

    def max_idx(self) -> int:
        return max(self._offsets) if self._offsets else -1

    def get(self, frame_idx: int) -> bytes:
        off = self._offsets[frame_idx]
        length = self._lengths[frame_idx]
        start = self._data_start + off
        return self._mm[start : start + length]  # mmap slice returns bytes copy

    def close(self) -> None:
        try:
            self._mm.close()
        except (OSError, ValueError):
            pass
        try:
            self._fd.close()
        except OSError:
            pass


# Thread-local reader caches avoid concurrent LRU updates.
_PACKED_READERS_LOCAL = _threading.local()
_PACKED_READERS_MAX = 256


def _packed_readers_dict() -> dict[str, _PackedJpegReader]:
    d = getattr(_PACKED_READERS_LOCAL, "readers", None)
    if d is None:
        d = {}
        _PACKED_READERS_LOCAL.readers = d
    return d

# Thread-local OpenCV handles keep video seek state isolated and bounded.
_CV2_CAPS_LOCAL = _threading.local()
_CV2_CAPS_MAX = 8


def _cv2_caps_dict() -> dict[str, "object"]:
    d = getattr(_CV2_CAPS_LOCAL, "caps", None)
    if d is None:
        d = {}
        _CV2_CAPS_LOCAL.caps = d
    return d


def _get_reader(packed_path: Path) -> _PackedJpegReader:
    """Per-thread LRU of packed readers. mmap-backed reads share OS page
    cache across threads; per-thread dict avoids races on LRU bookkeeping."""
    readers = _packed_readers_dict()
    key = str(packed_path)
    reader = readers.get(key)
    if reader is not None:
        return reader
    if len(readers) >= _PACKED_READERS_MAX:
        old_key = next(iter(readers))
        old_reader = readers.pop(old_key)
        try: old_reader.close()
        except Exception: pass
    reader = _PackedJpegReader(key)
    readers[key] = reader
    return reader


_CV2_THREADS_PINNED = False


def _get_cv2_cap(video_path: str):
    """Return a video capture from the thread-local bounded cache.

    Each thread owns its capture handles. Limit decoder threads to avoid
    CPU oversubscription when loading multiple videos concurrently."""
    global _CV2_THREADS_PINNED
    caps = _cv2_caps_dict()
    cap = caps.get(video_path)
    if cap is not None:
        return cap
    import cv2  # noqa: PLC0415  (lazy: pack-cache build script doesn't need cv2)
    if not _CV2_THREADS_PINNED:
        cv2.setNumThreads(1)  # also affects FFmpeg backend
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
        _CV2_THREADS_PINNED = True
    if len(caps) >= _CV2_CAPS_MAX:
        old_key = next(iter(caps))
        old_cap = caps.pop(old_key)
        try: old_cap.release()
        except Exception: pass
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 failed to open {video_path}")
    caps[video_path] = cap
    return cap


def _seek_frame(video_path: Path, target_idx: int, fps: int = DEFAULT_FPS) -> np.ndarray:
    """Decode a video frame as HWC uint8 RGB.

    Use the packed JPEG cache when available, otherwise decode the source
    video with OpenCV. Videos outside the configured cache root are decoded
    directly."""
    try:
        packed_path = _packed_path_for(Path(video_path))
    except ValueError:
        packed_path = None
    if packed_path is None or not packed_path.exists():
        # No packed cache — read straight from mp4 via cv2 (BGR→RGB).
        import cv2  # noqa: PLC0415
        cap = _get_cv2_cap(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = int(target_idx)
        if idx >= n:
            idx = max(0, n - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cv2 failed to read frame {idx} from {video_path}")
        return frame[:, :, ::-1].copy()  # BGR → RGB, contiguous
    reader = _get_reader(packed_path)
    idx = int(target_idx)
    if not reader.has(idx):
        # Index over-runs were clipped to last available frame at build time;
        # fall back the same way for safety.
        idx = reader.max_idx()
    jpg_bytes = reader.get(idx)
    with Image.open(io.BytesIO(jpg_bytes)) as img:
        return np.asarray(img.convert("RGB"))


class SkillSegmentDataset(Dataset):
    """6-head training dataset.

    Yields one sample per ``__getitem__`` call. Combine with
    ``WeightedRandomSampler`` (using ``sampler_weights.json`` from
    ``split_per_head.py``) to rebalance class frequency.

    Args:
        head_jsonl_paths: explicit list of jsonl shard paths. Either this OR
            ``skill_segments_dir`` must be provided.
        skill_segments_dir: directory containing ``head__{canonical}.jsonl``
            files; resolved via :func:`load_all_head_jsonls`.
        action_horizon: action chunk length in frames (30 Hz). Defaults to 30.
        random_window: when True (default) and ``n_frames > action_horizon``,
            picks a random start index inside the segment. When False, always
            starts at ``start_idx_30hz`` (deterministic; useful for eval / unit
            tests).
        seed: optional torch.Generator seed for the per-sample random window.
            Distinct per worker is recommended.
    """

    def __init__(
        self,
        *,
        head_jsonl_paths: Sequence[str | Path] | None = None,
        skill_segments_dir: str | Path | None = None,
        action_horizon: int = 30,
        random_window: bool = True,
        seed: int | None = None,
        canonical_heads: Sequence[str] = CANONICAL_HEADS,
        use_per_frame_state_delta: bool = False,
        prompt_style: str = "task_then_now",
        state_column: str = "observation.state",
        action_column: str = "action",
    ) -> None:
        if head_jsonl_paths is None and skill_segments_dir is None:
            raise ValueError("Must provide head_jsonl_paths or skill_segments_dir")
        if head_jsonl_paths is not None and skill_segments_dir is not None:
            raise ValueError("Provide head_jsonl_paths OR skill_segments_dir, not both")

        if skill_segments_dir is not None:
            self._rows = load_all_head_jsonls(skill_segments_dir, canonical_heads)
        else:
            self._rows = []
            for path in head_jsonl_paths:  # type: ignore[union-attr]
                self._rows.extend(load_head_jsonl(path))

        if not self._rows:
            raise ValueError("No rows loaded — check jsonl shard paths")

        # Validate canonical labels match expected vocab (so a stray row from a
        # mis-built shard fails loud, not silent).
        unknown = {r.skill_canonical for r in self._rows} - set(canonical_heads)
        if unknown:
            raise ValueError(f"Unknown skill_canonical values in shards: {unknown}")

        self._action_horizon = int(action_horizon)
        self._random_window = bool(random_window)
        self._rng = _stdlib_random.Random(seed)
        self._head_id = {n: i for i, n in enumerate(canonical_heads)}
        self._use_per_frame_state_delta = bool(use_per_frame_state_delta)
        self._prompt_style = str(prompt_style)
        self._state_column = str(state_column)
        self._action_column = str(action_column)

        # Preload state/action columns to avoid repeated parquet reads.
        # Set OPENPI_NO_PARQUET_PRELOAD=1 to decode columns on demand.
        if os.environ.get("OPENPI_NO_PARQUET_PRELOAD", "0") != "1":
            self._preload_parquet_cache()

    def _preload_parquet_cache(self) -> None:
        import time
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp

        unique_paths = sorted({r.parquet for r in self._rows})
        n = len(unique_paths)
        # Load parquet columns in separate processes.
        # Configure concurrency with OPENPI_PRELOAD_WORKERS.
        n_workers = int(os.environ.get("OPENPI_PRELOAD_WORKERS", "8"))
        n_workers = min(n_workers, mp.cpu_count(), max(1, n))
        logger.info(
            f"Preloading {n} parquet files into RAM (state + action) "
            f"with {n_workers} ProcessPool workers..."
        )
        t0 = time.time()
        cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        total_bytes = 0
        completed = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _read_parquet_state_action, p, self._state_column, self._action_column
                )
                for p in unique_paths
            ]
            for fut in as_completed(futures):
                p, state_arr, action_arr = fut.result()
                cache[p] = (state_arr, action_arr)
                total_bytes += state_arr.nbytes + action_arr.nbytes
                completed += 1
                if completed % 100 == 0 or completed == n:
                    logger.info(
                        f"  parquet preload {completed}/{n}  "
                        f"({total_bytes / (1024**3):.2f} GB so far, {time.time()-t0:.1f}s)"
                    )
        self._parquet_cache = cache
        logger.info(
            f"Parquet preload done: {n} files, "
            f"{total_bytes / (1024**3):.2f} GB in RAM, "
            f"took {time.time()-t0:.1f}s"
        )

    # ------------------------------------------------------------------ Dataset API

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict:
        row = self._rows[index]
        start_idx = self._pick_start(row)
        H = self._action_horizon

        # --- visual obs (single frame) ---
        head_image = _seek_frame(Path(row.head_video), start_idx)
        left_image = _seek_frame(Path(row.left_video), start_idx)
        right_image = _seek_frame(Path(row.right_video), start_idx)

        # --- proprio + action target (parquet) ---
        # Cached at __init__ in self._parquet_cache (see _preload_parquet_cache);
        # no per-sample disk I/O. Falls back to disk read if preload disabled
        # via OPENPI_NO_PARQUET_PRELOAD=1.
        cache = getattr(self, "_parquet_cache", None)
        if cache is not None and row.parquet in cache:
            state_arr, action_arr = cache[row.parquet]
            n_rows = state_arr.shape[0]
        else:
            tbl = pq.read_table(row.parquet, columns=[self._state_column, self._action_column])
            n_rows = tbl.num_rows
            state_arr = np.asarray(tbl[self._state_column].to_pylist(), dtype=np.float32)
            action_arr = np.asarray(tbl[self._action_column].to_pylist(), dtype=np.float32)
        # Clamp action window: short segments (n_frames <= H) start at
        # start_idx_30hz; we pad the tail by repeating the last frame's
        # parquet row so that downstream shape is always (H, 23).
        if row.end_idx_30hz > n_rows:
            raise ValueError(f"Segment exceeds parquet length: {row.sample_id}")
        idxs = [min(start_idx + k, row.end_idx_30hz - 1) for k in range(H)]
        state_idx = min(start_idx, n_rows - 1)

        state = state_arr[state_idx].astype(np.float32, copy=False)  # (256,)
        action = action_arr[idxs].astype(np.float32, copy=False)     # (H, 23)

        head_id = self._head_id[row.skill_canonical]
        out = {
            # --- existing openpi observation keys (consumed by data_loader/transforms) ---
            "observation/state": state,
            "observation/head_image": head_image,
            "observation/left_wrist_image": left_image,
            "observation/right_wrist_image": right_image,
            "actions": action,
            "prompt": build_prompt(row, self._prompt_style),
            # --- 6-head routing keys ---
            "skill_canonical_id": np.asarray(head_id, dtype=np.int64),
            "skill_canonical": row.skill_canonical,
            # --- bookkeeping (debug + DDP-stable shuffling) ---
            "sample_id": row.sample_id,
            "task_index": np.asarray(row.task_index, dtype=np.int64),
            "episode_index": np.asarray(row.episode_index, dtype=np.int64),
            "start_idx_30hz": np.asarray(start_idx, dtype=np.int64),
        }
        if self._use_per_frame_state_delta:
            out["action_delta_state"] = state_arr[idxs].astype(np.float32, copy=False)  # (H, 256)
        return out

    # ------------------------------------------------------------------ helpers

    @property
    def rows(self) -> tuple[SkillSegmentRow, ...]:
        return tuple(self._rows)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(r.sample_id for r in self._rows)

    @property
    def skill_canonical_ids(self) -> np.ndarray:
        """Per-row canonical id array — useful for debugging sampler distributions."""
        return np.asarray([self._head_id[r.skill_canonical] for r in self._rows], dtype=np.int64)

    @property
    def task_indices(self) -> np.ndarray:
        """Per-row task id as an integer, aligned with ``sample_ids``."""
        return np.asarray([r.task_index for r in self._rows], dtype=np.int64)

    @property
    def window_counts(self) -> np.ndarray:
        """Number of valid action-window starts for each segment row.

        This is ``max(1, segment_length - action_horizon + 1)`` and matches
        the support used by ``_pick_start`` when random windows are enabled.
        Samplers can use it to make head-internal sampling closer to
        chunk-uniform while keeping per-head batch counts fixed.
        """
        H = self._action_horizon
        return np.asarray(
            [max(1, (r.end_idx_30hz - r.start_idx_30hz) - H + 1) for r in self._rows],
            dtype=np.int64,
        )

    def _pick_start(self, row: SkillSegmentRow) -> int:
        H = self._action_horizon
        seg_len = row.end_idx_30hz - row.start_idx_30hz
        if seg_len <= H or not self._random_window:
            return row.start_idx_30hz
        # randint inclusive on both ends; slack = seg_len - H >= 1
        offset = self._rng.randint(0, seg_len - H)
        return row.start_idx_30hz + offset


def load_sampler_weights_in_order(
    sampler_weights_path: str | Path,
    sample_ids: Iterable[str],
) -> list[float]:
    """Reorder ``sampler_weights.json`` to match the dataset row order.

    ``WeightedRandomSampler(weights, num_samples, replacement=True)`` expects
    a list of weights *in the same order as the Dataset rows* — pass the result
    of this helper directly.
    """
    weights_map = json.loads(Path(sampler_weights_path).read_text())
    weights: list[float] = []
    missing: list[str] = []
    for sid in sample_ids:
        w = weights_map.get(sid)
        if w is None:
            missing.append(sid)
            continue
        weights.append(float(w))
    if missing:
        raise KeyError(
            f"sampler_weights.json missing {len(missing)} sample_ids "
            f"(first 3: {missing[:3]})"
        )
    return weights
