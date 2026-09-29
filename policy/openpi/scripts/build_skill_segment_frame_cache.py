"""Pre-decode every frame the 6-head skill_segment dataset needs into JPEGs.

Why
---
``SkillSegmentDataset._seek_frame`` calls decord per frame from DataLoader
workers. JAX+CUDA are already initialized in the parent at that point, and
fork()'ing a worker that subsequently imports decord SIGKILLs in our setup
(spawn / forkserver re-run scripts/train.py which transitively imports cv2
and crashes on missing libGL.so.1). The cleanest workaround is to do all
video decoding offline in a stand-alone process, store every needed frame
as a JPEG, and have the training-time worker just ``Image.open`` it.

Build cost (full train v8): ~31M frames × 224x224 JPEG q=92 ≈ 148 GB,
~30 min on 16 CPU workers.

Layout
------
``<cache_root>/<task>/<view>/<episode>/<idx:08d>.jpg``

where ``<task>``, ``<view>``, ``<episode>`` mirror the original mp4 path
under ``<video_root>``. The training-side ``_seek_frame`` reconstructs this
path from the row's ``head_video`` / ``left_video`` / ``right_video`` field
plus the resolved frame index.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def _iter_rows(jsonl_dir: Path) -> Iterable[dict]:
    for f in sorted(jsonl_dir.glob("head__*.jsonl")):
        with f.open() as fp:
            for line in fp:
                yield json.loads(line)


def collect_jobs(jsonl_dir: Path, action_horizon: int) -> dict[str, list[int]]:
    """Return ``{video_path: sorted_unique_indices}`` covering every frame
    ``_pick_start`` could possibly return for any row referencing that mp4.
    """
    jobs: dict[str, set[int]] = defaultdict(set)
    for row in _iter_rows(jsonl_dir):
        start = int(row["start_idx_30hz"])
        end = int(row["end_idx_30hz"])
        hi = max(start, end - action_horizon)  # inclusive upper bound on start_idx
        idxs = range(start, hi + 1)
        for vk in ("head_video", "left_video", "right_video"):
            jobs[row[vk]].update(idxs)
    return {k: sorted(v) for k, v in jobs.items()}


def cache_path_for(video_path: Path, frame_idx: int, video_root: Path, cache_root: Path) -> Path:
    rel = video_path.resolve().relative_to(video_root.resolve()).with_suffix("")
    return cache_root / rel / f"{int(frame_idx):08d}.jpg"


def _decode_one_video(args: tuple[str, list[int], str, str, int]) -> tuple[str, int, int]:
    video_path_str, indices, video_root_str, cache_root_str, jpeg_quality = args
    video_path = Path(video_path_str)
    video_root = Path(video_root_str)
    cache_root = Path(cache_root_str)

    out_dir = cache_path_for(video_path, 0, video_root, cache_root).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [i for i in indices if not (out_dir / f"{i:08d}.jpg").exists()]
    if not todo:
        return (video_path_str, 0, len(indices))

    import decord  # imported only inside worker

    decord.bridge.set_bridge("native")
    from decord import VideoReader, cpu

    reader = VideoReader(video_path_str, ctx=cpu(0))
    n_frames = len(reader)
    todo_clipped = sorted({min(i, n_frames - 1) for i in todo})

    batch = reader.get_batch(todo_clipped).asnumpy()  # (k, 224, 224, 3) uint8
    written = 0
    for resolved_idx, frame in zip(todo_clipped, batch, strict=True):
        Image.fromarray(frame).save(out_dir / f"{resolved_idx:08d}.jpg", quality=jpeg_quality)
        written += 1

    # Also create alias jpegs for indices that were clipped (idx >= n_frames)
    # so cache lookups for over-range requests succeed without a second decode.
    last = batch[-1]
    last_jpg = (out_dir / f"{n_frames - 1:08d}.jpg").read_bytes() if (out_dir / f"{n_frames - 1:08d}.jpg").exists() else None
    for i in todo:
        if i >= n_frames:
            target = out_dir / f"{i:08d}.jpg"
            if not target.exists():
                if last_jpg is not None:
                    target.write_bytes(last_jpg)
                else:
                    Image.fromarray(last).save(target, quality=jpeg_quality)
                written += 1

    return (video_path_str, written, len(indices))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir", type=Path, default=Path("data/segments"))
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("data/behavior/videos"),
    )
    parser.add_argument("--cache-root", type=Path, default=Path("data/frame_cache"))
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--limit-videos", type=int, default=0, help="for smoke testing only")
    args = parser.parse_args()

    args.cache_root.mkdir(parents=True, exist_ok=True)

    print(f"collecting jobs from {args.jsonl_dir}", flush=True)
    jobs = collect_jobs(args.jsonl_dir, action_horizon=args.action_horizon)
    total = sum(len(v) for v in jobs.values())
    print(f"-> {len(jobs)} unique videos, {total:,} frame indices to ensure cached", flush=True)

    items = list(jobs.items())
    if args.limit_videos > 0:
        items = items[: args.limit_videos]
        print(f"smoke test mode: limiting to first {len(items)} videos", flush=True)

    work = [
        (video, idxs, str(args.video_root), str(args.cache_root), args.jpeg_quality)
        for video, idxs in items
    ]

    written_total = 0
    skipped_total = 0
    ctx = mp.get_context("spawn")  # decord+spawn safe in this stand-alone script
    with ctx.Pool(args.workers) as pool:
        for i, (video, written, requested) in enumerate(pool.imap_unordered(_decode_one_video, work, chunksize=2), start=1):
            written_total += written
            if written == 0:
                skipped_total += 1
            if i % 50 == 0 or i == len(work):
                print(f"[{i:>4}/{len(work)}] wrote={written_total:,} skipped_videos={skipped_total} last={video}", flush=True)

    print(f"done. total_written={written_total:,} skipped_videos={skipped_total}", flush=True)


if __name__ == "__main__":
    main()
