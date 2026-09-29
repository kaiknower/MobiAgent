"""Repack the per-frame JPEG cache into one file per source mp4.

The original ``build_skill_segment_frame_cache.py`` writes ~31M small JPEGs to
a shared filesystem. At training time, the dataset opens hundreds of
files per batch; on yrfs each open is a network round-trip, which produces
~30-50 sec stalls every ~2 min and roughly 3× the average step time.

This packer keeps the JPEGs (compression unchanged) but concatenates each
video's frames into a single file with a small index header. File count
drops from 31M → 3K, eliminating the metadata-bound stalls. The dataset's
``_seek_frame`` is updated in lockstep to read packed entries.

File format (little-endian, all int64):
    [ num_entries ]                        (8 B)
    [ frame_idx, byte_offset, byte_length ] × num_entries  (24 B each)
    [ concatenated JPEG bytes ]            (variable)

Lookup: read header, read full index into a dict, then per-frame seek+read.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import struct
from pathlib import Path

PACK_HEADER = struct.Struct("<Q")
PACK_ENTRY = struct.Struct("<qqq")
PACK_SUFFIX = ".pak"


def list_episode_dirs(cache_root: Path) -> list[Path]:
    """Each leaf episode dir holds the per-frame ``XXXXXXXX.jpg`` files."""
    out: list[Path] = []
    for task_dir in sorted(cache_root.glob("task-*")):
        for view_dir in sorted(task_dir.iterdir()):
            if not view_dir.is_dir():
                continue
            for episode_dir in sorted(view_dir.iterdir()):
                if episode_dir.is_dir():
                    out.append(episode_dir)
    return out


def packed_path_for(episode_dir: Path, cache_root: Path, packed_root: Path) -> Path:
    rel = episode_dir.relative_to(cache_root)
    return packed_root / rel.parent / f"{rel.name}{PACK_SUFFIX}"


def _pack_one_episode(args: tuple[str, str, str, bool]) -> tuple[str, int, int]:
    episode_dir_str, cache_root_str, packed_root_str, delete_jpgs = args
    episode_dir = Path(episode_dir_str)
    cache_root = Path(cache_root_str)
    packed_root = Path(packed_root_str)

    out_path = packed_path_for(episode_dir, cache_root, packed_root)
    if out_path.exists():
        return (episode_dir_str, 0, 0)  # already packed

    # Discover frame indices from filenames.
    jpg_files = sorted(episode_dir.glob("*.jpg"))
    if not jpg_files:
        return (episode_dir_str, 0, 0)

    entries: list[tuple[int, bytes]] = []
    for jpg in jpg_files:
        try:
            idx = int(jpg.stem)
        except ValueError:
            continue
        entries.append((idx, jpg.read_bytes()))
    entries.sort(key=lambda x: x[0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    n = len(entries)
    with tmp_path.open("wb") as f:
        f.write(PACK_HEADER.pack(n))
        # Compute offsets (relative to start of data section).
        cur = 0
        for idx, jpg in entries:
            f.write(PACK_ENTRY.pack(idx, cur, len(jpg)))
            cur += len(jpg)
        for _idx, jpg in entries:
            f.write(jpg)

    tmp_path.rename(out_path)

    written = out_path.stat().st_size
    if delete_jpgs:
        for jpg in jpg_files:
            try:
                jpg.unlink()
            except OSError:
                pass
        try:
            episode_dir.rmdir()
        except OSError:
            pass

    return (episode_dir_str, n, written)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/frame_cache"))
    parser.add_argument("--packed-root", type=Path, default=Path("data/frame_cache_packed"))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--delete-jpgs", action="store_true", help="remove per-frame JPEGs after each pack succeeds")
    args = parser.parse_args()

    args.packed_root.mkdir(parents=True, exist_ok=True)

    episodes = list_episode_dirs(args.cache_root)
    print(f"found {len(episodes)} episode dirs to pack", flush=True)

    work = [(str(d), str(args.cache_root), str(args.packed_root), args.delete_jpgs) for d in episodes]

    total_frames = 0
    total_bytes = 0
    skipped = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for i, (ep, frames, written) in enumerate(pool.imap_unordered(_pack_one_episode, work, chunksize=4), start=1):
            total_frames += frames
            total_bytes += written
            if frames == 0 and written == 0:
                skipped += 1
            if i % 100 == 0 or i == len(work):
                print(
                    f"[{i:>4}/{len(work)}] frames_packed={total_frames:,} "
                    f"bytes_written={total_bytes/1e9:.1f} GB skipped={skipped} last={ep}",
                    flush=True,
                )

    print(f"done. total_frames={total_frames:,} total_bytes={total_bytes/1e9:.1f} GB skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
