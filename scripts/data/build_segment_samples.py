"""Build per-segment training samples from normalized predictions.

Input:  predictions.jsonl
Output: segments.jsonl  — one row per segment with all info training needs:
        sample_id, task_id, episode_id, segment_id, task_instruction,
        skill_canonical, skill_description, start_idx_30hz, end_idx_30hz,
        n_frames, head/left/right video paths, parquet path, meta path.

Temporal alignment:
  - 30 Hz action data (verified via parquet num_rows == meta length)
  - source_time_sec = compressed_time * time_scale (default 5.0)
  - idx = round(source_time_sec * 30)
  - clamp both endpoints to [0, parquet num_rows]; intervals are [start, end)
  - n_frames < 5 segments are flagged in a CSV diagnostic but kept in the JSONL
    (the per-expert splitting stage filters these segments)
  - skill_canonical derived from verb prefix (deterministic, 6 classes):
      'place X in Y'   -> place_in
      'place X on Y'   -> place_on
      'pick up ...'    -> pick_up_from
      'move to ...'    -> move_to
      'open X'         -> open
      'close X'        -> close
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

# Default paths are relative to the repository. Override via --in / --out
# / --dataset-root / --instructions-path CLI flags or env vars.
INPUT_PATH = Path(os.getenv("MOBIAGENT_PREDICTIONS",
    "data/segments/predictions.jsonl"))
OUTPUT_PATH = Path(os.getenv("CLAW_SEGMENTS_OUT",
    "data/segments/segments.jsonl"))
SUMMARY_PATH = Path(os.getenv("CLAW_PER_TASK_SUMMARY",
    "data/segments/per_task_summary.json"))
SHORT_CSV = Path(os.getenv("CLAW_SHORT_DIAG",
    "data/segments/short_segments_diagnostic.csv"))

DATASET_ROOT = Path(os.getenv("CLAW_DATASET_ROOT", "data/behavior"))
INSTRUCTIONS_PATH = Path(os.getenv("CLAW_TASK_INSTRUCTIONS",
    "configs/task_instructions.json"))
FPS = 30


def derive_canonical(desc: str) -> str:
    low = desc.lower().strip()
    if (low.startswith("place ") or low.startswith("put ")) and " on " in low:
        return "place_on"
    if low.startswith("place ") or low.startswith("put "):
        return "place_in"
    if low.startswith("move to") or low.startswith("move "):
        return "move_to"
    if low.startswith("pick up"):
        return "pick_up_from"
    if low.startswith("open "):
        return "open"
    if low.startswith("close "):
        return "close"
    return "other"  # noise filler ('manipulation', 'inspect X', etc.); excluded during per-expert splitting


def paths_for(task_id: str, episode_id: str) -> dict[str, Path]:
    return {
        "head_video":  DATASET_ROOT / "videos" / task_id / "observation.images.rgb.head" / f"{episode_id}.mp4",
        "left_video":  DATASET_ROOT / "videos" / task_id / "observation.images.rgb.left_wrist" / f"{episode_id}.mp4",
        "right_video": DATASET_ROOT / "videos" / task_id / "observation.images.rgb.right_wrist" / f"{episode_id}.mp4",
        "parquet":     DATASET_ROOT / "data" / task_id / f"{episode_id}.parquet",
        "meta":        DATASET_ROOT / "meta" / "episodes" / task_id / f"{episode_id}.json",
    }


def parquet_rows_cached(parquet: Path, cache: dict[Path, int]) -> int:
    if parquet not in cache:
        cache[parquet] = pq.read_metadata(parquet).num_rows
    return cache[parquet]


def main() -> int:
    global DATASET_ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="write outputs (default dry-run)")
    p.add_argument("--in", dest="in_path", default=str(INPUT_PATH),
                   help="predictions.jsonl input")
    p.add_argument("--out", dest="out_path", default=str(OUTPUT_PATH),
                   help="segments.jsonl output")
    p.add_argument("--dataset-root", default=str(DATASET_ROOT),
                   help="root of behavior_224_rgb-style data on this machine "
                        "(must contain videos/ data/ meta/ subdirs); env: CLAW_DATASET_ROOT")
    p.add_argument("--instructions-path", default=str(INSTRUCTIONS_PATH),
                   help="task_instructions.json path; env: CLAW_TASK_INSTRUCTIONS")
    p.add_argument("--summary-out", default=str(SUMMARY_PATH),
                   help="per_task_summary.json output")
    p.add_argument("--short-csv-out", default=str(SHORT_CSV),
                   help="short_segments_diagnostic.csv output")
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    dataset_root = Path(args.dataset_root)
    instructions_path = Path(args.instructions_path)
    summary_path = Path(args.summary_out)
    short_csv_path = Path(args.short_csv_out)

    if not instructions_path.exists():
        print(f"missing {instructions_path} — pass --instructions-path or set CLAW_TASK_INSTRUCTIONS",
              file=sys.stderr)
        return 1
    if not dataset_root.exists():
        print(f"missing dataset root {dataset_root} — pass --dataset-root or set CLAW_DATASET_ROOT",
              file=sys.stderr)
        return 1
    instructions = json.loads(instructions_path.read_text())

    # Override module-global so paths_for() picks up the new root
    DATASET_ROOT = dataset_root

    # one parquet sanity check per task (compare num_rows against video frame estimate)
    print("=== fps + parquet alignment sanity check (one episode per task) ===")
    seen_tasks: set[str] = set()
    parquet_rows_cache: dict[Path, int] = {}
    sample_segments_per_task: dict[str, dict] = {}

    rows: list[dict] = []
    short_rows: list[dict] = []
    per_task_counter: dict[str, dict] = {}
    missing_paths: list[str] = []
    canonical_counter: dict[str, int] = {}
    sample_ids: set[str] = set()
    validation_errors: list[str] = []

    for line in in_path.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        task_id = rec["task_id"]
        episode_id = rec["episode_id"]
        time_scale = float(rec.get("video_context", {}).get("time_scale", 5.0))
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError(f"{task_id}/{episode_id}: time_scale must be finite and positive")
        instruction = instructions.get(task_id, "")
        if not isinstance(instruction, str) or not instruction.strip():
            validation_errors.append(f"Missing task instruction for {task_id}")

        paths = paths_for(task_id, episode_id)
        missing = False
        for k, pth in paths.items():
            if not pth.is_file():
                missing_paths.append(f"{task_id}/{episode_id} :: {k} -> {pth}")
                missing = True
        if missing:
            continue
        try:
            n_rows = parquet_rows_cached(paths["parquet"], parquet_rows_cache)
            if n_rows <= 0:
                raise ValueError("empty parquet")
            meta = json.loads(paths["meta"].read_text())
            if "length" in meta and int(meta["length"]) != n_rows:
                raise ValueError(f"meta length {meta['length']} != parquet rows {n_rows}")
        except Exception as exc:
            validation_errors.append(f"{task_id}/{episode_id}: {exc}")
            continue

        # Sanity: one parquet per task, log num_rows
        if task_id not in seen_tasks and paths["parquet"].exists():
            try:
                n_rows = parquet_rows_cached(paths["parquet"], parquet_rows_cache)
                src_dur = rec.get("video_context", {}).get("source_duration_sec", n_rows / FPS)
                expected = int(src_dur * FPS)
                ok = abs(n_rows - expected) <= 2
                print(f"  {task_id}/{episode_id}: parquet={n_rows} rows, expected~{expected} ({'✓' if ok else '✗ mismatch'})")
                seen_tasks.add(task_id)
            except Exception as exc:
                print(f"  {task_id}/{episode_id}: parquet read fail: {exc}")

        # Process segments
        ts = per_task_counter.setdefault(task_id, {"total": 0, "by_canonical": {}, "n_frames": []})
        for seg in rec.get("skill_timeline", []):
            desc = seg.get("skill_description", "")
            if not isinstance(desc, str) or not desc.strip():
                continue
            if seg.get("segment_id") is None or not str(seg["segment_id"]).strip():
                validation_errors.append(f"{task_id}/{episode_id}: missing segment_id")
                continue
            seg_id = str(seg["segment_id"])
            try:
                s_compressed = float(seg["start_time_sec"])
                e_compressed = float(seg["end_time_sec"])
            except (KeyError, TypeError, ValueError):
                validation_errors.append(f"{task_id}/{episode_id}/{seg_id}: invalid timestamps")
                continue
            if not all(math.isfinite(x) for x in (s_compressed, e_compressed)) or e_compressed <= s_compressed:
                validation_errors.append(f"{task_id}/{episode_id}/{seg_id}: invalid time interval")
                continue
            s_source = s_compressed * time_scale
            e_source = e_compressed * time_scale
            s_idx = max(0, min(int(round(s_source * FPS)), n_rows))
            e_idx = max(0, min(int(round(e_source * FPS)), n_rows))

            if e_idx <= s_idx:
                continue
            n_frames = e_idx - s_idx
            canonical = derive_canonical(desc)
            sample_id = f"{task_id}/{episode_id}/{seg_id}"
            if sample_id in sample_ids:
                validation_errors.append(f"Duplicate sample_id: {sample_id}")
                continue
            sample_ids.add(sample_id)

            row = {
                "sample_id": sample_id,
                "task_id": task_id,
                "episode_id": episode_id,
                "segment_id": seg_id,
                "task_instruction": instruction,
                "skill_canonical": canonical,
                "skill_description": desc,
                "start_time_sec_source": round(s_idx / FPS, 3),
                "end_time_sec_source": round(e_idx / FPS, 3),
                "start_idx_30hz": s_idx,
                "end_idx_30hz": e_idx,
                "n_frames": n_frames,
                "head_video":  str(paths["head_video"]),
                "left_video":  str(paths["left_video"]),
                "right_video": str(paths["right_video"]),
                "parquet":     str(paths["parquet"]),
                "meta":        str(paths["meta"]),
            }
            rows.append(row)

            # bookkeeping
            ts["total"] += 1
            ts["by_canonical"][canonical] = ts["by_canonical"].get(canonical, 0) + 1
            ts["n_frames"].append(n_frames)
            canonical_counter[canonical] = canonical_counter.get(canonical, 0) + 1

            if n_frames < 5:
                short_rows.append(row)

    # report
    print()
    print(f"=== build report ===")
    print(f"  total segments:     {len(rows)}")
    print(f"  short (<5 frames):  {len(short_rows)}  ({100*len(short_rows)/max(1,len(rows)):.2f}%)")
    print(f"  missing paths:      {len(missing_paths)}")
    if missing_paths[:3]:
        for m in missing_paths[:3]:
            print(f"    - {m}")
    print()
    print(f"=== canonical distribution ===")
    for c in ("move_to", "pick_up_from", "place_in", "place_on", "open", "close", "other"):
        print(f"  {c:<14} {canonical_counter.get(c, 0)}")
    print()
    print(f"=== per-task ===")
    for tid in sorted(per_task_counter):
        ts = per_task_counter[tid]
        nf = ts["n_frames"]
        if not nf:
            print(f"  {tid}: empty")
            continue
        nf_sorted = sorted(nf)
        p50 = nf_sorted[len(nf_sorted)//2]
        p90 = nf_sorted[int(len(nf_sorted)*0.9)]
        print(f"  {tid}: total={ts['total']:<5}  median_frames={p50:<4} p90={p90:<4}  by_canonical={ts['by_canonical']}")

    if missing_paths or validation_errors:
        for error in validation_errors[:10]:
            print(error, file=sys.stderr)
        print("Invalid source data; no outputs written.", file=sys.stderr)
        return 1
    if not rows:
        print("No nonempty segments; no outputs written.", file=sys.stderr)
        return 1
    if args.apply:
        for path in (out_path, summary_path, short_csv_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"\nWROTE {out_path}  ({len(rows)} rows)")

        summary_path.write_text(json.dumps({
            "n_segments": len(rows),
            "n_short": len(short_rows),
            "canonical_counts": canonical_counter,
            "per_task": {tid: {"total": ts["total"], "by_canonical": ts["by_canonical"]} for tid, ts in per_task_counter.items()},
            "missing_paths": missing_paths[:50],
        }, indent=2))
        print(f"WROTE {summary_path}")

        if short_rows:
            with short_csv_path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["sample_id", "n_frames", "skill_canonical", "skill_description"])
                w.writeheader()
                for r in short_rows:
                    w.writerow({k: r[k] for k in w.fieldnames})
            print(f"WROTE {short_csv_path}  ({len(short_rows)} rows)")
    else:
        print("\nDRY-RUN — pass --apply to write outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
