"""Build chunk-uniform sampler weights for the skill_segments dataset.

The default sampler_weights.json gives every segment within a head equal
sampling probability. Combined with `_pick_start`'s uniform-random offset
within a segment, that produces *segment-uniform* sampling: each segment is
visited equally often, but a long segment (171 chunks) and a short segment
(1 chunk) are both picked the same number of times — so each chunk inside
a long segment is undersampled, while a short segment's single chunk gets
hammered K times.

This script produces *chunk-uniform within head* weights:

    weight(s) ∝ max(1, n_frames(s) - action_horizon + 1)
              = number of valid 30-frame windows you can extract from segment s

When the stratified sampler picks segments from head h's pool with these
weights, P(segment s) ∝ n_chunks(s); combined with `_pick_start`'s uniform
offset (P(chunk | segment) = 1/n_chunks(s)), we get
P(any chunk c in head h) = constant. That gives every chunk position in
the head equal expected sampling rate over training.

Output JSON keys = sample_id, values = float weights. Cross-head normalization
does not matter (the stratified sampler operates within each head pool
independently), so we keep weights in raw "n_chunks" units for easy
inspection.

Usage:

    python scripts/build_chunk_uniform_sampler_weights.py \
        --segments /path/to/skill_segments/segments.jsonl \
        --action-horizon 30 \
        --output /path/to/skill_segments/sampler_weights_chunk_uniform.json

Then point `SkillSegmentsDataConfig.sampler_weights_path` at the output file.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segments", type=pathlib.Path, required=True, help="Path to segments.jsonl")
    p.add_argument("--action-horizon", type=int, default=30, help="Action chunk length (default 30)")
    p.add_argument("--output", type=pathlib.Path, required=True, help="Output JSON path")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.segments.is_file():
        print(f"ERROR: --segments {args.segments} does not exist", file=sys.stderr)
        return 1

    H = int(args.action_horizon)
    if H < 1:
        print("ERROR: --action-horizon must be >= 1", file=sys.stderr)
        return 1

    weights: dict[str, float] = {}
    per_head_count: dict[str, int] = collections.Counter()
    per_head_chunks: dict[str, int] = collections.Counter()
    per_head_seg_lens: dict[str, list[int]] = collections.defaultdict(list)

    with args.segments.open() as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"WARN: line {lineno}: bad JSON ({e}); skipping", file=sys.stderr)
                continue

            sample_id = row.get("sample_id")
            head = row.get("skill_canonical")
            if sample_id is None or head is None:
                print(f"WARN: line {lineno}: missing sample_id or skill_canonical; skipping", file=sys.stderr)
                continue

            # Prefer explicit n_frames; fall back to (end - start).
            n_frames = row.get("n_frames")
            if n_frames is None:
                n_frames = int(row.get("end_idx_30hz", 0)) - int(row.get("start_idx_30hz", 0))
            n_frames = int(n_frames)

            n_chunks = max(1, n_frames - H + 1)
            weights[sample_id] = float(n_chunks)

            per_head_count[head] += 1
            per_head_chunks[head] += n_chunks
            per_head_seg_lens[head].append(n_frames)

    # Sort heads in canonical expert order.
    canonical_order = ["move_to", "pick_up_from", "place_in", "place_on", "open", "close"]
    print(f"\n{'head':>14s} {'segments':>10s} {'total chunks':>13s} {'avg seg_len':>12s} {'min':>5s} {'max':>5s}")
    print("-" * 70)
    for head in sorted(per_head_count, key=lambda h: canonical_order.index(h) if h in canonical_order else 99):
        n = per_head_count[head]
        chunks = per_head_chunks[head]
        lens = per_head_seg_lens[head]
        avg_len = sum(lens) / len(lens) if lens else 0
        print(f"{head:>14s} {n:>10d} {chunks:>13d} {avg_len:>12.1f} {min(lens):>5d} {max(lens):>5d}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(weights, f, separators=(",", ":"))

    print(f"\nWrote {len(weights)} weights to {args.output}")
    print(f"  total weight sum: {sum(weights.values()):.0f}  (= total chunks across dataset)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
