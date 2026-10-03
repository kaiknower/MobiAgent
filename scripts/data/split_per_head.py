"""Split segments.jsonl into 6 per-canonical-head shards + sampler weights.

Filtering and output:
  - Drop n_frames < 5
  - Drop skill_canonical == "other" (filler noise)
  - One JSONL per head: head__{move_to,pick_up_from,place_in,place_on,open,close}.jsonl
  - sampler_weights.json keyed by sample_id, weight = (1 / class_count[head]) ** alpha
  - manifest.json with file paths, counts, sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("data/segments/segments.jsonl")
OUT_DIR = Path("data/segments")

CANONICAL_HEADS = ("move_to", "pick_up_from", "place_in", "place_on", "open", "close")
ALPHA = 0.5  # sampler weight exponent: 1/count^alpha


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    p.add_argument("--in", dest="in_path", default=str(INPUT_PATH))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--alpha", type=float, default=ALPHA)
    args = p.parse_args()
    if not math.isfinite(args.alpha) or args.alpha < 0:
        p.error("--alpha must be finite and nonnegative")

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)

    rows: list[dict] = []
    dropped_other = 0
    dropped_short = 0
    seen_ids: set[str] = set()
    for line in in_path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        sample_id = r.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen_ids:
            raise ValueError(f"Missing or duplicate sample_id: {sample_id!r}")
        seen_ids.add(sample_id)
        start, end, length = (r.get(key) for key in ("start_idx_30hz", "end_idx_30hz", "n_frames"))
        if any(type(value) is not int for value in (start, end, length)) or start < 0 or end <= start or length != end - start:
            raise ValueError(f"Invalid frame interval for {sample_id}")
        if r.get("skill_canonical") == "other":
            dropped_other += 1
            continue
        if r.get("n_frames", 0) < 5:
            dropped_short += 1
            continue
        if r.get("skill_canonical") not in CANONICAL_HEADS:
            print(f"  WARN: unknown canonical {r.get('skill_canonical')!r} for {r.get('sample_id')}; dropped")
            continue
        rows.append(r)

    print(f"=== input ===")
    print(f"  total in:        {sum(1 for _ in in_path.open() if _.strip())}")
    print(f"  dropped other:   {dropped_other}")
    print(f"  dropped short:   {dropped_short}")
    print(f"  retained:        {len(rows)}")
    print()

    # Group by canonical
    by_head: dict[str, list[dict]] = {h: [] for h in CANONICAL_HEADS}
    for r in rows:
        by_head[r["skill_canonical"]].append(r)

    # Frame stats per head
    print(f"=== per-head distribution + frame stats ===")
    for h in CANONICAL_HEADS:
        rs = by_head[h]
        nf = sorted(r["n_frames"] for r in rs)
        if not nf:
            print(f"  {h:<14} count=0")
            continue
        p50 = nf[len(nf)//2]
        p90 = nf[int(len(nf)*0.9)]
        print(f"  {h:<14} count={len(rs):<6} median_frames={p50:<5} p90={p90:<5}  total_frames={sum(nf)}")
    print()

    # Sampler weights: (1/count) ** alpha for each head, applied per row
    counts = {h: len(by_head[h]) for h in CANONICAL_HEADS}
    raw_weight_per_head = {h: (1.0 / counts[h]) ** args.alpha if counts[h] > 0 else 0.0 for h in CANONICAL_HEADS}
    # normalize so weights sum approximately to len(rows) (just so the magnitudes are sensible)
    total_raw = sum(raw_weight_per_head[r["skill_canonical"]] for r in rows)
    norm = len(rows) / total_raw if total_raw > 0 else 1.0
    sample_weights = {r["sample_id"]: round(raw_weight_per_head[r["skill_canonical"]] * norm, 6) for r in rows}

    print(f"=== sampler weights (alpha={args.alpha}) ===")
    for h in CANONICAL_HEADS:
        w_norm = raw_weight_per_head[h] * norm
        print(f"  {h:<14} count={counts[h]:<6} raw_weight={raw_weight_per_head[h]:.6f}  normalized_weight={w_norm:.4f}")

    if args.apply:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"alpha": args.alpha, "n_total": len(rows), "heads": {}}
        for h in CANONICAL_HEADS:
            shard = out_dir / f"head__{h}.jsonl"
            with shard.open("w") as fh:
                for r in by_head[h]:
                    fh.write(json.dumps(r) + "\n")
            manifest["heads"][h] = {
                "path": str(shard),
                "count": counts[h],
                "sha256": sha256(shard),
                "raw_sampler_weight_per_sample": raw_weight_per_head[h],
                "normalized_sampler_weight_per_sample": raw_weight_per_head[h] * norm,
            }
            print(f"  wrote {shard}  ({counts[h]} rows)")

        weights_path = out_dir / "sampler_weights.json"
        weights_path.write_text(json.dumps(sample_weights))
        manifest["sampler_weights_path"] = str(weights_path)
        manifest["sampler_weights_count"] = len(sample_weights)
        print(f"  wrote {weights_path}  ({len(sample_weights)} entries)")

        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  wrote {manifest_path}")
    else:
        print("\nDRY-RUN — pass --apply to write shards.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
