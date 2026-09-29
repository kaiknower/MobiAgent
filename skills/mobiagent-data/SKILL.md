---
name: mobiagent-data
description: Convert BEHAVIOR temporal skill predictions into aligned training segments and per-expert shards with sampler weights. Use after offline skill discovery and before BEHAVIOR policy training.
---

# MobiAgent Data Preparation

Run commands from the repository root, two levels above this file.

## Inputs and data contract

- Prediction JSONL with `task_id`, `episode_id`, `video_context.time_scale`, and
  `skill_timeline` entries (`segment_id`, `skill_description`, `start_time_sec`,
  `end_time_sec`). Inspect the actual time scale; the builder's fallback is 5.
- Original BEHAVIOR dataset with `videos/`, `data/`, and `meta/`.
- Task-instruction JSON mapping task IDs to instructions.
- Destination directory for segments and expert shards.

The builder assumes 30 Hz action data and maps compressed video time to source
frames before clamping segment ends to the parquet row count. Resolve incompatible
sampling rates before running this workflow.

## Workflow

1. Run the segment builder without `--apply` to inspect alignment, canonical skill
   distribution and missing files:

   ```bash
   python scripts/data/build_segment_samples.py \
     --in /path/to/demo_skill_predictions.jsonl \
     --dataset-root /path/to/behavior \
     --instructions-path /path/to/task_instructions.json \
     --out data/segments/segments.jsonl \
     --summary-out data/segments/per_task_summary.json \
     --short-csv-out data/segments/short_segments_diagnostic.csv
   ```

2. Resolve missing source paths or mismatched timestamps, then repeat the command
   with `--apply` to write the outputs. Use the requested destination and avoid
   overwriting a different dataset's shards.
3. Split the segment file; inspect the report before adding `--apply`:

   ```bash
   python scripts/data/split_per_head.py \
     --in data/segments/segments.jsonl --out-dir data/segments --alpha 0.5
   ```

4. Verify the output manifest counts and expert coverage. The splitter filters
   segments shorter than five frames and unknown canonical skills. Report empty
   expert shards before training; do not manufacture annotations to fill them.

## Outputs

`segments.jsonl`, `head__*.jsonl`, `sampler_weights.json`, and `manifest.json`.
The expert order is `move_to`, `pick_up_from`, `place_in`, `place_on`, `open`, `close`.

For BEHAVIOR normalization and training, use
[MobiAgent Training](../mobiagent-training/SKILL.md). RoboCasa reads its six datasets
through its own [data recipe](../../configs/robocasa/data.example.json); do not feed
these BEHAVIOR shards to the RoboCasa reader.

## Implementation

[Segment builder](../../scripts/data/build_segment_samples.py) and
[per-expert splitter](../../scripts/data/split_per_head.py).
