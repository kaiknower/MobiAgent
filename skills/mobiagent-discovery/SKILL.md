---
name: mobiagent-discovery
description: Discover manipulation skills from BEHAVIOR demonstrations and export aligned per-expert training data. Use for temporal annotation, semantic grouping, or converting existing predictions into training segments and sampler weights.
---

# MobiAgent Skill Discovery

Run commands from the repository root, two levels above this file.
Read [data preparation](../../docs/data.md) for dependencies.

## Choose the stage

For raw demonstrations, begin with skill discovery. If reviewed predictions already
exist and the request is to prepare training data, go directly to training-data
export without repeating model inference. Run both stages when the requested
outcome is training-ready data from raw demonstrations.

## Skill discovery

### Inputs

- BEHAVIOR dataset root containing `videos/`, `data/`, and `meta/`.
- Task IDs to process and an output root.
- Configured GPT API or Gemini credentials; see [API configuration](../../README.md#api-configuration).

### Workflow

1. Install the discovery dependencies (`pip install -e '.[discovery]'`) in the
   working environment and ensure `ffmpeg` is available. Inspect the selected
   tasks' video, state and episode metadata before requesting model inference.
2. Run the offline pipeline for the requested task IDs:

   ```bash
   mobiagent-discover --platform behavior --dataset-root /path/to/behavior \
     --tasks task-0001 task-0003 --output-root outputs/discovery/behavior
   ```

3. Read the returned `inference_status`, `selected_demo_count`, `run_dir`, and
   artifact paths. `skipped_no_api_key`, `completed_no_selected_demos`, and `failed`
   do not establish that skills were successfully inferred. Investigate the
   reported condition before retrying; keep successful artifacts from partial runs.
4. Review temporal boundaries against the video's time scale and source trajectory.
   Keep skill descriptions grounded in observed actions. Inspect the pipeline's
   clustering status separately from temporal inference status.

### Outputs

Use the paths returned by the pipeline; do not assume a `latest/` directory exists.
A run includes:

- `manifests/selected_demos.json`: selected episodes.
- `predictions/demo_skill_predictions.jsonl`: temporal skill predictions.
- `review/`: timeline and readable annotations.
- `final/demo_skills.jsonl`: skill records.
- `clusters/`: frequency and semantic grouping artifacts.

## Training-data export

Use the discovery predictions as the input for this stage. The discovery pipeline
does not create the task-instruction mapping expected by the segment builder;
supply that mapping separately.

### Inputs and data contract

- Prediction JSONL with `task_id`, `episode_id`, `video_context.time_scale`, and
  `skill_timeline` entries (`segment_id`, `skill_description`, `start_time_sec`,
  `end_time_sec`). Inspect the actual time scale; the builder's fallback is 5.
- Original BEHAVIOR dataset with `videos/`, `data/`, and `meta/`.
- Task-instruction JSON mapping task IDs to instructions.
- Destination directory for segments and expert shards.

The builder assumes 30 Hz action data and maps compressed video time to source
frames before clamping both endpoints to the parquet row count. Intervals use
an inclusive start and exclusive end. Missing sources, instructions and duplicate
sample IDs fail before any outputs are written. Resolve incompatible
sampling rates before running this workflow.

### Workflow

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

### Outputs

`segments.jsonl`, `head__*.jsonl`, `sampler_weights.json`, and `manifest.json`.
The expert order is `move_to`, `pick_up_from`, `place_in`, `place_on`, `open`, `close`.

For BEHAVIOR normalization and training, use
[MobiAgent Training](../mobiagent-training/SKILL.md). RoboCasa reads its six datasets
through its own [data recipe](../../configs/robocasa/data.example.json); do not feed
these BEHAVIOR shards to the RoboCasa reader.

## Implementation

- [Discovery pipeline](../../src/mobiagent/discovery/behavior/pipeline.py) and
  [CLI](../../src/mobiagent/discovery/__main__.py).
- [Segment builder](../../scripts/data/build_segment_samples.py) and
  [per-expert splitter](../../scripts/data/split_per_head.py).
