# Data preparation

For official RoboCasa downloads, normalization and training recipes, see
[RoboCasa training data](robocasa.md#training-data).

Use the BEHAVIOR skill-discovery pipeline to process demonstration videos and
construct per-expert training segments. Both stages are covered by the
[MobiAgent Discovery skill](../skills/mobiagent-discovery/SKILL.md).

```bash
pip install -e '.[discovery]'
# ffmpeg must also be installed and available on PATH.
mobiagent-discover --platform behavior --dataset-root /path/to/behavior \
  --tasks task-0001 task-0003 --output-root outputs/discovery/behavior \
  --export-segment-clips
```

The pipelines align video, robot state and actions, call a configured
multimodal provider, and generate skill descriptions and segment annotations.
They support the GPT API and Gemini; supply your own credentials. Inspect the
returned status: a run with no configured API credentials does not perform model
inference.

`--export-segment-clips` writes decodable skill videos under the run's `segments/`
directory. Playback timestamps are converted back to source frames using the
recorded `video_context.time_scale`; exported clips and training rows share the
same frame intervals. Without this flag, the pipeline exports annotations only.

Use the returned run directory to export training segments and expert shards:

```bash
DISCOVERY_RUN=/path/to/discovery/run_YYYYMMDD_HHMMSS
python scripts/data/build_segment_samples.py \
  --in "$DISCOVERY_RUN/predictions/demo_skill_predictions.jsonl" \
  --instructions-path "$DISCOVERY_RUN/manifests/task_instructions.json" \
  --dataset-root /path/to/behavior --out data/segments/segments.jsonl \
  --summary-out data/segments/per_task_summary.json \
  --short-csv-out data/segments/short_segments_diagnostic.csv --apply
python scripts/data/split_per_head.py \
  --in data/segments/segments.jsonl --out-dir data/segments --apply
```

Run these commands without `--apply` first to inspect the data reports.
Before training, compute
normalization statistics and keep the resulting assets next to the checkpoint.
