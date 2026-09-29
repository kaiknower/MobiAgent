# Data preparation

The repository includes processing code, not demonstrations or annotations.

```bash
pip install -e '.[discovery]'
# ffmpeg must also be installed and available on PATH.
mobiagent-discover --platform behavior --dataset-root /path/to/behavior \
  --tasks task-0001 task-0003 --output-root outputs/discovery/behavior
```

The extracted pipelines align video, robot state and actions, call a configured
multimodal provider, and generate skill descriptions and segment annotations.
They support Azure OpenAI and Gemini; supply your own credentials. Inspect the
returned status: a run with no configured API credentials does not perform model
inference.

Segment construction and splitting are under `scripts/data/`:

```bash
python scripts/data/build_segment_samples.py --help
python scripts/data/split_per_head.py --help
```

Use your own prediction and task-instruction files. No historical annotations,
rewritten object assignments, fixed segment boundaries, or per-task correction
scripts are included. Before training, compute normalization statistics and keep
the resulting assets next to the checkpoint; they are not committed to Git.
