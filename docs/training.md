# Policy training and serving

The OpenPI-derived backend needs Linux, compatible NVIDIA GPUs, and its pinned
JAX/PyTorch dependencies. Its environment is separate from the lightweight agent.
The source dependency versions have been retained; they were not upgraded during
this extraction.

```bash
cd policy/openpi
uv sync
```

Set `MOBIAGENT_BASE_PARAMS`, `MOBIAGENT_SEGMENTS_DIR`, `MOBIAGENT_ASSETS_DIR`, and
`MOBIAGENT_CHECKPOINT_DIR` to your own paths. Paths are resolved relative to the
working directory unless absolute. The default base checkpoint URI is the
upstream OpenPI pi0.5 checkpoint; no trained weights are included.

## BEHAVIOR / S1

```bash
uv run scripts/compute_norm_stats.py --config-name mobiagent_behavior
uv run scripts/train.py mobiagent_behavior --exp-name my_run
# Use mobiagent_s1 for the three-expert S1 dataset.
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config mobiagent_behavior --policy.dir /path/to/checkpoint
```

The two portable baseline recipes freeze the shared vision-language backbone and
train routed action experts. They replace the historical numbered experiments;
batch sizes, schedules and data paths must be configured for the target run.

## RoboCasa

Copy the root `configs/robocasa_data.example.json` to a local recipe and fill in
the six skill datasets, normalization paths, and base model parameters. Export
`MOBIAGENT_ROBOCASA_RECIPE` as an absolute path to that file.

```bash
uv run scripts/train_robocasa.py --exp-name my_run
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config mobiagent_robocasa --policy.dir /path/to/checkpoint
```

`train_robocasa.py` retains the source six-GPU frozen-backbone schedule and
`train_robocasa_joint.py` retains the eight-GPU joint-training variant. They check
GPU availability before training. These entrypoints have not been executed as
part of repository preparation. They require external RoboCasa/Groot code,
datasets, normalization files and base weights. The example recipe contains no
training data or experimental results.

Keep per-expert normalization consistent between training and serving. Merely
changing the order of expert names without changing the routing and statistics
will produce incorrect actions.
