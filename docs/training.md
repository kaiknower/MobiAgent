# Policy training and serving

For RoboCasa training, serving and agent inference, use the dedicated
[RoboCasa guide](robocasa.md) and entrypoints in `scripts/robocasa/`.

## BEHAVIOR

The OpenPI-derived backend needs Linux, compatible NVIDIA GPUs, and the source
JAX/PyTorch dependencies. Its environment is separate from the lightweight agent.

```bash
cd policy/openpi
uv sync
```

Set `MOBIAGENT_BASE_PARAMS`, `MOBIAGENT_SEGMENTS_DIR`, `MOBIAGENT_ASSETS_DIR`, and
`MOBIAGENT_CHECKPOINT_DIR` to your own absolute paths. The default base checkpoint
URI is the upstream OpenPI pi0.5 checkpoint; no trained weights are included.

```bash
uv run scripts/compute_norm_stats.py --config-name mobiagent_behavior
uv run scripts/train.py mobiagent_behavior --exp-name my_run
uv run scripts/serve_policy.py --config mobiagent_behavior \
  --checkpoint /path/to/checkpoint --port 8000
```

The portable recipe freezes the shared vision-language backbone and trains routed
action experts. Batch sizes, schedules and data paths must be configured for the
target run. Keep expert order and normalization identical in training and serving.
