# Policy training and serving

For RoboCasa training, serving and agent inference, use the dedicated
[RoboCasa guide](robocasa.md) and entrypoints in `scripts/robocasa/`.

## BEHAVIOR

Source videos default to `datasets/behavior/videos` and the optional packed-frame
cache defaults to `data/behavior/frame_cache` in this repository. To use another
location, set absolute `OPENPI_SKILL_SEGMENT_VIDEO_ROOT` and
`OPENPI_SKILL_SEGMENT_PACKED_CACHE_ROOT` paths in `.env`. A packed cache is optional;
the reader decodes source videos directly when no matching cache is available.

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

The BEHAVIOR recipe trains the shared vision-language backbone and all six routed
action experts jointly (`freeze_filter=nnx.Nothing()`) on eight GPUs with 8-way
FSDP: a batch of 480 split across experts in proportion to their segment counts,
50,000 steps, 1,000 warmup steps, a peak learning rate of 1e-4, and one global
action normalizer. Data paths must be configured for the target run. Keep expert
order and normalization identical in training and serving.

BEHAVIOR segment intervals are `[start_idx_30hz, end_idx_30hz)`. Short action
chunks repeat the final frame within their own skill segment; they never extend
into the next skill. Normalization statistics use the same padding rule.
