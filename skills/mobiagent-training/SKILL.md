---
name: mobiagent-training
description: Train or resume MobiAgent policies on RoboCasa or BEHAVIOR with the shared OpenPI backend. Use for configuring datasets, normalization and checkpoints and launching the benchmark-specific training recipe.
---

# MobiAgent Policy Training

Run commands from the repository root, two levels above this file. Use the
OpenPI backend environment (`uv sync --project policy/openpi`). Read only the
selected benchmark's guide: [RoboCasa](../../docs/robocasa.md#training) or
[BEHAVIOR](../../docs/training.md).

## Inputs

Benchmark, dataset paths, base parameters, normalization assets, experiment name,
checkpoint destination, and whether this is a new or resumed run. Use the requested
compute environment; do not provision another host or launch additional sweeps.

## RoboCasa

1. Download the official demonstrations in the RoboCasa environment and generate
   normalization assets and the data recipe in the backend environment:

   ```bash
   python scripts/robocasa/download_data.py --dataset-root datasets/robocasa
   uv run --project policy/openpi scripts/robocasa/prepare_data.py \
     --manifest datasets/robocasa/datasets.json
   ```

   Read [training data setup](../../docs/robocasa.md#training-data) for custom
   task selections. Preserve `close`, `open`, `switch`, `manipulate`, `navigate`,
   `pnp` order. For existing prepared data, use the supplied recipe.
2. Validate the recipe, then run the joint trainer:

   ```bash
   uv run --project policy/openpi scripts/robocasa/train.py \
     --recipe configs/robocasa/data.json --check-data
   export MOBIAGENT_CHECKPOINT_DIR="$PWD/checkpoints/robocasa"
   uv run --project policy/openpi scripts/robocasa/train.py \
     --recipe configs/robocasa/data.json --exp-name my_run
   ```

3. This recipe trains the shared VLM and all six action experts together:
   `freeze_filter=nnx.Nothing()`. Keep the VLM trainable. It requires eight GPUs
   and checks their availability and checkpoint disk space before training.
   If those checks fail, report the resource requirement instead of changing to
   a frozen-backbone recipe or taking devices used by another job.
4. Resume the same experiment with `--resume`. `--stop-after` is an absolute end
   step (2–45000), not an additional step count. A two-step check still requires
   the full training hardware and base weights.

## BEHAVIOR

1. Set absolute `MOBIAGENT_SEGMENTS_DIR`, `MOBIAGENT_BASE_PARAMS`,
   `MOBIAGENT_ASSETS_DIR`, and `MOBIAGENT_CHECKPOINT_DIR`. Check shard coverage and
   expert order using the data manifest.
2. Compute normalization statistics, then train:

   ```bash
   uv run --project policy/openpi policy/openpi/scripts/compute_norm_stats.py \
     --config-name mobiagent_behavior
   uv run --project policy/openpi policy/openpi/scripts/train.py \
     mobiagent_behavior --exp-name my_run
   ```

3. The BEHAVIOR configuration freezes its VLM backbone and trains routed experts.
   This is distinct from RoboCasa's joint-training recipe. Resume the same
   BEHAVIOR experiment by adding `--resume`.

## Completion and handoff

Report the actual checkpoint directory, last completed step and any failure.
Preserve the checkpoint's normalization assets and training configuration.
A launched process or an initialized directory is not a completed training run.
On failure, use the error to determine whether resuming is valid; do not overwrite
an existing experiment or repeatedly restart a failing allocation.

For policy serving and task execution, use
[MobiAgent Execution](../mobiagent-execution/SKILL.md), passing the checkpoint,
benchmark, normalization recipe and server address.

## Implementation

[RoboCasa joint trainer](../../scripts/robocasa/train.py),
[shared training runner](../../policy/openpi/src/openpi/training/runner.py), and
[BEHAVIOR configuration](../../policy/openpi/src/openpi/training/config.py).
