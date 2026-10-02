# RoboCasa: data, training and inference

Run commands from the **repository root**. Use a RoboCasa environment for
simulation and official data downloads, and a separate Linux GPU environment
for the OpenPI policy backend.

## Code map

| Purpose | Code |
| --- | --- |
| Official dataset download | [`scripts/robocasa/download_data.py`](../scripts/robocasa/download_data.py) |
| Task-to-expert mapping | [`configs/robocasa/datasets.example.json`](../configs/robocasa/datasets.example.json) |
| Normalization and training recipe | [`scripts/robocasa/prepare_data.py`](../scripts/robocasa/prepare_data.py) |
| LeRobot episode reader | [`policy/openpi/src/openpi/training/robocasa_lerobot.py`](../policy/openpi/src/openpi/training/robocasa_lerobot.py) |
| Joint training and sampling | [`scripts/robocasa/train.py`](../scripts/robocasa/train.py) · [`robocasa_data.py`](../policy/openpi/src/openpi/training/robocasa_data.py) |
| Simulator adapter | [`src/mobiagent/environments/robocasa.py`](../src/mobiagent/environments/robocasa.py) |
| Inference and policy serving | [`scripts/robocasa/infer.py`](../scripts/robocasa/infer.py) · [`scripts/robocasa/serve.py`](../scripts/robocasa/serve.py) |

## Environment setup

Install RoboCasa using its [official installation guide](https://robocasa.ai/docs/build/html/introduction/installation.html).
The downloader targets the RoboCasa 1.0 registry and LeRobot v2 dataset format;
the reference upstream revision is `456174f62b89b8fca99eaaf33949c29fec9cfc2a`.
Install MobiAgent and the policy client in the simulator environment:

```bash
pip install -e .
pip install -e policy/openpi/packages/openpi-client
cp configs/robocasa/policy_servers.example.yaml configs/robocasa/policy_servers.yaml
```

For the GPU backend, install the locked JAX/PyTorch dependencies:

```bash
uv sync --project policy/openpi
```

The backend reads Parquet episodes and video directly through the local LeRobot
reader. Simulator assets are required for execution. Set `MUJOCO_GL` and
`MUJOCO_EGL_DEVICE_ID` for the renderer as needed.

## Training data

Use the [official RoboCasa datasets](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html).
The default selection uses human pretraining demonstrations and assigns official
atomic tasks to six experts. Edit a copy of the selection to change the task groups,
source (`human` or `mimicgen`) or split (`pretrain` or `target`). Every expert must
have data; a task belongs to one expert.

In the RoboCasa environment, preview the selected paths, then download:

```bash
python scripts/robocasa/download_data.py \
  --dataset-root datasets/robocasa --dry-run
python scripts/robocasa/download_data.py \
  --dataset-root datasets/robocasa
```

The script calls RoboCasa's official downloader and preserves its dataset layout:

```text
datasets/robocasa/
  datasets.json              Task names, expert assignments and absolute paths
  v1.0/pretrain/atomic/
    <Task>/<Version>/lerobot/
      meta/                 Episode, task and modality metadata
      data/                 Parquet state and action trajectories
      videos/               Three camera streams
```

Choose another `--dataset-root` for a larger filesystem. Use
`--selection configs/robocasa/datasets.json` for a custom selection. Existing
complete datasets are reused. The manifest is written only after all selected
datasets have their metadata, Parquet data and videos.

## Prepare the training recipe

In the GPU backend environment, compute per-expert statistics from the selected
demonstrations and write the trainer's configuration:

```bash
uv run --project policy/openpi scripts/robocasa/prepare_data.py \
  --manifest datasets/robocasa/datasets.json \
  --assets-dir assets/robocasa \
  --output configs/robocasa/data.json
```

The script reads each episode, extracts the 16-dimensional state and
12-dimensional action through `meta/modality.json`, pads both to 32 dimensions,
and computes mean, standard deviation and quantiles for each expert. It validates
camera-file coverage. Outputs are:

- `assets/robocasa/<expert>/norm_stats.json`: normalization for training and serving.
- `configs/robocasa/data.json`: dataset paths, normalization paths, base parameters
  and batch settings consumed by `train.py`.

The default base parameters are the upstream OpenPI pi0.5 checkpoint,
`gs://openpi-assets/checkpoints/pi05_base/params`, downloaded by the weight loader
when training starts. To use a local copy, pass
`--base-params /absolute/path/to/pi05_base/params`. Generated configuration paths
are absolute. Use new `--output` and `--assets-dir` locations for another recipe.

## Training

Validate one sample from each expert before starting training:

```bash
uv run --project policy/openpi scripts/robocasa/train.py \
  --recipe configs/robocasa/data.json --check-data
```

This checks metadata, RGB loading, action chunks and normalization, then exits
before allocating the training mesh. The joint trainer updates the shared VLM
and all six action experts (`freeze_filter=nnx.Nothing()`). It requires eight
available GPUs with at least 70 GiB free memory each.

```bash
export MOBIAGENT_CHECKPOINT_DIR="$PWD/checkpoints/robocasa"
uv run --project policy/openpi scripts/robocasa/train.py \
  --recipe configs/robocasa/data.json --exp-name my_run
```

Checkpoints are saved under
`checkpoints/robocasa/pi05_robocasa_shared_vlm_v3/my_run/<step>/` and include
per-expert normalization assets. Keep expert order `close`, `open`, `switch`,
`manipulate`, `navigate`, `pnp` identical in training and inference. Resume the
same experiment by adding `--resume`. `--stop-after` sets the absolute final
training step, up to 45,000.

## Download weights

The [MobiAgent checkpoint](https://huggingface.co/Liukaikai/MobiAgent) contains
the jointly trained VLM and six action experts at step 25,000. Request access,
wait for approval, then authenticate and download:

```bash
hf auth login
hf download Liukaikai/MobiAgent \
  --include 'params/**' --include 'assets/**' \
  --include '_CHECKPOINT_METADATA' --include 'checkpoint_info.json' \
  --local-dir checkpoints/robocasa-joint-25000
```

For the complete checkpoint, including training state, omit the `--include`
filters. If you have not prepared a training recipe, create one from the template:

```bash
cp configs/robocasa/data.example.json configs/robocasa/data.json
```

For this checkpoint, set each `norm_path` in `configs/robocasa/data.json`
to the absolute downloaded `assets/per_expert/<expert>/norm_stats.json` path.
Preserve expert order and the recipe's `use_quantile_norm` setting. Policy serving
uses the normalization settings; dataset paths are read when preparing or training.

## Inference

Export the recipe for the policy server:

```bash
export MOBIAGENT_ROBOCASA_RECIPE="$PWD/configs/robocasa/data.json"
uv run --project policy/openpi scripts/robocasa/serve.py \
  --checkpoint /absolute/path/to/checkpoint --port 8000
```

Set the server host and port in `configs/robocasa/policy_servers.yaml`.
Configure the planner and critic's Azure settings using
[API configuration](../README.md#api-configuration), then run in the simulator
environment:

```bash
python scripts/robocasa/infer.py \
  --task YOUR_ROBOCASA_TASK \
  --instruction 'YOUR TASK INSTRUCTION' \
  --max-episode-steps 3000 --max-ticks 200
```

The adapter supplies a 16-dimensional state and three cameras, pads state to
32 dimensions and executes 12-dimensional simulator actions. The policy predicts
50-step action chunks. `--policy-servers` overrides the server configuration;
`--help` lists runtime options.
