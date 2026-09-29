# RoboCasa: inference and training

Run the commands below from the **repository root**. The agent and GPU policy
backend use separate Python environments. RoboCasa demonstrations, pretrained
parameters, normalization files and trained checkpoints are external assets.

## Code map

| Purpose | Code |
| --- | --- |
| Agent inference entrypoint | [`scripts/robocasa/infer.py`](../scripts/robocasa/infer.py) |
| Simulator observations and actions | [`src/mobiagent/environments/robocasa.py`](../src/mobiagent/environments/robocasa.py) |
| Planner, visual critic and control loop | [`src/mobiagent/execution/`](../src/mobiagent/execution/) |
| Policy serving entrypoint | [`scripts/robocasa/serve.py`](../scripts/robocasa/serve.py) |
| Joint training (trainable VLM) | [`scripts/robocasa/train.py`](../scripts/robocasa/train.py) |
| Dataset reader and normalization | [`policy/openpi/src/openpi/training/robocasa_data.py`](../policy/openpi/src/openpi/training/robocasa_data.py) |
| Shared VLM and action experts | [`policy/openpi/src/openpi/models/pi0_six_head.py`](../policy/openpi/src/openpi/models/pi0_six_head.py) |

## Environment setup

For the agent, use Python 3.11 or later and install the local packages into the
RoboCasa simulator's environment:

```bash
pip install -e .
pip install -e policy/openpi/packages/openpi-client
cp configs/robocasa/policy_servers.example.yaml configs/robocasa/policy_servers.yaml
```

Install the RoboCasa version that provides `robocasa/<Task>` Gym registrations
and `robocasa.utils.groot_utils` for your datasets. Simulator assets and that
external implementation are not bundled. Set `MUJOCO_GL` and
`MUJOCO_EGL_DEVICE_ID` for your renderer when needed.

For the GPU policy backend, use a separate Linux environment with compatible
NVIDIA GPUs:

```bash
uv sync --project policy/openpi
cp configs/robocasa/data.example.json configs/robocasa/data.json
export MOBIAGENT_ROBOCASA_RECIPE="$PWD/configs/robocasa/data.json"
export MOBIAGENT_CHECKPOINT_DIR="$PWD/checkpoints/robocasa"
```

Fill in the six datasets, base parameters and normalization paths in `data.json`.
Use absolute asset paths. Install the matching RoboCasa/Groot dataset reader in
the backend environment as well. Export the recipe variable in each backend shell.

## Inference

First start the policy server with a trained checkpoint:

```bash
uv run --project policy/openpi scripts/robocasa/serve.py \
  --checkpoint /path/to/checkpoint --port 8000
```

Set the server host and port in `configs/robocasa/policy_servers.yaml`. In the
agent environment, export `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and
`AZURE_OPENAI_DEPLOYMENT`, then launch the closed-loop agent:

```bash
python scripts/robocasa/infer.py \
  --task YOUR_ROBOCASA_TASK \
  --instruction 'YOUR TASK INSTRUCTION' \
  --max-episode-steps 3000 --max-ticks 200
```

This entrypoint selects RoboCasa by default and uses the shared planner,
policy execution and visual reflection loop. `--policy-servers` overrides the
configuration path; `--help` lists the remaining runtime options.

The adapter passes a 16-dimensional state and three cameras to the policy,
pads the state to the model's 32-dimensional input, and executes 12-dimensional
actions. Expert order is `close`, `open`, `switch`, `manipulate`, `navigate`, `pnp`.
Keep that order and normalization statistics identical in training and inference.
Simulator success is exposed through `episode_score()` for terminal evaluation;
it does not override the visual critic's decision.

## Training

The shared VLM and all six action experts are trained jointly. The VLM is not
frozen (`freeze_filter=nnx.Nothing()`). The training recipe requires eight GPUs.

```bash
uv run --project policy/openpi scripts/robocasa/train.py --exp-name my_run
```

The entrypoint checks GPU availability before training. Use
`--resume` to continue a run. The checkpoints include normalization assets;
serving also requires the recipe's normalization paths to remain available.
