<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/mobiagent-logo-dark.svg">
  <img src="docs/assets/mobiagent-logo.svg" alt="MobiAgent" width="520">
</picture>

## MobiAgent: Dual-Loop Recursive Policy Self-Improvement for Long-Horizon Mobile Manipulation

[Chenzhi Liu](https://openreview.net/profile?id=~Chenzhi_Liu2)<sup>1,*</sup> · [Zhang Yue](https://openreview.net/profile?id=~Zhang_Yue_bolt1)<sup>1,*</sup> · [Jiehong Lin](https://openreview.net/profile?id=~Jiehong_Lin1)<sup>1,*,†</sup> · [Jianan Wang](https://openreview.net/profile?id=~Jianan_Wang2)<sup>2</sup><br>
[Bo Wang](https://openreview.net/profile?id=~Bo_Wang36)<sup>1</sup> · [Zhongrui Wang](https://openreview.net/profile?id=~Zhongrui_Wang1)<sup>3,‡</sup> · [Xiaojuan Qi](https://openreview.net/profile?id=~Xiaojuan_Qi4)<sup>1,‡</sup>

<sup>1</sup> The University of Hong Kong &nbsp; <sup>2</sup> Astribot &nbsp; <sup>3</sup> Southern University of Science and Technology<br>
<sup>*</sup> Equal contribution &nbsp; <sup>†</sup> Project leader &nbsp; <sup>‡</sup> Corresponding authors

[![Project Overview](https://img.shields.io/badge/Project-Overview-3366cc?style=plastic&logo=googlechrome&logoColor=white)](https://kaiknower.github.io/mobiagent/)
[![Paper: Coming soon](https://img.shields.io/badge/Paper-Coming_soon-b31b1b?style=plastic&logo=arxiv&logoColor=white)](#paper)
[![RoboCasa Training Dataset](https://img.shields.io/badge/Training_Data-RoboCasa-ffcc4d?style=plastic&logo=huggingface)](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html)
[![Hugging Face Model Weights](https://img.shields.io/badge/Model_Weights-Hugging_Face-ffcc4d?style=plastic&logo=huggingface)](https://huggingface.co/Liukaikai/MobiAgent)

<img src="docs/assets/framework.png" alt="MobiAgent dual-loop architecture: deployment through planning, skill execution and reflection, and offline policy evolution through skill discovery and training" width="100%">

</div>

MobiAgent connects a deployment loop of **planning, skill execution, and visual reflection** with an offline loop of **demonstration processing, skill discovery, and policy training**. A shared vision-language backbone supports multiple flow-matching action experts.

## Paper

The paper link will be added here when available.

## Model weights

The [RoboCasa joint-training checkpoint at step 25,000](https://huggingface.co/Liukaikai/MobiAgent) includes model
parameters, per-expert normalization assets, and training state. Request access
on Hugging Face; downloads become available after manual approval. See the
[download and inference guide](docs/robocasa.md#download-weights).

## Training data

RoboCasa training uses the [official RoboCasa human demonstration datasets](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html).
The [dataset selection](configs/robocasa/datasets.example.json) maps atomic tasks to
`close`, `open`, `switch`, `manipulate`, `navigate`, and `pnp` experts.

Run the downloader in the RoboCasa environment:

```bash
python scripts/robocasa/download_data.py --dataset-root datasets/robocasa
```

Datasets are stored under `datasets/robocasa/v1.0/pretrain/atomic/<Task>/<Version>/lerobot/`.
The downloader writes `datasets/robocasa/datasets.json`, recording each task's
expert and local path. In the GPU backend environment, compute normalization and
create the training recipe:

```bash
uv sync --project policy/openpi
uv run --project policy/openpi scripts/robocasa/prepare_data.py \
  --manifest datasets/robocasa/datasets.json
uv run --project policy/openpi scripts/robocasa/train.py \
  --recipe configs/robocasa/data.json --check-data
uv run --project policy/openpi scripts/robocasa/train.py \
  --recipe configs/robocasa/data.json --exp-name my_run
```

Preparation saves normalization files in `assets/robocasa/<expert>/norm_stats.json`
and records their absolute paths in `configs/robocasa/data.json`. Training updates
the shared VLM and all six experts jointly and saves checkpoints under
`checkpoints/robocasa/`. See the [RoboCasa setup and training guide](docs/robocasa.md#training-data)
for environment requirements, dataset selection and custom paths.

## Agent skills

Reusable agent workflows are defined in [`skills/`](skills/), each with a
`SKILL.md` and agent metadata. They call the shared Python implementation for
simulation execution and offline policy evolution.

| Skill | Capability |
| --- | --- |
| [mobiagent-execution](skills/mobiagent-execution/SKILL.md) | Policy serving, long-horizon execution and visual feedback |
| [mobiagent-discovery](skills/mobiagent-discovery/SKILL.md) | Skill discovery, temporal alignment and training-data export |
| [mobiagent-training](skills/mobiagent-training/SKILL.md) | RoboCasa joint training and BEHAVIOR policy training |

See [using agent skills](docs/skills.md) for invocation and implementation details.

## Repository layout

```text
skills/                 Agent workflow packages (SKILL.md + agent metadata)
  mobiagent-execution/  Long-horizon task execution and policy serving
  mobiagent-discovery/  Skill discovery and training-data export
  mobiagent-training/   Policy training and checkpoint serving
scripts/
  robocasa/           RoboCasa download, preparation, inference, serving and training
  data/               Demonstration segmentation and per-expert data splitting
configs/
  robocasa/           RoboCasa dataset and policy-server templates
src/mobiagent/
  execution/          Shared planner, skill routing, visual critic and control loop
  environments/       RoboCasa simulator adapter
  discovery/behavior/ BEHAVIOR offline skill discovery
policy/openpi/
  src/openpi/         Shared model, policy serving and training implementation
  scripts/            BEHAVIOR training, serving and preprocessing entrypoints
docs/                 Skill usage, benchmark setup and architecture
```

| Workflow | Entry point | Guide |
| --- | --- | --- |
| RoboCasa data download | [`scripts/robocasa/download_data.py`](scripts/robocasa/download_data.py) | [Training data](docs/robocasa.md#training-data) |
| RoboCasa preparation | [`scripts/robocasa/prepare_data.py`](scripts/robocasa/prepare_data.py) | [Normalization and recipe](docs/robocasa.md#prepare-the-training-recipe) |
| RoboCasa inference | [`scripts/robocasa/infer.py`](scripts/robocasa/infer.py) | [Inference setup](docs/robocasa.md#inference) |
| RoboCasa policy server | [`scripts/robocasa/serve.py`](scripts/robocasa/serve.py) | [Serving a checkpoint](docs/robocasa.md#inference) |
| RoboCasa training · trainable VLM | [`scripts/robocasa/train.py`](scripts/robocasa/train.py) | [Training setup](docs/robocasa.md#training) |
| BEHAVIOR execution and training | [`src/mobiagent/execution/`](src/mobiagent/execution/) · [`policy/openpi/scripts/`](policy/openpi/scripts/) | [Deployment](docs/deployment.md) · [Training](docs/training.md) |
| Offline skill discovery | [`src/mobiagent/discovery/behavior/`](src/mobiagent/discovery/behavior/) | [Data preparation](docs/data.md) |

## Quick start

Python 3.11 or later is required. The lightweight agent and the GPU policy backend
can run in separate environments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e policy/openpi/packages/openpi-client
cp configs/policy_servers.example.yaml configs/policy_servers.yaml
```

Run the agent loop with mock planning, policies and observations, without a GPU,
robot, dataset, or model API call:

```bash
mobiagent --env mock --mock-all --task example \
  --instruction 'Move an object into a container' --max-ticks 3
```

## API configuration

Copy [.env.example](.env.example) to `.env` and fill in your provider settings:

```bash
cp .env.example .env
# Edit .env, then export its variables in the current shell.
set -a
source .env
set +a
```

| Variable in `.env` | Used for | Value to provide |
| --- | --- | --- |
| `OPENAI_API_KEY` | GPT planning, visual reflection and BEHAVIOR skill naming | Your API key |
| `OPENAI_BASE_URL` | GPT API connection | Your API base URL including `/v1`, e.g. `https://api.openai.com/v1` |
| `OPENAI_MODEL` | GPT requests | A model supporting image inputs and JSON output |
| `GEMINI_API_KEY` | BEHAVIOR video skill discovery | Your Gemini API key |
| `DASHSCOPE_API_KEY` | Optional alternative video discovery provider | Your DashScope API key |

Set the three `OPENAI_*` values in `.env` before making GPT requests.
Optional `MOBIAGENT_PLANNER_MODEL`, `MOBIAGENT_JUDGE_MODEL`,
`MOBIAGENT_FRAME_MODEL`, and `MOBIAGENT_NAMING_MODEL` override the model for
individual stages. Blank overrides use `OPENAI_MODEL`.

The connection is configured in [`src/mobiagent/api.py`](src/mobiagent/api.py).
GPT request calls are in [`execution/llm_client.py`](src/mobiagent/execution/llm_client.py)
and [`discovery/behavior/api_client.py`](src/mobiagent/discovery/behavior/api_client.py).
The API key, base URL and model are empty in the template; fill them with
your own settings. `.env` is excluded from Git. Training uses demonstration
data and base model weights; provider credentials are used for planning,
reflection and skill discovery.

- [BEHAVIOR and RoboCasa deployment](docs/deployment.md)
- [Offline skill discovery and data preparation](docs/data.md)
- [Policy training and serving](docs/training.md)
- [RoboCasa inference and training](docs/robocasa.md)

## Citation

If you find MobiAgent useful in your research, please cite our paper:

```bibtex
@misc{liu2026mobiagent,
  title={MobiAgent: Dual-Loop Recursive Policy Self-Improvement for Long-Horizon Mobile Manipulation},
  author={Chenzhi Liu and Zhang Yue and Jiehong Lin and Jianan Wang and Bo Wang and Zhongrui Wang and Xiaojuan Qi},
  year={2026},
  eprint={26xx.xxxxx}, % Replace with the final arXiv ID.
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```
