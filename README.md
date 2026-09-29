<div align="center">

# MobiAgent
### From Execution to Evolution: A Dual-Loop Agentic System for Long-Horizon Mobile Manipulation

[Chenzhi Liu](https://openreview.net/profile?id=~Chenzhi_Liu2)<sup>1,*</sup> · [Zhang Yue](https://openreview.net/profile?id=~Zhang_Yue_bolt1)<sup>1,*</sup> · [Jiehong Lin](https://openreview.net/profile?id=~Jiehong_Lin1)<sup>1,*,†</sup> · [Jianan Wang](https://openreview.net/profile?id=~Jianan_Wang2)<sup>2</sup><br>
[Bo Wang](https://openreview.net/profile?id=~Bo_Wang36)<sup>1</sup> · [Zhongrui Wang](https://openreview.net/profile?id=~Zhongrui_Wang1)<sup>3,‡</sup> · [Xiaojuan Qi](https://openreview.net/profile?id=~Xiaojuan_Qi4)<sup>1,‡</sup>

<sup>1</sup> The University of Hong Kong &nbsp; <sup>2</sup> Astribot &nbsp; <sup>3</sup> Southern University of Science and Technology<br>
<sup>*</sup> Equal contribution &nbsp; <sup>†</sup> Project leader &nbsp; <sup>‡</sup> Corresponding authors

[![Overview](https://img.shields.io/badge/Project-Overview-3366cc)](docs/overview.md)
[![Setup](https://img.shields.io/badge/Code-Quick_Start-357a38)](#quick-start)
[![Training](https://img.shields.io/badge/Policy-Training-7952b3)](docs/training.md)
[![Status](https://img.shields.io/badge/Status-Private_Preview-555555)](docs/release-scope.md)

<img src="docs/assets/framework.png" alt="MobiAgent dual-loop architecture: deployment through planning, skill execution and reflection, and offline policy evolution through skill discovery and training" width="100%">

</div>

MobiAgent connects a deployment loop of **planning, skill execution, and visual reflection** with an offline loop of **demonstration processing, skill discovery, and policy training**. A shared vision-language backbone supports multiple flow-matching action experts.

This repository is a **private code preview**. It contains curated source code and documentation, with no experiment results, logs, datasets, trained weights, or credentials. The cleaned runtime and portable recipes are not a claim of exact reproduction of the paper's reported experiments. See [release scope and validation](docs/release-scope.md).

## Repository layout

```text
src/mobiagent/
  execution/          Planner, visual critic, skill routing, reactive runtime
  discovery/          Offline skill discovery for BEHAVIOR and S1 demonstrations
  environments/       RoboCasa observation/action adapter
  robots/s1/          Astribot S1 agent and robot bridge interface
policy/openpi/        Shared VLM, action experts, data loaders, training and serving
scripts/data/         Segment construction and per-head data splitting
configs/              Portable configuration examples
examples/             Robot bridge template
tests/               CPU unit and integration checks
docs/                Setup, architecture, source provenance and release scope
```

## Quick start

Python 3.11 or later is required. The lightweight agent and the GPU policy backend
can run in separate environments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pip install -e policy/openpi/packages/openpi-client
cp configs/policy_servers.example.yaml configs/policy_servers.yaml
```

Run the agent loop with mock planning, policies and observations, without a GPU,
robot, dataset, or model API call:

```bash
mobiagent --env mock --mock-all --task example \
  --instruction 'Move an object into a container' --max-ticks 3
```

For live planning and reflection, export your own `AZURE_OPENAI_API_KEY`,
`AZURE_OPENAI_ENDPOINT`, and `AZURE_OPENAI_DEPLOYMENT`. The `.env.example` file
lists supported settings; it is a template, not automatically loaded.

- [BEHAVIOR and RoboCasa deployment](docs/deployment.md)
- [Offline skill discovery and data preparation](docs/data.md)
- [Policy training and serving](docs/training.md)
- [Astribot S1 bridge](docs/deployment.md#astribot-s1)

## Validation

```bash
pip install -e '.[dev,discovery]'
pytest
```

The tests exercise routing, retries, replanning, observation contracts, and data
utilities without calling external model services or moving a robot. Full
simulation, GPU training, and hardware validation require the corresponding
external environments and assets.

## Acknowledgments

The work has been supported by Hong Kong Research Grant Council - General Research Fund Scheme (Grant No. 17202422, 17212923, 17215025) Theme-based Research (Grant No.T45-701/22-R), and Strategic Topics Grant (Grant No.STG3/E-605/25-N). Part of the described research work is conducted in the JC STEM Lab of Robotics for Soft Materials funded by The Hong Kong Jockey Club Charities Trust.

## License and release status

First-party code remains a private research preview; no public open-source license
has been selected yet. Third-party license notices are retained under
[`policy/openpi/`](policy/openpi/) and summarized in [NOTICE.md](NOTICE.md).
