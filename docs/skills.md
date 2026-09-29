# Agent skills

MobiAgent exposes four repository-local workflow skills. Each package contains
`SKILL.md` with a name, selection description, inputs, commands and output checks,
plus `agents/openai.yaml` with agent UI metadata.

| Skill | Workflow | Implementation |
| --- | --- | --- |
| [mobiagent-execution](../skills/mobiagent-execution/SKILL.md) | Serve a checkpoint and execute a long-horizon simulation task | `src/mobiagent/execution/`, `scripts/robocasa/` |
| [mobiagent-discovery](../skills/mobiagent-discovery/SKILL.md) | Annotate demonstration timelines and group skills | `src/mobiagent/discovery/behavior/` |
| [mobiagent-data](../skills/mobiagent-data/SKILL.md) | Align annotations, build segments and split expert shards | `scripts/data/` |
| [mobiagent-training](../skills/mobiagent-training/SKILL.md) | Train or resume a policy | `scripts/robocasa/train.py`, `policy/openpi/` |

## Using a skill

Open the repository in an agent that can read local files and run shell commands.
[AGENTS.md](../AGENTS.md) maps workflow requests to the relevant skill. For example:

> Read `skills/mobiagent-training/SKILL.md` and configure RoboCasa joint training
> using my dataset recipe and checkpoint directory.

A harness that supports registering skill directories can load these packages
and expose their names as `$mobiagent-training`, `$mobiagent-execution`, and so on.
When registering a skill, preserve access to this repository: commands run from
its root and references point to the shared code and guides.

## Architecture

The agent reads a skill, resolves its inputs and invokes the documented Python
entrypoint. Simulation planning, policy execution and visual reflection run inside
the existing Python control loop. The simulation planner does not automatically
load these Markdown files.

Workflow skills organize agent operations across execution and policy evolution.
The learned action experts (`open`, `close`, `navigate`, and other benchmark-specific
skills) remain model components selected by the runtime policy router.

RoboCasa training updates both the shared VLM and action experts. The BEHAVIOR
recipe retains its own backbone-freezing configuration.
