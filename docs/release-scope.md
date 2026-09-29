# Private preview scope

This repository was assembled from the project's BEHAVIOR execution / discovery
and policy workspaces on the 4090 host, and its S1, RoboCasa and shared-VLM training
workspaces on the A100 host. Original workspaces were read only and left unchanged.

Included:

- Reactive planner, policy routing, visual critic and retry/replan control loop.
- BEHAVIOR and S1 offline demonstration / skill-discovery pipelines.
- S1 robot bridge contract and RoboCasa observation/action conversion.
- Shared-VLM / multiple-action-expert model, data loaders, training and serving.
- Configuration templates, installation guides and CPU tests.

Excluded:

- Experiment results, aggregate tables, logs, rollout videos, caches, datasets,
  normalization values and trained checkpoints.
- Credentials, private service endpoints, hostnames and machine-specific paths.
- Benchmark batch launchers, fixed expert-routing overrides, scripted navigation
  pose corrections, forced arm poses, simulator pose rollback, verdict overrides,
  ground-truth subtask probes, and task-specific annotation repair scripts.
- Unrelated agent integrations, chat connectors, old baselines and experiment notes.

The cleaned deployment runtime uses task-independent visual judging; the portable
training recipes replace historical experiment variants. These intentional changes
mean this snapshot is **not an exact reproduction package for the paper's reported
numbers**. Documentation and tests distinguish source extraction from integration
validation. Full GPU training, simulator rollout and robot execution are not tested
in this preparation task.

Public project, paper, dataset, weight and license links remain pending. The GitHub
repository must remain private until its owner explicitly decides to publish it.

## Validation performed

- 32 CPU tests passed for the reactive loop, skill routing, retry/replan history,
  S1 state handling, RoboCasa policy payload, and offline data utilities.
- Source syntax, configuration syntax, and scans for credentials, private
  endpoints, machine paths, and excluded artifacts passed.
- GPU training, full simulator rollouts, and robot execution were not run.
