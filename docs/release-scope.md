# Private preview scope

This repository contains the project's simulation code, extracted from its
BEHAVIOR and RoboCasa workspaces. Original workspaces remain unchanged.

Included:

- Shared planner, policy routing, visual critic and retry/replan control loop.
- BEHAVIOR execution, offline demonstration processing and skill discovery.
- RoboCasa simulator adapter, inference entrypoint and policy server.
- RoboCasa frozen-backbone and joint-training entrypoints.
- Shared VLM, action experts, data loaders and configuration templates.

Excluded:

- Physical robot integration, S1-specific code and test suites.
- Results, logs, videos, caches, datasets, normalization values and checkpoints.
- Credentials, private endpoints and machine-specific paths.
- Fixed expert-routing overrides, scripted navigation corrections, forced poses,
  simulator pose rollback, verdict overrides and task-specific annotation repairs.

The cleaned runtime and portable recipes are not an exact reproduction package
for the paper's reported numbers. Source and interface checks do not establish
full simulator or GPU training performance; those runs require external assets.

Paper and citation details remain pending. The repository remains private until
its owner decides to publish it. No public license has been selected.
