---
name: mobiagent-execution
description: Execute long-horizon manipulation tasks with MobiAgent in RoboCasa or BEHAVIOR. Use when a task needs a running policy server, reactive subtask planning, visual reflection, and bounded retries in simulation.
---

# MobiAgent Execution

Run commands from the repository root, two levels above this file. Use the
simulator's Python environment for the agent and the OpenPI environment for serving.

## Inputs

- Benchmark (`robocasa` or `behavior`), simulator task ID, and task instruction.
- Trained checkpoint and policy-server configuration, or an existing server.
- Run directory and execution budget (`--max-ticks`, optionally `--max-episode-steps`).

Read [RoboCasa setup](../../docs/robocasa.md#inference) or
[BEHAVIOR setup](../../docs/deployment.md) for the selected benchmark only.

## Workflow

1. Check that the task exists in the selected simulator and the policy matches
   its observation/action contract. Use the supplied checkpoint and server;
   do not train another model just to satisfy an inference request.
2. If serving is needed, start the matching checkpoint in the backend environment:

   ```bash
   # RoboCasa: export MOBIAGENT_ROBOCASA_RECIPE first.
   uv run --project policy/openpi scripts/robocasa/serve.py \
     --checkpoint /path/to/checkpoint --port 8000
   # BEHAVIOR:
   uv run --project policy/openpi policy/openpi/scripts/serve_policy.py \
     --config mobiagent_behavior --checkpoint /path/to/checkpoint --port 8000
   ```

3. Export model API settings as described in the benchmark guide. Run one task:

   ```bash
   # RoboCasa:
   python scripts/robocasa/infer.py --task TASK_ID --instruction 'TASK GOAL' \
     --policy-servers configs/robocasa/policy_servers.yaml \
     --max-ticks 200 --max-episode-steps 3000 --run-dir runs/robocasa/TASK_ID
   # BEHAVIOR:
   mobiagent --env omni --task TASK_ID --instruction 'TASK GOAL' \
     --policy-servers configs/policy_servers.yaml --max-ticks 200 \
     --run-dir runs/behavior/TASK_ID
   ```

4. Let the runtime choose each subtask, execute action chunks and apply visual
   feedback. Its planner, executor and critic share the running simulator state;
   do not launch a new simulator process for every subtask or replace the selected
   expert with a fixed task-specific sequence.
5. Inspect `summary.json` in the run directory. Report `total_ticks`,
   `subtasks_completed`, `subtasks_failed`, `abort_reason`, and the recorded
   `success` value. Process exit code zero does not by itself mean task success.
   Report benchmark success only if a simulator score was actually observed.

RoboCasa expert order is `close`, `open`, `switch`, `manipulate`, `navigate`, `pnp`.
BEHAVIOR expert order is `move_to`, `pick_up_from`, `place_in`, `place_on`, `open`,
`close`. Preserve the selected benchmark's order and normalization.

For a runtime-only check, use `mobiagent --env mock --mock-all --task example
--instruction 'Move an object' --max-ticks 3`; label its output as a mock run.
If the run fails or reaches its budget, report its state and cause rather than
silently restarting another full episode.

## Implementation

- [Reactive runtime](../../src/mobiagent/execution/orchestrator.py)
- [Planner](../../src/mobiagent/execution/planner_vlm.py) and
  [visual critic](../../src/mobiagent/execution/judge_vlm.py)
- [RoboCasa adapter and expert routing](../../src/mobiagent/environments/robocasa.py)
