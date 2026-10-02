# Simulation deployment

For the complete RoboCasa workflow, see [RoboCasa inference and training](robocasa.md).

## BEHAVIOR

Install BEHAVIOR-1K / OmniGibson and its required simulator assets in a compatible
Linux environment. The adapter uses the BEHAVIOR `gello` simulation configuration
helpers. Start a policy server using the [training guide](training.md), then set
its address in `configs/policy_servers.yaml`.

Export `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` in the agent environment. Supply the task instruction:

```bash
mobiagent --env omni --task task-0001 --instruction 'YOUR TASK INSTRUCTION' \
  --policy-servers configs/policy_servers.yaml --max-ticks 200
```

The policy configuration and planner vocabulary must match. The default
BEHAVIOR setup uses `move_to`, `pick_up_from`, `place_in`, `place_on`, `open`, and
`close` in that order. Runtime output is excluded from Git.
