# Deployment

Start a policy server as described in [training.md](training.md), then put its
address into `configs/policy_servers.yaml`. Model API credentials must be exported
in the shell. Deployment invokes billable model APIs only when you run it.

## BEHAVIOR

Install BEHAVIOR-1K / OmniGibson and its required simulator assets in a compatible
Linux environment. The extracted adapter also uses the BEHAVIOR `gello` teleop
configuration helpers. Supply the benchmark instruction explicitly:

```bash
mobiagent --env omni --task task-0001 --instruction 'YOUR TASK INSTRUCTION' \
  --policy-servers configs/policy_servers.yaml --max-ticks 200
```

The policy configuration and planner vocabulary must match. The default
BEHAVIOR setup uses `move_to`, `pick_up_from`, `place_in`, `place_on`, `open`, and
`close` in that order. Runtime logging code remains available, but generated logs
are excluded from Git.

## RoboCasa

Install the RoboCasa version providing the `robocasa/<Task>` Gym registration and
Groot dataset reader used by your datasets. The adapter preserves the source
16-dimensional state, three camera observations, and 12-dimensional actions.
Set `MUJOCO_GL` and `MUJOCO_EGL_DEVICE_ID` for your own renderer when necessary.

```bash
mobiagent --env robocasa --task YOUR_ROBOCASA_TASK \
  --instruction 'YOUR TASK INSTRUCTION' \
  --policy-servers configs/policy_servers.yaml --max-episode-steps 3000
```

The shared RoboCasa policy orders its experts as `close`, `open`, `switch`,
`manipulate`, `navigate`, and `pnp`. The planner receives that vocabulary.
Simulator success is available through `episode_score()` for terminal evaluation;
it is not used to rewrite the visual critic's verdict.

## Astribot S1

Implement the camera, state, action execution, gripper, and termination methods in
[`examples/robot_bridge.py`](../examples/robot_bridge.py) for your robot SDK. The
repository contains the bridge interface, not a vendor robot driver.

```bash
mobiagent-s1 --task trash-general --robot my_bridge:make_robot \
  --config configs/policy_servers.yaml
```

The model uses three experts: `move_to`, `pick_up`, and `place`. Preserve the source
34-dimensional Cartesian robot contract. The policy preprocessing removes head
coordinates and uses 32-dimensional model actions; the client restores head
coordinates and returns a 34-dimensional chunk. Chassis actions are deltas.
