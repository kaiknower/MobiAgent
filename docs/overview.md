# MobiAgent

**From Execution to Evolution: A Dual-Loop Agentic System for Long-Horizon Mobile Manipulation**

![Framework](assets/framework.png)

The deployment loop asks a vision-language planner for the next skill, routes
that instruction to a policy, and uses visual feedback to decide whether to
continue, retry, or replan. The offline loop turns demonstration video and robot
state into skill segments and trains routed action experts over a shared VLM.

Agent-facing workflows live in `skills/`; see [the skill guide](skills.md).
Each skill calls the corresponding Python implementation.
BEHAVIOR and RoboCasa use simulator adapters. The RoboCasa training and inference
entrypoints are grouped under `scripts/robocasa/`; shared implementations stay in
`src/mobiagent/` and `policy/openpi/`.
