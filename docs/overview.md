# MobiAgent

**From Execution to Evolution: A Dual-Loop Agentic System for Long-Horizon Mobile Manipulation**

![Framework](assets/framework.png)

The deployment loop asks a vision-language planner for the next skill, routes
that instruction to a policy, and uses visual feedback to decide whether to
continue, retry, or replan. The offline loop turns demonstration video and robot
state into skill segments and trains routed action experts over a shared VLM.

The code is organized around those interfaces rather than individual experiments.
BEHAVIOR and RoboCasa use simulator adapters; S1 supplies the same control-loop
idea through a user-provided robot bridge.

The project webpage, paper link, datasets, weights, and public release are not
published by this repository. Public links can be added to the README when ready.
