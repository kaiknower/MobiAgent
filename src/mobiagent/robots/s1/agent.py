"""Agent runtime: drives planner → policy → robot → judge end-to-end.

The orchestration loop is fixed here so the user only has to provide a
robot bridge (camera, state, action chunk executor) and pick a task; no
copy-pasted Python boilerplate. See ``RobotBridge`` for the methods the
bridge must implement.

CLI: see ``mobiagent.robots.s1.__main__`` (``python -m mobiagent.robots.s1 --task ... --robot ...``).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .judge import judge as _judge
from .planner import next_subtask
from .policy_registry import PolicyRegistry
from .schemas import TASK_GOALS, build_full_prompt

logger = logging.getLogger("mobiagent.robots.s1.agent")


class RobotBridge(Protocol):
    """Methods the agent loop calls on the user-supplied robot bridge."""

    def read_head_camera(self) -> np.ndarray:
        """uint8 HWC head-camera frame (used by the planner + judge)."""

    def read_obs_dict(self) -> dict[str, Any]:
        """Return the policy-side observation dict. Must contain at least
        the three cameras + 34-dim cartesian state, e.g.::

            {
                "observation/egocentric_camera":  <H, W, 3 uint8>,
                "observation/wrist_image_left":   <H, W, 3 uint8>,
                "observation/wrist_image_right":  <H, W, 3 uint8>,
                "observation/state":              <34, float32>,
            }
        """

    def execute_chunk(self, actions: np.ndarray) -> None:
        """Execute a (T, 34) action chunk on the robot. Block until done."""

    def gripper_state(self) -> str:
        """Current gripper state for the judge ("OPEN" / "CLOSED")."""

    def prev_gripper_state(self) -> str:
        """Gripper state from the previous attempt (judge context)."""

    def task_is_done(self) -> bool:
        """Return True when the global task is finished (loop exits early)."""


def run_agent(
    robot: RobotBridge,
    *,
    task: str,
    config_path: Path | str = "configs/policy_servers.yaml",
    max_steps: int = 200,
) -> list[dict]:
    """Run the end-to-end loop: planner → policy → robot → judge.

    Stops when ``robot.task_is_done()`` returns True or after ``max_steps``
    subtask cycles. Returns the per-subtask outcome history.
    """
    if task not in TASK_GOALS:
        raise ValueError(
            f"unknown task {task!r}; valid task ids: {list(TASK_GOALS)}"
        )
    goal = TASK_GOALS[task]
    policies = PolicyRegistry.from_yaml(Path(config_path))
    history: list[dict] = []
    prev_obs: dict | None = None
    mid_obs: dict | None = None

    logger.info("agent start: task=%s goal=%r max_steps=%d", task, goal, max_steps)

    try:
        for step in range(max_steps):
            if robot.task_is_done():
                logger.info("robot.task_is_done() → exit at step %d", step)
                break

            subtask = next_subtask(
                global_goal=goal,
                head_image=robot.read_head_camera(),
                history=history,
            )
            logger.info(
                "step %d / subtask=%s prompt=%r stage=%s",
                step, subtask.id, subtask.prompt, subtask.stage_hint,
            )
            # Reset rolling inpaint prior when the active subtask changes —
            # otherwise the prior leaks across heads / skills.
            policies.reset_inpaint_state()

            for attempt in range(subtask.max_retries):
                obs = robot.read_obs_dict()
                chunk = policies.select(subtask.stage_hint).request_chunk(
                    obs=obs,
                    prompt=build_full_prompt(goal, subtask.prompt),
                    stage_hint=subtask.stage_hint,
                )
                robot.execute_chunk(chunk["actions"])
                obs_after = robot.read_obs_dict()

                verdict = _judge(
                    subtask=subtask,
                    obs_after=obs_after,
                    obs_prev=prev_obs,
                    obs_mid=mid_obs,
                    robot_info={
                        "gripper_now":  robot.gripper_state(),
                        "gripper_prev": robot.prev_gripper_state(),
                    },
                    history=history,
                )
                mid_obs, prev_obs = prev_obs, obs_after

                if verdict.verdict in ("complete", "error"):
                    history.append({
                        "subtask_id":    subtask.id,
                        "stage_hint":    subtask.stage_hint,
                        "outcome":       verdict.verdict,
                        "n_attempts":    attempt + 1,
                        "judge_reason":  verdict.reason,
                    })
                    break
            else:
                history.append({
                    "subtask_id":   subtask.id,
                    "stage_hint":   subtask.stage_hint,
                    "outcome":      "failed",
                    "n_attempts":   subtask.max_retries,
                    "judge_reason": "retry budget exhausted",
                })
    finally:
        try:
            policies.close()
        except Exception:
            pass

    return history


__all__ = ["RobotBridge", "run_agent"]
