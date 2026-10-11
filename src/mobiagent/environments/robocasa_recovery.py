from __future__ import annotations
from typing import Any
import numpy as np

SPOUT = "Move the spout to wash all locations of the sink basin."
NAVIGATE = "Navigate to the sink."
WINDOW_STEPS = 60
RATIO_THRESHOLD = 0.12


class RinseRecovery:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.phase = "waiting"
        self.first_subtask_id: str | None = None
        self.start_index: int | None = None
        self.ratio: float | None = None
        self.eef_path_m: float | None = None
        self.command_norm_mean: float | None = None
        self.trigger_count = 0

    def observe_chunk(
        self,
        *,
        task_name: str,
        subtask_id: str,
        prompt: str,
        attempt_number: int,
        states: Any,
        actions: Any,
        prompts: Any,
    ) -> bool:
        if (
            self.phase != "waiting"
            or task_name != "RinseSinkBasin"
            or prompt != SPOUT
            or (attempt_number != 1)
        ):
            return False
        if self.first_subtask_id is None:
            self.first_subtask_id = subtask_id
            try:
                self.start_index = next(
                    (i for i, p in enumerate(prompts) if p == SPOUT)
                )
            except (TypeError, StopIteration):
                return False
        if subtask_id != self.first_subtask_id or self.start_index is None:
            return False
        start = self.start_index
        if len(states) < start + WINDOW_STEPS or len(actions) < start + WINDOW_STEPS:
            return False
        if any((p != SPOUT for p in prompts[start : start + WINDOW_STEPS])):
            self.phase = "invalid_prompt_window"
            return False
        robot_states = np.asarray(
            states[start : start + WINDOW_STEPS], dtype=np.float64
        )
        policy_actions = np.asarray(
            actions[start : start + WINDOW_STEPS], dtype=np.float64
        )
        if (
            robot_states.shape != (WINDOW_STEPS, 16)
            or policy_actions.ndim != 2
            or policy_actions.shape[0] != WINDOW_STEPS
            or (policy_actions.shape[1] < 3)
            or (not np.isfinite(robot_states).all())
            or (not np.isfinite(policy_actions[:, :3]).all())
        ):
            self.phase = "invalid_robot_data"
            return False
        path = float(np.linalg.norm(np.diff(robot_states[:, :3], axis=0), axis=1).sum())
        command = float(np.linalg.norm(policy_actions[:, :3], axis=1).mean())
        self.eef_path_m = path
        self.command_norm_mean = command
        if command <= 0.05:
            self.phase = "low_command"
            return False
        self.ratio = path / command
        if self.ratio < RATIO_THRESHOLD:
            self.phase = "nav_pending"
            self.trigger_count = 1
            return True
        self.phase = "clear"
        return False

    def planner_intent(self, history: Any) -> str | None:
        if not history:
            return None
        last = history[-1]
        last_prompt = getattr(last.get("subtask"), "prompt", None)
        if (
            self.phase == "nav_pending"
            and last.get("outcome") == "failed"
            and (last_prompt == SPOUT)
        ):
            self.phase = "nav_emitted"
            return "navigate"
        if self.phase == "nav_emitted" and last_prompt == NAVIGATE:
            self.phase = "done"
            return "spout"
        return None
