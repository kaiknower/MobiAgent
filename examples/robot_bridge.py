"""Robot bridge template — copy this, rename, fill in the TODOs.

Six methods, all called by the agent loop on every chunk cycle. Once
filled in, register the factory in ``--robot=my_pkg.my_bridge:make_robot``
when launching the agent:

    python -m mobiagent.robots.s1 \
        --task trash-general \
        --robot my_pkg.my_bridge:make_robot

────────────────────────────────────────────────────────────────────────────
Image conventions (uint8 HWC; CHW or float arrays are auto-converted by
the policy client, but uint8 HWC is the recommended canonical form):

  * head camera     ─ ~640×360 or 1280×720 (whatever your driver emits)
  * left wrist      ─ same
  * right wrist     ─ same
  * the planner / judge only consume the head frame; the policy server
    consumes all three.

State convention (1-D float32, length 34) — raw cartesian EE pose, same
layout the model was trained on:

      idx 0-8    torso            (3 pos + 6d rot)
      idx 9-17   left_arm
      idx 18     left_gripper     (1 = open, 0 = closed; project-defined)
      idx 19-27  right_arm
      idx 28     right_gripper
      idx 29     head_yaw         ← model is trained without this; just
      idx 30     head_pitch       ← report current pose, agent re-inserts it
      idx 31-33  chassis (x, y, theta)   ← will be zeroed before send

Action chunk convention — what ``execute_chunk`` receives:

  * shape ``(T, 34)`` float32, same column layout as state.
  * idx 29-30 (head) are filled with the most recent obs head pose
    (so head holds still). Drive them directly.
  * idx 31-33 (chassis) is a **delta from current chassis pose**, not
    an absolute target. Add your robot's live chassis pose before
    sending to the base controller.
  * idx 0-28 (torso/arms/grippers) are absolute EE-pose targets in
    the project's standard frame.

Gripper state (str): "OPEN" or "CLOSED" — only consumed by the judge.

task_is_done(): return True when YOUR task-level success predicate
triggers (e.g. an absorbing condition in sim, or a manual stop signal
on real hardware); the loop also exits when the planner declares
completion.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from typing import Any

import numpy as np


class MyRobotBridge:
    """Implement RobotBridge for your robot. Replace every TODO."""

    def __init__(self) -> None:
        # TODO: open your camera / motor drivers / IK service here.
        self._gripper_prev = "OPEN"
        self._gripper_now = "OPEN"

    # ─────────── observation ───────────

    def read_head_camera(self) -> np.ndarray:
        """Return the head (egocentric) camera frame as uint8 HWC.

        Used by the planner (per subtask) and the judge (per attempt).
        """
        raise NotImplementedError  # TODO: pull frame from your camera bridge

    def read_obs_dict(self) -> dict[str, Any]:
        """Return the full policy-side observation dict.

        Required keys (canonical openpi names):
          * "observation/egocentric_camera"  — uint8 HWC
          * "observation/wrist_image_left"   — uint8 HWC
          * "observation/wrist_image_right"  — uint8 HWC
          * "observation/state"              — float32 (34,) cartesian pose
        """
        return {
            "observation/egocentric_camera": self.read_head_camera(),
            "observation/wrist_image_left":  self._read_left_wrist(),     # TODO
            "observation/wrist_image_right": self._read_right_wrist(),    # TODO
            "observation/state":             self._read_state_34().astype(np.float32),  # TODO
        }

    # ─────────── action ───────────

    def execute_chunk(self, actions: np.ndarray) -> None:
        """Execute a (T, 34) action chunk. Block until the chunk completes.

        Per-column handling reminder:
          * idx 0-28: absolute EE-pose targets → torso / arms / grippers
          * idx 29-30: head_yaw / head_pitch (already held at current pose
            by the policy client; just forward to your head driver)
          * idx 31-33: chassis **delta** (dx, dy, dtheta) — add to current
            chassis pose before commanding the base
        """
        # Cache previous gripper state for the judge before commanding.
        self._gripper_prev = self._gripper_now
        # TODO: send chunk to your IK / streaming controller; wait for done.
        # TODO: update self._gripper_now from the last action frame (e.g.
        #       "CLOSED" if actions[-1, 18] < 0.5 else "OPEN" — project-defined).
        raise NotImplementedError

    # ─────────── side channels for the judge ───────────

    def gripper_state(self) -> str:
        """Current gripper state ("OPEN" / "CLOSED"). Judge context."""
        return self._gripper_now

    def prev_gripper_state(self) -> str:
        """Gripper state at the previous attempt. Judge context."""
        return self._gripper_prev

    def task_is_done(self) -> bool:
        """Loop exits early when this returns True.

        Wire to a manual stop signal, a sim success flag, or always
        return False to let the planner / retry budget control exit.
        """
        return False

    # ─────────── private helpers (rename / inline as you like) ───────────

    def _read_left_wrist(self) -> np.ndarray:
        raise NotImplementedError  # TODO

    def _read_right_wrist(self) -> np.ndarray:
        raise NotImplementedError  # TODO

    def _read_state_34(self) -> np.ndarray:
        """Return the raw 34-dim cartesian EE-pose state. See module
        docstring for the index layout. Do NOT zero chassis here —
        the policy client zeros it before send."""
        raise NotImplementedError  # TODO


def make_robot() -> MyRobotBridge:
    """Factory invoked by the CLI: ``--robot my_pkg.my_bridge:make_robot``."""
    return MyRobotBridge()
