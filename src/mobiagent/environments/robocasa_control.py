from __future__ import annotations

import numpy as np

from .robocasa_config import action_kind


class AttemptControl:
    def __init__(self, config, subtask, obs):
        self.config = config
        self.kind = action_kind(subtask.stage_hint, subtask.prompt)
        self.start_gap = self.gap(obs)
        self.peak_gap = self.start_gap
        self.previous_gap = None
        self.confirmations = 0
        self.chunks = 0
        self.hold = None
        self.pick_handoff = None
        self.place_checked = False
        self.release_checked = False
        self.gap_history = []
        self.hold_baseline = (
            self.start_gap
            if self.start_gap is not None and 5 <= self.start_gap <= 68
            else None
        )

    @staticmethod
    def gap(obs):
        value = (obs or {}).get("observation/state")
        if value is None:
            return None
        state = np.asarray(value).reshape(-1)
        if state.size != 16 or not np.isfinite(state).all():
            return None
        return abs(float(state[14] - state[15])) * 1000

    def observe(self, obs):
        self.chunks += 1
        gap = self.gap(obs)
        if gap is None or self.start_gap is None:
            return None
        if self.kind == "pick" and self.hold is None:
            self.peak_gap = max(self.peak_gap, gap)
            contact = (
                self.peak_gap >= self.config.pick_open_min_mm
                and self.config.pick_contact_min_mm
                <= gap
                <= self.config.pick_contact_max_mm
                and gap <= self.peak_gap - self.config.pick_close_delta_mm
            )
            if (
                contact
                and self.previous_gap is not None
                and abs(gap - self.previous_gap) > self.config.pick_stable_delta_mm
            ):
                self.confirmations = 0
            self.confirmations = self.confirmations + 1 if contact else 0
            self.previous_gap = gap
            if self.confirmations >= self.config.freeze_confirmations:
                self.pick_handoff = {
                    "peak_gap_mm": self.peak_gap,
                    "contact_gap_mm": gap,
                    "confirmations": self.confirmations,
                }
                return "hold"
        if self.kind == "place" and not self.place_checked:
            released = self.start_gap <= 62 and gap >= 45 and gap - self.start_gap >= 10
            self.confirmations = self.confirmations + 1 if released else 0
            if self.confirmations >= self.config.freeze_confirmations:
                self.place_checked = True
                return "visual"
        if self.kind == "action" and not self.release_checked:
            previous = self.previous_gap
            self.previous_gap = gap
            self.gap_history = (self.gap_history + [gap])[-3:]
            if self.hold_baseline is None and len(self.gap_history) == 3:
                if (
                    all(5 <= value <= 68 for value in self.gap_history)
                    and max(self.gap_history) - min(self.gap_history) <= 2.5
                ):
                    self.hold_baseline = sum(self.gap_history) / 3
            if self.hold_baseline is not None and self.chunks >= 12:
                if gap - self.hold_baseline >= 5 or (
                    previous is not None and gap - previous >= 4
                ):
                    self.release_checked = True
                    return "release"
        minimum = {"switch": 12, "open": 18, "close": 12}.get(self.kind)
        if (
            minimum is not None
            and self.chunks >= minimum
            and (self.chunks - minimum) % self.config.visual_cadence == 0
        ):
            return "visual"
        return None


def robot_info(obs, previous=None, handoff=None):
    state = np.asarray((obs or {}).get("observation/state", []), dtype=float).reshape(
        -1
    )
    if state.size != 16 or not np.isfinite(state).all():
        return {}
    result = {
        "end_effector_position_relative": state[:3].tolist(),
        "end_effector_rotation_relative": state[3:7].tolist(),
        "base_position": state[7:10].tolist(),
        "base_rotation": state[10:14].tolist(),
        "gripper_finger_positions": state[14:16].tolist(),
        "jaw_gap_mm": AttemptControl.gap(obs),
    }
    prior = np.asarray(
        (previous or {}).get("observation/state", []), dtype=float
    ).reshape(-1)
    if prior.size == 16 and np.isfinite(prior).all():
        result["delta_end_effector_position"] = (state[:3] - prior[:3]).tolist()
        result["delta_base_position"] = (state[7:10] - prior[7:10]).tolist()
        result["delta_jaw_gap_mm"] = result["jaw_gap_mm"] - AttemptControl.gap(previous)
    if handoff:
        result["grasp_phase_contact"] = handoff
    return result
