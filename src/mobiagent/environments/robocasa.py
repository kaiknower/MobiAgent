"""RoboCasa camera, state and action adapter for simulation execution."""

from __future__ import annotations
import random
import numpy as np
from .robocasa_config import ExecutionConfig, SKILLS, STAGE_SKILLS

STATE_KEYS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.base_position",
    "state.base_rotation",
    "state.gripper_qpos",
)


class RoboCasaEnv:
    def __init__(
        self,
        task_name: str,
        *,
        seed: int = 7,
        split: str = "target",
        horizon: int | None = None,
    ):
        self.config = ExecutionConfig.from_environment()
        self.task_name, self.seed, self.split = task_name, seed, split
        self._explicit_horizon = horizon
        self.horizon = horizon
        self._env = None
        self._latest_obs = {}
        self._steps = 0
        self._done = False
        self._score = False
        self._last_action = None
        self._arm_home = None
        self._retry_key = None
        self._retry_count = 0
        self._motion_states = []
        self._motion_actions = []
        self._motion_prompts = []
        self._active_prompt = ""
        self.recovery = None

    def reset(self, task: str | None = None):
        import gymnasium as gym
        import robocasa  # noqa: F401

        self.close()
        self.task_name = task or self.task_name
        if self._explicit_horizon is None:
            from robocasa.utils.dataset_registry_utils import get_task_horizon

            self.horizon = max(
                1, round(get_task_horizon(self.task_name) * self.config.horizon_scale)
            )
        else:
            if self._explicit_horizon < 1:
                raise ValueError("horizon must be positive")
            self.horizon = self._explicit_horizon
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._env = gym.make(
            f"robocasa/{self.task_name}", split=self.split, seed=self.seed
        )
        value = self._env.reset(seed=self.seed)
        self._steps, self._done, self._score = 0, False, False
        self._last_action = None
        self._retry_key, self._retry_count = None, 0
        self._motion_states, self._motion_actions, self._motion_prompts = [], [], []
        if self.task_name == "RinseSinkBasin":
            from .robocasa_recovery import RinseRecovery

            self.recovery = RinseRecovery()
        else:
            self.recovery = None
        self._capture_arm_home()
        return self._map_obs(value[0] if isinstance(value, tuple) else value)

    def _map_obs(self, raw):
        from openpi_client import image_tools

        obs = {
            "observation/state": np.concatenate(
                [np.asarray(raw[k], dtype=np.float32).ravel() for k in STATE_KEYS]
            )
        }
        for name, key in [
            ("head", "video.robot0_agentview_left"),
            ("left_wrist", "video.robot0_eye_in_hand"),
            ("right_wrist", "video.robot0_agentview_right"),
        ]:
            image = np.asarray(raw[key])
            obs[f"observation/{name}_image_orig"] = image
            obs[f"observation/{name}_image"] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(image, 224, 224)
            )
        self._latest_obs = obs
        return obs

    def step(self, chunk):
        from robocasa.utils.env_utils import convert_action

        actions = np.asarray(
            chunk["actions"] if isinstance(chunk, dict) else chunk, dtype=np.float32
        )
        if (
            actions.ndim != 2
            or actions.shape[1] != 12
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Expected a finite (T, 12) RoboCasa action chunk")
        for row in actions:
            if self.is_done():
                break
            self._motion_states.append(self._latest_obs["observation/state"].copy())
            self._motion_actions.append(row.copy())
            self._motion_prompts.append(self._active_prompt)
            result = self._env.step(convert_action(row.astype(np.float64)))
            if len(result) == 5:
                raw, _, terminated, truncated, info = result
                self._done = bool(terminated or truncated)
            else:
                raw, _, self._done, info = result
            self._score = bool(info.get("success", False))
            self._steps += 1
            self._last_action = row.copy()
            self._map_obs(raw)
        return self._latest_obs

    def is_done(self):
        return self._done or self._steps >= self.horizon

    def is_success(self):
        # Benchmark predicates are reported separately after execution.
        return False

    def episode_score(self):
        checker = getattr(self._simulator(), "_check_success", None)
        if callable(checker):
            return bool(checker())
        return self._score

    def _simulator(self):
        base = self._env.unwrapped
        seen = set()
        while hasattr(base, "env") and id(base) not in seen:
            seen.add(id(base))
            base = base.env
        return base

    def hold_action(self):
        if self._last_action is None:
            return None
        row = np.zeros(12, dtype=np.float32)
        row[6], row[11] = self._last_action[6], self._last_action[11]
        return row

    def _capture_arm_home(self):
        self._arm_home = None
        if self.task_name != "KettleBoiling":
            return
        base = self._simulator()
        robot = base.robots[0]
        pos = list(robot._ref_arm_joint_pos_indexes)
        vel = list(robot._ref_arm_joint_vel_indexes)
        for arm in robot.arms:
            pos.extend(robot._ref_gripper_joint_pos_indexes.get(arm, []))
            vel.extend(robot._ref_gripper_joint_vel_indexes.get(arm, []))
        self._arm_home = (pos, vel, base.sim.data.qpos[pos].copy())

    def after_attempt(self, subtask, decision):
        from .robocasa_config import action_kind
        from .robocasa_control import AttemptControl

        if self.task_name != "KettleBoiling" or self._arm_home is None:
            return
        if action_kind(subtask.stage_hint, subtask.prompt) != "pick":
            return
        if not (
            decision.recommended_followup == "retry"
            or decision.verdict == "incomplete"
            and not decision.recommended_followup
        ):
            return
        gap = AttemptControl.gap(self._latest_obs)
        if gap is None or 5 < gap < 20:
            return
        if self._retry_key != subtask.id:
            self._retry_key, self._retry_count = subtask.id, 0
        self._retry_count += 1
        if self._retry_count % 3:
            return
        pos, vel, values = self._arm_home
        base = self._simulator()
        base.sim.data.qpos[pos] = values
        base.sim.data.qvel[vel] = 0
        base.sim.forward()

    def begin_attempt(self, subtask):
        self._active_prompt = subtask.prompt

    def observe_attempt(self, subtask, attempt_number):
        if self.recovery is None:
            return None
        stopped = self.recovery.observe_chunk(
            task_name=self.task_name,
            subtask_id=subtask.id,
            prompt=subtask.prompt,
            attempt_number=attempt_number,
            states=self._motion_states,
            actions=self._motion_actions,
            prompts=self._motion_prompts,
        )
        if stopped:
            from mobiagent.execution.schemas import JudgeDecision

            return JudgeDecision(
                verdict="error",
                reason="Robot motion stalled during the spout action.",
                recommended_followup="replan_keep_pose",
            )
        return None

    @property
    def steps_used(self):
        return self._steps

    def get_camera_frame(self, camera="head"):
        return self._latest_obs.get(f"observation/{camera}_image")

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


class RoboCasaPolicyRegistry:
    """Route all skills through one shared VLM / six-expert policy server."""

    def __init__(self, host="127.0.0.1", port=8000, *, execute_steps=5):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(host=host, port=port)
        if not 1 <= execute_steps <= 50:
            raise ValueError("execute_steps must be between 1 and 50")
        self.execute_steps = execute_steps
        self.task_name = None

    @classmethod
    def from_yaml(cls, path):
        import yaml

        data = yaml.safe_load(path.read_text())["shared"]
        return cls(
            data.get("host", "127.0.0.1"),
            int(data.get("port", 8000)),
            execute_steps=ExecutionConfig.from_environment().execute_steps,
        )

    def select(self, skill):
        skill = STAGE_SKILLS.get(skill, skill)
        if skill not in SKILLS:
            raise ValueError(f"Unknown RoboCasa skill: {skill}")
        return self

    def request_chunk(self, *, obs, prompt, stage_hint, timeout_s=None):
        from .robocasa_prompt_canon import canonicalize

        prompt, matched = canonicalize(prompt, stage_hint)
        if (
            stage_hint == "open_close"
            and "manipulate" in matched
            and "switch" not in matched
        ):
            stage_hint = "manipulate"
        execute_steps = self.execute_steps
        if self.task_name == "KettleBoiling":
            from .robocasa_config import action_kind
            from .robocasa_control import AttemptControl

            gap = AttemptControl.gap(obs)
            if (
                action_kind(stage_hint, prompt) == "place"
                and gap is not None
                and gap >= 25
            ):
                execute_steps = min(execute_steps, 3)
        stage_hint = STAGE_SKILLS.get(stage_hint, stage_hint)
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        if state.shape != (16,):
            raise ValueError("Expected the 16-dimensional RoboCasa state")
        payload = {
            "state": np.pad(state, (0, 16)),
            "prompt": prompt,
            "skill_canonical_ids": np.int32(SKILLS.index(stage_hint)),
            "image": {
                "base_0_rgb": obs["observation/head_image"],
                "left_wrist_0_rgb": obs["observation/left_wrist_image"],
                "right_wrist_0_rgb": obs["observation/right_wrist_image"],
            },
            "image_mask": {
                k: True for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            },
        }
        prediction = self.client.infer(payload)
        actions = np.asarray(prediction["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < 12 or not np.isfinite(actions).all():
            raise ValueError("Expected finite RoboCasa policy actions")
        return {"actions": actions[:execute_steps, :12]}

    def reset_inpaint_state(self):
        pass

    def close(self):
        close = getattr(self.client, "close", None)
        if close:
            close()
