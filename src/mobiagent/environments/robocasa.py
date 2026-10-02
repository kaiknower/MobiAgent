"""RoboCasa camera, state and action adapter for simulation execution."""
from __future__ import annotations
import random
import numpy as np

STATE_KEYS = ('state.end_effector_position_relative', 'state.end_effector_rotation_relative',
              'state.base_position', 'state.base_rotation', 'state.gripper_qpos')
SKILLS = ('close', 'open', 'switch', 'manipulate', 'navigate', 'pnp')


class RoboCasaEnv:
    def __init__(self, task_name: str, *, seed: int = 7, split: str = 'target', horizon: int = 3000):
        self.task_name, self.seed, self.split, self.horizon = task_name, seed, split, horizon
        self._env = None
        self._latest_obs = {}
        self._steps = 0
        self._done = False
        self._score = False

    def reset(self, task: str | None = None):
        import gymnasium as gym
        import robocasa  # Registers the simulator's Gym environments.
        self.close()
        self.task_name = task or self.task_name
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._env = gym.make(f'robocasa/{self.task_name}', split=self.split, seed=self.seed)
        value = self._env.reset(seed=self.seed)
        self._steps, self._done, self._score = 0, False, False
        return self._map_obs(value[0] if isinstance(value, tuple) else value)

    def _map_obs(self, raw):
        from openpi_client import image_tools
        obs = {'observation/state': np.concatenate([np.asarray(raw[k], dtype=np.float32).ravel() for k in STATE_KEYS])}
        for name, key in [('head', 'video.robot0_agentview_left'),
                          ('left_wrist', 'video.robot0_eye_in_hand'),
                          ('right_wrist', 'video.robot0_agentview_right')]:
            image = np.asarray(raw[key])
            obs[f'observation/{name}_image_orig'] = image
            obs[f'observation/{name}_image'] = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
        self._latest_obs = obs
        return obs

    def step(self, chunk):
        from robocasa.utils.env_utils import convert_action
        actions = np.asarray(chunk['actions'] if isinstance(chunk, dict) else chunk, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 12 or not np.isfinite(actions).all():
            raise ValueError('Expected a finite (T, 12) RoboCasa action chunk')
        for row in actions:
            if self.is_done(): break
            result = self._env.step(convert_action(row.astype(np.float64)))
            if len(result) == 5:
                raw, _, terminated, truncated, info = result
                self._done = bool(terminated or truncated)
            else:
                raw, _, self._done, info = result
            self._score |= bool(info.get('success', False))
            self._steps += 1
            self._map_obs(raw)
        return self._latest_obs

    def is_done(self):
        return self._done or self._steps >= self.horizon

    def is_success(self):
        # Benchmark predicates are reported separately after execution.
        return False

    def episode_score(self):
        return self._score

    def get_camera_frame(self, camera='head'):
        return self._latest_obs.get(f'observation/{camera}_image')

    def close(self):
        if self._env is not None:
            self._env.close(); self._env = None


class RoboCasaPolicyRegistry:
    """Route all skills through one shared VLM / six-expert policy server."""
    def __init__(self, host='127.0.0.1', port=8000):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.client = WebsocketClientPolicy(host=host, port=port)

    @classmethod
    def from_yaml(cls, path):
        import yaml
        data = yaml.safe_load(path.read_text())['shared']
        return cls(data.get('host', '127.0.0.1'), int(data.get('port', 8000)))

    def select(self, skill):
        if skill not in SKILLS: raise ValueError(f'Unknown RoboCasa skill: {skill}')
        return self

    def request_chunk(self, *, obs, prompt, stage_hint, timeout_s=None):
        state = np.asarray(obs['observation/state'], dtype=np.float32)
        if state.shape != (16,): raise ValueError('Expected the 16-dimensional RoboCasa state')
        payload = {'state': np.pad(state, (0, 16)), 'prompt': prompt,
                   'skill_canonical_ids': np.int32(SKILLS.index(stage_hint)),
                   'image': {'base_0_rgb': obs['observation/head_image'],
                             'left_wrist_0_rgb': obs['observation/left_wrist_image'],
                             'right_wrist_0_rgb': obs['observation/right_wrist_image']},
                   'image_mask': {k: True for k in ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')}}
        prediction = self.client.infer(payload)
        return {'actions': np.asarray(prediction['actions'], dtype=np.float32)[:, :12]}

    def reset_inpaint_state(self):
        pass

    def close(self):
        close = getattr(self.client, 'close', None)
        if close: close()
