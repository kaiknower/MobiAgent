import numpy as np
from mobiagent.environments.robocasa import RoboCasaEnv, RoboCasaPolicyRegistry, SKILLS


def test_shared_expert_contract():
    class Client:
        def infer(self, payload):
            assert payload['skill_canonical_ids'] == SKILLS.index('pnp')
            assert payload['state'].shape == (32,)
            assert set(payload['image']) == {'base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'}
            return {'actions': np.zeros((50, 32), dtype=np.float32)}
    registry = object.__new__(RoboCasaPolicyRegistry)
    registry.client = Client()
    obs = {'observation/state': np.ones(16, dtype=np.float32)}
    for name in ('head', 'left_wrist', 'right_wrist'):
        obs[f'observation/{name}_image'] = np.zeros((224,224,3),dtype=np.uint8)
    chunk = registry.request_chunk(obs=obs, prompt='pick up cup', stage_hint='pnp')
    assert chunk['actions'].shape == (50, 12)


def test_simulator_score_is_separate_from_agent_control():
    env = RoboCasaEnv('test')
    env._score = True
    assert env.episode_score() is True
    assert env.is_success() is False
    assert env.is_done() is False
