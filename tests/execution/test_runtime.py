import numpy as np
import pytest
from mobiagent.execution.env_mock import MockEnv
from mobiagent.execution.orchestrator import Orchestrator
from mobiagent.execution.policy_registry import PolicyRegistry
from mobiagent.execution.policy_client import build_policy_observation
from mobiagent.execution.schemas import JudgeDecision, Subtask


def run_loop(tmp_path, monkeypatch, verdicts, max_ticks=3):
    monkeypatch.setenv('CLAW_CHUNKS_PER_ATTEMPT', '1')
    env = MockEnv(success_after_steps=100)
    planner_calls = []
    answers = iter(verdicts)
    def planner(**kwargs):
        planner_calls.append(kwargs['history'])
        return Subtask(id='step', prompt='move to object', stage_hint='move_to',
                       success_check='object reachable', max_retries=1)
    def judge(**kwargs):
        return JudgeDecision(verdict=next(answers), reason='visible evidence')
    controller = Orchestrator(env=env, global_goal='put object away', sim_task_name='test',
        registry=PolicyRegistry.mock(), planner_fn=planner, judge_fn=judge,
        run_dir=tmp_path, max_ticks=max_ticks)
    result = controller.run(env.reset('test'))
    return result, planner_calls, controller


def test_retry_then_advance(tmp_path, monkeypatch):
    result, calls, controller = run_loop(tmp_path, monkeypatch, ['incomplete', 'complete', 'complete'])
    assert result['total_ticks'] == 3
    assert result['subtasks_completed'] == 2
    assert len(calls) == 2
    assert len(calls[1]) == 1
    assert calls[1][0]['n_attempts'] == 2


def test_failed_subtask_replans_from_observed_state(tmp_path, monkeypatch):
    result, calls, controller = run_loop(tmp_path, monkeypatch, ['error', 'complete', 'complete'])
    assert result['subtasks_failed'] == 1
    assert len(calls) == 3
    assert calls[1][0]['outcome'] == 'failed'
    assert 'visible evidence' in calls[1][0]['judge_reason']
    assert not hasattr(controller, '_restore_fn')


def test_router_preserves_planner_choice(monkeypatch):
    # A stale experiment variable must not silently override the chosen skill.
    monkeypatch.setenv('CLAW_FORCE_STAGE', 'close')
    observation = MockEnv().reset('test')
    payload = build_policy_observation(observation, prompt='pick up object', stage_hint='pick_up_from')
    assert payload['stage_hint'] == payload['stage_override'] == 1
    with pytest.raises(ValueError):
        build_policy_observation(observation, prompt='bad', stage_hint='unknown')


def test_s1_router_and_chassis_state(monkeypatch):
    from mobiagent.robots.s1.policy_client import build_policy_observation as build_s1
    monkeypatch.setenv('S1_FORCE_STAGE', 'move_to')
    obs = MockEnv().reset('test')
    obs['observation/state'] = np.ones(34, dtype=np.float32)
    result = build_s1(obs, prompt='place object', stage_hint='place')
    assert result['stage_hint'] == 2
    np.testing.assert_array_equal(result['observation/state'][31:34], 0)


def test_mock_cli_completes_ticks(tmp_path, monkeypatch):
    import sys
    from mobiagent.execution.run import main
    monkeypatch.setenv('CLAW_CHUNKS_PER_ATTEMPT', '1')
    monkeypatch.setattr(sys, 'argv', ['mobiagent', '--task', 'example', '--instruction', 'move object',
        '--mock-all', '--max-ticks', '3', '--run-dir', str(tmp_path)])
    assert main() == 0
    import json
    result = json.loads((tmp_path/'summary.json').read_text())
    assert result['total_ticks'] == 3
    assert result['subtasks_completed'] == 3
