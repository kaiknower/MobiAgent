from types import SimpleNamespace
import sys

import numpy as np
import pytest

from mobiagent.environments.robocasa import RoboCasaEnv, RoboCasaPolicyRegistry
from mobiagent.environments.robocasa_brain import RoboCasaBrain, composite
from mobiagent.environments.robocasa_config import ExecutionConfig, action_kind
from mobiagent.environments.robocasa_control import AttemptControl, robot_info
from mobiagent.environments.robocasa_prompt_canon import canonicalize
from mobiagent.environments.robocasa_recovery import RinseRecovery
from mobiagent.execution.schemas import JudgeDecision, Subtask


def observation(gap):
    state = np.zeros(16, dtype=np.float32)
    state[14:16] = (gap / 2000, -gap / 2000)
    return {"observation/state": state}


def subtask(stage="pick_up_from", prompt="Pick up the kettle from the counter."):
    return Subtask(
        id="action", prompt=prompt, success_check="vlm_judge", stage_hint=stage
    )


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_scale(scale):
    with pytest.raises(ValueError):
        ExecutionConfig(horizon_scale=scale)


def test_budget_override(monkeypatch):
    monkeypatch.setenv("CLAW_HORIZON_SCALE", "1.5")
    monkeypatch.setenv("CLAW_CHUNKS_BY_KIND", "open=20")
    config = ExecutionConfig.from_environment()
    assert config.horizon_scale == 1.5
    assert config.window("open", "Open the drawer.") == 20
    assert config.window("pick_up_from", "Pick up the straw from the drawer.") == 50
    assert config.retry_budget("move_to", "Navigate to the sink.", 50) == 1


def test_adaptive_place_budget():
    config = ExecutionConfig()
    assert config.effective_steps("KettleBoiling", "place_on", observation(30)) == 3
    assert config.effective_steps("KettleBoiling", "place_on", observation(10)) == 5
    assert config.effective_steps("OtherTask", "place_on", observation(30)) == 5
    assert (
        action_kind(
            "open_close", "Move the spout to wash all locations of the sink basin."
        )
        == "action"
    )


def test_empty_jaw_does_not_freeze():
    control = AttemptControl(ExecutionConfig(), subtask(), observation(80))
    assert all(control.observe(observation(0)) is None for _ in range(10))


def test_pick_requires_stable_contact():
    control = AttemptControl(ExecutionConfig(), subtask(), observation(80))
    assert control.observe(observation(40)) is None
    assert control.observe(observation(20)) is None
    for _ in range(3):
        assert control.observe(observation(20)) is None
    assert control.observe(observation(20)) == "hold"
    assert control.pick_handoff["confirmations"] == 5


def test_release_requests_visual_confirmation():
    control = AttemptControl(
        ExecutionConfig(),
        subtask("place_on", "Place the kettle on the stove burner."),
        observation(20),
    )
    for _ in range(4):
        assert control.observe(observation(60)) is None
    assert control.observe(observation(60)) == "visual"
    assert control.observe(observation(60)) is None


def test_release_guard_does_not_assert_success():
    control = AttemptControl(
        ExecutionConfig(),
        subtask("manipulate", "Stir the vegetables."),
        observation(20),
    )
    for _ in range(11):
        assert control.observe(observation(20)) is None
    assert control.observe(observation(26)) == "release"


def test_hold_preserves_gripper_and_mode():
    env = RoboCasaEnv("OtherTask", horizon=10)
    env._last_action = np.arange(12, dtype=np.float32)
    hold = env.hold_action()
    assert hold[6] == 6 and hold[11] == 11
    assert np.count_nonzero(hold) == 2


def test_terminal_score_is_separate_from_control():
    env = RoboCasaEnv("OtherTask", horizon=10)
    core = SimpleNamespace(_check_success=lambda: True)
    env._env = SimpleNamespace(unwrapped=SimpleNamespace(env=core))
    assert env.is_success() is False
    assert env.episode_score() is True


def test_robot_info_uses_single_arm_state():
    info = robot_info(observation(20), observation(40))
    assert info["jaw_gap_mm"] == pytest.approx(20)
    assert info["delta_jaw_gap_mm"] == pytest.approx(-20)
    assert "carry_item" not in info and "carry_hand" not in info


def test_policy_routes_and_executes_prefix():
    registry = RoboCasaPolicyRegistry.__new__(RoboCasaPolicyRegistry)
    registry.execute_steps, registry.task_name = 5, "KettleBoiling"
    requests = []
    registry.client = SimpleNamespace(
        infer=lambda payload: requests.append(payload)
        or {"actions": np.zeros((50, 32))}
    )
    chunk = registry.request_chunk(
        obs={
            **observation(30),
            "observation/head_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/left_wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/right_wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        prompt="Place the kettle on the stove burner.",
        stage_hint="place_on",
    )
    assert chunk["actions"].shape == (3, 12)
    assert int(requests[0]["skill_canonical_ids"]) == 5
    assert requests[0]["prompt"].endswith(".")
    assert registry.select("move_to") is registry


def test_instruction_assets_are_available():
    prompt, skills = canonicalize("navigate to sink (2/3)", "move_to")
    assert prompt == "Navigate to the sink."
    assert "navigate" in skills


def test_task_prompts_and_composite():
    env = SimpleNamespace(task_name="KettleBoiling")
    brain = RoboCasaBrain(env)
    assert "PandaOmron" in brain.planner_prompt
    image = composite(
        {"observation/head_image": np.full((224, 224, 3), 100, dtype=np.uint8)},
        "Current",
    )
    assert image.shape == (540, 512, 3)
    assert image[100, 100].tolist() == [100, 100, 100]


def test_rinse_recovery_only_once():
    recovery = RinseRecovery()
    prompt = "Move the spout to wash all locations of the sink basin."
    st = subtask("manipulate", prompt)
    kwargs = dict(
        task_name="RinseSinkBasin",
        subtask_id=st.id,
        prompt=prompt,
        attempt_number=1,
        states=np.zeros((60, 16)),
        actions=np.ones((60, 12)),
        prompts=[prompt] * 60,
    )
    assert recovery.observe_chunk(**kwargs)
    assert not recovery.observe_chunk(**kwargs)
    assert recovery.planner_intent([{"subtask": st, "outcome": "failed"}]) == "navigate"
    nav = subtask("move_to", "Navigate to the sink.")
    assert recovery.planner_intent([{"subtask": nav, "outcome": "complete"}]) == "spout"
    assert recovery.planner_intent([{"subtask": nav, "outcome": "complete"}]) is None


def test_pick_reset_preserves_base_and_loaded_gripper():
    env = RoboCasaEnv("KettleBoiling", horizon=10)
    data = SimpleNamespace(qpos=np.arange(10, dtype=float), qvel=np.ones(10))
    sim = SimpleNamespace(data=data, forward=lambda: None)
    env._env = SimpleNamespace(unwrapped=SimpleNamespace(sim=sim))
    env._arm_home = ([2, 3], [2, 3], np.array([50, 60]))
    env._latest_obs = observation(10)
    st, decision = subtask(), JudgeDecision(verdict="incomplete", reason="")
    for _ in range(3):
        env.after_attempt(st, decision)
    assert data.qpos[2] == 2
    env._latest_obs = observation(0)
    for _ in range(3):
        env.after_attempt(st, decision)
    assert data.qpos[[2, 3]].tolist() == [50, 60]
    assert data.qpos[0] == 0 and data.qvel[0] == 1


@pytest.mark.parametrize(
    "explicit,scale,expected", [(None, "2", 3000), (None, "1", 1500), (100, "2", 100)]
)
def test_reset_uses_task_budget(monkeypatch, explicit, scale, expected):
    monkeypatch.setenv("CLAW_HORIZON_SCALE", scale)
    raw = {
        key: np.zeros(size)
        for key, size in zip(
            (
                "state.end_effector_position_relative",
                "state.end_effector_rotation_relative",
                "state.base_position",
                "state.base_rotation",
                "state.gripper_qpos",
            ),
            (3, 4, 3, 4, 2),
        )
    }
    for key in (
        "video.robot0_agentview_left",
        "video.robot0_eye_in_hand",
        "video.robot0_agentview_right",
    ):
        raw[key] = np.zeros((224, 224, 3), dtype=np.uint8)
    simulator = SimpleNamespace(reset=lambda seed: (raw, {}), close=lambda: None)
    simulator.unwrapped = simulator
    monkeypatch.setitem(
        sys.modules,
        "gymnasium",
        SimpleNamespace(make=lambda *args, **kwargs: simulator),
    )
    monkeypatch.setitem(sys.modules, "robocasa", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "robocasa.utils", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "robocasa.utils.dataset_registry_utils",
        SimpleNamespace(get_task_horizon=lambda task: 1500),
    )
    monkeypatch.setitem(
        sys.modules,
        "openpi_client",
        SimpleNamespace(
            image_tools=SimpleNamespace(
                resize_with_pad=lambda image, *args: image,
                convert_to_uint8=lambda image: image,
            )
        ),
    )
    env = RoboCasaEnv("OtherTask", horizon=explicit)
    env.reset()
    assert env.horizon == expected
    assert env.steps_used == 0


def test_orchestrator_stops_at_step_budget(tmp_path):
    from mobiagent.execution.orchestrator import Orchestrator

    class Environment:
        task_name = "OtherTask"
        horizon = 13
        steps_used = 0
        config = ExecutionConfig(windows={"pick": 2})

        def is_done(self):
            return self.steps_used >= self.horizon

        def is_success(self):
            return False

        def step(self, chunk):
            self.steps_used += min(
                len(chunk["actions"]), self.horizon - self.steps_used
            )
            return observation(0)

    requests = []
    client = SimpleNamespace(
        request_chunk=lambda **kwargs: requests.append(kwargs)
        or {"actions": np.zeros((5, 12))}
    )
    registry = SimpleNamespace(
        select=lambda stage: client, reset_inpaint_state=lambda: None
    )
    env = Environment()
    orch = Orchestrator(
        env=env,
        global_goal="Pick up the object.",
        sim_task_name=env.task_name,
        registry=registry,
        planner_fn=lambda **kwargs: subtask(),
        judge_fn=lambda **kwargs: JudgeDecision(verdict="complete", reason="visible"),
        run_dir=tmp_path,
        max_ticks=100,
    )
    result = orch.run(observation(0))
    assert env.steps_used == 13
    assert len(requests) == 3
    assert result["abort_reason"] == "horizon_exhausted"


def test_all_task_resources_load():
    from mobiagent.environments.robocasa_brain import TASK_PROFILES

    for task in (*TASK_PROFILES, "DeliverStraw"):
        brain = RoboCasaBrain(SimpleNamespace(task_name=task))
        assert brain.planner_prompt and brain.judge_prompt


def test_planner_rejects_empty_gripper_handoff(monkeypatch):
    from mobiagent.environments import robocasa_brain

    env = SimpleNamespace(
        task_name="KettleBoiling", _latest_obs=observation(0), recovery=None
    )
    brain = RoboCasaBrain(env)
    prior = subtask()
    monkeypatch.setattr(
        robocasa_brain,
        "chat_completion_json",
        lambda **kwargs: {
            "prompt": "Place the kettle on the stove burner.",
            "stage_hint": "place_on",
            "max_retries": 2,
        },
    )
    result = brain.plan(
        global_goal="Boil water.",
        sim_task_name="KettleBoiling",
        history=[{"subtask": prior, "outcome": "failed"}],
        head_image=None,
    )
    assert result.stage_hint == "pick_up_from"
    assert result.prompt == prior.prompt
