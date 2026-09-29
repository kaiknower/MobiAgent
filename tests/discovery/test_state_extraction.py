import numpy as np
import pytest

from mobiagent.discovery.behavior.state_extraction import STATE_SCHEMA, summarize_state_row


def test_state_schema_contains_semantic_slice_names() -> None:
    assert STATE_SCHEMA["base_vel"] == (253, 256)
    assert STATE_SCHEMA["trunk_qpos"] == (236, 240)
    assert STATE_SCHEMA["arm_left_qpos"] == (158, 165)


def test_summarize_state_row_extracts_expected_fields() -> None:
    state = np.zeros(256, dtype=np.float32)
    task_info = np.arange(82, dtype=np.float32)
    state[253:256] = [1.0, 2.0, 3.0]
    state[236:240] = [4.0, 5.0, 6.0, 7.0]
    state[193:195] = [0.1, 0.2]
    summary = summarize_state_row(state=state, task_info=task_info)
    assert summary["base_vel"] == [1.0, 2.0, 3.0]
    assert summary["trunk_qpos"] == [4.0, 5.0, 6.0, 7.0]
    assert summary["gripper_left"] == pytest.approx(0.3)
    assert summary["task_info_full"][0] == 0.0


@pytest.mark.parametrize(
    ("state", "task_info", "match"),
    [
        (np.zeros(255, dtype=np.float32), np.arange(82, dtype=np.float32), r"state must have at least 256 elements"),
        (np.zeros(256, dtype=np.float32), np.arange(81, dtype=np.float32), r"task_info must have at least 82 elements"),
    ],
)
def test_summarize_state_row_rejects_short_or_malformed_inputs(
    state: np.ndarray, task_info: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        summarize_state_row(state=state, task_info=task_info)
