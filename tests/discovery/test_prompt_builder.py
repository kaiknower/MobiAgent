from mobiagent.discovery.behavior.prompt_builder import build_timeline_prompt


def test_build_timeline_prompt_mentions_state_semantics() -> None:
    state_payload = {
        "task_info_full": [0.0, 1.5],
        "sampled_state_rows": [
            {
                "base_vel": [1.0, 2.0, 3.0],
                "gripper_left": 0.3,
            }
        ],
    }
    prompt = build_timeline_prompt(
        task_name="picking_up_trash",
        merged_video_description="left wrist | head | right wrist",
        meta_summary={"scene_instance": "scene"},
        state_schema_text="base_vel: robot base velocity",
        state_payload=state_payload,
    )
    assert "left wrist | head | right wrist" in prompt["user_text"]
    assert "task_info_full" in prompt["user_text"]
    assert "scene_instance" in prompt["user_text"] or "meta_summary" in prompt["user_text"]
    assert '"task_info_full": [\n    0.0,\n    1.5\n  ]' in prompt["user_text"]
    assert '"sampled_state_rows": [\n    {\n      "base_vel": [\n        1.0,\n        2.0,\n        3.0\n      ],\n      "gripper_left": 0.3\n    }\n  ]' in prompt["user_text"]
    assert "base_vel: robot base velocity" in prompt["user_text"]
    assert "skill_timeline" in prompt["user_text"]
    assert "Return valid JSON only." in prompt["user_text"]
    assert "Do not normalize skill names early" in prompt["user_text"]
