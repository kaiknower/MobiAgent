from pathlib import Path

from mobiagent.discovery.behavior.review_io import finalize_timelines
from mobiagent.discovery.behavior.review_io import build_review_rows
from mobiagent.discovery.behavior.review_io import build_timeline_review_text
from mobiagent.discovery.behavior.review_io import write_finalized_timelines
from mobiagent.discovery.behavior.review_io import write_timeline_review


def test_build_review_rows_requires_skill_timeline() -> None:
    try:
        build_review_rows([{"episode_id": "episode_00010010"}])
    except ValueError as exc:
        assert str(exc) == "prediction['skill_timeline'] is required"
    else:
        raise AssertionError("Expected ValueError for missing skill_timeline")


def test_build_review_rows_preserves_task_id() -> None:
    rows = build_review_rows(
        [
            {
                "task_id": "task-0001",
                "episode_id": "episode_00010010",
                "skill_timeline": [{"segment_id": "seg_0"}],
            }
        ]
    )

    assert rows[0]["task_id"] == "task-0001"
    assert rows[0]["episode_id"] == "episode_00010010"


def test_build_timeline_review_text_formats_task_episode_and_times() -> None:
    text = build_timeline_review_text(
        [
            {
                "task_id": "task-0001",
                "episode_id": "episode_00010010",
                "final_skill_timeline": [
                    {
                        "skill_description": "move to target",
                        "start_time_sec": 0,
                        "end_time_sec": 1.234,
                    }
                ],
            }
        ]
    )

    assert "task-0001 / episode_00010010" in text
    assert "- move to target [0.00s - 1.23s]" in text


def test_write_timeline_review_writes_text_file(tmp_path: Path) -> None:
    output_path = tmp_path / "review" / "timeline_review.txt"

    write_timeline_review(
        output_path,
        [
            {
                "task_id": "task-0001",
                "episode_id": "episode_00010010",
                "final_skill_timeline": [
                    {
                        "skill_description": "pick up object",
                        "start_time_sec": 2.0,
                        "end_time_sec": 3.0,
                    }
                ],
            }
        ],
    )

    assert output_path.read_text(encoding="utf-8") == (
        "task-0001 / episode_00010010\n"
        "- pick up object [2.00s - 3.00s]\n"
    )


def test_finalize_timelines_prefers_reviewed_timeline_when_present(tmp_path: Path) -> None:
    review_path = tmp_path / "review.jsonl"
    review_path.write_text(
        '{"episode_id":"episode_00010010","auto_skill_timeline":[{"segment_id":"seg_0"}],"final_skill_timeline":[{"segment_id":"seg_human"}]}\n',
        encoding="utf-8",
    )
    rows = finalize_timelines(review_path)
    assert set(rows[0].keys()) == {"episode_id", "final_skill_timeline"}
    assert rows[0]["final_skill_timeline"][0]["segment_id"] == "seg_human"
    assert "auto_skill_timeline" not in rows[0]


def test_finalize_timelines_falls_back_to_auto_timeline_when_final_missing(tmp_path: Path) -> None:
    review_path = tmp_path / "review.jsonl"
    review_path.write_text(
        '{"episode_id":"episode_00010011","auto_skill_timeline":[{"segment_id":"seg_auto"}]}\n',
        encoding="utf-8",
    )

    rows = finalize_timelines(review_path)

    assert set(rows[0].keys()) == {"episode_id", "final_skill_timeline"}
    assert rows[0]["final_skill_timeline"][0]["segment_id"] == "seg_auto"
    assert "auto_skill_timeline" not in rows[0]


def test_finalize_timelines_preserves_empty_final_timeline_override(tmp_path: Path) -> None:
    review_path = tmp_path / "review.jsonl"
    review_path.write_text(
        '{"episode_id":"episode_00010012","auto_skill_timeline":[{"segment_id":"seg_auto"}],"final_skill_timeline":[]}\n',
        encoding="utf-8",
    )

    rows = finalize_timelines(review_path)

    assert set(rows[0].keys()) == {"episode_id", "final_skill_timeline"}
    assert rows[0]["final_skill_timeline"] == []
    assert "auto_skill_timeline" not in rows[0]


def test_write_finalized_timelines_rewrites_final_file_after_review_edit(tmp_path: Path) -> None:
    review_path = tmp_path / "review.jsonl"
    final_path = tmp_path / "final.jsonl"
    review_path.write_text(
        '{"episode_id":"episode_00010013","auto_skill_timeline":[{"segment_id":"seg_auto"}]}\n',
        encoding="utf-8",
    )

    first_rows = write_finalized_timelines(review_path, final_path)
    assert first_rows == [
        {
            "episode_id": "episode_00010013",
            "final_skill_timeline": [{"segment_id": "seg_auto"}],
        }
    ]
    assert final_path.read_text(encoding="utf-8") == (
        '{"episode_id":"episode_00010013","final_skill_timeline":[{"segment_id":"seg_auto"}]}\n'
    )

    review_path.write_text(
        '{"episode_id":"episode_00010013","auto_skill_timeline":[{"segment_id":"seg_auto"}],"final_skill_timeline":[{"segment_id":"seg_human"}]}\n',
        encoding="utf-8",
    )

    second_rows = write_finalized_timelines(review_path, final_path)
    assert second_rows == [
        {
            "episode_id": "episode_00010013",
            "final_skill_timeline": [{"segment_id": "seg_human"}],
        }
    ]
    assert final_path.read_text(encoding="utf-8") == (
        '{"episode_id":"episode_00010013","final_skill_timeline":[{"segment_id":"seg_human"}]}\n'
    )
