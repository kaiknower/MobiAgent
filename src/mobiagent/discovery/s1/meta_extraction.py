"""Read s1_mobile dataset metadata.

Each dataset_id has:
- meta/info.json       — schema (features, fps, total_episodes, robot_type)
- meta/episodes.jsonl  — one row per episode (episode_index, tasks, length,
                         is_episode_success)
- meta/tasks.jsonl     — one row per task_index (task_index, task instruction)
"""

from __future__ import annotations

import json
from pathlib import Path


def _read_jsonl_row(jsonl_path: Path, predicate) -> dict | None:
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if predicate(row):
                return row
    return None


def extract_meta_summary(meta_path: Path, episode_index: int | None = None) -> dict:
    """Summarize s1_mobile metadata.

    Arguments:
        meta_path: either a path to info.json, episodes.jsonl, or the meta
                   directory itself; the function resolves to the meta dir.
        episode_index: when provided, include the matching episodes.jsonl row.
    """
    meta_dir = meta_path if meta_path.is_dir() else meta_path.parent
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"

    info = json.loads(info_path.read_text(encoding="utf-8"))

    summary: dict = {
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "total_videos": info.get("total_videos"),
        "feature_keys": list(info.get("features", {}).keys()),
    }

    if tasks_path.exists():
        instructions: list[dict] = []
        with tasks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    instructions.append(json.loads(line))
        summary["tasks"] = instructions

    if episode_index is not None and episodes_path.exists():
        episode_row = _read_jsonl_row(
            episodes_path, lambda row: row.get("episode_index") == episode_index
        )
        if episode_row is None:
            raise ValueError(
                f"episode_index={episode_index} not found in {episodes_path}"
            )
        summary["episode"] = {
            "episode_index": episode_row.get("episode_index"),
            "tasks": episode_row.get("tasks"),
            "length": episode_row.get("length"),
            "is_episode_success": episode_row.get("is_episode_success"),
        }

    return summary
