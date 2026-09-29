"""Look up the natural-language instruction for a s1_mobile task.

A task_id (e.g. "trash-bottle") maps to one or more dataset_ids, each with its
own tasks.jsonl (one row per task_index). For s1_mobile the per-dataset_id
tasks.jsonl currently always contains a single row whose instruction is shared
across all dataset_ids backing the same task, so reading the first dataset_id
is sufficient. If a future dataset_id ever has multiple task_indexes or
differing instructions, this function will raise.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import (
    DEFAULT_DATASET_FAMILY_DIRNAME,
    LEROBOT_SUBDIR,
    META_DIRNAME,
    TASK_TO_DATASET_IDS,
    TASK_TO_FAMILY,
)


def load_task_instruction(*, dataset_root: Path, task_id: str) -> str:
    try:
        dataset_ids = TASK_TO_DATASET_IDS[task_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown task_id {task_id!r}; expected one of {sorted(TASK_TO_DATASET_IDS)}"
        ) from exc

    family = TASK_TO_FAMILY.get(task_id, DEFAULT_DATASET_FAMILY_DIRNAME)
    instructions: set[str] = set()
    for dataset_id in dataset_ids:
        tasks_path = (
            dataset_root
            / family
            / dataset_id
            / LEROBOT_SUBDIR
            / META_DIRNAME
            / "tasks.jsonl"
        )
        if not tasks_path.exists():
            continue
        with tasks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("task")
                if isinstance(text, str) and text.strip():
                    instructions.add(text.strip())

    if not instructions:
        raise ValueError(f"No task instructions found for task_id={task_id!r}")
    if len(instructions) > 1:
        raise ValueError(
            f"Conflicting instructions across dataset_ids for task_id={task_id!r}: "
            f"{sorted(instructions)}"
        )
    return next(iter(instructions))


__all__ = ["load_task_instruction"]
