from __future__ import annotations

import json
from pathlib import Path


def load_task_instruction(*, dataset_root: Path, task_id: str) -> str:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_index = row.get("task_index")
            if row.get("task_id") == task_id or (
                isinstance(task_index, int) and f"task-{task_index:04d}" == task_id
            ):
                instruction = row.get("task")
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError(f"Missing task instruction for {task_id}")
                return instruction
    raise ValueError(f"Unable to find task instruction for {task_id}")


__all__ = ["load_task_instruction"]
