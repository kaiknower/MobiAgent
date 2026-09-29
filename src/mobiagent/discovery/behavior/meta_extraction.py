from __future__ import annotations

import json
from pathlib import Path


def extract_meta_summary(meta_path: Path) -> dict:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    config = json.loads(payload["config"])
    return {
        "task_name": config["task"]["activity_name"],
        "scene_instance": config["scene"]["scene_instance"],
        "n_steps": int(payload.get("n_steps", 0)),
    }
