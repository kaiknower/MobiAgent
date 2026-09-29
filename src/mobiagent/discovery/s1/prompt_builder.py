from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)


def build_timeline_prompt(
    *,
    task_name: str,
    merged_video_description: str,
    meta_summary: Mapping[str, Any],
    state_schema_text: str,
    state_payload: Mapping[str, Any],
) -> dict[str, str]:
    system_text = (
        "You are Claw's offline skill discovery assistant.\n"
        "Extract a time-ordered skill timeline from the provided demo inputs and return strict JSON."
    )
    user_text = "\n".join(
        [
            f"Task name: {task_name}",
            f"Merged video layout: {merged_video_description}",
            "The payload includes `task_info_full`; preserve it as provided.",
            "State schema:",
            state_schema_text,
            "Do not normalize skill names early. Keep the original wording until the final `skill_timeline` JSON output.",
            "Required JSON output keys: task_name, meta_summary, state_payload, skill_timeline.",
            "Return valid JSON only.",
            "meta_summary:",
            _json_text(meta_summary),
            "state_payload:",
            _json_text(state_payload),
        ]
    )
    return {
        "system_text": system_text,
        "user_text": user_text,
    }
