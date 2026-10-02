"""Visual reflection critic shared by the clean model loop."""
from __future__ import annotations
import json
import os
from typing import Any
from .llm_client import chat_completion_json, make_user_content
from .schemas import Attempt, DynamicPlan, JudgeDecision, Subtask


def judge(*, subtask: Subtask, obs_after: dict[str, Any],
          obs_prev: dict[str, Any] | None = None,
          obs_mid: dict[str, Any] | None = None,
          robot_info: dict[str, Any] | None = None,
          history: list[Attempt], plan: DynamicPlan, current_idx: int,
          model: str | None = None) -> JudgeDecision:
    """Judge visible evidence without task-specific success rules or simulator probes."""
    images = []
    labels = []
    for label, obs in [('two attempts ago', obs_mid), ('previous attempt', obs_prev), ('current', obs_after)]:
        if obs is None:
            continue
        for camera in ('head', 'left_wrist', 'right_wrist'):
            for key in (f'observation/{camera}_image_orig', f'{camera}_image_orig',
                        f'observation/{camera}_image', f'{camera}_image'):
                if obs.get(key) is not None:
                    images.append(obs[key]); labels.append(f'{label}: {camera}'); break
    text = json.dumps({
        'global_goal': plan.global_goal,
        'subtask': subtask.prompt,
        'skill': subtask.stage_hint,
        'success_check': subtask.success_check,
        'failure_cues': subtask.failure_cues,
        'image_order': labels,
        'robot_proprioception': robot_info or {},
        'attempts_so_far': len(history),
    }, ensure_ascii=False, default=str)
    result = chat_completion_json(
        system_text=(
            'You are a visual reflection critic for mobile manipulation. Compare the current '
            'camera observations with earlier observations and the requested subtask. '
            'Return complete only when visible evidence supports the requested outcome. '
            'Use incomplete when progress or the outcome is uncertain, and error when the '
            'observed state calls for replanning. Do not infer success from elapsed time, '
            'a task name, or an earlier verdict. Return one JSON object with verdict '
            '(complete/incomplete/error), reason, evidence (a list of visible cues), '
            'and recommended_followup (next/retry/replan_keep_pose).'),
        user_content=make_user_content(text=text, images=images or None),
        model=model or os.getenv('MOBIAGENT_JUDGE_MODEL') or os.getenv('OPENAI_MODEL'),
        max_completion_tokens=4096,
    )
    verdict = str(result.get('verdict', '')).lower().strip()
    if verdict not in {'complete', 'incomplete', 'error'}:
        verdict = 'incomplete'
    followup = str(result.get('recommended_followup', '')).lower().strip()
    if followup not in {'next', 'retry', 'replan_keep_pose', ''}:
        followup = ''
    evidence = result.get('evidence', [])
    return JudgeDecision(verdict=verdict, reason=str(result.get('reason', '')),
                         evidence=evidence if isinstance(evidence, list) else [str(evidence)],
                         recommended_followup=followup)
