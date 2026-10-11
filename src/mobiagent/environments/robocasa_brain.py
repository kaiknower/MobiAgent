from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mobiagent.execution.llm_client import chat_completion_json, make_user_content
from mobiagent.execution.planner_vlm import _parse_subtask_json
from mobiagent.execution.schemas import JudgeDecision

from .robocasa_control import AttemptControl, robot_info
from .robocasa_prompt_canon import canonicalize


TASK_PROFILES = {
    "KettleBoiling": "kettle",
    "LoadDishwasher": "dishwasher",
    "PrepareCoffee": "coffee",
    "RinseSinkBasin": "rinse",
    "StackBowlsCabinet": "stack",
    "WashLettuce": "lettuce",
}


def composite(obs, label):
    keys = ("head", "left_wrist", "right_wrist")
    captions = ("[1] LEFT VIEW", "[2] HAND CAM", "[3] RIGHT VIEW")
    canvas = Image.new("RGB", (512, 540))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 6), label, font=font, fill="white")
    for index, (key, caption) in enumerate(zip(keys, captions)):
        image = obs.get(f"observation/{key}_image_orig")
        if image is None:
            image = obs.get(f"observation/{key}_image")
        x, y = index % 2 * 256, 28 + index // 2 * 256
        if image is not None:
            canvas.paste(
                Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((256, 256)),
                (x, y),
            )
        draw.rectangle((x, y, x + 124, y + 20), fill="black")
        draw.text((x + 4, y + 4), caption, font=font, fill="yellow")
    return np.asarray(canvas)


class RoboCasaBrain:
    def __init__(self, env):
        self.env = env
        profile = TASK_PROFILES.get(env.task_name, "default")
        root = Path(__file__).with_name("robocasa_templates")
        planner_profile = "default" if profile in ("coffee", "stack") else profile
        self.planner_prompt = (root / f"{planner_profile}_planner.txt").read_text()
        self.judge_prompt = (root / f"{profile}_judge.txt").read_text()

    def plan(self, *, global_goal, sim_task_name, history, head_image):
        rows = [
            {
                "prompt": entry["subtask"].prompt,
                "stage_hint": entry["subtask"].stage_hint,
                "outcome": entry["outcome"],
                "judge_reason": entry.get("judge_reason", ""),
                "plan_sketch": entry["subtask"].plan_sketch,
            }
            for entry in history
        ]
        obs = self.env._latest_obs
        text = json.dumps(
            {
                "global_goal": global_goal,
                "history": rows,
                "robot_proprioception": robot_info(obs),
            },
            ensure_ascii=False,
        )
        value = chat_completion_json(
            system_text=self.planner_prompt,
            user_content=make_user_content(
                text=text, images=[composite(obs, "Current")]
            ),
            model=os.getenv("MOBIAGENT_PLANNER_MODEL") or os.getenv("OPENAI_MODEL"),
        )
        subtask = _parse_subtask_json(value, fallback_id=f"subtask-{len(history) + 1}")
        subtask.prompt, matched = canonicalize(
            subtask.prompt, subtask.stage_hint, global_goal
        )
        subtask.plan_sketch = [
            canonicalize(item, None, global_goal)[0] for item in subtask.plan_sketch
        ]
        if subtask.stage_hint == "open_close":
            subtask.stage_hint = (
                "manipulate"
                if "manipulate" in matched and "switch" not in matched
                else "switch"
            )
        if sim_task_name == "KettleBoiling" and history:
            previous = history[-1]
            prior = previous["subtask"]
            if (
                previous["outcome"] == "complete"
                and prior.stage_hint == "place_on"
                and subtask.stage_hint == "switch"
            ):
                burners = [
                    item
                    for item in prior.plan_sketch
                    if item.startswith("Turn on ") and " burner" in item
                ]
                if len(set(burners)) == 1:
                    subtask.prompt = burners[0]
                    subtask.plan_sketch = [burners[0]]
            if (
                previous["outcome"] == "failed"
                and prior.stage_hint == "pick_up_from"
                and prior.prompt == "Pick up the kettle from the counter."
                and subtask.prompt == "Place the kettle on the stove burner."
            ):
                gap = AttemptControl.gap(obs)
                if gap is not None and gap <= 5:
                    subtask.prompt, subtask.stage_hint = prior.prompt, prior.stage_hint
                    subtask.plan_sketch = list(prior.plan_sketch)
        recovery = getattr(self.env, "recovery", None)
        if recovery is not None:
            intent = recovery.planner_intent(history)
            if intent:
                subtask.prompt = (
                    "Navigate to the sink."
                    if intent == "navigate"
                    else "Move the spout to wash all locations of the sink basin."
                )
                subtask.stage_hint = "move_to" if intent == "navigate" else "manipulate"
                subtask.plan_sketch = [subtask.prompt]
        return subtask

    def judge(
        self,
        *,
        subtask,
        obs_after,
        history,
        plan,
        current_idx,
        obs_prev=None,
        obs_mid=None,
        robot_info=None,
    ):
        images, labels = [], []
        for label, obs in (
            ("Start", obs_mid),
            ("Previous", obs_prev),
            ("Current", obs_after),
        ):
            if obs is not None:
                images.append(composite(obs, label))
                labels.append(label)
        text = json.dumps(
            {
                "global_goal": plan.global_goal,
                "subtask": subtask.prompt,
                "skill": subtask.stage_hint,
                "success_check": subtask.success_check,
                "failure_cues": subtask.failure_cues,
                "image_order": labels,
                "robot_proprioception": robot_info or {},
            },
            ensure_ascii=False,
        )
        value = chat_completion_json(
            system_text=self.judge_prompt,
            user_content=make_user_content(text=text, images=images),
            max_completion_tokens=4096,
            model=os.getenv("MOBIAGENT_JUDGE_MODEL") or os.getenv("OPENAI_MODEL"),
        )
        verdict = str(value.get("verdict", "")).lower().strip()
        if verdict not in ("complete", "incomplete", "error"):
            verdict = "incomplete"
        followup = str(value.get("recommended_followup", "")).lower().strip()
        if followup not in (
            "next",
            "retry",
            "replan_keep_pose",
            "replan_plan_deviated",
            "",
        ):
            followup = ""
        evidence = value.get("evidence") or []
        gap = AttemptControl.gap(obs_after)
        if (
            verdict == "complete"
            and len(
                {id(obs) for obs in (obs_mid, obs_prev, obs_after) if obs is not None}
            )
            < 3
        ):
            verdict, followup = "incomplete", "retry"
            value["reason"] = (
                "Three distinct robot observations are required to confirm the outcome."
            )
        if (
            subtask.stage_hint in ("pick_up_from", "pick")
            and verdict == "complete"
            and gap is not None
            and gap <= 5
        ):
            verdict, followup = "incomplete", "retry"
            value["reason"] = (
                "The gripper is fully shut; a grasp is not supported by robot contact evidence."
            )
        return JudgeDecision(
            verdict=verdict,
            reason=str(value.get("reason", "")),
            evidence=evidence if isinstance(evidence, list) else [str(evidence)],
            recommended_followup=followup,
        )
