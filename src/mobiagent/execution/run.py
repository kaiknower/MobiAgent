"""CLI entry: `python -m mobiagent.execution.run --task task-0001 ...`

Modes (one of):
  --dry-run         : ask the planner for ONE subtask and exit (Step 6a)
  --mock-all        : full Orchestrator loop with mock policy + mock planner +
                      mock judge (Step 7 dev-test)
  (default)         : full eval — needs --policy-servers + Azure OpenAI + real env

Mock judge rules (selectable via --mock-judge):
  always_complete   : every chunk is judged 'complete' (planner walks forward)
  alternating       : alternate complete / incomplete to exercise retries
  fail_then_replan  : the 5th tick gets 'incomplete' until max_retries → planner
                      sees last_failed and revises
"""
from __future__ import annotations

import argparse
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .orchestrator import Orchestrator
from .planner_vlm import next_subtask
from .policy_registry import PolicyRegistry
from .schemas import Attempt, DynamicPlan, JudgeDecision, Subtask


# ---------- mock planner / judge ----------


def _make_mock_planner(seq: list[tuple[str, str]] | None = None):
    """Cycle through a hard-coded sequence of (stage_hint, prompt) tuples.

    By default this gives a complete pick→move→place→… loop suitable for
    --mock-all dev runs (all 6 stage_hints exercised at least once).
    """
    seq = seq or [
        ("move_to",      "move to source"),
        ("pick_up_from", "pick up object"),
        ("move_to",      "move to target"),
        ("place_in",     "place object in target"),
        ("open",         "open something"),
        ("close",        "close something"),
    ]
    state = {"i": 0}

    def _planner(*, global_goal: str, sim_task_name: str,
                 head_image, history=None, completed_subtasks=None, last_failed=None) -> Subtask:
        idx = state["i"] % len(seq)
        state["i"] += 1
        stage, prompt = seq[idx]
        return Subtask(
            id=f"mock-{state['i']:03d}",
            prompt=prompt,
            success_check="vlm_judge",
            stage_hint=stage,  # type: ignore[arg-type]
            max_retries=1,
            target_object_name=None,
        )

    return _planner


def _make_mock_judge(rule: str):
    state = {"counter": 0}

    def _judge(*, subtask: Subtask, obs_after: dict[str, Any], history: list[Attempt],
               plan: DynamicPlan, current_idx: int) -> JudgeDecision:
        state["counter"] += 1
        if rule == "always_complete":
            return JudgeDecision(verdict="complete", reason="mock always_complete", evidence=["mock"])
        if rule == "alternating":
            verdict = "complete" if state["counter"] % 2 == 1 else "incomplete"
            return JudgeDecision(verdict=verdict, reason="mock alternating", evidence=["mock"])
        if rule == "fail_then_replan":
            # First 4 ticks: complete on first try.
            # 5th onward: incomplete repeatedly (force retries → planner sees last_failed).
            if state["counter"] <= 4:
                return JudgeDecision(verdict="complete", reason="mock easy", evidence=["mock"])
            return JudgeDecision(
                verdict="incomplete", reason="mock forces planner replan",
                evidence=["mock"], recommended_followup="",
            )
        raise ValueError(f"unknown mock judge rule: {rule!r}")

    return _judge


# ---------- env factory ----------

from .env_factory import make_env  # noqa: E402  (re-export so --env mock|omni both work)


def load_task_instruction(sim_task_name: str) -> str:
    candidates = [Path("configs/task_instructions.json")]
    for p in candidates:
        if p.exists():
            try:
                d = json.loads(p.read_text())
                if sim_task_name in d:
                    return d[sim_task_name]
            except Exception:
                continue
    raise RuntimeError(
        f"Could not find task instruction for {sim_task_name!r}. Pass --instruction directly."
    )


# ---------- CLI ----------

def main(argv=None, *, default_env="mock", default_policy_servers=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="sim task name e.g. task-0001")
    ap.add_argument("--env", default=default_env, choices=["mock", "omni", "robocasa"])
    ap.add_argument("--instruction", default=None, help="override the global goal")
    ap.add_argument("--run-dir", default=None, help="output dir; default = runs/<task>_<timestamp>")
    ap.add_argument("--dry-run", action="store_true",
                    help="call planner once + write subtask.json + exit (no env stepping)")
    ap.add_argument("--mock-all", action="store_true",
                    help="run full Orchestrator with mock policy registry + mock planner + mock judge")
    ap.add_argument("--mock-judge", default="always_complete",
                    choices=["always_complete", "alternating", "fail_then_replan"])
    ap.add_argument("--policy-servers", default=default_policy_servers,
                    help="path to policy_servers.yaml (when not --mock-all)")
    ap.add_argument("--max-ticks", type=int, default=200)
    ap.add_argument("--instance-id", type=int, default=None,
                    help="public test instance id (TRO state overlay); --env omni only")
    ap.add_argument("--max-episode-steps", type=int, default=None,
                    help="Simulator step cap (OmniGibson default 5000; RoboCasa default 3000)")
    args = ap.parse_args(argv)

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path("runs") / f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir: {run_dir}", flush=True)

    instruction = args.instruction or load_task_instruction(args.task)
    print(f"global_goal: {instruction!r}", flush=True)

    env_kwargs: dict[str, Any] = {}
    if args.env == "omni" and args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = int(args.max_episode_steps)
    # NOTE: `instance_id` is intentionally NOT passed to the env constructor.
    # The constructor's instance_id feeds `activity_instance_id` in the scene
    # config → OmniGibson builds `..._0_<id>_template.json`, which only exists
    # for id 0. Per-instance variation is applied as a `tro_state.json` overlay
    # AFTER a base `_0_0_template` reset — so the id only goes to reset() below.
    if args.env == "robocasa":
        env_kwargs["task_name"] = args.task
        if args.max_episode_steps is not None:
            env_kwargs["horizon"] = args.max_episode_steps
        os.environ["MOBIAGENT_SKILLS"] = "close, open, switch, manipulate, navigate, pnp"
        os.environ["CLAW_PROMPT_STYLE"] = "skill_only"
    env = make_env(args.env, **env_kwargs)
    reset_kwargs: dict[str, Any] = {}
    if args.env == "omni" and args.instance_id is not None:
        reset_kwargs["tro_instance_id"] = int(args.instance_id)
    obs = env.reset(args.task, **reset_kwargs)
    print(f"env={args.env} reset (instance_id={args.instance_id}); "
          f"obs keys: {sorted(obs.keys())[:8]}...", flush=True)

    # Avoid `or` because the value is an ndarray — bool() raises.
    head_image = obs.get("observation/head_image")
    if head_image is None:
        head_image = obs.get("head_image")

    # ---------- --dry-run: one planner call and exit ----------

    if args.dry_run:
        print("\n--dry-run: calling next_subtask once ...", flush=True)
        st = next_subtask(
            global_goal=instruction, sim_task_name=args.task,
            completed_subtasks=[], last_failed=None, head_image=head_image,
        )
        out = run_dir / "subtask.json"
        out.write_text(json.dumps({
            "id": st.id, "prompt": st.prompt, "stage_hint": st.stage_hint,
            "max_retries": st.max_retries,
            "target_object_name": st.target_object_name,
            "failure_cues": st.failure_cues,
            "success_check": st.success_check,
            "rationale": st.rationale,
        }, indent=2), encoding="utf-8")
        print(f"  [{st.stage_hint:<13}] {st.id}: {st.prompt}")
        print(f"wrote {out}")
        env.close()
        return 0

    # ---------- wire registry + planner + judge ----------

    if args.mock_all:
        registry = PolicyRegistry.mock()
        planner_fn = _make_mock_planner()
        judge_fn = _make_mock_judge(args.mock_judge)
    else:
        if not args.policy_servers:
            print("ERROR: --policy-servers is required (or use --dry-run / --mock-all).", flush=True)
            env.close()
            return 2
        if args.env == "robocasa":
            from mobiagent.environments.robocasa import RoboCasaPolicyRegistry
            registry = RoboCasaPolicyRegistry.from_yaml(Path(args.policy_servers))
        else:
            registry = PolicyRegistry.from_yaml(Path(args.policy_servers))
        from .judge_vlm import judge as judge_real
        planner_fn = next_subtask  # the real Azure planner

        def judge_fn(**kwargs):
            return judge_real(**kwargs)

    orch = Orchestrator(
        env=env,
        global_goal=instruction,
        sim_task_name=args.task,
        registry=registry,
        planner_fn=planner_fn,
        judge_fn=judge_fn,
        run_dir=run_dir,
        max_ticks=args.max_ticks,
    )
    print("\n=== orchestrator running ===", flush=True)
    summary = orch.run(initial_obs=obs)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))

    registry.close()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
