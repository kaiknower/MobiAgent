"""Global timing accumulator for the lhe_omni pipeline (entry-agnostic).

Each component adds to T:
  - VLA inference  : WebsocketPolicyClient.infer   (server round-trip)
  - pure physics   : env_omni.step  (og.sim.step with rendering OFF)
  - render         : env_omni.step  (separate og.sim.render — EXCLUDED from report)
  - planner / judge: planner_vlm.next_subtask / judge_vlm.judge (Azure VLM calls)

On process exit, dumps T to $CLAW_TIMING_OUT (default ./claw_timing.json) with
derived totals. `reported_infer_plus_phys_s` = infer + phys (render EXCLUDED) —
the metric directly comparable to the champion's number.
"""
from __future__ import annotations

import atexit
import functools
import json
import os
import time
from contextlib import contextmanager

T = {
    "infer": 0.0, "phys": 0.0, "render": 0.0, "planner": 0.0, "judge": 0.0,
    "n_infer": 0, "n_steps": 0, "n_planner": 0, "n_judge": 0,
}


@contextmanager
def acc(key: str, count_key: str | None = None):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        T[key] += time.perf_counter() - t0
        if count_key:
            T[count_key] += 1
        _dump()  # persist after every infer/planner/judge (survives shutdown segfault)


def add_phys_render(t_phys: float, t_render: float, n: int = 1) -> None:
    T["phys"] += t_phys
    T["render"] += t_render
    T["n_steps"] += n
    # Write incrementally (every 50 steps) so a shutdown SEGFAULT — which bypasses
    # atexit — does not lose the breakdown. By the time the sim shuts down all
    # stepping is already on disk.
    if T["n_steps"] % 50 == 0:
        _dump()


def timed(key: str, count_key: str | None = None):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            with acc(key, count_key):
                return fn(*a, **k)
        return wrapper
    return deco


def _dump() -> None:
    out = dict(T)
    out["reported_infer_plus_phys_s"] = round(T["infer"] + T["phys"], 3)
    out["with_vlm_s"] = round(T["infer"] + T["phys"] + T["planner"] + T["judge"], 3)
    out["render_excluded_s"] = round(T["render"], 3)
    path = os.environ.get("CLAW_TIMING_OUT", "claw_timing.json")
    try:
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass
    print("[claw-timing] " + json.dumps(out), flush=True)


atexit.register(_dump)
