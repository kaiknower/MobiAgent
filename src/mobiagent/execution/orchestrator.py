"""Behavior-1K reactive tick loop — planner-per-step + judge.

Flow per tick (one tick == one "attempt"):
  1. If there is no current subtask, ask the planner for ONE next subtask
     (passing completed history + last failure context).
  2. Route the subtask to the policy server matching its `stage_hint`,
     request CLAW_CHUNKS_PER_ATTEMPT (default 10) chunks back-to-back, replaying
     each through the env — so each attempt is several seconds of motion.
  3. Ask the judge for a verdict on the final obs, also showing it the final
     frames of the previous two attempts (attempt-resolution comparison).
  4. complete       → mark subtask done, clear current, planner gets called next tick.
     incomplete     → retry same subtask (no planner call) until attempts == max_retries.
     retries done   → clear current, stash failure context for the planner's next call.
  5. The orchestrator owns termination, NOT the planner. Stop when ANY of:
       - env.is_success() returns True (BDDL goal predicate)
       - total_ticks >= max_ticks
       - max consecutive planner failures (sanity guard)

Logging: every tick writes one row to {run_dir}/logs/ticks.jsonl.
"""
from __future__ import annotations
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from .env_protocol import EnvProtocol
from .policy_registry import PolicyRegistry
from .schemas import Attempt, DynamicPlan, JudgeDecision, RunMemory, Subtask, build_full_prompt
PlannerFn = Callable[..., Subtask]
JudgeFn = Callable[..., JudgeDecision]

def _now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')

class Orchestrator:

    def __init__(self, *, env: EnvProtocol, global_goal: str, sim_task_name: str, registry: PolicyRegistry, planner_fn: PlannerFn, judge_fn: JudgeFn, run_dir: Path, max_ticks: int=200, max_consecutive_planner_failures: int=3, chunk_timeout_s: float=30.0, chunk_done_hook: Callable[..., None] | None=None) -> None:
        self.env = env
        self.registry = registry
        self.planner_fn = planner_fn
        self.judge_fn = judge_fn
        self.run_dir = Path(run_dir)
        self.max_ticks = max_ticks
        self.max_consecutive_planner_failures = max_consecutive_planner_failures
        self.chunk_timeout_s = chunk_timeout_s
        self._chunk_done_hook = chunk_done_hook
        plan = DynamicPlan(global_goal=global_goal, sim_task_name=sim_task_name, subtasks=[], plan_revision=0, rationale='reactive — planner emits one subtask per call', scene_summary='')
        self.memory = RunMemory(current_plan=plan)
        self._history: list[dict[str, Any]] = []
        self._needs_planner: bool = True
        self._planner_failures: int = 0
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'logs').mkdir(parents=True, exist_ok=True)
        self._ticks_log = self.run_dir / 'logs' / 'ticks.jsonl'
        self._ticks_fh = self._ticks_log.open('a', encoding='utf-8')
        self._abort_reason: str | None = None
        self._latest_obs: dict[str, Any] | None = None
        self._prev_obs_within_subtask: dict[str, Any] | None = None
        self._prev2_obs_within_subtask: dict[str, Any] | None = None
        self._chunks_per_attempt = max(1, int(os.environ.get('CLAW_CHUNKS_PER_ATTEMPT', '10') or '10'))
        self._prev_ended_stage_hint: str | None = None
        self._consecutive_chunk_fails: int = 0
        self._max_consecutive_chunk_fails: int = 3

    def run(self, initial_obs: dict[str, Any]) -> dict[str, Any]:
        self._latest_obs = initial_obs
        try:
            while not self._should_stop():
                self.tick()
        except Exception as exc:
            self._abort_reason = f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
        finally:
            try:
                self._ticks_fh.flush()
                self._ticks_fh.close()
            except Exception:
                pass
        return self.summary()

    def tick(self) -> None:
        if self._needs_planner:
            try:
                next_st = self.planner_fn(global_goal=self.memory.current_plan.global_goal, sim_task_name=self.memory.current_plan.sim_task_name, history=list(self._history), head_image=self._latest_head_image())
            except Exception as exc:
                self._planner_failures += 1
                self._record_planner_failure(exc)
                if self._planner_failures >= self.max_consecutive_planner_failures:
                    self._abort_reason = f'planner_failed_{self._planner_failures}_times: {exc}'
                return
            self._planner_failures = 0
            seq = len(self.memory.current_plan.subtasks) + 1
            next_st.id = f'{self.memory.current_plan.sim_task_name}-subtask-{seq:03d}'
            self.memory.current_plan.subtasks.append(next_st)
            self.memory.current_idx = len(self.memory.current_plan.subtasks) - 1
            self._needs_planner = False
            self._record_planner_emit(next_st, history_len=len(self._history))
            try:
                self.registry.reset_inpaint_state()
            except Exception:
                pass
        st = self.memory.current_subtask
        if st is None:
            self._abort_reason = 'no_current_subtask'
            return
        attempts = self.memory.attempts_per_subtask.get(st.id, 0)
        attempt = Attempt(subtask_id=st.id, attempt_number=attempts + 1, plan_revision=self.memory.current_plan.plan_revision, started_at=_now_iso())
        try:
            client = self.registry.select(st.stage_hint)
        except Exception as exc:
            attempt.judge = JudgeDecision(verdict='incomplete', reason=f'router_fail: {exc}', evidence=[])
            attempt.finished_at = _now_iso()
            attempt.notes = 'router_fail'
            self._record_attempt(attempt)
            self._post_judge(st, attempt, fatal_for_run=True)
            return
        full_prompt = build_full_prompt(self.memory.current_plan.global_goal, st.prompt)
        try:
            obs_after = self._latest_obs or {}
            _total_actions = 0
            for _ci in range(self._chunks_per_attempt):
                chunk = client.request_chunk(obs=obs_after, prompt=full_prompt, stage_hint=st.stage_hint, timeout_s=self.chunk_timeout_s)
                _total_actions += len(chunk.get('actions', [])) if isinstance(chunk, dict) else 0
                obs_after = self.env.step(chunk)
                self._latest_obs = obs_after
                if self._chunk_done_hook is not None:
                    try:
                        self._chunk_done_hook(env=self.env, subtask=st, attempt_number=attempt.attempt_number, obs_after=obs_after)
                    except Exception:
                        logger = __import__('logging').getLogger(__name__)
                        logger.exception('chunk_done_hook raised; continuing')
            attempt.chunk_size = _total_actions
            _prev_for_next_attempt = obs_after
            self._consecutive_chunk_fails = 0
        except Exception as exc:
            attempt.judge = JudgeDecision(verdict='incomplete', reason=f'chunk_fail: {exc}', evidence=[])
            attempt.finished_at = _now_iso()
            attempt.notes = 'chunk_fail'
            self._record_attempt(attempt)
            self._consecutive_chunk_fails += 1
            msg = str(exc).lower()
            hard_dead = 'name or service not known' in msg or 'errno -2' in msg or 'connection refused' in msg or ('errno 111' in msg) or ('nodename nor servname' in msg)
            if hard_dead or self._consecutive_chunk_fails >= self._max_consecutive_chunk_fails:
                self._abort_reason = f"chunk_fail_{('hard_network' if hard_dead else f'x{self._consecutive_chunk_fails}')}: {exc}"
                self._post_judge(st, attempt, fatal_for_run=True)
                return
            self._post_judge(st, attempt)
            return
        obs_mid = self._prev2_obs_within_subtask
        (_carry_side, _carry_item) = self._carry_hand_info(obs_after)
        robot_info = {'gripper_now': self._gripper_state_text(obs_after), 'gripper_prev': self._gripper_state_text(self._prev_obs_within_subtask), 'gripper_2back': self._gripper_state_text(obs_mid), 'gripper_cmd': self._gripper_cmd_text(chunk), 'carry_hand': _carry_side, 'carry_item': _carry_item}
        try:
            decision = self.judge_fn(subtask=st, obs_after=obs_after, obs_prev=self._prev_obs_within_subtask, obs_mid=obs_mid, robot_info=robot_info, history=self.memory.history, plan=self.memory.current_plan, current_idx=self.memory.current_idx)
        except TypeError:
            try:
                decision = self.judge_fn(subtask=st, obs_after=obs_after, obs_prev=self._prev_obs_within_subtask, obs_mid=obs_mid, history=self.memory.history, plan=self.memory.current_plan, current_idx=self.memory.current_idx)
            except TypeError:
                try:
                    decision = self.judge_fn(subtask=st, obs_after=obs_after, obs_prev=self._prev_obs_within_subtask, history=self.memory.history, plan=self.memory.current_plan, current_idx=self.memory.current_idx)
                except TypeError:
                    decision = self.judge_fn(subtask=st, obs_after=obs_after, history=self.memory.history, plan=self.memory.current_plan, current_idx=self.memory.current_idx)
        except Exception as exc:
            decision = JudgeDecision(verdict='incomplete', reason=f'judge_fail: {exc}', evidence=[], recommended_followup='')
        attempt.judge = decision
        attempt.finished_at = _now_iso()
        self._record_attempt(attempt)
        self._prev2_obs_within_subtask = self._prev_obs_within_subtask
        self._prev_obs_within_subtask = _prev_for_next_attempt
        self._post_judge(st, attempt)

    def summary(self) -> dict[str, Any]:
        plan = self.memory.current_plan
        completed = sum((1 for h in self._history if h.get('outcome') == 'complete'))
        failed = sum((1 for h in self._history if h.get('outcome') == 'failed'))
        return {'sim_task_name': plan.sim_task_name, 'global_goal': plan.global_goal, 'success': self._env_succeeded(), 'subtasks_emitted': len(plan.subtasks), 'subtasks_completed': completed, 'subtasks_failed': failed, 'total_ticks': self.memory.total_ticks, 'abort_reason': self._abort_reason}

    def _should_stop(self) -> bool:
        if self._abort_reason is not None:
            return True
        try:
            if self.env.is_done():
                meta = (getattr(self.env, '_latest_obs', None) or {}).get('_meta', {})
                self._abort_reason = 'env_truncated' if meta.get('truncated') else 'env_terminated'
                return True
        except (AttributeError, TypeError):
            pass
        if self.memory.total_ticks >= self.max_ticks:
            self._abort_reason = 'max_ticks'
            return True
        if self._env_succeeded():
            self._abort_reason = 'env_success'
            return True
        return False

    def _env_succeeded(self) -> bool:
        try:
            return bool(self.env.is_success())
        except Exception:
            return False

    @staticmethod
    def _gripper_word(n: float) -> str:
        if n > 0.6:
            return 'OPEN'
        if n > 0.0:
            return 'PARTLY-OPEN'
        if n > -0.6:
            return 'PARTLY-CLOSED'
        return 'CLOSED'

    def _gripper_state_text(self, obs: Any) -> str:
        """'left=CLOSED(-0.78) right=OPEN(+0.99)' from an obs's proprio, or '?'."""
        if not obs:
            return '?'
        v = obs.get('observation/state')
        if v is None:
            return '?'
        try:
            import numpy as _np
            a = _np.asarray(v, dtype=_np.float32).reshape(-1)
            ln = 2.0 * (float(a[193:195].sum()) / 0.1) - 1.0
            rn = 2.0 * (float(a[232:234].sum()) / 0.1) - 1.0
            return f'left={self._gripper_word(ln)}({ln:+.2f}) right={self._gripper_word(rn)}({rn:+.2f})'
        except Exception:
            return '?'

    def _carry_hand_info(self, obs: Any) -> tuple[str | None, str | None]:
        """(side, item_prompt) — which gripper ('LEFT'/'RIGHT') is holding a
        previously-picked, not-yet-placed item, or (None, None). 'side' is None
        if we can't tell which gripper (both open/closed) even when carrying."""
        last_pick = None
        for h in reversed(self._history):
            if h.get('outcome') != 'complete':
                continue
            sh = getattr(h.get('subtask'), 'stage_hint', None)
            if sh in ('place_in', 'place_on', 'place'):
                return (None, None)
            if sh == 'pick_up_from':
                last_pick = getattr(h.get('subtask'), 'prompt', None) or 'the picked-up item'
                break
        if last_pick is None:
            return (None, None)
        v = (obs or {}).get('observation/state')
        if v is None:
            return (None, last_pick)
        try:
            import numpy as _np
            a = _np.asarray(v, dtype=_np.float32).reshape(-1)
            l_closed = 2.0 * (float(a[193:195].sum()) / 0.1) - 1.0 < 0.5
            r_closed = 2.0 * (float(a[232:234].sum()) / 0.1) - 1.0 < 0.5
            if l_closed and (not r_closed):
                return ('LEFT', last_pick)
            if r_closed and (not l_closed):
                return ('RIGHT', last_pick)
            return (None, last_pick)
        except Exception:
            return (None, last_pick)

    def _gripper_cmd_text(self, chunk: Any) -> str:
        """The policy's last commanded gripper this attempt, e.g. 'left=-0.85 right=+1.00'."""
        if not isinstance(chunk, dict):
            return '?'
        acts = chunk.get('actions')
        if acts is None:
            return '?'
        try:
            import numpy as _np
            a = _np.asarray(acts)
            if a.ndim != 2 or a.shape[-1] < 23:
                return '?'
            last = a[-1]
            return f'left={float(last[14]):+.2f} right={float(last[22]):+.2f}'
        except Exception:
            return '?'

    def _post_judge(self, st: Subtask, attempt: Attempt, *, fatal_for_run: bool=False) -> None:
        self.memory.history.append(attempt)
        self.memory.total_ticks += 1
        if fatal_for_run:
            return
        d = attempt.judge or JudgeDecision(verdict='incomplete', reason='')
        if d.verdict == 'complete' and d.recommended_followup != 'retry':
            attempts_so_far = self.memory.attempts_per_subtask.get(st.id, 0) + 1
            self._history.append({'subtask': st, 'outcome': 'complete', 'judge_reason': d.reason or '', 'n_attempts': attempts_so_far})
            self.memory.attempts_per_subtask.pop(st.id, None)
            self._needs_planner = True
            self._prev_obs_within_subtask = None
            self._prev2_obs_within_subtask = None
            self._prev_ended_stage_hint = st.stage_hint
            return
        attempts = self.memory.attempts_per_subtask.get(st.id, 0) + 1
        self.memory.attempts_per_subtask[st.id] = attempts
        deviated = d.verdict == 'error' or d.recommended_followup in ('replan_plan_deviated', 'replan_keep_pose')
        keep_pose = d.recommended_followup == 'replan_keep_pose'
        budget_exhausted = attempts > st.max_retries
        if deviated or budget_exhausted:
            failure_reason = d.reason or '(no judge reason recorded)'
            if budget_exhausted:
                failure_reason = f'retries exhausted ({attempts} attempts); last judge said: {failure_reason}'
            elif deviated:
                self.registry.reset_inpaint_state()
                failure_reason = f'early replan after {attempts} attempts; last judge said: {failure_reason}'
            self._history.append({'subtask': st, 'outcome': 'failed', 'judge_reason': failure_reason, 'n_attempts': attempts})
            self.memory.attempts_per_subtask.pop(st.id, None)
            self._needs_planner = True
            self._prev_obs_within_subtask = None
            self._prev2_obs_within_subtask = None
            self._prev_ended_stage_hint = st.stage_hint

    def _latest_head_image(self) -> Any:
        if self._latest_obs is None:
            return None
        from .policy_client import OPENPI_HEAD_KEY
        for k in (OPENPI_HEAD_KEY, 'observation/head_image', 'head_image'):
            v = self._latest_obs.get(k)
            if v is not None:
                return v
        return None

    def _record_attempt(self, a: Attempt) -> None:
        st = self.memory.current_subtask
        self._ticks_fh.write(json.dumps({'ts': _now_iso(), 'subtask_id': a.subtask_id, 'attempt': a.attempt_number, 'plan_revision': a.plan_revision, 'stage_hint': st.stage_hint if st else None, 'subtask_prompt': st.prompt if st else None, 'chunk_size': a.chunk_size, 'verdict': a.judge.verdict if a.judge else None, 'reason': a.judge.reason if a.judge else None, 'evidence': list(a.judge.evidence) if a.judge else None, 'recommended_followup': a.judge.recommended_followup if a.judge else None, 'notes': a.notes, 'started_at': a.started_at, 'finished_at': a.finished_at}) + '\n')
        self._ticks_fh.flush()

    def _record_planner_failure(self, exc: Exception) -> None:
        self._ticks_fh.write(json.dumps({'ts': _now_iso(), 'event': 'planner_failure', 'consecutive': self._planner_failures, 'error': f'{type(exc).__name__}: {exc}'}) + '\n')
        self._ticks_fh.flush()

    def _record_planner_emit(self, st: Subtask, *, history_len: int) -> None:
        """Log a planner_emit event to ticks.jsonl. Lets review tooling show
        the planner's stated rationale + the history snapshot it saw."""
        global_goal = self.memory.current_plan.global_goal if self.memory.current_plan else ''
        full_prompt = build_full_prompt(global_goal, st.prompt)
        self._ticks_fh.write(json.dumps({'ts': _now_iso(), 'event': 'planner_emit', 'subtask_id': st.id, 'stage_hint': st.stage_hint, 'prompt': st.prompt, 'global_goal': global_goal, 'server_instruction': full_prompt, 'rationale': st.rationale, 'plan_sketch': list(st.plan_sketch), 'target_object_name': st.target_object_name, 'history_len_when_emitted': history_len}) + '\n')
        self._ticks_fh.flush()
__all__ = ['Orchestrator']
