"""Behavior-1K long-horizon eval runner — DIMOS-free.

Replaces dimos_pi0_5GT/ with a thin orchestrator built around:
  - A VLM-generated DynamicPlan at startup (no hardcoded fixed_stage_plan)
  - A planner-judge outer loop that walks the plan by index
  - Retry/replan semantics on judge failure (not just stage±1)
  - Six-checkpoint-server routing via Subtask.stage_hint

Public surface (use these from outside the package):
  schemas: Subtask, DynamicPlan, PlannerDecision, JudgeDecision, RunMemory
  env_protocol: EnvProtocol (ABC)
  env_mock: MockEnv (no OmniGibson, for dev)
  planner_vlm: build_dynamic_plan(global_goal, sim_task_name, scene_obs) -> DynamicPlan
  run: __main__ entry point (`python -m behavior_1k_eval.run --task task-0001 ...`)
"""
