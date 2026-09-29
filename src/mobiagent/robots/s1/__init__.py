"""S1-mobile real-machine skill stack.

Three modules:
  - planner  : reactive VLM planner emitting one subtask at a time
  - judge    : per-attempt VLM judge of subtask completion
  - policy   : msgpack-over-websocket client to a 3-head openpi server

Stage IDs (3-head): move_to=0, pick_up=1, place=2.
"""
from .schemas import (
    CANONICAL_STAGE_HINTS,
    DynamicPlan,
    JudgeDecision,
    JudgeFollowup,
    JudgeVerdict,
    RunMemory,
    STAGE_HINT_TO_INT,
    Subtask,
    TASK_GOALS,
    build_full_prompt,
)
from .planner import next_subtask
from .judge import judge
from .policy_client import (
    MockPolicyClient,
    PolicyClientProtocol,
    WebsocketPolicyClient,
    build_policy_observation,
)
from .policy_registry import PolicyRegistry
from .agent import RobotBridge, run_agent

__all__ = [
    "CANONICAL_STAGE_HINTS",
    "DynamicPlan",
    "JudgeDecision",
    "JudgeFollowup",
    "JudgeVerdict",
    "MockPolicyClient",
    "PolicyClientProtocol",
    "PolicyRegistry",
    "RobotBridge",
    "RunMemory",
    "STAGE_HINT_TO_INT",
    "Subtask",
    "TASK_GOALS",
    "WebsocketPolicyClient",
    "build_full_prompt",
    "build_policy_observation",
    "judge",
    "next_subtask",
    "run_agent",
]
