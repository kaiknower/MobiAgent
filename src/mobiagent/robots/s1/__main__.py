"""``python -m mobiagent.robots.s1`` — run the full planner→policy→robot→judge loop.

Usage:

  python -m mobiagent.robots.s1 \
      --task   trash-general \
      --robot  my_pkg.my_bridge:make_robot \
      --config configs/policy_servers.yaml

``--robot`` is a dotted import path to a factory: ``pkg.module:factory``.
The factory is called with no arguments and must return an object that
satisfies the :class:`mobiagent.robots.s1.agent.RobotBridge` protocol.

See :doc:`examples/robot_bridge_template.py` for a starter bridge.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys

from .agent import run_agent
from .schemas import TASK_GOALS


def _load_robot(spec: str):
    if ":" not in spec:
        raise SystemExit(
            f"--robot expects 'package.module:factory_callable', got {spec!r}"
        )
    mod_name, factory_name = spec.split(":", 1)
    try:
        module = importlib.import_module(mod_name)
    except ImportError as exc:
        raise SystemExit(f"cannot import {mod_name!r}: {exc}") from exc
    try:
        factory = getattr(module, factory_name)
    except AttributeError as exc:
        raise SystemExit(
            f"{mod_name!r} has no attribute {factory_name!r}"
        ) from exc
    return factory()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="s1-agent",
        description="Drive the S1-mobile 3-head policy end-to-end "
                    "(planner → policy → robot → judge).",
    )
    parser.add_argument(
        "--task", required=True,
        choices=sorted(TASK_GOALS.keys()),
        help="task id from TASK_GOALS (sets the planner's global goal)",
    )
    parser.add_argument(
        "--robot", required=True,
        help="dotted path to a no-arg factory returning a RobotBridge, "
             "e.g. 'my_pkg.bridge:make_robot'",
    )
    parser.add_argument(
        "--config", default="configs/policy_servers.yaml",
        help="path to the policy-server YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=200,
        help="max planner cycles before forced exit (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    robot = _load_robot(args.robot)
    history = run_agent(
        robot,
        task=args.task,
        config_path=args.config,
        max_steps=args.max_steps,
    )
    print(json.dumps(history, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
