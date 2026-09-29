"""Run MobiAgent planning, policy execution and visual reflection in RoboCasa."""
from mobiagent.execution.run import main

if __name__ == "__main__":
    raise SystemExit(main(default_env="robocasa",
                         default_policy_servers="configs/robocasa/policy_servers.yaml"))
