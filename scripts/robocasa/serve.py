"""Serve the six-expert RoboCasa policy checkpoint."""
from openpi.serving.serve_policy import main

if __name__ == "__main__":
    main(default_config="mobiagent_robocasa")
