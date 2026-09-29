"""Train a shared policy using a named BEHAVIOR configuration."""
from openpi.training.runner import main
from openpi.training import config

if __name__ == "__main__":
    main(config.cli())
