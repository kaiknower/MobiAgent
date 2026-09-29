"""Serve a BEHAVIOR or RoboCasa checkpoint over WebSocket."""
import argparse
import logging


def main(argv=None, *, default_config="mobiagent_behavior"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=["mobiagent_behavior", "mobiagent_robocasa"], default=default_config)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training.config import get_config
    logging.basicConfig(level=logging.INFO)
    policy = create_trained_policy(get_config(args.config), args.checkpoint)
    WebsocketPolicyServer(policy=policy, host=args.host, port=args.port,
                          metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
