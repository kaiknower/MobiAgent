"""Run offline skill discovery on a local demonstration dataset."""
import argparse
import importlib
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--platform", choices=["behavior", "s1"], default="behavior")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/discovery"))
    args = parser.parse_args()
    pipeline = importlib.import_module(f"mobiagent.discovery.{args.platform}.pipeline")
    result = pipeline.run_demo_skill_discovery(dataset_root=args.dataset_root, task_ids=args.tasks, output_root=args.output_root)
    print(result)

if __name__ == "__main__":
    main()
