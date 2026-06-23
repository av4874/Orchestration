"""
Trigger Argo Workflow via API. Called in Jenkins final stage.

Usage:
    python pipeline/trigger_argo.py --workflow full-canary --round 1 --model-path models/v1/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.tools.routing_tools import trigger_argo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", default="full-canary",
                        choices=["full-canary", "fast-promote", "emergency-rollback"])
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    model_path = args.model_path or f"models/v{args.round}/"
    result = trigger_argo(json.dumps({
        "workflow": args.workflow,
        "round": args.round,
        "model_path": model_path,
    }))
    print(result)
    if result.startswith("ERROR"):
        sys.exit(1)


if __name__ == "__main__":
    main()
