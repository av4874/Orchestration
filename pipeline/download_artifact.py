"""
Download GitHub Actions artifact (attack samples) for a given round.
Called in Jenkins Stage 1.

Usage:
    python pipeline/download_artifact.py --round 1
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.tools.github_tools import download_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()

    result = download_artifact(json.dumps({"round": args.round}))
    print(result)
    if result.startswith("ERROR"):
        sys.exit(1)


if __name__ == "__main__":
    main()
