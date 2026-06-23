"""
Deploy versioned models to staging and write pipeline_status.json for the Space dashboard.
Called in Argo stage-deploy step.

Usage:
    python siem/deploy.py --round 1 --model-path models/v1/ --staging
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ENTERPRISE_ROOT = Path(__file__).parent.parent
STAGING_MODELS = ENTERPRISE_ROOT / "models" / "staging"
RESULTS_DIR = ENTERPRISE_ROOT / "results"


def write_pipeline_status(round_num: int, model_path: Path):
    """Write pipeline_status.json — consumed by the Orchestration Space dashboard tab."""
    evasion_report = {}
    evasion_path = RESULTS_DIR / "evasion_report.json"
    if evasion_path.exists():
        with open(evasion_path) as f:
            evasion_report = json.load(f)

    decision = {}
    decision_path = RESULTS_DIR / "pipeline_decision.json"
    if decision_path.exists():
        with open(decision_path) as f:
            decision = json.load(f)

    memory = {}
    memory_path = ENTERPRISE_ROOT / "pipeline" / "attack_memory.json"
    if memory_path.exists():
        with open(memory_path) as f:
            memory = json.load(f)

    per_family = evasion_report.get("per_family", {})
    families_tried = [
        {"name": fam, "samples": info.get("total", 0), "evasion_pct": info.get("evasion_rate", 0)}
        for fam, info in per_family.items()
    ]

    per_detector = evasion_report.get("per_detector", {})
    weakness_scores = {det: info.get("evasion_rate", 0) for det, info in per_detector.items()}

    retrain = decision.get("models_to_retrain", [])

    status = {
        "round": round_num,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "staged_model_path": str(model_path),
        "red_team": {
            "families_tried": families_tried,
        },
        "blue_team": {
            "weakness_scores": weakness_scores,
            "retrain_priority": retrain,
        },
        "orchestrator": {
            "action": decision.get("action", ""),
            "severity": decision.get("severity", ""),
            "confidence": decision.get("confidence", 0),
            "argo_workflow": decision.get("argo_workflow", ""),
            "reason": decision.get("reason", ""),
        },
        "memory": {
            "current_focus": memory.get("current_focus", []),
            "known_blind_spots": memory.get("known_blind_spots", []),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    status_path = RESULTS_DIR / "pipeline_status.json"
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)
    print(f"pipeline_status.json written: {status_path}")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--staging", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path or f"models/v{args.round}/")
    if not model_path.exists():
        # In early rounds, model path may not exist yet — warn but don't fail
        print(f"WARNING: model path not found: {model_path} — staging dir unchanged")
    else:
        STAGING_MODELS.mkdir(parents=True, exist_ok=True)
        shutil.copytree(model_path, STAGING_MODELS, dirs_exist_ok=True)
        print(f"Deployed {model_path} -> {STAGING_MODELS}")

    staging_url = os.environ.get("STAGING_URL", "http://guardrail-staging.jenkins.svc.cluster.local:7860")
    print(f"Staging endpoint: {staging_url}")

    write_pipeline_status(args.round, model_path)

    print("Stage deploy complete.")


if __name__ == "__main__":
    main()
