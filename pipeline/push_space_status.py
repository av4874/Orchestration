"""
push_space_status.py — Push pipeline round results to HF Space pipeline_status.json.
Called by orchestrator after each round to update the Space dashboard.

Usage:
    python pipeline/push_space_status.py --round 1
    python pipeline/push_space_status.py --round 1 --dry-run
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ENTERPRISE_ROOT = Path(__file__).parent.parent
RESULTS_DIR = ENTERPRISE_ROOT / "results"

HF_TOKEN = os.environ.get("HF_TOKEN", "")
SPACE_REPO = "Builder117/Orchestration"
SPACE_FILE = "pipeline_status.json"
HF_API_BASE = "https://huggingface.co/api"


def build_status(round_num: int) -> dict:
    """Assemble pipeline_status.json payload from local result files."""
    # Evasion report
    evasion_path = RESULTS_DIR / "evasion_report.json"
    weakness_scores = {"injection": None, "jailbreak": None, "insecure_output": None, "indirect_injection": None}
    retrain_priority = []
    if evasion_path.exists():
        er = json.loads(evasion_path.read_text(encoding="utf-8"))
        for det, v in er.get("per_detector", {}).items():
            weakness_scores[det] = round(v.get("evasion_rate", 0), 4)
        retrain_priority = [d for d, v in weakness_scores.items() if v and v >= 0.25]
        retrain_priority.sort(key=lambda d: weakness_scores[d], reverse=True)

    # Red team message
    red_msg_path = ENTERPRISE_ROOT / "agent_workspace" / "red_to_orchestrator.json"
    families_tried = []
    if red_msg_path.exists():
        red = json.loads(red_msg_path.read_text(encoding="utf-8"))
        body = red.get("body", {})
        for fam in body.get("families_tried", [body.get("top_family", "unknown")]):
            families_tried.append({
                "name": fam,
                "samples": body.get("samples_generated", body.get("sample_count", 0)),
                "evasion_pct": body.get("expected_evasion", 0.0),
            })

    # Orchestrator decision
    decision_path = RESULTS_DIR / "pipeline_decision.json"
    orch = {"action": "none", "severity": "none", "confidence": None, "argo_workflow": "none", "reason": "Pending."}
    if decision_path.exists():
        d = json.loads(decision_path.read_text(encoding="utf-8"))
        orch = {
            "action": d.get("action", "none"),
            "severity": d.get("severity", "none"),
            "confidence": d.get("confidence"),
            "argo_workflow": d.get("argo_workflow", "none"),
            "reason": d.get("reason", ""),
        }

    overall = round_num > 0 and any(v for v in weakness_scores.values() if v)
    status = "complete" if overall else "awaiting_first_run"

    return {
        "round": round_num,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "red_team": {"families_tried": families_tried},
        "blue_team": {"weakness_scores": weakness_scores, "retrain_priority": retrain_priority},
        "orchestrator": orch,
    }


LINEAGE_FILE = "model_lineage.json"


def build_lineage_entry(round_num: int) -> dict | None:
    """Build one round's lineage entry from local result files. Returns None if no retrain happened."""
    retrain_path = RESULTS_DIR / "retrain_report.json"
    if not retrain_path.exists():
        return None
    retrain = json.loads(retrain_path.read_text(encoding="utf-8"))
    if retrain.get("action") in ("skip", "fast_promote") or not retrain.get("detectors_retrained"):
        return None

    evasion_path = RESULTS_DIR / "evasion_report.json"
    pre_evasion = {}
    attack_family = "unknown"
    if evasion_path.exists():
        er = json.loads(evasion_path.read_text(encoding="utf-8"))
        for det, v in er.get("per_detector", {}).items():
            pre_evasion[det] = round(v.get("evasion_rate", 0), 4)
        attack_family = er.get("attack_family", "unknown")

    decision_path = RESULTS_DIR / "pipeline_decision.json"
    argo_workflow = "unknown"
    if decision_path.exists():
        d = json.loads(decision_path.read_text(encoding="utf-8"))
        argo_workflow = d.get("argo_workflow", "unknown")

    version = round_num + 3  # matches retrain.py VERSION_OFFSET
    detectors_built = {}
    for det in retrain.get("detectors_retrained", []):
        base_id = {
            "injection":          "Builder117/distilbert-prompt-injection",
            "jailbreak":          "Builder117/distilbert-jailbreak",
            "insecure_output":    "Builder117/distilbert-insecure-output",
            "indirect_injection": "Builder117/distilbert-indirect-injection",
        }.get(det, det)
        detectors_built[det] = {
            "pre_evasion": pre_evasion.get(det),
            "post_evasion": retrain.get("post_evasion", {}).get(det),
            "hf_model": f"{base_id}-v{version}",
        }

    return {
        "round": round_num,
        "hf_version": version,
        "attack_family": attack_family,
        "argo_workflow": argo_workflow,
        "detectors": detectors_built,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def merge_lineage(existing_json: str | None, new_entry: dict) -> dict:
    """Append new_entry to lineage, replacing any existing entry for same round."""
    if existing_json:
        try:
            lineage = json.loads(existing_json)
        except json.JSONDecodeError:
            lineage = {"rounds": []}
    else:
        lineage = {"rounds": []}

    lineage["rounds"] = [r for r in lineage["rounds"] if r.get("round") != new_entry["round"]]
    lineage["rounds"].append(new_entry)
    lineage["rounds"].sort(key=lambda r: r["round"])
    return lineage


def push_to_space(payload: dict, dry_run: bool, round_num: int) -> bool:
    """Push pipeline_status.json (and model_lineage.json if retrain happened) to HF Space."""
    status_content = json.dumps(payload, indent=2)
    lineage_entry = build_lineage_entry(round_num)

    if dry_run:
        print(f"[DRY-RUN] Would push to {SPACE_REPO}/{SPACE_FILE}:")
        print(status_content[:400])
        if lineage_entry:
            print(f"[DRY-RUN] Would append to {LINEAGE_FILE}: round {lineage_entry['round']}, family={lineage_entry['attack_family']}")
        return True

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set — cannot push to Space")
        return False

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed — run: pip install huggingface_hub")
        return False

    api = HfApi(token=HF_TOKEN)

    try:
        api.upload_file(
            path_or_fileobj=status_content.encode("utf-8"),
            path_in_repo=SPACE_FILE,
            repo_id=SPACE_REPO,
            repo_type="space",
            commit_message=f"chore: update dashboard round {payload['round']}",
        )
        print(f"[Space] Pushed {SPACE_FILE} — round {payload['round']}")

        if lineage_entry:
            try:
                existing_raw = api.hf_hub_download(
                    repo_id=SPACE_REPO, repo_type="space", filename=LINEAGE_FILE
                )
                existing_json = Path(existing_raw).read_text(encoding="utf-8")
            except Exception:
                existing_json = None
            lineage = merge_lineage(existing_json, lineage_entry)
            api.upload_file(
                path_or_fileobj=json.dumps(lineage, indent=2).encode("utf-8"),
                path_in_repo=LINEAGE_FILE,
                repo_id=SPACE_REPO,
                repo_type="space",
                commit_message=f"chore: update model lineage round {round_num}",
            )
            print(f"[Space] {LINEAGE_FILE} updated — {len(lineage['rounds'])} rounds tracked")

        return True

    except Exception as e:
        print(f"ERROR: HF upload failed — {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = build_status(args.round)
    print(f"[Space] Built status for round {args.round}: status={payload['status']}")
    success = push_to_space(payload, args.dry_run, args.round)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
