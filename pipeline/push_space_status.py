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


def push_to_space(payload: dict, dry_run: bool) -> bool:
    """Push pipeline_status.json to HF Space repo via HF Hub API."""
    import base64
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    content = json.dumps(payload, indent=2)

    if dry_run:
        print(f"[DRY-RUN] Would push to {SPACE_REPO}/{SPACE_FILE}:")
        print(content[:500])
        return True

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set — cannot push to Space")
        return False

    # Get current file SHA (needed for update)
    url = f"{HF_API_BASE}/spaces/{SPACE_REPO}/resolve/main/{SPACE_FILE}"
    sha = None
    try:
        r = requests.get(
            f"https://huggingface.co/api/repos/{SPACE_REPO}/contents/{SPACE_FILE}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            verify=False, timeout=15,
        )
        if r.ok:
            sha = r.json().get("sha")
    except Exception:
        pass

    # Commit via HF Hub API
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    body = {
        "message": f"chore: update pipeline_status.json round {payload['round']}",
        "content": encoded,
    }
    if sha:
        body["sha"] = sha

    try:
        r = requests.put(
            f"https://huggingface.co/api/spaces/{SPACE_REPO}/raw/main/{SPACE_FILE}",
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
            json=body,
            verify=False,
            timeout=30,
        )
        if r.ok:
            print(f"[Space] pipeline_status.json pushed — round {payload['round']}")
            return True
        # Fallback: huggingface_hub library
        raise RuntimeError(f"HF API {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"  API push failed ({e}), trying huggingface_hub...")

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write(content)
            tmp = f.name
        api.upload_file(
            path_or_fileobj=tmp,
            path_in_repo=SPACE_FILE,
            repo_id=SPACE_REPO,
            repo_type="space",
            commit_message=f"chore: update pipeline_status.json round {payload['round']}",
        )
        _os.unlink(tmp)
        print(f"[Space] pipeline_status.json pushed via huggingface_hub — round {payload['round']}")
        return True
    except Exception as e2:
        print(f"ERROR: both push methods failed: {e2}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = build_status(args.round)
    print(f"[Space] Built status for round {args.round}: status={payload['status']}")
    success = push_to_space(payload, args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
