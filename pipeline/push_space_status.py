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
        # Body may use families_tried/top_family (old) or valid_samples[].attack_family (new)
        if body.get("families_tried"):
            fam_list = body["families_tried"]
        elif body.get("top_family"):
            fam_list = [body["top_family"]]
        elif body.get("valid_samples"):
            # Aggregate by attack_family from valid_samples
            from collections import Counter
            counts = Counter(s.get("attack_family", "unknown") for s in body["valid_samples"])
            fam_list = list(counts.keys())
        else:
            fam_list = ["unknown"]
        for fam in fam_list:
            # Count samples for this family from valid_samples if available
            valid = body.get("valid_samples", [])
            fam_count = sum(1 for s in valid if s.get("attack_family") == fam) if valid else body.get("samples_generated", body.get("sample_count", 0))
            evasion = body.get("expected_evasion", 0.0)
            if valid:
                fam_evasion = sum(1 for s in valid if s.get("attack_family") == fam and s.get("expected_evasion")) / max(fam_count, 1)
                evasion = fam_evasion
            families_tried.append({
                "name": fam,
                "samples": fam_count,
                "evasion_pct": evasion,
            })

    # Orchestrator decision
    decision_path = RESULTS_DIR / "pipeline_decision.json"
    orch = {"action": "none", "confidence": None, "reason": "Pending.", "models_to_retrain": []}
    if decision_path.exists():
        d = json.loads(decision_path.read_text(encoding="utf-8"))
        orch = {
            "action": d.get("action", "none"),
            "confidence": d.get("confidence"),
            "reason": d.get("reason", ""),
            "models_to_retrain": d.get("models_to_retrain", []),
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
    """Build one round's lineage entry from local result files. Returns None if no retrain happened.

    Handles three retrain report formats:
    - new per-detector format: per_detector dict with per-detector results (retrain_kernel v2)
    - old kernel format: hub_pushed + output_model + eval_f1 (single combined model)
    - legacy format: detectors_retrained list + post_evasion
    """
    retrain_path = RESULTS_DIR / "retrain_report.json"
    if not retrain_path.exists():
        return None
    retrain = json.loads(retrain_path.read_text(encoding="utf-8"))

    # Detect format
    is_per_detector_format = bool(retrain.get("per_detector"))
    is_kernel_format = not is_per_detector_format and retrain.get("hub_pushed") and retrain.get("output_model")
    is_legacy_format = not is_per_detector_format and not is_kernel_format and bool(retrain.get("detectors_retrained"))

    # No retrain happened
    if not is_per_detector_format and not is_kernel_format and not is_legacy_format:
        return None
    if retrain.get("status") in ("no_retrain_needed",):
        return None
    if retrain.get("action") in ("skip", "fast_promote"):
        return None

    evasion_path = RESULTS_DIR / "evasion_report.json"
    pre_evasion = {}
    attack_family = "unknown"
    if evasion_path.exists():
        er = json.loads(evasion_path.read_text(encoding="utf-8"))
        for det, v in er.get("per_detector", {}).items():
            pre_evasion[det] = round(v.get("evasion_rate", 0), 4)
        attack_family = list(er.get("per_family", {}).keys())[0] if er.get("per_family") else "unknown"

    decision_path = RESULTS_DIR / "pipeline_decision.json"
    argo_workflow = "unknown"
    if decision_path.exists():
        d = json.loads(decision_path.read_text(encoding="utf-8"))
        argo_workflow = d.get("argo_workflow", "unknown")

    if is_per_detector_format:
        # New format: per_detector dict — each detector trained and pushed independently
        per_det = retrain["per_detector"]
        detectors_built = {}
        for det, dr in per_det.items():
            detectors_built[det] = {
                "pre_evasion": dr.get("evasion_rate_before", pre_evasion.get(det)),
                "post_evasion": dr.get("evasion_rate_after"),
                "hf_model": dr.get("output_model"),
            }
        # Aggregate eval metrics across retrained detectors
        f1_vals = [r["eval_f1"] for r in per_det.values() if r.get("eval_f1") is not None]
        acc_vals = [r["eval_accuracy"] for r in per_det.values() if r.get("eval_accuracy") is not None]
        loss_vals = [r["eval_loss"] for r in per_det.values() if r.get("eval_loss") is not None]
        size_vals = [r["train_size"] for r in per_det.values() if r.get("train_size") is not None]
        return {
            "round": round_num,
            "hf_version": round_num,
            "attack_family": attack_family,
            "argo_workflow": argo_workflow,
            "detectors": detectors_built,
            "eval_f1": round(sum(f1_vals) / len(f1_vals), 4) if f1_vals else None,
            "eval_accuracy": round(sum(acc_vals) / len(acc_vals), 4) if acc_vals else None,
            "eval_loss": round(sum(loss_vals) / len(loss_vals), 4) if loss_vals else None,
            "train_size": sum(size_vals) if size_vals else None,
            "output_model": list(per_det.values())[0].get("output_model") if per_det else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    if is_kernel_format:
        # Old single-model kernel format
        output_model = retrain["output_model"]
        post_evasion_all = retrain.get("evasion_rate_after")
        targeted = [det for det, v in pre_evasion.items() if v and v >= 0.25] or list(pre_evasion.keys())
        detectors_built = {
            det: {
                "pre_evasion": pre_evasion.get(det),
                "post_evasion": post_evasion_all,
                "hf_model": output_model,
            }
            for det in targeted
        }
        return {
            "round": round_num,
            "hf_version": round_num,
            "attack_family": attack_family,
            "argo_workflow": argo_workflow,
            "detectors": detectors_built,
            "eval_f1": retrain.get("eval_f1"),
            "eval_accuracy": retrain.get("eval_accuracy"),
            "eval_loss": retrain.get("eval_loss"),
            "train_size": retrain.get("train_size"),
            "output_model": output_model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Legacy per-detector format
    version = round_num + 3
    base_ids = {
        "injection":          "Builder117/distilbert-prompt-injection",
        "jailbreak":          "Builder117/distilbert-jailbreak",
        "insecure_output":    "Builder117/distilbert-insecure-output",
        "indirect_injection": "Builder117/distilbert-indirect-injection",
    }
    detectors_built = {
        det: {
            "pre_evasion": pre_evasion.get(det),
            "post_evasion": retrain.get("post_evasion", {}).get(det),
            "hf_model": f"{base_ids.get(det, det)}-v{version}",
        }
        for det in retrain.get("detectors_retrained", [])
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
