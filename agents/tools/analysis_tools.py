"""
analysis_tools.py — Blue Team tools for reading evasion reports and analyzing detector weaknesses.
"""
import json
import os
from pathlib import Path

from langchain_core.tools import tool

RESULTS_DIR = Path(os.environ.get("ENTERPRISE_ROOT", ".")) / "results"
_HF_REPO = "Builder117/enterprise-adversarial-samples"


def _fetch_evasion_report_from_hf() -> dict | None:
    """Download evasion_report.json from HF Hub when local file is absent (Kaggle kernel context)."""
    try:
        from huggingface_hub import hf_hub_download
        hf_token = os.environ.get("HF_TOKEN", "")
        local = hf_hub_download(
            _HF_REPO, "results/evasion_report.json",
            repo_type="dataset", token=hf_token or None,
            local_dir=str(RESULTS_DIR.parent),
        )
        return json.loads(Path(local).read_text(encoding="utf-8"))
    except Exception:
        return None

DETECTORS = ["injection", "jailbreak", "insecure_output", "indirect_injection"]
CRITICAL_THRESHOLD = 0.40
HIGH_THRESHOLD = 0.25
MEDIUM_THRESHOLD = 0.10


@tool
def read_evasion_report(filter_detector: str = "") -> str:
    """
    Read evasion_report.json from Jenkins merge stage.
    Pass filter_detector to focus on one detector or leave empty for full report.
    """
    report_path = RESULTS_DIR / "evasion_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = _fetch_evasion_report_from_hf()
        if report is None:
            return "ERROR: evasion_report.json not found locally or on HF Hub."

    if filter_detector:
        if filter_detector not in DETECTORS:
            return f"ERROR: unknown detector '{filter_detector}'. Valid: {DETECTORS}"
        data = report.get("per_detector", {}).get(filter_detector, {})
        return json.dumps({
            "detector": filter_detector,
            "evasion_rate": data.get("evasion_rate"),
            "evaded": data.get("evaded"),
            "total": data.get("total"),
            "round": report.get("round"),
        }, indent=2)

    # Return slim summary — full report stays on disk at results/evasion_report.json
    per_det = report.get("per_detector", {})
    return json.dumps({
        "round": report.get("round"),
        "per_detector": {
            d: {"evasion_rate": v.get("evasion_rate"), "evaded": v.get("evaded"), "total": v.get("total")}
            for d, v in per_det.items()
        },
    }, indent=2)


@tool
def analyze_weakness(analysis_input: str) -> str:
    """
    Analyze detector weaknesses from evasion_report.json.
    Input JSON: {"detector": "injection", "family": "base64_encoding"} or {"detector": "all"}.
    Returns weakness scores per detector/family with severity and retrain recommendations.
    """
    try:
        args = json.loads(analysis_input)
    except json.JSONDecodeError as e:
        return f"ERROR: invalid JSON — {e}"

    report_path = RESULTS_DIR / "evasion_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = _fetch_evasion_report_from_hf()
    if report is None:
        return json.dumps({
            "note": "No evasion_report.json — using prior-round estimates.",
            "weaknesses": [
                {"detector": "injection", "family": "unicode_homograph", "evasion": 0.70, "severity": "CRITICAL"},
                {"detector": "indirect_injection", "family": "html_comment_smuggling", "evasion": 0.35, "severity": "HIGH"},
                {"detector": "jailbreak", "family": "roleplay_framing", "evasion": 0.22, "severity": "LOW"},
                {"detector": "insecure_output", "family": "context_flooding", "evasion": 0.18, "severity": "LOW"},
            ],
            "retrain_priority": ["injection", "indirect_injection"],
        }, indent=2)

    per_detector = report.get("per_detector", {})
    per_family = report.get("per_family", {})

    target_detector = args.get("detector", "all")
    target_family = args.get("family")

    # per_family is keyed by attack family name, not detector name.
    # Each entry has a "detectors" sub-dict with per-detector breakdowns.
    weaknesses = []
    for fam, fam_data in per_family.items():
        if target_family and fam != target_family:
            continue

        # Determine which detectors this family hits
        det_map = fam_data.get("detectors", {})
        for det, det_data in det_map.items():
            if target_detector != "all" and det != target_detector:
                continue
            evasion = det_data.get("evasion_rate", 0.0)
            if evasion == 0 and det_data.get("total", 0) == 0:
                continue
            severity = (
                "CRITICAL" if evasion >= CRITICAL_THRESHOLD else
                "HIGH"     if evasion >= HIGH_THRESHOLD else
                "MEDIUM"   if evasion >= MEDIUM_THRESHOLD else
                "LOW"
            )
            weaknesses.append({
                "detector": det,
                "family": fam,
                "evasion": evasion,
                "samples_evaded": det_data.get("evaded", 0),
                "samples_total": det_data.get("total", 0),
                "severity": severity,
                "gap_score": round(evasion * 1.15, 3),
            })

    weaknesses.sort(key=lambda x: x["evasion"], reverse=True)
    retrain_priority = list(dict.fromkeys(w["detector"] for w in weaknesses if w["severity"] in ("CRITICAL", "HIGH")))

    # Strip unused fields — LLM only needs detector/family/evasion/severity to route
    slim = [{"detector": w["detector"], "family": w["family"], "evasion": w["evasion"], "severity": w["severity"]}
            for w in weaknesses]

    return json.dumps({
        "round": report.get("round") if report_path.exists() else None,
        "weaknesses": slim,
        "retrain_priority": retrain_priority,
        "recommendation": f"Retrain {retrain_priority} — {sum(1 for w in weaknesses if w['severity']=='CRITICAL')} critical gaps.",
    }, indent=2)
