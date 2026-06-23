"""
Segment 5: Generate per-round HTML scorecard from evasion_report.json + attack_memory.json.

Usage:
    python reporting/scorecard.py --round 1 --output results/scorecard.html
    python reporting/scorecard.py --round 1  # outputs to results/scorecard.html by default
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

ENTERPRISE_ROOT = Path(os.environ.get("ENTERPRISE_ROOT", Path(__file__).parent.parent))
RESULTS_DIR = ENTERPRISE_ROOT / "results"
TEMPLATES_DIR = Path(__file__).parent / "templates"

DETECTOR_LABELS = {
    "injection": "Prompt Injection",
    "jailbreak": "Jailbreak",
    "insecure_output": "Insecure Output",
    "indirect_injection": "Indirect Injection",
}

SEVERITY_COLORS = {
    "critical": "#dc2626",
    "high":     "#ea580c",
    "medium":   "#d97706",
    "low":      "#16a34a",
    "none":     "#6b7280",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def build_context(round_num: int) -> dict:
    evasion = load_json(RESULTS_DIR / "evasion_report.json")
    memory  = load_json(ENTERPRISE_ROOT / "pipeline" / "attack_memory.json")
    decision = load_json(RESULTS_DIR / "pipeline_decision.json")

    # Per-detector rows
    detectors = []
    per_detector = evasion.get("per_detector", {})
    for det, info in per_detector.items():
        rate = info.get("evasion_rate", 0)
        detectors.append({
            "name": det,
            "label": DETECTOR_LABELS.get(det, det),
            "evasion_pct": f"{rate:.1%}",
            "evasion_raw": rate,
            "evaded": info.get("evaded", 0),
            "total": info.get("total_scored", 0),
            "bar_width": int(rate * 100),
            "bar_color": "#dc2626" if rate >= 0.40 else "#ea580c" if rate >= 0.25 else "#16a34a",
        })

    # Per-family rows
    families = []
    for fam, info in evasion.get("per_family", {}).items():
        rate = info.get("evasion_rate", 0)
        # per_family has a "detectors" sub-dict; pick the detector with most samples as primary
        det_map = info.get("detectors", {})
        primary_detector = max(det_map, key=lambda d: det_map[d].get("total", 0)) if det_map else info.get("detector", "")
        families.append({
            "name": fam,
            "detector": primary_detector,
            "evasion_pct": f"{rate:.1%}",
            "evasion_raw": rate,
            "evaded": info.get("evaded", 0),
            "total": info.get("total", 0),
        })
    families.sort(key=lambda x: x["evasion_raw"], reverse=True)

    # Round history from memory
    rounds_history = memory.get("rounds", [])

    # Orchestrator decision
    severity = decision.get("severity", "none")

    return {
        "round": round_num,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "overall_evasion": evasion.get("overall_evasion_rate", 0),
        "overall_evasion_pct": f"{evasion.get('overall_evasion_rate', 0):.1%}",
        "severity": severity,
        "severity_color": SEVERITY_COLORS.get(severity, "#6b7280"),
        "action": decision.get("action", "n/a"),
        "argo_workflow": decision.get("argo_workflow", "n/a"),
        "reason": decision.get("reason", ""),
        "models_to_retrain": decision.get("models_to_retrain", []),
        "confidence": decision.get("confidence", decision.get("agent_confidence", 0)),
        "detectors": detectors,
        "families": families,
        "rounds_history": rounds_history,
        "current_focus": memory.get("current_focus", []),
        "known_blind_spots": memory.get("known_blind_spots", []),
    }


def render(context: dict, output_path: Path):
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("scorecard.html.j2")
    html = template.render(**context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Scorecard written: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else RESULTS_DIR / "scorecard.html"
    context = build_context(args.round)
    render(context, output_path)


if __name__ == "__main__":
    main()
