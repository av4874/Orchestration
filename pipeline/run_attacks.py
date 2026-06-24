"""
Segment 4: Run adversarial samples against LLM Threat Shield's 4 DistilBERT detectors.
Calls Builder117/Orchestration HF Space models via HF Inference API â€” no local clone needed.

Usage:
    python pipeline/run_attacks.py --samples results/round_1_samples.json
    python pipeline/run_attacks.py --samples results/round_1_samples.json --detector injection
    python pipeline/run_attacks.py --samples results/round_1_samples.json --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.shield_utils import MODELS, score_sample

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_detector(samples: list, detector: str) -> dict:
    results = []
    evaded_count = 0
    total = 0

    for s in samples:
        result = score_sample(s["prompt"], detector)
        total += 1
        if result.get("evaded"):
            evaded_count += 1
        results.append({
            "id": s["id"],
            "attack_family": s["attack_family"],
            "detector": detector,
            "prompt": s["prompt"],
            "prompt_preview": s["prompt"][:120],
            **result,
        })

    evasion_rate = round(evaded_count / total, 4) if total else 0.0
    return {
        "detector": detector,
        "total_scored": total,
        "evaded": evaded_count,
        "evasion_rate": evasion_rate,
        "samples": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="results/round_1_samples.json")
    parser.add_argument("--detector", default="all", choices=["all"] + list(MODELS.keys()))
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    samples_path = Path(args.samples)
    if not samples_path.exists():
        print(f"ERROR: samples file not found: {samples_path}")
        sys.exit(1)

    with open(samples_path) as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples from {samples_path}")

    if args.dry_run:
        print("DRY-RUN: generating mock scores (no HF API calls)")
        for detector in MODELS:
            mock_results = []
            for i, s in enumerate(samples):
                mock_results.append({
                    "id": s["id"],
                    "attack_family": s["attack_family"],
                    "detector": detector,
                    "prompt_preview": s["prompt"][:120],
                    "score": round(0.15 + (i % 3) * 0.2, 4),
                    "evaded": (i % 3) != 2,
                    "skipped": False,
                })
            evaded = sum(1 for r in mock_results if r["evaded"])
            out = {
                "detector": detector,
                "total_scored": len(mock_results),
                "evaded": evaded,
                "evasion_rate": round(evaded / len(mock_results), 4) if mock_results else 0.0,
                "samples": mock_results,
            }
            out_path = RESULTS_DIR / f"{detector}_results.json"
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  [{detector}] evasion_rate={out['evasion_rate']:.1%} (mock) -> {out_path}")
        return

    if not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN not set. Use --dry-run or set HF_TOKEN.")
        sys.exit(1)

    detectors_to_run = list(MODELS.keys()) if args.detector == "all" else [args.detector]

    for detector in detectors_to_run:
        print(f"  Scoring against {detector} detector...")
        result = run_detector(samples, detector)
        out_path = RESULTS_DIR / f"{detector}_results.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [{detector}] evasion_rate={result['evasion_rate']:.1%} ({result['evaded']}/{result['total_scored']}) -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()

