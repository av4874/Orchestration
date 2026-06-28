"""
Download retrained DistilBERT weights from HF Hub into models/v{round}/.
Run after Kaggle retrain completes, before Argo canary deploy.

Kaggle notebook pushes: Builder117/distilbert-{detector}-v{round+3}
This script pulls those weights into: models/v{round}/{detector}/

Usage:
    python pipeline/download_models.py --round 2
    python pipeline/download_models.py --round 2 --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

ENTERPRISE_ROOT = Path(__file__).parent.parent
RESULTS_DIR = ENTERPRISE_ROOT / "results"

# Must match retrain.py: new_version = round_num + 3
VERSION_OFFSET = 3

MODEL_IDS = {
    "injection":          "Builder117/distilbert-prompt-injection",
    "jailbreak":          "Builder117/distilbert-jailbreak",
    "insecure_output":    "Builder117/distilbert-insecure-output",
    "indirect_injection": "Builder117/distilbert-indirect-injection",
}


def download_model(detector: str, hf_model_id: str, local_dir: Path, dry_run: bool) -> dict:
    if dry_run:
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text(
            json.dumps({"model_type": "distilbert", "detector": detector, "dry_run": True}),
            encoding="utf-8",
        )
        print(f"  [dry-run] Would download {hf_model_id} -> {local_dir}")
        return {"status": "dry_run", "model_id": hf_model_id, "local_dir": str(local_dir)}

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    print(f"  Downloading {hf_model_id} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=hf_model_id,
            local_dir=str(local_dir),
            token=hf_token,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        return {"status": "ok", "model_id": hf_model_id, "local_dir": str(local_dir)}
    except Exception as e:
        return {"status": "error", "model_id": hf_model_id, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    round_num = args.round
    version = round_num + VERSION_OFFSET

    # Read retrain_report to know which detectors were actually retrained
    retrain_report_path = RESULTS_DIR / "retrain_report.json"
    if retrain_report_path.exists():
        with open(retrain_report_path) as f:
            report = json.load(f)
        detectors = report.get("detectors_retrained", [])
        if not detectors:
            print(f"No detectors retrained in round {round_num} — nothing to download.")
            return
    else:
        print(f"WARNING: retrain_report.json not found — downloading all 4 detectors")
        detectors = list(MODEL_IDS.keys())

    print(f"Round {round_num} | Downloading v{version} weights for: {detectors}")

    models_dir = ENTERPRISE_ROOT / "models" / f"v{round_num}"
    results = []

    for detector in detectors:
        base_id = MODEL_IDS[detector]
        versioned_id = f"{base_id}-v{version}"
        local_dir = models_dir / detector
        result = download_model(detector, versioned_id, local_dir, args.dry_run)
        results.append({"detector": detector, **result})
        if result["status"] == "error":
            print(f"  ERROR {detector}: {result['error']}")
        else:
            print(f"  OK {detector}: {result['status']}")

    failed = [r for r in results if r["status"] == "error"]

    # Manifest inside models/v{round}/ for Argo package-model-tpl
    manifest = {
        "round": round_num,
        "version": version,
        "detectors": detectors,
        "models_dir": str(models_dir),
        "results": results,
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = models_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written: {manifest_path}")

    # Signal file for pipeline checklist — results/round_{N}_models_ready.json
    from datetime import datetime, timezone
    RESULTS_DIR.mkdir(exist_ok=True)
    signal = {
        "round": round_num,
        "ready": len(failed) == 0,
        "detectors_downloaded": [r["detector"] for r in results if r["status"] != "error"],
        "detectors_failed": [r["detector"] for r in failed],
        "models_dir": str(models_dir),
        "hf_version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    signal_path = RESULTS_DIR / f"round_{round_num}_models_ready.json"
    signal_path.write_text(json.dumps(signal, indent=2), encoding="utf-8")
    print(f"Signal written:   {signal_path}")

    if failed:
        print(f"\nFAILED: {[r['detector'] for r in failed]}")
        sys.exit(1)

    print(f"All weights in {models_dir} — Argo can now package.")


if __name__ == "__main__":
    main()
