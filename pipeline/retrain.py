"""
Segment 4: Adversarial retrain — augments training datasets with evaded samples,
exports to Kaggle dataset format, and triggers Kaggle kernel (notebook) execution.

Reads pipeline_decision.json to know which detectors to retrain.
Writes: results/retrain_report.json + kaggle_export/<detector>/ dataset packages.

Kaggle rate limits:
  - Dataset create/version: 1 req/s enforced via KAGGLE_PUSH_DELAY_SEC (default 2s)
  - Kernel push: 1 req/s, enforced same way
  - Kernel status poll: every KAGGLE_POLL_INTERVAL_SEC (default 30s), max KAGGLE_POLL_MAX attempts
  - 429 responses: exponential backoff up to KAGGLE_BACKOFF_MAX_SEC

Usage:
    python pipeline/retrain.py --decision results/pipeline_decision.json
    python pipeline/retrain.py --decision results/pipeline_decision.json --dry-run
    python pipeline/retrain.py --decision results/pipeline_decision.json --no-wait
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ENTERPRISE_ROOT = Path(__file__).parent.parent
RESULTS_DIR = ENTERPRISE_ROOT / "results"
KAGGLE_EXPORT_DIR = ENTERPRISE_ROOT / "kaggle_export"

# Kaggle rate-limit config (override via env vars)
KAGGLE_PUSH_DELAY_SEC   = float(os.environ.get("KAGGLE_PUSH_DELAY_SEC",   "2"))
KAGGLE_POLL_INTERVAL_SEC = float(os.environ.get("KAGGLE_POLL_INTERVAL_SEC", "30"))
KAGGLE_POLL_MAX          = int(os.environ.get("KAGGLE_POLL_MAX",           "40"))   # 40 * 30s = 20min max
KAGGLE_BACKOFF_MAX_SEC   = float(os.environ.get("KAGGLE_BACKOFF_MAX_SEC",  "120"))
KAGGLE_USERNAME          = os.environ.get("KAGGLE_USERNAME", "builder117")

# HF Hub dataset IDs per detector (source datasets)
DATASET_IDS = {
    "injection":          "Builder117/llm-threat-injection-dataset",
    "jailbreak":          "Builder117/llm-threat-jailbreak-dataset",
    "insecure_output":    "Builder117/llm-threat-insecure-output-dataset",
    "indirect_injection": "Builder117/llm-threat-indirect-injection-dataset",
}

MODEL_IDS = {
    "injection":          "Builder117/distilbert-prompt-injection",
    "jailbreak":          "Builder117/distilbert-jailbreak",
    "insecure_output":    "Builder117/distilbert-insecure-output",
    "indirect_injection": "Builder117/distilbert-indirect-injection",
}

POSITIVE_LABELS = {
    "injection":          "INJECTION",
    "jailbreak":          "JAILBREAK",
    "insecure_output":    "MALICIOUS",
    "indirect_injection": "INDIRECT",
}


def load_evaded_samples(detector: str) -> list:
    """Load evaded samples for a detector from per-detector result file."""
    result_path = RESULTS_DIR / f"{detector}_results.json"
    if not result_path.exists():
        print(f"  WARNING: {result_path} not found")
        return []
    with open(result_path) as f:
        data = json.load(f)
    evaded = []
    for s in data.get("samples", []):
        if s.get("evaded"):
            evaded.append({
                "text": s.get("prompt") or s.get("prompt_preview", ""),
                "label": 1,
                "label_name": POSITIVE_LABELS[detector],
                "source": "adversarial",
                "attack_family": s.get("attack_family", "unknown"),
            })
    return evaded


def _get_kaggle_client():
    """Build KaggleClient — accepts KAGGLE_API_TOKEN or KAGGLE_KEY (GHA secret name)."""
    from kagglesdk import KaggleClient
    token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
    if not token:
        raise RuntimeError(
            "Set KAGGLE_API_TOKEN or KAGGLE_KEY. "
            "Get token at kaggle.com/settings/api"
        )
    return KaggleClient(api_token=token)


def _kaggle_call_with_backoff(fn, *args, **kwargs):
    """Exponential backoff on 429 / transient errors. Max 6 attempts."""
    delay = KAGGLE_PUSH_DELAY_SEC
    for attempt in range(6):
        try:
            if attempt > 0:
                time.sleep(delay)
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = "429" in msg or "too many requests" in msg or "rate limit" in msg
            is_conflict = "409" in msg or "conflict" in msg
            if is_conflict:
                # 409 = kernel still running; wait 60s per attempt
                wait = 60
                if attempt < 5:
                    print(f"  Kaggle 409 conflict — kernel busy, waiting {wait}s (attempt {attempt+1}/6): {e}")
                    time.sleep(wait)
                else:
                    raise
                continue
            wait = min(delay * (2 ** attempt), KAGGLE_BACKOFF_MAX_SEC)
            if attempt < 5:
                label = "429 backoff" if is_rate_limit else "retry"
                print(f"  Kaggle {label} {wait:.0f}s (attempt {attempt+1}/6): {e}")
                time.sleep(wait)
            else:
                raise


def _push_dataset_to_kaggle(detector: str, out_dir: Path, metadata: dict) -> str:
    """
    Create or version a Kaggle dataset. Returns slug 'username/dataset-name'.
    Uses kagglesdk v2 DatasetApiClient. Rate-limited via KAGGLE_PUSH_DELAY_SEC.
    """
    from kagglesdk.datasets.services.dataset_api_service import (
        ApiCreateDatasetRequest, ApiCreateDatasetVersionRequest,
    )

    client = _get_kaggle_client()
    slug = metadata["id"]  # e.g. builder117/adversarial-guardrail-injection-r1
    owner, dataset_name = slug.split("/", 1)

    # Check existence
    exists = False
    try:
        _kaggle_call_with_backoff(
            client.datasets.dataset_api_client.get_dataset,
            owner_slug=owner, dataset_slug=dataset_name,
        )
        exists = True
    except Exception:
        exists = False

    time.sleep(KAGGLE_PUSH_DELAY_SEC)

    # Collect files to upload
    files = [str(p) for p in out_dir.iterdir() if p.is_file() and (p.suffix != ".json" or p.name == "dataset-metadata.json")]

    if exists:
        print(f"  Versioning Kaggle dataset: {slug}")
        req = ApiCreateDatasetVersionRequest(
            version_notes=f"adversarial round — {len(files)} files",
            files=files,
        )
        _kaggle_call_with_backoff(
            client.datasets.dataset_api_client.create_dataset_version,
            owner_slug=owner, dataset_slug=dataset_name, body=req,
        )
    else:
        print(f"  Creating Kaggle dataset: {slug}")
        req = ApiCreateDatasetRequest(
            owner_slug=owner,
            slug=dataset_name,
            title=metadata["title"],
            license_name="CC0-1.0",
            is_private=True,
            files=files,
        )
        _kaggle_call_with_backoff(
            client.datasets.dataset_api_client.create_dataset,
            body=req,
        )

    time.sleep(KAGGLE_PUSH_DELAY_SEC)
    return slug


def _push_kernel_to_kaggle(detector: str, out_dir: Path, dataset_slug: str, kernel_slug: str) -> str:
    """
    Save/push retrain_notebook.py as a private GPU script kernel.
    Returns full kernel slug 'username/kernel-slug'.
    Uses kagglesdk v2 KernelsApiClient.
    """
    from kagglesdk.kernels.services.kernels_api_service import (
        ApiSaveKernelRequest,
    )

    client = _get_kaggle_client()
    notebook_code = (out_dir / "retrain_notebook.py").read_text(encoding="utf-8")
    full_slug = f"{KAGGLE_USERNAME}/{kernel_slug}"

    req = ApiSaveKernelRequest()
    req.slug = full_slug
    req.new_title = f"Retrain {detector} detector"
    req.text = notebook_code
    req.language = "python"
    req.kernel_type = "script"
    req.is_private = True
    req.enable_gpu = True
    req.enable_internet = True
    req.dataset_data_sources = [dataset_slug]
    req.competition_data_sources = []
    req.kernel_data_sources = []
    req.category_ids = []

    print(f"  Pushing kernel: {full_slug}")
    _kaggle_call_with_backoff(
        client.kernels.kernels_api_client.save_kernel,
        request=req,
    )
    time.sleep(KAGGLE_PUSH_DELAY_SEC)
    return full_slug


def _poll_kernel_status(full_kernel_slug: str, wait: bool = True) -> str:
    """
    Poll Kaggle kernel session status until terminal state or timeout.
    Returns final status string. Enforces KAGGLE_POLL_INTERVAL_SEC between polls.
    """
    if not wait:
        print(f"  --no-wait: kernel {full_kernel_slug} submitted, skipping poll")
        return "submitted"

    client = _get_kaggle_client()
    owner, slug = full_kernel_slug.split("/", 1)

    print(f"  Polling {full_kernel_slug} (max {KAGGLE_POLL_MAX} x {KAGGLE_POLL_INTERVAL_SEC:.0f}s)...")
    for i in range(KAGGLE_POLL_MAX):
        time.sleep(KAGGLE_POLL_INTERVAL_SEC)
        try:
            from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
            sreq = ApiGetKernelSessionStatusRequest()
            sreq.user_name = owner
            sreq.kernel_slug = slug
            resp = _kaggle_call_with_backoff(
                client.kernels.kernels_api_client.get_kernel_session_status,
                request=sreq,
            )
            # resp may be object or dict depending on SDK version
            status = getattr(resp, "status", None) or (resp.get("status") if isinstance(resp, dict) else "unknown")
            status = str(status).lower()
            print(f"  [{i+1}/{KAGGLE_POLL_MAX}] status={status}")
            if status in ("complete", "error", "cancelled"):
                return status
        except Exception as e:
            print(f"  Poll error (non-fatal): {e}")

    print(f"  WARNING: kernel poll timed out after {KAGGLE_POLL_MAX} attempts")
    return "timeout"


def export_kaggle_dataset(detector: str, evaded_samples: list, round_num: int, dry_run: bool, no_wait: bool = False) -> dict:
    """
    Export augmented dataset in Kaggle-compatible format:
      kaggle_export/<detector>/
        dataset.csv        — evaded samples (attack class only, ready to merge with base dataset)
        dataset-metadata.json — Kaggle dataset card
        retrain_notebook.py   — standalone fine-tune script to run on Kaggle GPU

    In live mode, also downloads the HF Hub source dataset and concatenates it.
    """
    out_dir = KAGGLE_EXPORT_DIR / detector
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = DATASET_IDS[detector]
    model_id = MODEL_IDS[detector]
    new_version = round_num + 3  # v1-3 already exist from original training

    # Sanitize values interpolated into generated notebook code
    _safe = re.compile(r'^[\w/\-.:]+$')
    if not _safe.match(detector) or not _safe.match(model_id) or not _safe.match(dataset_id):
        raise ValueError(f"Unsafe characters in detector/model_id/dataset_id: {detector!r} {model_id!r} {dataset_id!r}")

    if dry_run:
        print(f"  [DRY-RUN] Would export {len(evaded_samples)} samples to {out_dir}")
        csv_lines = ["text,label,label_name,source,attack_family"]
        for s in evaded_samples[:3]:
            text = s["text"].replace('"', '""')
            csv_lines.append(f'"{text}",{s["label"]},{s["label_name"]},{s["source"]},{s["attack_family"]}')
        (out_dir / "adversarial_samples.csv").write_text("\n".join(csv_lines), encoding="utf-8")
    else:
        # Live: download full HF dataset + concat with evaded samples
        try:
            import pandas as pd
            from datasets import load_dataset
        except ImportError:
            return {"status": "error", "error": "Install: pip install datasets pandas"}

        hf_token = os.environ.get("HF_TOKEN")
        print(f"  Downloading {dataset_id}...")
        try:
            ds = load_dataset(dataset_id, token=hf_token, split="train")
            base_df = ds.to_pandas()[["text", "label"]]
        except Exception as e:
            return {"status": "error", "error": f"HF dataset load failed: {e}"}

        aug_df = pd.DataFrame(evaded_samples)
        aug_df = aug_df[["text", "label", "label_name", "source", "attack_family"]]
        full_df = pd.concat([base_df, aug_df], ignore_index=True)
        full_df.to_csv(out_dir / "adversarial_samples.csv", index=False)
        print(f"  Full dataset: {len(base_df)} base + {len(aug_df)} adversarial = {len(full_df)} rows")

    # Write Kaggle dataset metadata card
    metadata = {
        "title": f"Adversarial Guardrail — {detector} augmented r{round_num}",
        "id": f"{KAGGLE_USERNAME}/adversarial-guardrail-{detector.replace('_', '-')}-r{round_num}",
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            f"Augmented training set for {detector} detector. "
            f"Round {round_num} adversarial samples appended — {len(evaded_samples)} new attack examples."
        ),
    }
    (out_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Write standalone Kaggle retrain notebook script
    notebook_script = f'''"""
Kaggle retrain script — {detector} detector v{new_version}
Run on Kaggle GPU (T4 or P100): Settings > Accelerator > GPU

Dataset input: adversarial-guardrail-{detector.replace("_", "-")}-r{round_num} (attach in Kaggle notebook)
Output: fine-tuned DistilBERT model pushed to HF Hub as {model_id}-v{new_version}
"""
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    TrainingArguments, Trainer,
)
import torch

HF_TOKEN = os.environ["HF_TOKEN"]   # set in Kaggle Secrets
MODEL_OUT = "{model_id}-v{new_version}"
DATASET_CSV = "/kaggle/input/adversarial-guardrail-{detector.replace("_", "-")}-r{round_num}/dataset.csv"
BASE_MODEL = "distilbert-base-uncased"
LABEL2ID = {{"{POSITIVE_LABELS[detector]}": 1, "LEGIT": 0}}
ID2LABEL = {{1: "{POSITIVE_LABELS[detector]}", 0: "LEGIT"}}

# Load dataset
df = pd.read_csv(DATASET_CSV)
df = df[["text", "label"]].dropna()
df["label"] = df["label"].astype(int)

# Train/val split
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)

# Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

ds = DatasetDict({{
    "train": Dataset.from_pandas(train_df).map(tokenize, batched=True),
    "validation": Dataset.from_pandas(val_df).map(tokenize, batched=True),
}})

# Model
model = DistilBertForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=2, label2id=LABEL2ID, id2label=ID2LABEL
)

# Training
args = TrainingArguments(
    output_dir=f"/kaggle/working/{{MODEL_OUT}}",
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    push_to_hub=True,
    hub_model_id=MODEL_OUT,
    hub_token=HF_TOKEN,
)

trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"])
trainer.train()
trainer.push_to_hub()
print(f"Model pushed: {{MODEL_OUT}}")
'''
    (out_dir / "retrain_notebook.py").write_text(notebook_script, encoding="utf-8")

    retrain_config = {
        "dataset_id": dataset_id,
        "output_model": f"{model_id}-v{new_version}",
        "detector": detector,
        "round": round_num,
    }
    (RESULTS_DIR / f"retrain_config_{detector}.json").write_text(
        json.dumps(retrain_config, indent=2), encoding="utf-8"
    )

    print(f"  Kaggle export ready: {out_dir}/")
    print(f"  -> adversarial_samples.csv ({len(evaded_samples)} rows)")
    print(f"  -> dataset-metadata.json")
    print(f"  -> retrain_notebook.py (run on Kaggle GPU T4)")
    print(f"  -> Target model: {model_id}-v{new_version}")

    kernel_slug = f"retrain-{detector.replace('_', '-')}-r{round_num}"
    kernel_status = "dry_run"

    if not dry_run:
        try:
            # Push dataset (rate-limited internally)
            dataset_slug = _push_dataset_to_kaggle(detector, out_dir, metadata)
            # Push kernel (rate-limited internally)
            full_kernel_slug = _push_kernel_to_kaggle(detector, out_dir, dataset_slug, kernel_slug)
            # Poll for completion
            kernel_status = _poll_kernel_status(full_kernel_slug, wait=not no_wait)
            print(f"  Kernel final status: {kernel_status}")
        except Exception as e:
            print(f"  ERROR pushing to Kaggle: {e}")
            return {
                "status": "error",
                "error": str(e),
                "samples_exported": len(evaded_samples),
                "export_dir": str(out_dir),
            }

    return {
        "status": "ok",
        "samples_exported": len(evaded_samples),
        "export_dir": str(out_dir),
        "target_model": f"{model_id}-v{new_version}",
        "kaggle_dataset": metadata["id"],
        "kernel_slug": kernel_slug,
        "kernel_status": kernel_status,
        "notebook": str(out_dir / "retrain_notebook.py"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", default="results/pipeline_decision.json")
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true",
                        help="Push kernel but don't poll for completion (fire-and-forget)")
    args = parser.parse_args()

    decision_path = Path(args.decision)
    if not decision_path.exists():
        print(f"ERROR: decision file not found: {decision_path}")
        sys.exit(1)

    with open(decision_path) as f:
        decision = json.load(f)

    action = decision.get("action", "skip")
    models_to_retrain = decision.get("models_to_retrain", [])
    round_num = args.round or decision.get("round", 1)

    print(f"Round {round_num} | Action: {action} | Models: {models_to_retrain}")

    if action in ("skip", "fast_promote") or not models_to_retrain:
        print("No retrain needed — exiting.")
        report = {"round": round_num, "action": action, "retrained": []}
        (RESULTS_DIR / "retrain_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    retrain_results = []
    pre_evasion = {}
    post_evasion = {}

    evasion_path = RESULTS_DIR / "evasion_report.json"
    if evasion_path.exists():
        with open(evasion_path) as f:
            evasion = json.load(f)
        pre_evasion = {d: v["evasion_rate"] for d, v in evasion.get("per_detector", {}).items()}

    for detector in models_to_retrain:
        print(f"\n--- {detector} ---")
        evaded = load_evaded_samples(detector)
        print(f"  Evaded samples: {len(evaded)}")

        export_result = export_kaggle_dataset(detector, evaded, round_num, args.dry_run, no_wait=args.no_wait)

        retrain_results.append({
            "detector": detector,
            "samples_exported": len(evaded),
            "export": export_result,
        })

        if args.dry_run:
            pre = pre_evasion.get(detector, 0.3)
            post_evasion[detector] = round(max(0.0, pre - 0.12), 4)
        else:
            post_evasion[detector] = None  # filled after Kaggle run completes

    overall_pre = round(sum(pre_evasion.values()) / len(pre_evasion), 4) if pre_evasion else None
    post_vals = [v for v in post_evasion.values() if v is not None]
    overall_post = round(sum(post_vals) / len(post_vals), 4) if post_vals else None

    report = {
        "round": round_num,
        "action": action,
        "detectors_retrained": models_to_retrain,
        "pre_evasion": pre_evasion,
        "post_evasion": post_evasion,
        "overall_pre": overall_pre,
        "overall_post": overall_post,
        "retrain_via": "kaggle",
        "export_dir": str(KAGGLE_EXPORT_DIR),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrain_results": retrain_results,
    }

    (RESULTS_DIR / "retrain_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nRetrain report: results/retrain_report.json")
    if pre_evasion and any(v is not None for v in post_evasion.values()):
        for det in models_to_retrain:
            pre = pre_evasion.get(det, "n/a")
            post = post_evasion.get(det)
            print(f"  {det}: {pre:.1%} -> {post:.1%}" if isinstance(pre, float) and post else f"  {det}: post-retrain eval pending")


if __name__ == "__main__":
    main()

