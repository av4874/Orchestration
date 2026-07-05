"""
trigger_agents.py — Trigger a Kaggle T4 kernel that runs a ReAct agent with Qwen3-8B INT4.

Replaces direct `python agents/<agent>_agent.py` calls in Jenkins Phase 2 (live run).
GHA dry-run is unchanged — it calls agents directly with --dry-run (no LLM needed).

Usage:
    python pipeline/trigger_agents.py --agent red_team --round 1
    python pipeline/trigger_agents.py --agent blue_team --round 2
    python pipeline/trigger_agents.py --agent orchestrator --round 2
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ENTERPRISE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ENTERPRISE_ROOT))

# Reuse Kaggle helpers from retrain.py
from pipeline.retrain import (
    _get_kaggle_client,
    _kaggle_call_with_backoff,
    _poll_kernel_status,
    KAGGLE_USERNAME,
    KAGGLE_PUSH_DELAY_SEC,
)

AGENTS_EXPORT_DIR = ENTERPRISE_ROOT / "kaggle_export" / "agents"

# GHA run ID suffix keeps kernel slugs unique across retries — avoids 409 on rerun
_GHA_RUN_SUFFIX = os.environ.get("GITHUB_RUN_ID", "")[-6:] if os.environ.get("GITHUB_RUN_ID") else ""

# Kaggle dataset that caches Qwen3-8B weights — avoids 15-min HF download every run.
# Mounted at /kaggle/input/qwen3-8b-cache/ inside the kernel.
# Set to "" to disable (e.g. before the dataset has been created via save_model_kernel).
QWEN3_CACHE_DATASET = "amatullahvakhariya/qwen3-8b-cache"


def _inject_round_into_notebook(notebook_json: str, round_num: int) -> str:
    """
    Patch the notebook source so ROUND is hardcoded to round_num.
    Kaggle SaveKernel has no env-var injection — only way to pass ROUND is via code.
    Matches escaped-quote variant in raw JSON: ROUND = int(os.environ.get(\\"ROUND\\"...))
    """
    patched = re.sub(
        r'ROUND\s*=\s*int\(os\.environ\.get\([^)]*\)\)',
        f'ROUND = {round_num}',
        notebook_json,
        count=1,
    )
    if patched == notebook_json:
        print(f"  WARNING: ROUND injection pattern not found in notebook — kernel will use default ROUND")
    else:
        print(f"  Injected ROUND={round_num} into notebook code")
    return patched


def _inject_hf_token_into_notebook(notebook_json: str) -> str:
    """
    Inject HF_TOKEN from GHA environment into the notebook so the kernel can push to HF Hub.
    Kaggle secrets can't be set via API — patching is the only option for GHA-triggered kernels.
    The notebooks are private so embedding the token is acceptable.
    """
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("  WARNING: HF_TOKEN not set in environment — kernel will fail at HF Hub upload")
        return notebook_json
    # Replace the fallback assignment in the except block — UserSecretsClient wins if secret attached,
    # our injected value wins in GHA-triggered runs where no Kaggle secret is present.
    patched = notebook_json.replace(
        'HF_TOKEN = os.environ.get("HF_TOKEN", "")',
        f'HF_TOKEN = "{hf_token}"',
    )
    if patched == notebook_json:
        print("  WARNING: HF_TOKEN injection pattern not found in notebook")
    else:
        print("  Injected HF_TOKEN into notebook code")
    return patched


def _delete_stale_agent_kernels(client, round_num: int, extra_slugs: list = None):
    """
    Delete all known enterprise agent kernel slugs by brute-force.
    list_kernels doesn't return private kernels, so we enumerate slug patterns directly.
    Kaggle 409 on SaveKernel = another kernel running on account (free-tier limit).
    extra_slugs: additional exact slugs to delete (e.g. the current GHA-suffixed slug on retry).
    """
    from kagglesdk.kernels.types.kernels_api_service import ApiDeleteKernelRequest

    agents = ["red-team", "blue-team", "orchestrator"]
    deleted_any = False

    # All slug patterns ever used across old and new format
    candidate_slugs = list(extra_slugs or [])
    for ag in agents:
        for r in range(1, round_num + 2):
            candidate_slugs.append(f"enterprise-agent-{ag}-r{r}")   # old format
            candidate_slugs.append(f"enterprise-{ag}-r{r}")          # new format
            candidate_slugs.append(f"enterprise-{ag}-agent-round-{r}")  # title-derived format

    for slug in candidate_slugs:
        try:
            dreq = ApiDeleteKernelRequest()
            dreq.user_name = KAGGLE_USERNAME
            dreq.kernel_slug = slug
            client.kernels.kernels_api_client.delete_kernel(request=dreq)
            print(f"  Deleted stale kernel: {KAGGLE_USERNAME}/{slug}")
            deleted_any = True
        except Exception as e:
            msg = str(e).lower()
            if "404" in msg or "not found" in msg:
                pass  # didn't exist, fine
            elif "403" in msg or "forbidden" in msg:
                pass  # doesn't belong to us, fine
            else:
                print(f"  Delete {slug}: {e}")

    if deleted_any:
        print("  Waiting 15s for Kaggle to release slot...")
        time.sleep(15)


def _wait_for_kernel_idle(client, kernel_slug: str, max_wait_sec: int = 1800):
    """Wait for an existing kernel to reach a terminal state before pushing a new version."""
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
    owner, slug = kernel_slug.split("/", 1)
    print(f"  Checking if {kernel_slug} is already running...")
    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        try:
            req = ApiGetKernelSessionStatusRequest()
            req.user_name = owner
            req.kernel_slug = slug
            resp = _kaggle_call_with_backoff(
                client.kernels.kernels_api_client.get_kernel_session_status,
                request=req,
            )
            status = getattr(resp, "status", None) or (resp.get("status") if isinstance(resp, dict) else "unknown")
            status = str(status).lower()
            if "." in status:
                status = status.split(".")[-1]
            if status in ("complete", "error", "cancelled"):
                print(f"  Kernel idle (status={status}), safe to push.")
                return
            print(f"  Kernel still {status}, waiting 30s...")
            time.sleep(30)
        except Exception as e:
            msg = str(e).lower()
            if "404" in msg or "not found" in msg or "403" in msg or "forbidden" in msg:
                # No active session — safe to create/push
                print(f"  No active session — safe to push.")
                return
            print(f"  Status check error (proceeding anyway): {e}")
            return
    print(f"  WARNING: timed out waiting for {kernel_slug} to be idle after {max_wait_sec}s — attempting push anyway")


def _push_agent_kernel(agent: str, round_num: int) -> str:
    """Push the agent's Kaggle kernel notebook and return the full kernel slug."""
    from kagglesdk.kernels.services.kernels_api_service import ApiSaveKernelRequest

    notebook_path = AGENTS_EXPORT_DIR / f"{agent}_kernel.ipynb"
    if not notebook_path.exists():
        raise FileNotFoundError(f"Kernel notebook not found: {notebook_path}")

    notebook_code = notebook_path.read_text(encoding="utf-8")

    # Inject ROUND and HF_TOKEN — Kaggle API has no env var field, must patch notebook code
    notebook_code = _inject_round_into_notebook(notebook_code, round_num)
    notebook_code = _inject_hf_token_into_notebook(notebook_code)

    # Unique slug per GHA run avoids 409 when previous run's kernel is still alive
    slug_suffix = f"-{_GHA_RUN_SUFFIX}" if _GHA_RUN_SUFFIX else ""
    slug_only = f"enterprise-{agent.replace('_', '-')}-r{round_num}{slug_suffix}"
    kernel_slug = f"{KAGGLE_USERNAME}/{slug_only}"
    # Title must slugify to slug_only (Kaggle derives slug from title)
    kernel_title = slug_only.replace("-", " ").title()

    client = _get_kaggle_client()
    # Pass current slug so retry deletes the GHA-suffixed kernel too (not just base patterns)
    _delete_stale_agent_kernels(client, round_num, extra_slugs=[slug_only])
    _wait_for_kernel_idle(client, kernel_slug)

    req = ApiSaveKernelRequest()
    req.slug = kernel_slug
    req.new_title = kernel_title
    req.text = notebook_code
    req.language = "python"
    req.kernel_type = "notebook"
    req.is_private = True
    req.enable_gpu = True
    req.machine_shape = "NvidiaTeslaT4"
    req.enable_internet = True
    # Attach Qwen3-8B cache dataset if available — skips 15-min HF download per run
    dataset_sources = [QWEN3_CACHE_DATASET] if QWEN3_CACHE_DATASET else []
    req.dataset_data_sources = dataset_sources
    req.competition_data_sources = []
    req.kernel_data_sources = []
    req.category_ids = []

    print(f"  Pushing kernel: {kernel_slug}")
    _kaggle_call_with_backoff(client.kernels.kernels_api_client.save_kernel, request=req)
    time.sleep(KAGGLE_PUSH_DELAY_SEC)
    return kernel_slug


def _download_results(agent: str, round_num: int):
    """Download agent result files from HF Hub into local results/ and agent_traces/."""
    try:
        from huggingface_hub import hf_hub_download
        hf_token = os.environ.get("HF_TOKEN")
        repo_id = "Builder117/enterprise-adversarial-samples"

        result_files = {
            "red_team":     [f"results/round_{round_num}_samples.json",
                             f"agent_traces/round_{round_num}_red_team.json"],
            "blue_team":    [f"agent_workspace/blue_to_orchestrator.json",
                             f"agent_traces/round_{round_num}_blue_team.json"],
            "orchestrator": [f"results/pipeline_decision.json",
                             f"agent_traces/round_{round_num}_orchestrator.json",
                             f"pipeline/attack_memory.json"],
        }

        for remote_path in result_files.get(agent, []):
            local_path = ENTERPRISE_ROOT / remote_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                    token=hf_token,
                    local_dir=str(ENTERPRISE_ROOT),
                )
                print(f"  Downloaded: {remote_path}")
            except Exception as e:
                print(f"  WARNING: could not download {remote_path}: {e}")
    except ImportError:
        print("  WARNING: huggingface_hub not installed — skipping result download")


def trigger_agent(agent: str, round_num: int, no_wait: bool = False,
                  max_attempts: int = int(os.environ.get("KAGGLE_MAX_ATTEMPTS", "4"))):
    """Push kernel, poll until complete, download results. Retries up to 4x — P100 fails fast, retry re-rolls for T4."""
    print(f"[trigger_agents] {agent} round {round_num}")

    last_slug = None
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(f"  Retry {attempt}/{max_attempts} — kernel errored, re-pushing (hoping for T4)...")
            time.sleep(30)

        kernel_slug = _push_agent_kernel(agent, round_num)
        last_slug = kernel_slug
        status = _poll_kernel_status(kernel_slug, wait=not no_wait)

        if status == "complete":
            print(f"  Kernel complete. Downloading results...")
            _download_results(agent, round_num)
            return {"agent": agent, "round": round_num, "kernel": kernel_slug, "status": status}
        elif status in ("error", "cancelled"):
            if attempt < max_attempts:
                print(f"  Kernel {status} on attempt {attempt} — will retry")
                continue
            raise RuntimeError(f"Kaggle kernel {kernel_slug} ended with status={status} after {max_attempts} attempts")
        else:
            # timeout = kernel never finished in poll window — treat as failure
            raise RuntimeError(f"Kaggle kernel {kernel_slug} timed out after {max_attempts} poll cycles")

    raise RuntimeError(f"All {max_attempts} attempts failed for {agent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["red_team", "blue_team", "orchestrator"], required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--no-wait", action="store_true", help="Submit kernel, do not poll")
    args = parser.parse_args()
    result = trigger_agent(args.agent, args.round, no_wait=args.no_wait)
    print(json.dumps(result, indent=2))
