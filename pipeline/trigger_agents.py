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
DATASET_SOURCE = f"{KAGGLE_USERNAME}/enterprise-adversarial-samples"


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
            if status in ("complete", "error", "cancelled"):
                print(f"  Kernel idle (status={status}), safe to push.")
                return
            print(f"  Kernel still {status}, waiting 30s...")
            time.sleep(30)
        except Exception as e:
            msg = str(e).lower()
            if "404" in msg or "not found" in msg:
                print(f"  Kernel doesn't exist yet, safe to create.")
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
    kernel_slug = f"{KAGGLE_USERNAME}/enterprise-agent-{agent.replace('_', '-')}-r{round_num}"

    client = _get_kaggle_client()
    _wait_for_kernel_idle(client, kernel_slug)

    req = ApiSaveKernelRequest()
    req.slug = kernel_slug
    req.new_title = f"Enterprise {agent} agent round {round_num}"
    req.text = notebook_code
    req.language = "python"
    req.kernel_type = "notebook"
    req.is_private = True
    req.enable_gpu = True
    req.enable_internet = True
    req.dataset_data_sources = [DATASET_SOURCE]
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
        from huggingface_hub import hf_hub_download, list_repo_files
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
                downloaded = hf_hub_download(
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


def trigger_agent(agent: str, round_num: int, no_wait: bool = False):
    """Push kernel, poll until complete, download results."""
    print(f"[trigger_agents] {agent} round {round_num}")

    kernel_slug = _push_agent_kernel(agent, round_num)
    status = _poll_kernel_status(kernel_slug, wait=not no_wait)

    if status == "complete":
        print(f"  Kernel complete. Downloading results...")
        _download_results(agent, round_num)
    elif status in ("error", "cancelled"):
        raise RuntimeError(f"Kaggle kernel {kernel_slug} ended with status={status}")
    else:
        print(f"  Kernel status={status} — results may not be ready")

    return {"agent": agent, "round": round_num, "kernel": kernel_slug, "status": status}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["red_team", "blue_team", "orchestrator"], required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--no-wait", action="store_true", help="Submit kernel, do not poll")
    args = parser.parse_args()
    result = trigger_agent(args.agent, args.round, no_wait=args.no_wait)
    print(json.dumps(result, indent=2))
