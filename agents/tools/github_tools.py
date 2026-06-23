"""
GitHub artifact download tool — used by Jenkins download stage.
Kept minimal: primary GH interaction is via routing_tools.open_github_issue.
"""

import json
import os
import zipfile
from pathlib import Path

import requests

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def download_artifact(input_json: str) -> str:
    """
    Download a GitHub Actions artifact by round number.
    Input: JSON with keys: round (int), repo (optional, falls back to GITHUB_REPO env).
    Writes artifact to results/round_N_samples.json.
    Requires GITHUB_TOKEN env var.
    """
    try:
        params = json.loads(input_json)
    except json.JSONDecodeError as e:
        return f"ERROR: invalid JSON — {e}"

    round_num = int(params.get("round", 1))
    repo = params.get("repo") or os.environ.get("GITHUB_REPO")
    token = os.environ.get("GITHUB_TOKEN")

    if not token:
        return "DRY_RUN: GITHUB_TOKEN not set — would download artifact from GitHub Actions."
    if not repo:
        return "ERROR: GITHUB_REPO not set."

    artifact_name = f"round-{round_num}-attack-artifacts"

    try:
        resp = requests.get(
            f"https://api.github.com/repos/{repo}/actions/artifacts",
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"},
            params={"name": artifact_name, "per_page": 5},
            timeout=15,
        )
        if resp.status_code != 200:
            return f"ERROR: GitHub API {resp.status_code}: {resp.text[:200]}"

        artifacts = resp.json().get("artifacts", [])
        if not artifacts:
            return f"ERROR: no artifact named '{artifact_name}' found."

        artifact = artifacts[0]
        download_url = artifact["archive_download_url"]

        dl = requests.get(
            download_url,
            headers={"Authorization": f"token {token}"},
            timeout=60,
            stream=True,
        )

        zip_path = RESULTS_DIR / f"round_{round_num}_artifact.zip"
        with open(zip_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RESULTS_DIR)

        zip_path.unlink(missing_ok=True)
        return f"Artifact '{artifact_name}' downloaded and extracted to {RESULTS_DIR}"

    except requests.RequestException as e:
        return f"ERROR: download failed — {e}"
