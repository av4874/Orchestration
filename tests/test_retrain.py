"""
Retrain gate: validates post-retrain evasion lower than pre-retrain.
For LLM Threat Shield: compares detector evasion rates before/after HF dataset augmentation.
Run: pytest tests/test_retrain.py -v
"""

import json
import sys
from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).parent.parent / "results"


def test_retrain_report_exists():
    path = RESULTS_DIR / "retrain_report.json"
    assert path.exists(), "retrain_report.json not found. Run pipeline/retrain.py first."


def test_post_retrain_evasion_lower():
    path = RESULTS_DIR / "retrain_report.json"
    if not path.exists():
        pytest.skip("retrain_report.json not found")
    with open(path) as f:
        report = json.load(f)

    if report.get("action") == "skip":
        pytest.skip("No retrain was performed this round")

    overall_pre = report.get("overall_pre")
    overall_post = report.get("overall_post")

    if overall_post is None:
        pytest.skip("Post-retrain evasion not yet available — retrain may be in progress on Colab")

    assert overall_post < overall_pre, (
        f"Post-retrain evasion {overall_post:.1%} not lower than pre-retrain {overall_pre:.1%}. "
        "Retrain did not improve robustness — check augmentation samples."
    )


def test_per_detector_improvement():
    path = RESULTS_DIR / "retrain_report.json"
    if not path.exists():
        pytest.skip("retrain_report.json not found")
    with open(path) as f:
        report = json.load(f)

    if report.get("action") == "skip":
        pytest.skip("No retrain was performed this round")

    pre = report.get("pre_evasion", {})
    post = report.get("post_evasion", {})

    for detector in report.get("detectors_retrained", []):
        pre_rate = pre.get(detector, 0)
        post_rate = post.get(detector)
        if post_rate is None:
            pytest.skip(f"{detector}: post-retrain eval pending")
        assert post_rate <= pre_rate, (
            f"{detector}: post-retrain evasion {post_rate:.1%} >= pre-retrain {pre_rate:.1%}"
        )


def test_retrain_configs_written():
    path = RESULTS_DIR / "retrain_report.json"
    if not path.exists():
        pytest.skip("retrain_report.json not found")
    with open(path) as f:
        report = json.load(f)

    if report.get("action") == "skip":
        pytest.skip("No retrain was performed this round")

    for detector in report.get("detectors_retrained", []):
        config_path = RESULTS_DIR / f"retrain_config_{detector}.json"
        assert config_path.exists(), f"retrain_config_{detector}.json not found — retrain.py did not write config"
        with open(config_path) as f:
            config = json.load(f)
        assert "dataset_id" in config, f"retrain_config_{detector}.json missing dataset_id"
        assert "output_model" in config, f"retrain_config_{detector}.json missing output_model"
