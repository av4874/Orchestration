"""
Quality gate: validates pipeline_decision.json schema and routing logic.
Run: pytest tests/test_guardrail.py -v
"""

import json
import sys
from pathlib import Path

import jsonschema
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"

VALID_DETECTORS = {"injection", "jailbreak", "insecure_output", "indirect_injection"}

DECISION_SCHEMA = {
    "type": "object",
    "required": ["round", "action", "models_to_retrain", "severity", "confidence", "reason", "argo_workflow"],
    "properties": {
        "round":             {"type": "integer"},
        "action":            {"type": "string", "enum": ["retrain", "partial_retrain", "fast_promote", "emergency_rollback", "skip"]},
        "models_to_retrain": {"type": "array", "items": {"type": "string"}},
        "severity":          {"type": "string", "enum": ["critical", "high", "medium", "low", "none"]},
        "confidence":        {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason":            {"type": "string", "minLength": 10},
        "argo_workflow":     {"type": "string", "enum": ["full-canary", "fast-promote", "emergency-rollback", "none"]},
    },
}

# action -> allowed argo_workflows
WORKFLOW_ROUTING = {
    "retrain":            ["full-canary"],
    "partial_retrain":    ["full-canary"],
    "fast_promote":       ["fast-promote"],
    "emergency_rollback": ["emergency-rollback"],
    "skip":               ["fast-promote", "none"],
}


def test_decision_file_exists():
    path = RESULTS_DIR / "pipeline_decision.json"
    assert path.exists(), f"pipeline_decision.json not found at {path}. Run orchestrator_agent.py first."


def test_decision_schema_valid():
    path = RESULTS_DIR / "pipeline_decision.json"
    if not path.exists():
        pytest.skip("pipeline_decision.json not found")
    with open(path) as f:
        decision = json.load(f)
    try:
        jsonschema.validate(decision, DECISION_SCHEMA)
    except jsonschema.ValidationError as e:
        pytest.fail(f"pipeline_decision.json schema invalid: {e.message}")


def test_routing_consistency():
    path = RESULTS_DIR / "pipeline_decision.json"
    if not path.exists():
        pytest.skip("pipeline_decision.json not found")
    with open(path) as f:
        decision = json.load(f)
    action = decision["action"]
    workflow = decision["argo_workflow"]
    allowed = WORKFLOW_ROUTING.get(action, [])
    assert workflow in allowed, (
        f"Inconsistent routing: action='{action}' should use {allowed}, got '{workflow}'"
    )


def test_retrain_has_valid_detectors():
    path = RESULTS_DIR / "pipeline_decision.json"
    if not path.exists():
        pytest.skip("pipeline_decision.json not found")
    with open(path) as f:
        decision = json.load(f)
    if decision["action"] in ("retrain", "partial_retrain"):
        models = decision["models_to_retrain"]
        assert len(models) > 0, f"action='{decision['action']}' but models_to_retrain is empty"
        invalid = [m for m in models if m not in VALID_DETECTORS]
        assert not invalid, f"Unknown detectors: {invalid}. Valid: {VALID_DETECTORS}"


def test_reason_not_empty():
    path = RESULTS_DIR / "pipeline_decision.json"
    if not path.exists():
        pytest.skip("pipeline_decision.json not found")
    with open(path) as f:
        decision = json.load(f)
    assert len(decision.get("reason", "")) >= 10, "reason field must be at least 10 characters"


def test_confidence_in_range():
    path = RESULTS_DIR / "pipeline_decision.json"
    if not path.exists():
        pytest.skip("pipeline_decision.json not found")
    with open(path) as f:
        decision = json.load(f)
    conf = decision.get("confidence", -1)
    assert 0.0 <= conf <= 1.0, f"confidence {conf} out of range [0, 1]"
