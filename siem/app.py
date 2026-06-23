"""
Staging FastAPI server — wraps DistilBERT detectors via HF Space API.
Deployed in K8s as the STAGING_URL endpoint for smoke_test.py and canary_monitor.py.

Usage (local):
    HF_TOKEN=hf_xxx uvicorn siem.app:app --host 0.0.0.0 --port 8000

Usage (Docker/K8s):
    See jenkins/k8s/staging-deployment.yaml
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.shield_utils import score_sample, MODELS

app = FastAPI(title="Adversarial Guardrail — Staging Scorer", version="1.0")

VALID_MODELS = list(MODELS.keys())


class ScoreRequest(BaseModel):
    text: str
    model: str = "injection"


class ScoreResponse(BaseModel):
    label: str
    confidence: float
    evaded: bool
    model: str
    skipped: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "models": VALID_MODELS}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    if req.model not in VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{req.model}'. Valid: {VALID_MODELS}")

    if not os.environ.get("HF_TOKEN"):
        raise HTTPException(status_code=503, detail="HF_TOKEN not set — staging server cannot score")

    result = score_sample(req.text, req.model)

    if result.get("skipped"):
        raise HTTPException(status_code=503, detail=f"Scoring failed: {result.get('reason', 'unknown error')}")

    score_val = result["score"]
    evaded = result["evaded"]

    # Map detector to positive label name
    detector_labels = {
        "injection":          "INJECTION",
        "jailbreak":          "JAILBREAK",
        "insecure_output":    "MALICIOUS",
        "indirect_injection": "INDIRECT",
    }
    positive_label = detector_labels[req.model]
    label = "LEGIT" if evaded else positive_label

    return ScoreResponse(
        label=label,
        confidence=score_val,
        evaded=evaded,
        model=req.model,
        skipped=False,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
