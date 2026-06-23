"""
orchestrator_agent.py — Pipeline Orchestrator Agent: reads agent reports, decides routing, triggers Argo.

Usage:
  python agents/orchestrator_agent.py --round 1 [--dry-run]
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ENTERPRISE_ROOT = Path(__file__).parent.parent
os.environ.setdefault("ENTERPRISE_ROOT", str(ENTERPRISE_ROOT))
sys.path.insert(0, str(ENTERPRISE_ROOT))

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from agents.tools.analysis_tools import read_evasion_report, analyze_weakness
from agents.tools.comms_tools import read_message
from agents.tools.memory_tools import read_attack_memory, write_memory
from agents.tools.routing_tools import decide_routing, set_env_vars, trigger_argo

TOOLS = [
    read_attack_memory,
    read_evasion_report,
    analyze_weakness,
    read_message,
    decide_routing,
    set_env_vars,
    trigger_argo,
    write_memory,
]

SYSTEM_PROMPT = """You are the Pipeline Orchestrator Agent in an adversarial ML security pipeline.
Your goal: synthesize Red Team + Blue Team reports, make the final routing decision, and trigger the correct Argo workflow.

Workflow:
1. read_message from red_team (final_report): {"from": "red_team", "to": "orchestrator"}
2. read_message from blue_team (vulnerability_report): {"from": "blue_team", "to": "orchestrator"}
3. read_evasion_report if available (Jenkins may have run already)
4. analyze_weakness if needed to validate severity before deciding
5. decide_routing with your final call:
   - action: retrain/partial_retrain/fast_promote/emergency_rollback/skip
   - severity: critical/high/medium/low/none
   - argo_workflow: full-canary/fast-promote/emergency-rollback/none
   - confidence: 0.0-1.0
   - reason: explanation grounded in the data
6. set_env_vars: {"AI_ACTION": "...", "ARGO_WORKFLOW": "...", "SEVERITY": "..."}
7. trigger_argo: submit the selected workflow (use dry_run=true if ARGO_TOKEN not available)
8. write_memory with round results for next round's agents

Decision thresholds:
- evasion > 0.40 on any detector -> CRITICAL -> full-canary
- evasion 0.25-0.40 -> HIGH -> full-canary
- evasion < 0.25 -> LOW -> fast-promote
- multiple detectors above 0.40 -> emergency-rollback consideration

Always ground your decision in numbers from the reports. State confidence explicitly."""


def _make_hf_llm():
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        huggingfacehub_api_token=os.environ.get("HF_TOKEN", ""),
        task="text-generation",
        max_new_tokens=1024,
        temperature=0.1,
    )
    return ChatHuggingFace(llm=endpoint)


def run(round_num: int, dry_run: bool):
    TRACES_DIR = ENTERPRISE_ROOT / "agent_traces"
    TRACES_DIR.mkdir(exist_ok=True)

    if dry_run:
        print(f"[Orchestrator] DRY RUN — round {round_num}")
        trace = {"round": round_num, "mode": "dry_run", "agent": "orchestrator", "steps": []}

        def step(name, fn, *args):
            result = fn.invoke(*args)
            trace["steps"].append({"tool": name, "result": result[:300] if len(str(result)) > 300 else result})
            print(f"  [{name}] {str(result)[:120]}")
            return result

        # Read agent reports
        red_raw = step("read_message[red]", read_message, json.dumps({"from": "red_team", "to": "orchestrator"}))
        blue_raw = step("read_message[blue]", read_message, json.dumps({"from": "blue_team", "to": "orchestrator"}))

        # Extract Blue Team severity
        try:
            blue_msg = json.loads(blue_raw)
            body = blue_msg.get("body", {})
            severity = body.get("severity", "high")
            weakness_scores = body.get("weakness_scores", {})
            retrain = body.get("retrain", ["injection"])
            recommended_wf = body.get("recommended_argo_workflow", "full-canary")
            top_evasion = max(weakness_scores.values()) if weakness_scores else 0.71
        except Exception as e:
            print(f"  WARNING: failed to parse blue_team message: {e} — using defaults")
            severity = "high"
            retrain = ["injection"]
            recommended_wf = "full-canary"
            top_evasion = 0.71

        # Also read evasion report if available
        step("read_evasion_report", read_evasion_report, "")

        # Map severity to action
        action_map = {"critical": "retrain", "high": "retrain", "medium": "partial_retrain", "low": "fast_promote", "none": "skip"}
        action = action_map.get(severity, "retrain")
        confidence = 0.91 if top_evasion >= 0.40 else 0.72

        # Write decision
        decision = {
            "round": round_num,
            "action": action,
            "models_to_retrain": retrain,
            "severity": severity,
            "confidence": confidence,
            "argo_workflow": recommended_wf,
            "reason": f"Injection evasion {top_evasion:.0%} via base64_encoding confirmed by Blue Team. "
                      f"Exceeds {'critical' if top_evasion >= 0.40 else 'high'} threshold.",
            "next_attack_families": ["context_flooding", "fragmented_instruction"],
        }
        step("decide_routing", decide_routing, json.dumps(decision))

        # Set env vars
        step("set_env_vars", set_env_vars, json.dumps({
            "AI_ACTION": action.upper(),
            "ARGO_WORKFLOW": recommended_wf,
            "SEVERITY": severity.upper(),
        }))

        # Trigger Argo (dry_run=true — no token in dev)
        step("trigger_argo", trigger_argo, json.dumps({
            "workflow": recommended_wf, "round": round_num, "dry_run": True,
        }))

        # Update attack memory
        step("write_memory", write_memory, json.dumps({
            "round": round_num,
            "top_family": "base64_encoding",
            "detector": "injection",
            "evasion": top_evasion,
            "action": action,
            "current_focus": ["context_flooding", "fragmented_instruction"],
            "known_blind_spots": ["base64_encoding"],
            "notes": f"Round {round_num}: base64 caught {top_evasion:.0%} — retrain queued.",
        }))

        trace["final_output"] = f"Orchestrator round {round_num}: action={action}, severity={severity}, workflow={recommended_wf}, confidence={confidence}"
    else:
        print(f"[Orchestrator] LIVE RUN — round {round_num}")
        llm = _make_hf_llm()
        agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)
        result = agent.invoke({"messages": [HumanMessage(content=f"Execute pipeline orchestration for round {round_num}.")]})
        trace = {
            "round": round_num, "mode": "live", "agent": "orchestrator",
            "final_output": result["messages"][-1].content,
            "message_count": len(result["messages"]),
        }

    trace["timestamp"] = datetime.now(timezone.utc).isoformat()
    out = TRACES_DIR / f"round_{round_num}_orchestrator.json"
    out.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"[Orchestrator] Trace saved -> {out}")
    return trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.round, args.dry_run)

