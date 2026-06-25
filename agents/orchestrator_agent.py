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

SYSTEM_PROMPT = """You are the Pipeline Orchestrator. Read agent reports, decide routing, trigger Argo.

Thresholds: evasion >0.40=CRITICAL/full-canary | 0.25-0.40=HIGH/full-canary | <0.25=LOW/fast-promote | multiple >0.40=emergency-rollback.

TOOLS (in order):
1. read_message from red_team
2. read_message from blue_team
3. read_evasion_report
4. analyze_weakness {"detector":"all"} — once only
5. decide_routing {action,severity,argo_workflow,confidence,reason}
6. set_env_vars {AI_ACTION,ARGO_WORKFLOW,SEVERITY}
7. trigger_argo {workflow,round,dry_run}
8. write_memory with round summary

Ground decisions in numbers. No repeated tool calls."""


def _make_llm():
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        temperature=0.1,
        max_tokens=512,
    )


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
            "reason": f"Injection evasion {top_evasion:.0%} via unicode_homograph confirmed by Blue Team. "
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
            "top_family": "unicode_homograph",
            "detector": "injection",
            "evasion": top_evasion,
            "action": action,
            "current_focus": ["context_flooding", "fragmented_instruction"],
            "known_blind_spots": ["base64_encoding", "unicode_homograph"],
            "notes": f"Round {round_num}: unicode_homograph caught {top_evasion:.0%} — retrain queued.",
        }))

        trace["final_output"] = f"Orchestrator round {round_num}: action={action}, severity={severity}, workflow={recommended_wf}, confidence={confidence}"
    else:
        print(f"[Orchestrator] LIVE RUN — round {round_num}")
        # Load own session memory + structured agent summaries (clean context, no message bleed)
        WORKSPACE = ENTERPRISE_ROOT / "agent_workspace"
        session_mem_path = WORKSPACE / "orchestrator_session.json"
        prior_context = ""
        if session_mem_path.exists():
            try:
                prior = json.loads(session_mem_path.read_text(encoding="utf-8"))
                prior_context = (
                    f" Prior round {prior.get('round')}: action={prior.get('action')}, "
                    f"severity={prior.get('severity')}, workflow={prior.get('workflow')}."
                )
            except Exception:
                pass

        llm = _make_llm()
        agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)
        result = agent.invoke(
            {"messages": [HumanMessage(content=(
                f"Execute orchestration for round {round_num}.{prior_context} "
                "Call tools only. "
                "Step 1: read_message from red_team. "
                "Step 2: read_message from blue_team. "
                "Step 3: read_evasion_report. "
                "Step 4: decide_routing. "
                "Step 5: set_env_vars. "
                "Step 6: trigger_argo dry_run=true if no ARGO_TOKEN. "
                "Step 7: write_memory."
            ))]},
            config={"recursion_limit": 10},
        )

        # Write compact session memory
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        # Read real routing decision from attack_memory.json last round entry
        action_saved, severity_saved, workflow_saved = "unknown", "unknown", "unknown"
        try:
            from agents.tools.memory_tools import MEMORY_PATH
            if MEMORY_PATH.exists():
                mem = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
                last = mem.get("rounds", [{}])[-1]
                action_saved = last.get("action", "unknown")
                severity_saved = last.get("evasion", "unknown")
                workflow_saved = last.get("top_family", "unknown")
        except Exception:
            pass
        session_mem_path.write_text(json.dumps({
            "round": round_num,
            "action": action_saved,
            "severity": severity_saved,
            "workflow": workflow_saved,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")

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

