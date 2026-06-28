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


def _make_llm(llm=None):
    if llm is None:
        raise RuntimeError("No LLM — pass llm= arg or use --dry-run")
    return llm


def run(round_num: int, dry_run: bool, llm=None):
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

        # Extract Red Team family actually used
        try:
            red_msg = json.loads(red_raw)
            top_family = red_msg.get("top_family") or red_msg.get("families_tried", ["unknown"])[0]
        except Exception:
            top_family = "unknown"

        # Extract Blue Team severity
        try:
            blue_msg = json.loads(blue_raw)
            severity = blue_msg.get("severity", "high")
            weakness_scores = blue_msg.get("weakness_scores", {})
            retrain = blue_msg.get("retrain", ["injection"])
            recommended_wf = blue_msg.get("recommended_argo_workflow", "full-canary")
            top_evasion = max(weakness_scores.values()) if weakness_scores else 0.71
            top_detector = max(weakness_scores, key=weakness_scores.get) if weakness_scores else "injection"
        except Exception as e:
            print(f"  WARNING: failed to parse blue_team message: {e} — using defaults")
            severity = "high"
            retrain = ["injection"]
            recommended_wf = "full-canary"
            top_evasion = 0.71
            top_detector = "injection"

        # Also read evasion report if available
        step("read_evasion_report", read_evasion_report, "")

        # Map severity to action
        action_map = {"critical": "retrain", "high": "retrain", "medium": "partial_retrain", "low": "fast_promote", "none": "skip"}
        action = action_map.get(severity, "retrain")
        confidence = 0.91 if top_evasion >= 0.40 else 0.72

        # Rotate families: used family moves to known_blind_spots, next family takes focus
        from agents.tools.memory_tools import MEMORY_PATH
        try:
            mem = json.loads(MEMORY_PATH.read_text(encoding="utf-8")) if MEMORY_PATH.exists() else {}
        except Exception:
            mem = {}
        current_focus = mem.get("current_focus", [])
        known_blind_spots = list(set(mem.get("known_blind_spots", [])) | {top_family})
        next_focus = [f for f in current_focus if f != top_family]
        # Replenish focus from remaining families if running low
        all_families = ["fragmented_instruction", "multilingual", "roleplay_framing",
                        "html_comment_smuggling", "context_flooding", "unicode_homograph"]
        available = [f for f in all_families if f not in known_blind_spots and f not in next_focus]
        while len(next_focus) < 2 and available:
            next_focus.append(available.pop(0))

        # Write decision
        decision = {
            "round": round_num,
            "action": action,
            "models_to_retrain": retrain,
            "severity": severity,
            "confidence": confidence,
            "argo_workflow": recommended_wf,
            "reason": f"{top_detector} evasion {top_evasion:.0%} via {top_family} confirmed by Blue Team.",
            "next_attack_families": next_focus,
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

        # Update attack memory with real values
        step("write_memory", write_memory, json.dumps({
            "round": round_num,
            "top_family": top_family,
            "detector": top_detector,
            "evasion": top_evasion,
            "action": action,
            "current_focus": next_focus,
            "known_blind_spots": sorted(known_blind_spots),
            "notes": f"Round {round_num}: {top_family} evasion {top_evasion:.0%} on {top_detector} — {action} queued.",
        }))

        # Push round results to HF Space dashboard
        try:
            import subprocess
            push_cmd = [
                sys.executable, str(ENTERPRISE_ROOT / "pipeline" / "push_space_status.py"),
                "--round", str(round_num),
            ] + (["--dry-run"] if dry_run else [])
            subprocess.run(push_cmd, check=False, timeout=60)
        except Exception as e:
            print(f"  [Space push] WARNING: {e}")

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

        llm = _make_llm(llm)
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
        action_saved, severity_saved, workflow_saved = "unknown", "unknown", "unknown"
        try:
            decision_path = ENTERPRISE_ROOT / "results" / "pipeline_decision.json"
            if decision_path.exists():
                dec = json.loads(decision_path.read_text(encoding="utf-8"))
                action_saved = dec.get("action", "unknown")
                severity_saved = dec.get("severity", "unknown")
                workflow_saved = dec.get("argo_workflow", "unknown")
        except Exception:
            pass
        session_mem_path.write_text(json.dumps({
            "round": round_num,
            "action": action_saved,
            "severity": severity_saved,
            "workflow": workflow_saved,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")

        # Push round results to HF Space dashboard
        try:
            import subprocess as _sp
            _sp.run([
                sys.executable, str(ENTERPRISE_ROOT / "pipeline" / "push_space_status.py"),
                "--round", str(round_num),
            ], check=False, timeout=60)
        except Exception as _e:
            print(f"  [Space push] WARNING: {_e}")

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

