"""
comms_tools.py — agent-to-agent message passing via JSON files in agent_workspace/.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

WORKSPACE = Path(os.environ.get("ENTERPRISE_ROOT", ".")) / "agent_workspace"

VALID_ROUTES = {
    ("red_team", "blue_team"): "red_to_blue.json",
    ("blue_team", "red_team"): "blue_to_red.json",
    ("blue_team", "orchestrator"): "blue_to_orchestrator.json",
    ("red_team", "orchestrator"): "red_to_orchestrator.json",
}


@tool
def send_message(message_json: str) -> str:
    """
    Send a structured message to another agent via agent_workspace/ files.
    Input JSON: {"from": "red_team", "to": "blue_team", "round": 1,
                 "type": "attack_proposal", "body": {...}, "requires_response": true}
    Valid routes: red_team->blue_team, blue_team->red_team,
                  blue_team->orchestrator, red_team->orchestrator.
    """
    try:
        msg = json.loads(message_json)
    except json.JSONDecodeError as e:
        return f"ERROR: invalid JSON — {e}"

    sender = msg.get("from", "")
    recipient = msg.get("to", "")
    route = (sender, recipient)

    if route not in VALID_ROUTES:
        return f"ERROR: invalid route {sender}->{recipient}. Valid: {list(VALID_ROUTES.keys())}"

    msg["timestamp"] = datetime.now(timezone.utc).isoformat()
    filename = VALID_ROUTES[route]

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    out_path = WORKSPACE / filename
    out_path.write_text(json.dumps(msg, indent=2), encoding="utf-8")
    return f"Message sent {sender}->{recipient}, saved to agent_workspace/{filename}"


@tool
def read_message(route_json: str) -> str:
    """
    Read a message from agent_workspace/.
    Input JSON: {"from": "red_team", "to": "blue_team"} — reads the file for that route.
    Or: {"file": "red_to_blue.json"} to read by filename directly.
    Returns message content or error if not yet written.
    """
    try:
        args = json.loads(route_json)
    except json.JSONDecodeError as e:
        return f"ERROR: invalid JSON — {e}"

    if "file" in args:
        path = WORKSPACE / args["file"]
    else:
        sender = args.get("from", "")
        recipient = args.get("to", "")
        route = (sender, recipient)
        if route not in VALID_ROUTES:
            return f"ERROR: invalid route {sender}->{recipient}"
        path = WORKSPACE / VALID_ROUTES[route]

    if not path.exists():
        return f"No message yet at {path.name} — agent may not have sent yet."

    msg = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(msg.get("body", msg), indent=2)
