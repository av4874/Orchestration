"""
memory_tools.py — read/write persistent attack memory across rounds.
attack_memory.json: committed to repo, updated by Argo notify step each round.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain.tools import tool

MEMORY_PATH = Path(os.environ.get("ENTERPRISE_ROOT", ".")) / "pipeline" / "attack_memory.json"

_DEFAULT_MEMORY = {
    "rounds": [],
    "current_focus": [],
    "known_blind_spots": [],
}


def _load() -> dict:
    if MEMORY_PATH.exists():
        return json.loads(MEMORY_PATH.read_text(encoding="utf-8-sig"))
    return dict(_DEFAULT_MEMORY)


@tool
def read_attack_memory(query: str = "") -> str:
    """
    Read attack_memory.json — returns round history, current focus families,
    and known blind spots. Pass query="" for full memory or a keyword to filter rounds.
    """
    mem = _load()
    rounds_summary = [
        {"round": r.get("round"), "family": r.get("top_family"), "evasion": r.get("evasion_rate"), "action": r.get("action")}
        for r in mem.get("rounds", [])
    ][-3:]  # last 3 rounds only — caps token cost as history grows
    if query:
        rounds_summary = [r for r in rounds_summary if query.lower() in json.dumps(r).lower()]
        return json.dumps({"filtered_rounds": rounds_summary, "current_focus": mem.get("current_focus", [])}, indent=2)
    return json.dumps({
        "rounds": rounds_summary,
        "current_focus": mem.get("current_focus", []),
        "known_blind_spots": mem.get("known_blind_spots", []),
    }, indent=2)


@tool
def write_memory(update_json: str) -> str:
    """
    Merge update_json dict into attack_memory.json.
    update_json must be a JSON string with keys: round (int), top_family (str),
    detector (str), evasion (float), action (str), notes (str, optional).
    Appends to rounds list and updates current_focus if provided.
    """
    try:
        update = json.loads(update_json)
    except json.JSONDecodeError as e:
        return f"ERROR: invalid JSON — {e}"

    mem = _load()
    round_entry = {
        "round": update.get("round"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_family": update.get("top_family"),
        "detector": update.get("detector"),
        "evasion": update.get("evasion"),
        "action": update.get("action"),
        "notes": update.get("notes", ""),
    }
    mem.setdefault("rounds", []).append(round_entry)

    if "current_focus" in update:
        mem["current_focus"] = update["current_focus"]
    if "known_blind_spots" in update:
        existing = set(mem.get("known_blind_spots", []))
        existing.update(update["known_blind_spots"])
        mem["known_blind_spots"] = sorted(existing)

    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(json.dumps(mem, indent=2), encoding="utf-8")
    return f"Memory updated — round {round_entry['round']} recorded."

