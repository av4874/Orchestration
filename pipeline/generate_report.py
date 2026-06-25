"""
generate_report.py — Build static HTML demo from pipeline trace/results files.

Usage:
  python pipeline/generate_report.py [--round 5] [--out docs/index.html]
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ENTERPRISE_ROOT = Path(__file__).parent.parent

CSS = """
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #e6edf3; }
  header { background: #161b22; border-bottom: 1px solid #30363d; padding: 20px 32px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.3rem; font-weight: 600; }
  .badge { background: #238636; color: #fff; font-size: 0.72rem; padding: 2px 8px; border-radius: 12px; font-weight: 600; }
  .badge.warn { background: #9e6a03; }
  .badge.danger { background: #da3633; }
  .badge.info { background: #1f6feb; }
  .container { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 32px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }
  .card h3 { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
  .card .value { font-size: 2rem; font-weight: 700; }
  .card .sub { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }
  .value.green { color: #3fb950; }
  .value.yellow { color: #d29922; }
  .value.red { color: #f85149; }
  .value.blue { color: #58a6ff; }
  section { margin-bottom: 32px; }
  section h2 { font-size: 1rem; font-weight: 600; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #30363d; }
  .agent-flow { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }
  .agent-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px 18px; min-width: 160px; }
  .agent-box .name { font-size: 0.85rem; font-weight: 600; margin-bottom: 6px; }
  .agent-box .detail { font-size: 0.75rem; color: #8b949e; }
  .agent-box.red .name { color: #f85149; }
  .agent-box.blue .name { color: #58a6ff; }
  .agent-box.orch .name { color: #3fb950; }
  .arrow { color: #8b949e; font-size: 1.2rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
  th { background: #161b22; padding: 10px 14px; text-align: left; color: #8b949e; font-weight: 500; border-bottom: 1px solid #30363d; }
  td { padding: 10px 14px; border-bottom: 1px solid #21262d; vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  .prompt-cell { font-family: monospace; font-size: 0.78rem; color: #79c0ff; word-break: break-all; max-width: 380px; }
  .score-cell { font-weight: 600; }
  .evaded { color: #3fb950; }
  .caught { color: #f85149; }
  .step-list { list-style: none; }
  .step-list li { display: flex; gap: 12px; padding: 10px 0; border-bottom: 1px solid #21262d; font-size: 0.84rem; }
  .step-list li:last-child { border-bottom: none; }
  .step-num { background: #1f6feb22; color: #58a6ff; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 700; flex-shrink: 0; }
  .step-tool { color: #d2a8ff; font-family: monospace; font-weight: 600; min-width: 180px; }
  .step-result { color: #8b949e; word-break: break-all; }
  .decision-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .decision-row { display: flex; flex-direction: column; gap: 4px; }
  .decision-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; }
  .decision-value { font-size: 0.95rem; font-weight: 600; }
  .reasoning { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; font-size: 0.84rem; color: #8b949e; line-height: 1.6; margin-top: 16px; max-height: 220px; overflow-y: auto; }
  footer { text-align: center; padding: 32px; color: #8b949e; font-size: 0.78rem; border-top: 1px solid #30363d; margin-top: 32px; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; background: #1f6feb22; color: #58a6ff; }
  .tag.unicode { background: #d2a8ff22; color: #d2a8ff; }
  .tag.base64 { background: #ffa65722; color: #ffa657; }
  .note { background: #1f6feb11; border: 1px solid #1f6feb33; border-radius: 6px; padding: 12px 16px; font-size: 0.82rem; color: #8b949e; margin-bottom: 20px; }
  .note strong { color: #58a6ff; }
"""


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_report(round_num: int, out_path: Path):
    RESULTS = ENTERPRISE_ROOT / "results"
    TRACES = ENTERPRISE_ROOT / "agent_traces"

    evasion = load_json(RESULTS / "evasion_report.json")
    red_trace = load_json(TRACES / f"round_{round_num}_red_team.json")
    orch_trace = load_json(TRACES / f"round_{round_num}_orchestrator.json")
    inj_results = load_json(RESULTS / "injection_results.json")
    decision = load_json(RESULTS / "pipeline_decision.json")

    samples_path = RESULTS / f"round_{round_num}_samples.json"
    if not samples_path.exists():
        samples_path = RESULTS / "validated_samples.json"
    raw_samples = load_json(samples_path)
    sample_list = raw_samples if isinstance(raw_samples, list) else raw_samples.get("samples", [])

    overall_evasion = evasion.get("overall_evasion_rate", 0.0)
    total_samples = evasion.get("total_samples_scored", 0) or len(sample_list)
    per_detector = evasion.get("per_detector", {})
    per_family = evasion.get("per_family", {})
    families_tried = list(per_family.keys()) or ["unicode_homograph"]
    top_family = families_tried[0] if families_tried else "unicode_homograph"

    action = decision.get("action", "retrain")
    argo_workflow = decision.get("argo_workflow", "full-canary")
    severity = decision.get("severity", "high").upper()
    confidence = decision.get("confidence", 0.91)

    severity_badge = {"CRITICAL": "danger", "HIGH": "warn", "MEDIUM": "warn", "LOW": "info", "NONE": "info"}.get(severity, "info")
    evasion_color = "red" if overall_evasion >= 0.4 else ("yellow" if overall_evasion >= 0.1 else "green")
    action_color = "red" if action in ("retrain", "emergency_rollback") else "yellow"
    confidence_str = f"{confidence:.0%}" if isinstance(confidence, float) else str(confidence)

    # Sample rows
    inj_samples = inj_results.get("samples", []) if inj_results else []
    display_samples = inj_samples if inj_samples else sample_list[:10]
    sample_rows = ""
    for s in display_samples[:10]:
        prompt = s.get("prompt", s.get("prompt_preview", ""))
        prompt_display = esc(prompt[:80]) + ("…" if len(prompt) > 80 else "")
        fam = s.get("attack_family", top_family)
        tag_cls = "unicode" if "unicode" in fam else ("base64" if "base64" in fam else "")
        det = s.get("detector", "injection")
        score = s.get("score", None)
        evaded = s.get("evaded", False)
        score_str = f"{score:.3f}" if score is not None else "—"
        evaded_html = '<span class="evaded">✓ YES</span>' if evaded else '<span class="caught">✗ NO</span>'
        score_cls = "evaded" if evaded else "caught"
        sample_rows += (
            f'<tr>'
            f'<td style="font-family:monospace;font-size:0.75rem;color:#8b949e">{esc(s.get("id",""))}</td>'
            f'<td class="prompt-cell">{prompt_display}</td>'
            f'<td><span class="tag {tag_cls}">{esc(fam)}</span></td>'
            f'<td>{esc(det)}</td>'
            f'<td class="score-cell {score_cls}">{score_str}</td>'
            f'<td>{evaded_html}</td>'
            f'</tr>\n'
        )

    # Detector rows
    detector_rows = ""
    for det, stats in per_detector.items():
        rate = stats.get("evasion_rate", 0.0)
        rate_cls = "evaded" if rate >= 0.4 else ("yellow" if rate > 0 else "caught")
        tested = stats.get("total_scored", 0)
        evaded_n = stats.get("evaded", 0)
        status = "⚠️ Vulnerable" if rate >= 0.4 else ("⚡ Partial" if rate > 0 else "✓ Holding")
        detector_rows += (
            f'<tr><td style="font-weight:600">{esc(det)}</td>'
            f'<td>{tested}</td><td>{evaded_n}</td>'
            f'<td class="score-cell {rate_cls}">{rate:.1%}</td>'
            f'<td>{status}</td></tr>\n'
        )
    if not detector_rows:
        detector_rows = "<tr><td colspan='5' style='color:#8b949e'>No detector results yet</td></tr>"

    # Red team steps
    red_steps_html = ""
    steps = red_trace.get("steps", [])
    if steps:
        for i, step in enumerate(steps, 1):
            tool = esc(step.get("tool", ""))
            result = esc(str(step.get("result", ""))[:120])
            red_steps_html += (
                f'<li><span class="step-num">{i}</span>'
                f'<span class="step-tool">{tool}</span>'
                f'<span class="step-result">{result}</span></li>\n'
            )
    else:
        final = esc(red_trace.get("final_output", "")[:400])
        red_steps_html = f'<li><span class="step-num">→</span><span class="step-tool">live_run ({red_trace.get("message_count",0)} messages)</span><span class="step-result">{final}</span></li>'

    orch_reasoning = esc(orch_trace.get("final_output", "No orchestrator trace.")[:800])
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Enterprise Adversarial ML Pipeline — Demo</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <div><h1>🛡️ Enterprise Adversarial ML Pipeline</h1></div>
  <span class="badge info">DEMO — Cached Run</span>
  <span class="badge {severity_badge}">{severity}</span>
</header>
<div class="container">
  <div class="note">
    <strong>How this works:</strong> Red Team agents generate adversarial prompts →
    scored against DistilBERT threat detectors (HuggingFace Space) →
    Blue Team analyzes weaknesses → Orchestrator routes to Argo workflow.
    This page shows results from a real pipeline run (cached, no live API calls).
  </div>

  <div class="grid">
    <div class="card"><h3>Round</h3><div class="value blue">{round_num}</div><div class="sub">Pipeline cycle</div></div>
    <div class="card"><h3>Samples Tested</h3><div class="value blue">{total_samples}</div><div class="sub">Against 4 detectors</div></div>
    <div class="card"><h3>Attack Families</h3><div class="value blue">{len(families_tried)}</div><div class="sub">{esc(", ".join(families_tried))}</div></div>
    <div class="card"><h3>Evasion Rate</h3><div class="value {evasion_color}">{overall_evasion:.1%}</div><div class="sub">Avg across detectors</div></div>
    <div class="card"><h3>Argo Decision</h3><div class="value {action_color}">{esc(action.upper())}</div><div class="sub">{esc(argo_workflow)}</div></div>
    <div class="card"><h3>Confidence</h3><div class="value blue">{confidence_str}</div><div class="sub">Orchestrator confidence</div></div>
  </div>

  <section>
    <h2>Agent Communication Flow</h2>
    <div class="agent-flow">
      <div class="agent-box red">
        <div class="name">🔴 Red Team Agent</div>
        <div class="detail">Generated {total_samples} samples<br>Family: {esc(top_family)}</div>
      </div>
      <div class="arrow">→</div>
      <div class="agent-box blue">
        <div class="name">🔵 Blue Team Agent</div>
        <div class="detail">Analyzed detector gaps<br>Severity: {severity}</div>
      </div>
      <div class="arrow">→</div>
      <div class="agent-box orch">
        <div class="name">🟢 Orchestrator Agent</div>
        <div class="detail">Decision: {esc(action)}<br>Workflow: {esc(argo_workflow)}</div>
      </div>
    </div>
  </section>

  <section>
    <h2>Attack Samples — {esc(top_family)}</h2>
    <table>
      <thead><tr><th>ID</th><th>Prompt</th><th>Family</th><th>Detector</th><th>Score</th><th>Evaded</th></tr></thead>
      <tbody>{sample_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Detector Results</h2>
    <table>
      <thead><tr><th>Detector</th><th>Samples Tested</th><th>Evaded</th><th>Evasion Rate</th><th>Status</th></tr></thead>
      <tbody>{detector_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Red Team Agent — Tool Call Trace</h2>
    <ul class="step-list">{red_steps_html}</ul>
  </section>

  <section>
    <h2>Orchestrator Decision</h2>
    <div class="decision-box">
      <div class="decision-row"><span class="decision-label">Action</span><span class="decision-value">{esc(action.upper())}</span></div>
      <div class="decision-row"><span class="decision-label">Severity</span><span class="decision-value">{severity}</span></div>
      <div class="decision-row"><span class="decision-label">Argo Workflow</span><span class="decision-value">{esc(argo_workflow)}</span></div>
      <div class="decision-row"><span class="decision-label">Confidence</span><span class="decision-value">{confidence_str}</span></div>
    </div>
    <div class="reasoning"><strong>Orchestrator reasoning:</strong><br><br>{orch_reasoning}</div>
  </section>
</div>

<footer>
  Generated {generated_at} ·
  <a href="https://huggingface.co/Builder117" style="color:#58a6ff;">HuggingFace</a> ·
  Detector: DistilBERT (HF Space) ·
  Dataset: Builder117/enterprise-adversarial-samples (300 samples, 6 families)
</footer>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=5)
    parser.add_argument("--out", type=str, default="docs/index.html")
    args = parser.parse_args()
    build_report(args.round, ENTERPRISE_ROOT / args.out)
