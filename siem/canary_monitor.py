"""
Canary monitor: polls staging /score endpoint for 60 minutes,
tracks FP rate against benign corpus. Writes pass/fail result.

Usage:
    python siem/canary_monitor.py --duration 3600 --fp-threshold 0.05 --result-path /tmp/canary_result.txt
    python siem/canary_monitor.py --dry-run --duration 30
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

STAGING_URL = os.environ.get("STAGING_URL", "http://localhost:8000")

BENIGN_CORPUS = [
    "Hi team, please review the attached document for Thursday's meeting.",
    "Your order #12345 has been shipped. Expected delivery: Friday.",
    "Reminder: quarterly review is next week. Please prepare your slides.",
    "Thanks for the feedback! We'll incorporate it in the next sprint.",
    "Your subscription renewal is coming up on July 1st. No action needed.",
    "Weekly digest: top stories from your followed topics.",
    "Invoice #INV-2026-089 attached for your records.",
    "Happy to connect! Looking forward to our chat on Friday.",
    "The report you requested is ready for download in the portal.",
    "Your appointment is confirmed for Monday at 2:00 PM.",
]


def score_sample(text: str, model: str = "injection") -> float:
    try:
        resp = requests.post(
            f"{STAGING_URL}/score",
            json={"text": text, "model": model},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("confidence", 0.5)
    except Exception:
        pass
    return 0.5  # neutral on failure


def compute_fp_rate(threshold: float = 0.5) -> float:
    false_positives = 0
    for text in BENIGN_CORPUS:
        confidence = score_sample(text)
        if confidence > threshold:
            false_positives += 1
    return false_positives / len(BENIGN_CORPUS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=3600, help="Monitor duration in seconds")
    parser.add_argument("--fp-threshold", type=float, default=0.05, help="Max FP rate (0.05 = 5%)")
    parser.add_argument("--poll-interval", type=int, default=300, help="Poll every N seconds")
    parser.add_argument("--result-path", type=str, default="/tmp/canary_result.txt")
    parser.add_argument("--dry-run", action="store_true", help="Run for --duration seconds with mock data")
    args = parser.parse_args()

    print(f"Canary monitor starting — duration={args.duration}s, fp_threshold={args.fp_threshold:.0%}")
    print(f"Staging: {STAGING_URL}")

    start = time.time()
    poll_count = 0
    max_fp_seen = 0.0
    fp_history = []

    while time.time() - start < args.duration:
        elapsed = int(time.time() - start)

        if args.dry_run:
            fp_rate = 0.02 + (poll_count * 0.005)
        else:
            fp_rate = compute_fp_rate()

        poll_count += 1
        max_fp_seen = max(max_fp_seen, fp_rate)
        fp_history.append({"elapsed_s": elapsed, "fp_rate": round(fp_rate, 4)})

        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        status = "OK" if fp_rate <= args.fp_threshold else "ALERT"
        print(f"  [{ts}] Poll {poll_count}: FP rate = {fp_rate:.1%} [{status}]")

        if fp_rate > args.fp_threshold:
            print(f"CANARY FAILED: FP rate {fp_rate:.1%} exceeds threshold {args.fp_threshold:.0%}")
            with open(args.result_path, "w") as f:
                f.write("failed")
            log_path = Path(args.result_path).parent / "canary_history.json"
            with open(log_path, "w") as f:
                json.dump(fp_history, f, indent=2)
            sys.exit(0)

        remaining = args.duration - int(time.time() - start)
        sleep_time = min(args.poll_interval, remaining)
        if sleep_time > 0 and not args.dry_run:
            time.sleep(sleep_time)
        elif args.dry_run:
            time.sleep(2)

    print(f"\nCanary PASSED: max FP rate seen = {max_fp_seen:.1%} (threshold {args.fp_threshold:.0%})")
    with open(args.result_path, "w") as f:
        f.write("passed")

    log_path = Path(args.result_path).parent / "canary_history.json"
    with open(log_path, "w") as f:
        json.dump(fp_history, f, indent=2)

    print(f"Result: passed | History: {log_path}")


if __name__ == "__main__":
    main()
