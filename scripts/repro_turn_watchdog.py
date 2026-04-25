#!/usr/bin/env python3
"""Local repro for the iteration=0 cached-loop bug.

Run from the repo root WITHOUT a live gateway:

    ./venv/bin/python scripts/repro_turn_watchdog.py

What it shows:
1. WITHOUT the watchdog: a stuck cached agent emits 60 heartbeats at
   "iteration 0/60 (cached)" over 30 minutes, never killed by the
   inactivity timeout (default 1800s), because every cached reuse
   resets _last_activity_ts to now.
2. WITH the watchdog: the loop is killed within ~120s of the cached
   marker appearing, and the kill is recorded in
   $HERMES_STATE_DIR/turn_watchdog.log (default
   ~/.local/state/hermes/turn_watchdog.log).

No external services required. Pure simulation.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Allow running from anywhere; resolve repo root from this script's path.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from gateway.turn_watchdog import (  # noqa: E402
    HeartbeatThrottle,
    REASON_CACHE_LOOP,
    TurnWatchdog,
    audit_log_kill,
)


class StuckCachedAgent:
    """Simulates the bug: cached AIAgent that never makes its first API call.

    Mirrors the state set in gateway/run.py at the cached-reuse path:
        agent._last_activity_ts = time.time()
        agent._last_activity_desc = "starting new turn (cached)"
        agent._api_call_count = 0
        agent._is_cached_turn = True   # added by the watchdog fix
    """

    def __init__(self):
        self.session_id = "discord:repro-session-001"
        self.model = "claude-opus-4-7"
        self._iter = 0
        self._cached = True
        self._touched_at = time.monotonic()  # simulated wall clock
        self.interrupted_with = None

    def get_activity_summary(self):
        return {
            "api_call_count": self._iter,
            "max_iterations": 60,
            "is_cached_turn": self._cached,
            "last_activity_desc": "starting new turn (cached)",
            "current_tool": None,
        }

    def interrupt(self, msg):
        self.interrupted_with = msg


def run_without_watchdog(notify_interval_s: int, total_s: int) -> dict:
    """Pre-fix behavior: heartbeats forever, no kill."""
    agent = StuckCachedAgent()
    heartbeats = 0
    elapsed = 0
    while elapsed < total_s:
        elapsed += notify_interval_s
        s = agent.get_activity_summary()
        heartbeats += 1
    return {
        "heartbeats_emitted": heartbeats,
        "killed": False,
        "kill_at_s": None,
        "iterations_seen": agent._iter,
    }


def run_with_watchdog(notify_interval_s: int, total_s: int,
                      audit_dir: Path) -> dict:
    """Post-fix behavior: watchdog kills the stuck turn."""
    os.environ["HERMES_STATE_DIR"] = str(audit_dir)
    agent = StuckCachedAgent()
    wd = TurnWatchdog(cached_zero_threshold_s=90)
    throttle = HeartbeatThrottle()
    heartbeats_emitted = 0
    elapsed = 0
    kill_at = None
    while elapsed < total_s:
        elapsed += notify_interval_s
        s = agent.get_activity_summary()
        reason = wd.evaluate(s, elapsed_s=elapsed)
        if reason:
            agent.interrupt("Watchdog kill (stuck turn)")
            audit_log_kill(
                reason,
                turn_id=agent.session_id,
                model=agent.model,
                iteration=agent._iter,
                max_iterations=60,
                elapsed_s=elapsed,
                session_key=agent.session_id,
            )
            kill_at = elapsed
            break
        if throttle.should_emit(s, elapsed_s=elapsed):
            heartbeats_emitted += 1
    return {
        "heartbeats_emitted": heartbeats_emitted,
        "killed": kill_at is not None,
        "kill_at_s": kill_at,
        "kill_reason": REASON_CACHE_LOOP if kill_at else None,
        "interrupted_with": agent.interrupted_with,
    }


def _print_section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    notify_interval = 30   # heartbeat every 30s (simulated)
    total_window = 1800    # 30-min observation window (matches inactivity timeout)

    _print_section("REPRO: stuck cached turn — iteration stays at 0/60")
    print(f"notify_interval = {notify_interval}s, observation window = {total_window}s")
    print(f"existing inactivity timeout (HERMES_AGENT_TIMEOUT) = 1800s default")

    _print_section("BEFORE FIX (no watchdog)")
    before = run_without_watchdog(notify_interval, total_window)
    print(json.dumps(before, indent=2))
    if not before["killed"]:
        print("=> No kill. Heartbeats spam forever; user is stuck.")

    with tempfile.TemporaryDirectory() as td:
        audit_dir = Path(td)
        _print_section("AFTER FIX (TurnWatchdog + HeartbeatThrottle + audit log)")
        after = run_with_watchdog(notify_interval, total_window, audit_dir)
        print(json.dumps(after, indent=2))

        log = audit_dir / "turn_watchdog.log"
        if log.exists():
            print("\nAudit log written to:", log)
            for line in log.read_text().splitlines():
                rec = json.loads(line)
                print("  ", json.dumps(rec, indent=2))

    _print_section("ASSERTIONS")
    assert before["killed"] is False, "before-fix should never kill"
    assert after["killed"] is True, "after-fix must kill the stuck turn"
    assert after["kill_at_s"] <= 120, (
        f"after-fix should kill within ~120s, got {after['kill_at_s']}s"
    )
    assert after["kill_at_s"] < 1800, "must fire before inactivity timeout"
    assert after["interrupted_with"] == "Watchdog kill (stuck turn)"
    print("OK — all assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
