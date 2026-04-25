"""Tests for gateway.turn_watchdog — kills stuck cached turns.

Repro for the iteration=0 cached-loop bug:

When a turn is interrupted mid-flight and the next turn reuses the
cached AIAgent, gateway/run.py resets ``_api_call_count`` to 0 and sets
``_last_activity_desc = "starting new turn (cached)"``. Critically, it
also resets ``_last_activity_ts`` to ``time.time()``, which perpetually
pushes the inactivity-based timeout deadline forward. If the new turn
never makes its first API call (e.g., a follow-up interrupt arrives, or
the agent deadlocks in pre-flight), the heartbeat keeps logging
"Still working... iteration 0/60, starting new turn (cached)" forever
and the existing 1800s inactivity timeout never fires.

The watchdog catches this independently of the inactivity timer by
inspecting the heartbeat state tuple (iteration, cached, elapsed_bucket).
"""

import json
from pathlib import Path

import pytest

from gateway.turn_watchdog import (
    HeartbeatThrottle,
    REASON_CACHE_LOOP,
    REASON_STALL,
    TurnWatchdog,
    audit_log_kill,
    heartbeat_state,
)


def _summary(
    *,
    iteration: int,
    cached: bool,
    desc: str = "starting new turn (cached)",
    max_iterations: int = 60,
    current_tool=None,
):
    return {
        "api_call_count": iteration,
        "max_iterations": max_iterations,
        "is_cached_turn": cached,
        "last_activity_desc": desc,
        "current_tool": current_tool,
    }


class TestWatchdogCachedZeroKill:
    """Rule 1: iteration==0 AND cached AND elapsed > 90s -> kill."""

    def test_does_not_fire_under_threshold(self):
        wd = TurnWatchdog(cached_zero_threshold_s=90)
        assert wd.evaluate(_summary(iteration=0, cached=True), elapsed_s=60) is None

    def test_fires_just_over_threshold(self):
        wd = TurnWatchdog(cached_zero_threshold_s=90)
        assert wd.evaluate(_summary(iteration=0, cached=True), elapsed_s=91) == REASON_CACHE_LOOP

    def test_does_not_fire_when_not_cached(self):
        wd = TurnWatchdog(cached_zero_threshold_s=90)
        assert wd.evaluate(_summary(iteration=0, cached=False), elapsed_s=300) is None

    def test_does_not_fire_when_iteration_advanced(self):
        """Critical: do not break legitimate long Codex turns where
        iteration is genuinely progressing."""
        wd = TurnWatchdog(cached_zero_threshold_s=90)
        assert wd.evaluate(_summary(iteration=1, cached=True), elapsed_s=300) is None
        assert wd.evaluate(_summary(iteration=5, cached=True), elapsed_s=600) is None
        assert wd.evaluate(_summary(iteration=42, cached=False), elapsed_s=3600) is None


class TestWatchdogStallDetection:
    """Rule 2: identical (iteration, cached, elapsed_bucket) twice -> kill."""

    def test_two_identical_heartbeats_kill(self):
        wd = TurnWatchdog(bucket_size_s=300)  # 5-min buckets
        # Both heartbeats fall in the same 5-min bucket with identical state.
        assert wd.evaluate(_summary(iteration=3, cached=False), elapsed_s=60) is None
        assert wd.evaluate(_summary(iteration=3, cached=False), elapsed_s=240) == REASON_STALL

    def test_progressing_iteration_does_not_kill(self):
        wd = TurnWatchdog(bucket_size_s=300)
        assert wd.evaluate(_summary(iteration=3, cached=False), elapsed_s=60) is None
        assert wd.evaluate(_summary(iteration=4, cached=False), elapsed_s=240) is None
        assert wd.evaluate(_summary(iteration=5, cached=False), elapsed_s=420) is None

    def test_changing_bucket_does_not_kill(self):
        """Smaller buckets mean consecutive heartbeats land in different
        buckets and the stall rule cannot fire."""
        wd = TurnWatchdog(bucket_size_s=60)
        assert wd.evaluate(_summary(iteration=3, cached=False), elapsed_s=60) is None
        assert wd.evaluate(_summary(iteration=3, cached=False), elapsed_s=130) is None


class TestStuckTurnRepro:
    """End-to-end simulated repro of the user-reported symptom.

    Three consecutive heartbeats at iteration=0 with the cached marker
    must trigger a watchdog kill. The kill must occur while the existing
    1800s inactivity timeout is still far away.
    """

    def test_three_heartbeats_at_iter_zero_trigger_kill(self):
        """Notify interval = 30s -> heartbeats at 30, 60, 90, 120s.

        Asserts:
          * watchdog fires within 120s (well under the 1800s inactivity
            timeout), proving the fix catches the loop the existing
            timeout misses.
          * the fired reason is REASON_CACHE_LOOP, so the operator log
            attributes the kill to the right root cause.
        """
        wd = TurnWatchdog(cached_zero_threshold_s=90)
        decisions = []
        for elapsed in (30, 60, 90, 120):
            decisions.append(
                (elapsed, wd.evaluate(_summary(iteration=0, cached=True), elapsed_s=elapsed))
            )

        assert decisions[0] == (30, None), "first heartbeat below threshold"
        assert decisions[1] == (60, None), "second heartbeat below threshold"
        assert decisions[2] == (90, None), "threshold is strict greater-than"
        assert decisions[3] == (120, REASON_CACHE_LOOP), "fourth heartbeat past 90s -> kill"
        assert decisions[3][0] < 1800, "must fire before existing inactivity timeout"

    def test_three_consecutive_identical_heartbeats(self):
        """Even if cached_zero rule is bypassed, the stall rule catches
        a turn that emits 3+ heartbeats with no state change at all."""
        wd = TurnWatchdog(cached_zero_threshold_s=10_000, bucket_size_s=600)
        kills = []
        for elapsed in (60, 180, 300):  # all in same 600s bucket
            r = wd.evaluate(_summary(iteration=2, cached=False), elapsed_s=elapsed)
            if r:
                kills.append((elapsed, r))
        assert kills, "stall rule must fire on repeated identical heartbeats"
        assert kills[0][1] == REASON_STALL


class TestHeartbeatThrottle:
    def test_first_emit_passes(self):
        t = HeartbeatThrottle(bucket_size_s=60)
        assert t.should_emit(_summary(iteration=2, cached=False), elapsed_s=30) is True

    def test_duplicate_key_suppressed(self):
        t = HeartbeatThrottle(bucket_size_s=60)
        assert t.should_emit(_summary(iteration=2, cached=False), elapsed_s=30) is True
        assert t.should_emit(_summary(iteration=2, cached=False), elapsed_s=45) is False

    def test_iteration_change_re_emits(self):
        t = HeartbeatThrottle(bucket_size_s=60)
        assert t.should_emit(_summary(iteration=2, cached=False), elapsed_s=30) is True
        assert t.should_emit(_summary(iteration=3, cached=False), elapsed_s=45) is True

    def test_bucket_change_re_emits(self):
        t = HeartbeatThrottle(bucket_size_s=60)
        assert t.should_emit(_summary(iteration=2, cached=False), elapsed_s=30) is True
        assert t.should_emit(_summary(iteration=2, cached=False), elapsed_s=130) is True


class TestHeartbeatStateDetectsCached:
    def test_explicit_flag_takes_precedence(self):
        s = {"api_call_count": 0, "is_cached_turn": True, "last_activity_desc": "anything"}
        _, cached, _ = heartbeat_state(s, elapsed_s=10)
        assert cached is True

    def test_legacy_desc_string_fallback(self):
        s = {"api_call_count": 0, "last_activity_desc": "starting new turn (cached)"}
        _, cached, _ = heartbeat_state(s, elapsed_s=10)
        assert cached is True

    def test_non_cached(self):
        s = {"api_call_count": 5, "last_activity_desc": "executing tool: terminal"}
        _, cached, _ = heartbeat_state(s, elapsed_s=10)
        assert cached is False


class TestAuditLog:
    def test_writes_jsonl(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_STATE_DIR", str(tmp_path))
        audit_log_kill(
            REASON_CACHE_LOOP,
            turn_id="t-123",
            model="claude-opus-4-7",
            iteration=0,
            max_iterations=60,
            elapsed_s=120,
            session_key="discord:1234567890123456789012345678901234567890",
        )
        log_file = tmp_path / "turn_watchdog.log"
        assert log_file.exists()
        record = json.loads(log_file.read_text().strip())
        assert record["reason"] == REASON_CACHE_LOOP
        assert record["turn_id"] == "t-123"
        assert record["model"] == "claude-opus-4-7"
        assert record["iteration"] == 0
        assert record["max_iterations"] == 60
        assert record["elapsed_s"] == 120
        # session_key is truncated to 30 chars to avoid leaking long IDs.
        assert len(record["session_key"]) <= 30

    def test_appends_multiple_records(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_STATE_DIR", str(tmp_path))
        for i in range(3):
            audit_log_kill(
                REASON_STALL,
                turn_id=f"t-{i}",
                model="x",
                iteration=i,
                max_iterations=60,
                elapsed_s=100 + i,
            )
        log_file = tmp_path / "turn_watchdog.log"
        lines = [l for l in log_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 3
        assert [json.loads(l)["turn_id"] for l in lines] == ["t-0", "t-1", "t-2"]

    def test_failure_swallowed(self, tmp_path, monkeypatch):
        """Audit log must never break the turn loop. A bad path is silent."""
        # Point at a path whose parent is a regular file, so mkdir fails.
        bogus_parent = tmp_path / "not_a_dir"
        bogus_parent.write_text("blocking file")
        monkeypatch.setenv("HERMES_STATE_DIR", str(bogus_parent / "subdir"))
        # Must not raise.
        audit_log_kill(
            REASON_CACHE_LOOP,
            turn_id="t-x",
            model="m",
            iteration=0,
            max_iterations=60,
            elapsed_s=100,
        )


class TestWatchdogIntegrationWithFakeAgent:
    """Mirror the FakeAgent pattern from test_gateway_inactivity_timeout.py.

    Asserts the watchdog produces the kill signal when fed activity
    summaries from a stuck cached agent across simulated heartbeats.
    """

    class StuckCachedAgent:
        def __init__(self, max_iterations=60):
            self._iter = 0
            self._max = max_iterations
            self._cached = True
            self._desc = "starting new turn (cached)"
            self.interrupted_with = None
            self.session_id = "fake-session-abc"
            self.model = "claude-opus-4-7"

        def get_activity_summary(self):
            return {
                "api_call_count": self._iter,
                "max_iterations": self._max,
                "is_cached_turn": self._cached,
                "last_activity_desc": self._desc,
                "current_tool": None,
            }

        def interrupt(self, msg):
            self.interrupted_with = msg

    def test_stuck_agent_killed_within_120s(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_STATE_DIR", str(tmp_path))
        agent = self.StuckCachedAgent()
        wd = TurnWatchdog(cached_zero_threshold_s=90)

        kill_at = None
        for elapsed in range(30, 1801, 30):  # heartbeat every 30s up to 30 min
            reason = wd.evaluate(agent.get_activity_summary(), elapsed_s=elapsed)
            if reason:
                # Caller is responsible for the side effects; simulate them.
                agent.interrupt("Watchdog kill (stuck turn)")
                audit_log_kill(
                    reason,
                    turn_id=agent.session_id,
                    model=agent.model,
                    iteration=agent._iter,
                    max_iterations=agent._max,
                    elapsed_s=elapsed,
                )
                kill_at = elapsed
                break

        assert kill_at is not None, "watchdog must kill stuck cached turn"
        assert kill_at <= 120, f"kill should occur near the 90s threshold, got {kill_at}s"
        assert agent.interrupted_with == "Watchdog kill (stuck turn)"
        # Audit log was written.
        records = (tmp_path / "turn_watchdog.log").read_text().splitlines()
        assert len(records) == 1
        rec = json.loads(records[0])
        assert rec["reason"] == REASON_CACHE_LOOP
        assert rec["model"] == "claude-opus-4-7"
