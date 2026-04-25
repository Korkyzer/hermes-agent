"""Turn-level watchdog for the gateway agent loop.

Detects and kills stuck cached turns that the existing inactivity-based
timeout misses, because cached-agent reuse (gateway/run.py) resets
``_last_activity_ts`` to ``time.time()`` on every new turn — perpetually
pushing the inactivity deadline forward.

Two kill rules:
  1. ``iteration == 0`` AND ``cached`` AND ``elapsed > 90s``
     (the "iteration 0/60 (cached)" loop reported by users).
  2. The ``(iteration, cached, elapsed_bucket)`` tuple is observed twice
     consecutively in heartbeats (generic stall detection).

A separate :class:`HeartbeatThrottle` suppresses heartbeat sends when
``(iteration, elapsed_bucket)`` is unchanged so the user does not see
duplicate "Still working..." messages.

Kill events are appended to a JSONL audit log at
``$HERMES_STATE_DIR/turn_watchdog.log`` (default
``~/.local/state/hermes/turn_watchdog.log``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple


CACHED_ZERO_KILL_THRESHOLD_S = 90.0
DEFAULT_BUCKET_SIZE_S = 60

REASON_CACHE_LOOP = "cache_loop"
REASON_STALL = "stall"
REASON_INTERRUPT_ORPHAN = "interrupt_orphan"


def _audit_log_path() -> Path:
    base = os.environ.get("HERMES_STATE_DIR")
    if base:
        return Path(base) / "turn_watchdog.log"
    return Path.home() / ".local" / "state" / "hermes" / "turn_watchdog.log"


_audit_lock = Lock()


def audit_log_kill(
    reason: str,
    *,
    turn_id: str,
    model: str,
    iteration: int,
    max_iterations: int,
    elapsed_s: float,
    session_key: str = "",
    extra: Optional[dict] = None,
) -> None:
    """Append a kill record to the watchdog audit log (JSONL).

    Failure to write is swallowed — the audit log must never break the
    turn loop. The log lives outside the project tree so a corrupted
    repo state cannot lose kill history.
    """
    record = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": reason,
        "turn_id": turn_id,
        "model": model,
        "iteration": iteration,
        "max_iterations": max_iterations,
        "elapsed_s": round(elapsed_s, 1),
        "session_key": (session_key or "")[:30],
    }
    if extra:
        record.update(extra)
    path = _audit_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _audit_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def heartbeat_state(
    summary: dict,
    elapsed_s: float,
    bucket_size_s: int = DEFAULT_BUCKET_SIZE_S,
) -> Tuple[int, bool, int]:
    """Build the ``(iteration, cached, elapsed_bucket)`` state tuple.

    ``cached`` is True when this turn started by reusing a previously-
    cached AIAgent. Detected via the explicit ``is_cached_turn`` field
    if the activity summary exposes it; otherwise via the legacy
    "(cached)" substring in ``last_activity_desc`` for backwards-compat.
    """
    iteration = int(summary.get("api_call_count", 0) or 0)
    if "is_cached_turn" in summary:
        cached = bool(summary.get("is_cached_turn"))
    else:
        desc = summary.get("last_activity_desc") or ""
        cached = "(cached)" in desc
    bucket = int(elapsed_s // max(1, bucket_size_s))
    return (iteration, cached, bucket)


class TurnWatchdog:
    """Per-turn watchdog evaluating heartbeat states for stuck-turn signals.

    The watchdog is purely advisory. ``evaluate`` returns the kill reason
    or ``None``. The caller is responsible for actually interrupting the
    agent and writing the audit log.
    """

    def __init__(
        self,
        *,
        cached_zero_threshold_s: float = CACHED_ZERO_KILL_THRESHOLD_S,
        bucket_size_s: int = DEFAULT_BUCKET_SIZE_S,
    ):
        self.cached_zero_threshold_s = float(cached_zero_threshold_s)
        self.bucket_size_s = int(bucket_size_s)
        self._last_state: Optional[Tuple[int, bool, int]] = None

    def evaluate(self, summary: dict, elapsed_s: float) -> Optional[str]:
        state = heartbeat_state(summary, elapsed_s, self.bucket_size_s)
        iteration, cached, _bucket = state

        # Cache-loop territory owns this case exclusively. The stall
        # rule must NOT short-circuit it with a less specific
        # diagnosis when consecutive heartbeats happen to land in the
        # same elapsed bucket — operators need the cache_loop reason
        # to know to look at the agent-cache reuse path.
        if iteration == 0 and cached:
            self._last_state = state
            if elapsed_s > self.cached_zero_threshold_s:
                return REASON_CACHE_LOOP
            return None

        # Generic stall: identical (iteration, cached, bucket) tuple
        # observed twice consecutively. Catches non-cached hangs.
        if self._last_state is not None and self._last_state == state:
            return REASON_STALL

        self._last_state = state
        return None


class HeartbeatThrottle:
    """Suppress heartbeat sends when (iteration, elapsed_bucket) is unchanged."""

    def __init__(self, *, bucket_size_s: int = DEFAULT_BUCKET_SIZE_S):
        self.bucket_size_s = int(bucket_size_s)
        self._last_key: Optional[Tuple[int, int]] = None

    def should_emit(self, summary: dict, elapsed_s: float) -> bool:
        iteration = int(summary.get("api_call_count", 0) or 0)
        bucket = int(elapsed_s // max(1, self.bucket_size_s))
        key = (iteration, bucket)
        if key == self._last_key:
            return False
        self._last_key = key
        return True
