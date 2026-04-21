"""In-memory registry of long-running admin jobs (retrain + drift).

Holds transient state for background jobs so the UI can stream logs and
progress without the client needing to block the HTTP call for the full
3–10 minute training window. Runs are keyed by a short UUID and evicted
once the registry fills (oldest-first) so we do not leak memory on
long-lived servers.

Design notes
------------
- Purposefully process-local. A single FastAPI worker owns its runs; if
  we ever scale to multiple workers, we'd back this with Redis instead
  of introducing sticky sessions.
- Logs use a bounded deque (default 500 lines) to avoid unbounded growth
  from chatty subprocesses. The SSE endpoint replays the buffer on
  connect, then tails new events.
- `subscribers` is a list of asyncio.Queue that the SSE handler creates
  and tears down per connection. We use queues (not broadcast channels)
  so slow clients backpressure on their own queue, never the producer.
- A single class backs both the retrain and drift flows; each flow
  instantiates its own ``RetrainRunStore`` with its own stage tuple so
  the progress-bar percent is driven by the right pipeline.
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Total stages = five models trained + finalise. Keep in sync with the
# MODEL_FACTORIES dict in backend/app/ml/models/__init__.py.
STAGES: tuple[str, ...] = (
    "logreg",
    "random_forest",
    "xgboost",
    "lightgbm",
    "mlp",
    "evaluate",
)

# Stages for the drift → auto-retrain flow. ``drift_check`` is the
# upload-and-compare step; once drift clears the configured threshold
# the pipeline dives into the retrain stages above plus a final
# ``evaluate``.
DRIFT_STAGES: tuple[str, ...] = (
    "drift_check",
    "logreg",
    "random_forest",
    "xgboost",
    "lightgbm",
    "mlp",
    "evaluate",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RetrainRun:
    run_id: str
    trigger: str
    started_at: str
    ended_at: str | None = None
    state: str = "running"          # running | succeeded | failed
    stage: str = "queued"            # current model being trained or "evaluate"/"done"
    percent: int = 0                 # 0-100
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=500))
    result: dict[str, Any] | None = None
    error: str | None = None
    subscribers: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "trigger": self.trigger,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "state": self.state,
            "stage": self.stage,
            "percent": self.percent,
            "log_count": len(self.logs),
            "result": self.result,
            "error": self.error,
        }


class RetrainRunStore:
    """Thread-safe registry with a bounded number of historical runs.

    ``stages`` declares the ordered pipeline this store tracks. Each store
    instance owns its own tuple so a drift flow and a retrain flow can
    compute their own progress percentages without interfering.
    """

    def __init__(
        self,
        max_runs: int = 20,
        stages: tuple[str, ...] = STAGES,
    ) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, RetrainRun] = {}
        self._order: deque[str] = deque()
        self._max_runs = max_runs
        self._active: str | None = None
        self._stages = stages

    # ------------------------------------------------------------------ lifecycle

    def create(self, trigger: str) -> RetrainRun:
        run = RetrainRun(
            run_id=uuid.uuid4().hex[:12],
            trigger=trigger,
            started_at=_now_iso(),
        )
        with self._lock:
            self._runs[run.run_id] = run
            self._order.append(run.run_id)
            self._active = run.run_id
            while len(self._order) > self._max_runs:
                victim = self._order.popleft()
                self._runs.pop(victim, None)
        return run

    def finalise(self, run_id: str, *, state: str, error: str | None = None,
                 result: dict[str, Any] | None = None) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.state = state
            run.error = error
            run.result = result
            run.ended_at = _now_iso()
            run.percent = 100 if state == "succeeded" else run.percent
            run.stage = "done" if state == "succeeded" else "failed"
            if self._active == run_id:
                self._active = None
            self._broadcast(run, event="state")

    # ------------------------------------------------------------------ streaming

    def append_log(self, run_id: str, line: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.logs.append(line)
            self._broadcast(run, event="log", line=line)

    def set_stage(self, run_id: str, stage: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.stage = stage
            if stage in self._stages:
                # Hitting stage N means the (N-1)th stage finished.
                completed = self._stages.index(stage)
                run.percent = int(completed / len(self._stages) * 100)
            self._broadcast(run, event="stage")

    # ------------------------------------------------------------------ subscribers

    def subscribe(self, run_id: str) -> asyncio.Queue[dict[str, Any]] | None:
        """Attach an SSE queue; returns None if run unknown."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
            # Replay existing logs so reconnects do not miss history.
            for line in list(run.logs):
                q.put_nowait({"event": "log", "line": line})
            q.put_nowait({"event": "state", "snapshot": run.snapshot()})
            run.subscribers.append(q)
            return q

    def unsubscribe(self, run_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None and queue in run.subscribers:
                run.subscribers.remove(queue)

    def _broadcast(self, run: RetrainRun, *, event: str, **payload: Any) -> None:
        data: dict[str, Any] = {"event": event, "snapshot": run.snapshot(), **payload}
        for q in list(run.subscribers):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                # Drop the event for this subscriber rather than blocking.
                # Slow client will reconnect and get the log replay.
                pass

    # ------------------------------------------------------------------ lookups

    def get(self, run_id: str) -> RetrainRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def list(self) -> list[RetrainRun]:
        with self._lock:
            return [self._runs[rid] for rid in self._order if rid in self._runs]

    def active(self) -> RetrainRun | None:
        with self._lock:
            if self._active is None:
                return None
            return self._runs.get(self._active)


RUN_STORE = RetrainRunStore()
DRIFT_RUN_STORE = RetrainRunStore(stages=DRIFT_STAGES)
