"""Prometheus metrics emitted by the API."""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

PREDICT_LATENCY = Histogram(
    "predict_latency_seconds",
    "Latency of a single /predict call.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY,
)

PREDICTION_TOTAL = Counter(
    "prediction_total",
    "Number of predictions returned, labelled by predicted risk level.",
    labelnames=("risk_level",),
    registry=REGISTRY,
)

BATCH_PREDICTIONS = Counter(
    "batch_predictions_total",
    "Number of rows scored in /predict/batch calls.",
    registry=REGISTRY,
)

MODEL_MACRO_F1 = Gauge(
    "model_macro_f1",
    "Macro-F1 of the currently-deployed model on the holdout set.",
    registry=REGISTRY,
)

DRIFT_SCORE = Gauge(
    "drift_score",
    "Most recent Evidently dataset drift share (0..1).",
    registry=REGISTRY,
)

RETRAIN_TOTAL = Counter(
    "retrain_runs_total",
    "Number of retrain attempts.",
    labelnames=("outcome",),  # promoted | rejected | failed
    registry=REGISTRY,
)


_APP_START_MONOTONIC: float = time.monotonic()
_PREDICTION_TIMESTAMPS: deque[float] = deque(maxlen=200_000)


def record_prediction() -> None:
    """Record a single prediction wallclock timestamp for time-windowed KPIs."""
    _PREDICTION_TIMESTAMPS.append(time.time())


def uptime_seconds() -> int:
    return int(time.monotonic() - _APP_START_MONOTONIC)


def _trim(cutoff: float) -> None:
    while _PREDICTION_TIMESTAMPS and _PREDICTION_TIMESTAMPS[0] < cutoff:
        _PREDICTION_TIMESTAMPS.popleft()


def predictions_last_hour() -> int:
    _trim(time.time() - 3600)
    return len(_PREDICTION_TIMESTAMPS)


def predictions_today() -> int:
    start_of_day = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()
    _trim(start_of_day)
    return sum(1 for t in _PREDICTION_TIMESTAMPS if t >= start_of_day)
