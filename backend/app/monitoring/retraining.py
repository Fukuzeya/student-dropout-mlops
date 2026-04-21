"""Champion-vs-challenger retraining loop.

Triggered either on demand (via `/api/v1/retrain`, JWT-gated) or by a
scheduled job. The loop:

1. Loads the frozen holdout and the current production bundle.
2. Trains a challenger using the same trainer the DVC pipeline runs.
3. Evaluates both champion and challenger on the same holdout, capturing
   *paired* predictions for the McNemar test.
4. Calls :func:`registry.compare_for_promotion` with paired inputs so the
   promotion decision combines effect-size, no-class-regression, and
   statistical-significance gates.
5. On promotion: atomically swaps the joblib bundle and refreshes
   Prometheus gauges.
6. Always: appends an :class:`AuditEntry` to ``reports/retraining/history.jsonl``
   so the governance trail survives across processes.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from backend.app.core.config import get_settings
from backend.app.core.metrics import MODEL_MACRO_F1, RETRAIN_TOTAL
from backend.app.ml.evaluate import evaluate_predictions
from backend.app.ml.registry import (
    PromotionDecision,
    compare_for_promotion,
    register_and_promote,
)
from backend.app.ml.schemas import TARGET_CLASSES, feature_columns
from backend.app.monitoring.audit import AuditEntry, append_audit, utc_now_iso

log = logging.getLogger(__name__)
DEFAULT_AUDIT_LOG = Path("reports/retraining/history.jsonl")


@dataclass(slots=True)
class RetrainOutcome:
    decision: PromotionDecision
    challenger_metrics: dict[str, float]
    champion_metrics: dict[str, float]
    audit_entry: AuditEntry


def _predict_and_evaluate(
    model_path: Path, test_df: pd.DataFrame
) -> tuple[dict[str, Any], np.ndarray]:
    """Score `model_path` on `test_df`; return metrics + decoded predictions."""
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    classes: list[str] = bundle.get("classes", TARGET_CLASSES)
    y_true = test_df["Target"].astype(str).to_numpy()
    y_pred_idx = pipe.predict(test_df[feature_columns()])
    y_pred = np.asarray([classes[int(i)] for i in y_pred_idx])
    metrics = evaluate_predictions(y_true, y_pred, classes=classes)
    return metrics, y_pred


def _stream_subprocess(
    cmd: list[str],
    *,
    log_cb: Callable[[str], None] | None,
    stage_cb: Callable[[str], None] | None,
) -> None:
    """Run `cmd` with line-buffered stdout, streaming each line to callbacks.

    stdout + stderr are merged so the caller gets a single chronological
    stream. `stage_cb` is fired when a ``Training <name> ...`` marker (emitted
    by :mod:`backend.app.ml.train`) appears, which drives the UI progress bar.
    """
    log.info("Training challenger: %s", " ".join(cmd))
    # Force the child's stdout+stderr unbuffered so the UI sees each log line
    # as it is emitted rather than in a single flush at the end of training.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None  # for mypy; PIPE always yields a stream
    try:
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            log.debug("[train] %s", line)
            if log_cb is not None:
                log_cb(line)
            if stage_cb is not None:
                marker = _stage_from_line(line)
                if marker is not None:
                    stage_cb(marker)
    finally:
        proc.stdout.close()
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


_STAGE_MARKERS: dict[str, str] = {
    "Training logreg": "logreg",
    "Training random_forest": "random_forest",
    "Training xgboost": "xgboost",
    "Training lightgbm": "lightgbm",
    "Training mlp": "mlp",
    "Champion:": "evaluate",
}


def _stage_from_line(line: str) -> str | None:
    for marker, stage in _STAGE_MARKERS.items():
        if marker in line:
            return stage
    return None


def run_retraining(
    *,
    test_path: Path,
    champion_path: Path,
    challenger_out: Path,
    train_cmd: list[str] | None = None,
    trigger: str = "manual",
    audit_log_path: Path | None = None,
    log_cb: Callable[[str], None] | None = None,
    stage_cb: Callable[[str], None] | None = None,
) -> RetrainOutcome:
    """Run the full retrain → evaluate → promote-if-better cycle.

    ``log_cb`` / ``stage_cb`` are optional hooks used by the streaming
    admin endpoint to surface training output to the UI in real time.
    """
    test_df = pd.read_parquet(test_path)
    y_true = test_df["Target"].astype(str).to_numpy()
    champion_metrics, champion_pred = _predict_and_evaluate(champion_path, test_df)

    cmd = train_cmd or [
        sys.executable, "-m", "backend.app.ml.train", "run",
        "--train", "data/processed/train.parquet",
        "--val", "data/processed/val.parquet",
        "--models-out", str(challenger_out),
    ]
    _stream_subprocess(cmd, log_cb=log_cb, stage_cb=stage_cb)

    challenger_metrics, challenger_pred = _predict_and_evaluate(
        challenger_out / "model.joblib", test_df
    )

    decision = compare_for_promotion(
        champion_metrics,
        challenger_metrics,
        paired_predictions=(y_true, champion_pred, challenger_pred),
    )

    registered_version: str | None = None
    registered_name: str | None = None
    challenger_run_id: str | None = None

    if decision.promoted:
        log.info("Promoting challenger over champion: %s", decision.reason)
        # Atomic swap: write to temp file then rename onto the bundle path.
        tmp = champion_path.with_suffix(".joblib.new")
        shutil.copy2(challenger_out / "model.joblib", tmp)
        tmp.replace(champion_path)
        meta_target = champion_path.with_name("metadata.json")
        meta_source = challenger_out / "metadata.json"
        if meta_source.exists():
            shutil.copy2(meta_source, meta_target)
            try:
                meta = json.loads(meta_source.read_text())
                challenger_run_id = meta.get("champion_run_id")
            except Exception:  # noqa: BLE001
                challenger_run_id = None
        MODEL_MACRO_F1.set(challenger_metrics["macro_f1"])
        RETRAIN_TOTAL.labels(outcome="promoted").inc()

        # Mirror the local promotion in MLflow's Model Registry so a reviewer
        # browsing http://<host>:5000/#/models sees every champion ever shipped.
        if challenger_run_id:
            settings = get_settings()
            registered_name = settings.mlflow_registered_model_name
            registered_version = register_and_promote(
                run_id=challenger_run_id,
                model_name=registered_name,
                artifact_path="model",
                description=(
                    f"Promoted by retrain trigger={trigger!r} — "
                    f"macro-F1 {decision.challenger_macro_f1:.4f} "
                    f"(Δ {decision.challenger_macro_f1 - decision.champion_macro_f1:+.4f})"
                ),
                tags={
                    "trigger": trigger,
                    "macro_f1": f"{decision.challenger_macro_f1:.4f}",
                },
            )
    else:
        log.info("Challenger rejected: %s", decision.reason)
        RETRAIN_TOTAL.labels(outcome="rejected").inc()

    audit = _build_audit_entry(
        decision,
        trigger=trigger,
        n_test=int(len(test_df)),
        registered_model_name=registered_name,
        registered_model_version=registered_version,
    )
    append_audit(audit, audit_log_path or DEFAULT_AUDIT_LOG)

    return RetrainOutcome(
        decision=decision,
        challenger_metrics={
            k: v for k, v in challenger_metrics.items() if isinstance(v, (int, float))
        },
        champion_metrics={
            k: v for k, v in champion_metrics.items() if isinstance(v, (int, float))
        },
        audit_entry=audit,
    )


def _build_audit_entry(
    decision: PromotionDecision,
    *,
    trigger: str,
    n_test: int,
    registered_model_name: str | None = None,
    registered_model_version: str | None = None,
) -> AuditEntry:
    mc = decision.mcnemar
    return AuditEntry(
        timestamp=utc_now_iso(),
        trigger=trigger,
        promoted=decision.promoted,
        reason=decision.reason,
        champion_macro_f1=float(decision.champion_macro_f1),
        challenger_macro_f1=float(decision.challenger_macro_f1),
        macro_f1_delta=float(decision.challenger_macro_f1 - decision.champion_macro_f1),
        per_class_deltas={k: float(v) for k, v in decision.per_class_deltas.items()},
        mcnemar_p_value=float(mc.p_value) if mc is not None else None,
        mcnemar_b=int(mc.b) if mc is not None else None,
        mcnemar_c=int(mc.c) if mc is not None else None,
        mcnemar_significant=bool(mc.significant_at_05) if mc is not None else None,
        n_test=n_test,
        registered_model_name=registered_model_name,
        registered_model_version=registered_model_version,
    )


def write_outcome(outcome: RetrainOutcome, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "promoted": outcome.decision.promoted,
        "reason": outcome.decision.reason,
        "champion_macro_f1": outcome.decision.champion_macro_f1,
        "challenger_macro_f1": outcome.decision.challenger_macro_f1,
        "per_class_deltas": outcome.decision.per_class_deltas,
        "audit": outcome.audit_entry.as_dict(),
    }
    path.write_text(json.dumps(payload, indent=2))
