"""Monitoring endpoints — drift, model health, Prometheus metrics passthrough."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from backend.app.api.v1.cohort import COHORT_CACHE
from backend.app.api.v1.model_registry import MODEL_STORE
from backend.app.api.v1.schemas import (
    DashboardKpis,
    DriftAutoRetrainResponse,
    DriftReportSummary,
    DriftResponse,
    DriftRunStart,
    DriftRunStatus,
    EvaluationSummary,
    HealthResponse,
    RetrainResponse,
)
from backend.app.core.config import Settings, get_settings
from backend.app.core.metrics import (
    DRIFT_SCORE,
    MODEL_MACRO_F1,
    RETRAIN_TOTAL,
    predictions_last_hour,
    predictions_today,
    uptime_seconds,
)
from backend.app.core.security import TokenPayload, require_admin
from backend.app.monitoring.drift import compute_drift, latest_report
from backend.app.monitoring.retrain_runs import DRIFT_RUN_STORE, RetrainRun
from backend.app.monitoring.retraining import run_retraining

log = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


def _model_version(loaded: Any) -> str:
    run_id = str(loaded.metadata.get("champion_run_id", "") or "")
    return run_id[:12] if run_id else str(loaded.model_name)


def _champion_macro_f1(loaded: Any) -> float:
    metrics = loaded.metadata.get("metrics", {}) or {}
    return float(metrics.get("macro_f1", 0.0) or 0.0)


@router.get("/health", response_model=HealthResponse, summary="Liveness + model status")
def health() -> HealthResponse:
    if not MODEL_STORE.is_loaded():
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_name=None,
            uptime_seconds=uptime_seconds(),
            model_version="unknown",
            champion_macro_f1=0.0,
            predictions_last_hour=predictions_last_hour(),
        )
    loaded = MODEL_STORE.get()
    macro_f1 = _champion_macro_f1(loaded)
    if macro_f1:
        MODEL_MACRO_F1.set(macro_f1)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=loaded.model_name,
        uptime_seconds=uptime_seconds(),
        model_version=_model_version(loaded),
        champion_macro_f1=macro_f1,
        predictions_last_hour=predictions_last_hour(),
    )


_TS_PATTERN = re.compile(r"drift_(\d{8}T\d{6}Z)\.html$")


def _parse_report_timestamp(path: Path) -> str:
    """Extract the ISO timestamp embedded in the report filename.

    Filenames are written as ``drift_YYYYMMDDTHHMMSSZ.html``; we return an
    ISO 8601 string so the frontend can render it with a standard Date.
    Falls back to the file's mtime if the filename is unexpectedly shaped.
    """
    match = _TS_PATTERN.search(path.name)
    if match:
        raw = match.group(1)
        return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _summarise_drift_report(path: Path) -> DriftReportSummary:
    """Read an Evidently HTML sibling-JSON to extract the headline numbers.

    Evidently 0.4 writes the machine-readable summary alongside the HTML
    (``report.save_json``), but our pipeline only persists HTML. To keep the
    backend independent of Evidently internals, we re-parse by running a
    lightweight regex against the embedded inline-JSON payload the HTML
    contains. When parsing fails we still return a valid response with
    zeros so the UI can render a placeholder card.
    """
    generated_at = _parse_report_timestamp(path)
    text = path.read_text(encoding="utf-8", errors="ignore")

    drift_share = 0.0
    n_drifted = 0
    n_total = 0
    target_drift_detected = False

    # Evidently renders the metrics dict in a <script> block; both the old
    # and new layouts expose the same keys.
    share_match = re.search(r'"share_of_drifted_columns"\s*:\s*([0-9.]+)', text)
    if share_match:
        drift_share = float(share_match.group(1))
    drifted_match = re.search(r'"number_of_drifted_columns"\s*:\s*(\d+)', text)
    if drifted_match:
        n_drifted = int(drifted_match.group(1))
    total_match = re.search(r'"number_of_columns"\s*:\s*(\d+)', text)
    if total_match:
        n_total = int(total_match.group(1))
    target_match = re.search(r'"target_drift"\s*:\s*(true|false)', text)
    if target_match:
        target_drift_detected = target_match.group(1) == "true"

    return DriftReportSummary(
        generated_at=generated_at,
        drift_score=drift_share,
        features_drifted=n_drifted,
        features_total=n_total,
        target_drift_detected=target_drift_detected,
        report_url="/api/v1/monitoring/drift/latest",
    )


@router.get(
    "/drift",
    response_model=DriftReportSummary,
    summary="Summary of the most recent drift report",
)
def drift_summary(
    settings: Annotated[Settings, Depends(get_settings)],
) -> DriftReportSummary:
    path = latest_report(settings.drift_report_dir)
    if path is None:
        return DriftReportSummary(
            generated_at="",
            drift_score=0.0,
            features_drifted=0,
            features_total=0,
            target_drift_detected=False,
            report_url="/api/v1/monitoring/drift/latest",
        )
    summary = _summarise_drift_report(path)
    DRIFT_SCORE.set(summary.drift_score)
    return summary


@router.get(
    "/kpis",
    response_model=DashboardKpis,
    summary="Top-line KPI tiles for the dashboard home page",
)
def kpis(
    settings: Annotated[Settings, Depends(get_settings)],
) -> DashboardKpis:
    if not MODEL_STORE.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded — run the training pipeline first.",
        )
    test_path = Path("data/processed/test.parquet")
    if not test_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Holdout split missing — run `dvc repro preprocess` first.",
        )

    loaded = MODEL_STORE.get()
    rows = COHORT_CACHE.get_or_build(test_path)
    total = len(rows)
    high_risk = sum(1 for r in rows if r.prediction.risk_level == "High")
    active_interventions = sum(
        len(r.prediction.recommended_interventions)
        for r in rows
        if r.prediction.risk_level in {"High", "Medium"}
    )

    drift_path = latest_report(settings.drift_report_dir)
    if drift_path is not None:
        drift_summary_row = _summarise_drift_report(drift_path)
        last_drift_check = drift_summary_row.generated_at
        drift_score = drift_summary_row.drift_score
    else:
        last_drift_check = ""
        drift_score = 0.0

    return DashboardKpis(
        total_students=total,
        high_risk_count=high_risk,
        high_risk_pct=(high_risk / total * 100.0) if total else 0.0,
        predictions_today=predictions_today(),
        active_interventions=active_interventions,
        model_macro_f1=_champion_macro_f1(loaded),
        model_version=_model_version(loaded),
        last_drift_check=last_drift_check,
        drift_score=drift_score,
    )


@router.post(
    "/drift",
    response_model=DriftResponse,
    summary="Run a drift check against reference data",
)
async def run_drift(
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File(..., description="Production batch CSV to compare")],
) -> DriftResponse:
    if not settings.reference_data_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Reference snapshot missing — run `dvc repro preprocess` first.",
        )
    reference = pd.read_parquet(settings.reference_data_path)
    raw = await file.read()
    try:
        current = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    result = compute_drift(
        reference=reference,
        current=current,
        output_dir=settings.drift_report_dir,
    )
    DRIFT_SCORE.set(result.drift_share)
    return DriftResponse(
        detected=result.detected,
        drift_share=result.drift_share,
        n_drifted=result.n_drifted,
        n_total=result.n_total,
        report_url="/api/v1/monitoring/drift/latest",
    )


@router.get(
    "/drift/latest",
    summary="Inline HTML of the most recent Evidently drift report",
    response_class=HTMLResponse,
)
def latest(settings: Annotated[Settings, Depends(get_settings)]) -> FileResponse:
    path = latest_report(settings.drift_report_dir)
    if path is None:
        raise HTTPException(status_code=404, detail="No drift report available yet")
    return FileResponse(path, media_type="text/html")


@router.get(
    "/evaluation",
    response_model=EvaluationSummary,
    summary="Unified rigor report (CIs, calibration, threshold, cost, fairness)",
)
def evaluation(settings: Annotated[Settings, Depends(get_settings)]) -> EvaluationSummary:
    if not settings.evaluation_report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No evaluation report found — run `dvc repro evaluate` to "
                "generate reports/evaluation.json."
            ),
        )
    payload = json.loads(settings.evaluation_report_path.read_text())
    fairness = payload.get("fairness", {}) or {}

    figure_urls: dict[str, str] = {}
    pre_png = settings.figures_dir / "calibration_pre.png"
    post_png = settings.figures_dir / "calibration_post.png"
    if pre_png.exists():
        figure_urls["reliability_pre"] = "/api/v1/monitoring/evaluation/figures/calibration_pre.png"
    if post_png.exists():
        figure_urls["reliability_post"] = "/api/v1/monitoring/evaluation/figures/calibration_post.png"

    return EvaluationSummary(
        model_name=str(payload.get("model_name", "unknown")),
        n_test=int(payload.get("n_test", 0)),
        macro_f1=float(payload.get("macro_f1", 0.0)),
        macro_f1_lower=float(payload.get("macro_f1_lower", 0.0)),
        macro_f1_upper=float(payload.get("macro_f1_upper", 0.0)),
        dropout_recall_argmax=float(payload.get("dropout_recall_argmax", 0.0)),
        dropout_recall_tuned=float(payload.get("dropout_recall_tuned", 0.0)),
        chosen_threshold=float(payload.get("chosen_threshold", 0.5)),
        expected_utility_argmax=float(payload.get("expected_utility_argmax", 0.0)),
        expected_utility_tuned=float(payload.get("expected_utility_tuned", 0.0)),
        fairness_max_gap=float(payload.get("fairness_max_gap", 0.0)),
        fairness_summary_attribute=str(fairness.get("summary_attribute", "n/a")),
        calibration_ece=float(payload.get("calibration_ece", 0.0)),
        calibration_ece_post=payload.get("calibration_ece_post"),
        temperature=payload.get("temperature"),
        figure_urls=figure_urls,
        details=payload,
    )


@router.post(
    "/drift/auto-retrain",
    response_model=DriftAutoRetrainResponse,
    summary=(
        "Run drift on a production batch; if drift is detected (or `force=true`), "
        "trigger a champion-vs-challenger retrain in the same call."
    ),
)
async def drift_auto_retrain(
    settings: Annotated[Settings, Depends(get_settings)],
    _admin: Annotated[TokenPayload, Depends(require_admin)],
    file: Annotated[UploadFile, File(..., description="Production batch CSV")],
    threshold: float = Query(0.30, ge=0.0, le=1.0),
    force: bool = Query(False, description="Skip the drift gate and retrain unconditionally"),
) -> DriftAutoRetrainResponse:
    if not settings.reference_data_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reference snapshot missing — run `dvc repro preprocess` first.",
        )
    test_path = Path("data/processed/test.parquet")
    if not test_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Holdout split missing — run `dvc repro preprocess` first.",
        )

    reference = pd.read_parquet(settings.reference_data_path)
    raw = await file.read()
    try:
        current = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    drift_result = compute_drift(
        reference=reference,
        current=current,
        output_dir=settings.drift_report_dir,
        threshold=threshold,
    )
    DRIFT_SCORE.set(drift_result.drift_share)
    drift_payload = DriftResponse(
        detected=drift_result.detected,
        drift_share=drift_result.drift_share,
        n_drifted=drift_result.n_drifted,
        n_total=drift_result.n_total,
        report_url="/api/v1/monitoring/drift/latest",
    )

    if not (drift_result.detected or force):
        return DriftAutoRetrainResponse(
            drift=drift_payload,
            retrain=None,
            skipped=True,
            skip_reason=(
                f"drift_share={drift_result.drift_share:.3f} below threshold={threshold:.2f}"
            ),
            threshold=threshold,
        )

    challenger_dir = Path("models/staging")
    challenger_dir.mkdir(parents=True, exist_ok=True)
    trigger_tag = (
        f"drift:share={drift_result.drift_share:.3f}"
        if drift_result.detected
        else "drift:forced"
    )
    try:
        outcome = run_retraining(
            test_path=test_path,
            champion_path=settings.model_path,
            challenger_out=challenger_dir,
            trigger=trigger_tag,
            audit_log_path=settings.retrain_history_path,
        )
    except Exception as exc:
        RETRAIN_TOTAL.labels(outcome="failed").inc()
        raise HTTPException(status_code=500, detail=f"Auto-retrain failed: {exc}") from exc

    if outcome.decision.promoted:
        MODEL_STORE.load(
            model_path=settings.model_path,
            metadata_path=settings.metadata_path,
            reference_path=settings.reference_data_path,
        )

    audit = outcome.audit_entry
    retrain_payload = RetrainResponse(
        promoted=outcome.decision.promoted,
        reason=outcome.decision.reason,
        champion_macro_f1=outcome.decision.champion_macro_f1,
        challenger_macro_f1=outcome.decision.challenger_macro_f1,
        per_class_deltas=outcome.decision.per_class_deltas,
        timestamp=audit.timestamp,
        trigger=audit.trigger,
        mcnemar_p_value=audit.mcnemar_p_value,
        mcnemar_significant=audit.mcnemar_significant,
        n_test=audit.n_test,
    )
    return DriftAutoRetrainResponse(
        drift=drift_payload,
        retrain=retrain_payload,
        skipped=False,
        skip_reason=None,
        threshold=threshold,
    )


@router.get(
    "/evaluation/figures/{name}",
    summary="Reliability-diagram PNG produced by the evaluation stage",
)
def evaluation_figure(
    name: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> FileResponse:
    # Path-traversal guard — only serve files directly under figures_dir.
    if "/" in name or "\\" in name or not name.endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid figure name")
    path = settings.figures_dir / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Figure '{name}' not found")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# Async drift → auto-retrain
#
# The synchronous /drift/auto-retrain endpoint above blocks the HTTP call
# for the full 3–10 minute training window. That is the root cause of the
# "network error" the UI surfaces: browsers/reverse-proxies kill long
# requests before the server can respond. The /start variant below mirrors
# the retrain pattern: kick off a background thread, stream progress via
# SSE, expose an /active endpoint so reloads reconnect.
# ---------------------------------------------------------------------------


def _run_drift_in_thread(
    run: RetrainRun,
    *,
    settings: Settings,
    csv_bytes: bytes,
    filename: str,
    threshold: float,
    force: bool,
) -> None:
    """Background body for the drift→auto-retrain pipeline."""
    try:
        DRIFT_RUN_STORE.set_stage(run.run_id, "drift_check")
        DRIFT_RUN_STORE.append_log(
            run.run_id,
            f"Loading reference snapshot from {settings.reference_data_path} ...",
        )
        if not settings.reference_data_path.exists():
            raise RuntimeError(
                "Reference snapshot missing — run `dvc repro preprocess` first."
            )
        reference = pd.read_parquet(settings.reference_data_path)
        try:
            current = pd.read_csv(io.BytesIO(csv_bytes))
        except Exception as exc:
            raise RuntimeError(f"Could not parse CSV: {exc}") from exc

        DRIFT_RUN_STORE.append_log(
            run.run_id,
            f"Scoring drift on {len(current)} rows from {filename} "
            f"(threshold={threshold:.2f}) ...",
        )
        drift_result = compute_drift(
            reference=reference,
            current=current,
            output_dir=settings.drift_report_dir,
            threshold=threshold,
        )
        DRIFT_SCORE.set(drift_result.drift_share)
        DRIFT_RUN_STORE.append_log(
            run.run_id,
            (
                f"Drift result: drift_share={drift_result.drift_share:.3f} "
                f"n_drifted={drift_result.n_drifted}/{drift_result.n_total} "
                f"detected={drift_result.detected}"
            ),
        )
        drift_payload = DriftResponse(
            detected=drift_result.detected,
            drift_share=drift_result.drift_share,
            n_drifted=drift_result.n_drifted,
            n_total=drift_result.n_total,
            report_url="/api/v1/monitoring/drift/latest",
        )
        run.result = {
            "drift": drift_payload.model_dump(),
            "retrain": None,
            "skipped": None,
            "skip_reason": None,
            "threshold": threshold,
        }

        if not (drift_result.detected or force):
            skip_reason = (
                f"drift_share={drift_result.drift_share:.3f} "
                f"below threshold={threshold:.2f}"
            )
            DRIFT_RUN_STORE.append_log(run.run_id, f"Skipping retrain: {skip_reason}")
            run.result.update(
                {"skipped": True, "skip_reason": skip_reason}
            )
            DRIFT_RUN_STORE.finalise(
                run.run_id, state="succeeded", result=run.result
            )
            return

        test_path = Path("data/processed/test.parquet")
        if not test_path.exists():
            raise RuntimeError(
                "Holdout split missing — run `dvc repro preprocess` first."
            )
        challenger_dir = Path("models/staging")
        challenger_dir.mkdir(parents=True, exist_ok=True)
        trigger_tag = (
            f"drift:share={drift_result.drift_share:.3f}"
            if drift_result.detected
            else "drift:forced"
        )
        DRIFT_RUN_STORE.append_log(
            run.run_id, f"Launching retrain (trigger={trigger_tag}) ..."
        )

        outcome = run_retraining(
            test_path=test_path,
            champion_path=settings.model_path,
            challenger_out=challenger_dir,
            trigger=trigger_tag,
            audit_log_path=settings.retrain_history_path,
            log_cb=lambda line: DRIFT_RUN_STORE.append_log(run.run_id, line),
            stage_cb=lambda stage: DRIFT_RUN_STORE.set_stage(run.run_id, stage),
        )

        if outcome.decision.promoted:
            MODEL_STORE.load(
                model_path=settings.model_path,
                metadata_path=settings.metadata_path,
                reference_path=settings.reference_data_path,
            )

        audit = outcome.audit_entry
        retrain_payload = RetrainResponse(
            promoted=outcome.decision.promoted,
            reason=outcome.decision.reason,
            champion_macro_f1=outcome.decision.champion_macro_f1,
            challenger_macro_f1=outcome.decision.challenger_macro_f1,
            per_class_deltas=outcome.decision.per_class_deltas,
            timestamp=audit.timestamp,
            trigger=audit.trigger,
            mcnemar_p_value=audit.mcnemar_p_value,
            mcnemar_significant=audit.mcnemar_significant,
            n_test=audit.n_test,
            registered_model_name=audit.registered_model_name,
            registered_model_version=audit.registered_model_version,
        )
        run.result.update(
            {
                "retrain": retrain_payload.model_dump(),
                "skipped": False,
                "skip_reason": None,
            }
        )
        DRIFT_RUN_STORE.finalise(
            run.run_id, state="succeeded", result=run.result
        )
    except Exception as exc:  # noqa: BLE001 — surface everything to the UI
        log.exception("Drift auto-retrain failed")
        DRIFT_RUN_STORE.append_log(run.run_id, f"[error] {exc}")
        RETRAIN_TOTAL.labels(outcome="failed").inc()
        DRIFT_RUN_STORE.finalise(
            run.run_id, state="failed", error=str(exc)
        )


@router.post(
    "/drift/start",
    response_model=DriftRunStart,
    summary=(
        "Launch the drift→auto-retrain pipeline in the background and return "
        "a run id. Use /drift/runs/{run_id}/logs to stream progress."
    ),
)
async def start_drift_auto_retrain(
    settings: Annotated[Settings, Depends(get_settings)],
    _admin: Annotated[TokenPayload, Depends(require_admin)],
    file: Annotated[UploadFile, File(..., description="Production batch CSV")],
    threshold: float = Query(0.30, ge=0.0, le=1.0),
    force: bool = Query(False, description="Skip the drift gate and retrain unconditionally"),
) -> DriftRunStart:
    active = DRIFT_RUN_STORE.active()
    if active is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Drift check already running (run_id={active.run_id}); "
                "wait for it to finish."
            ),
        )
    raw = await file.read()
    trigger_tag = f"drift-ui:{file.filename or 'upload.csv'}"
    run = DRIFT_RUN_STORE.create(trigger=trigger_tag)
    DRIFT_RUN_STORE.append_log(
        run.run_id,
        f"Accepted {len(raw)} bytes from {file.filename or 'upload.csv'} "
        f"(threshold={threshold:.2f}, force={force})",
    )
    thread = threading.Thread(
        target=_run_drift_in_thread,
        kwargs={
            "run": run,
            "settings": settings,
            "csv_bytes": raw,
            "filename": file.filename or "upload.csv",
            "threshold": threshold,
            "force": force,
        },
        daemon=True,
        name=f"drift-{run.run_id}",
    )
    thread.start()
    base = "/api/v1/monitoring/drift/runs"
    return DriftRunStart(
        run_id=run.run_id,
        status_url=f"{base}/{run.run_id}",
        logs_url=f"{base}/{run.run_id}/logs",
    )


def _drift_run_to_status(
    run: RetrainRun, *, include_logs: bool = True
) -> DriftRunStatus:
    result = run.result or {}
    drift_raw = result.get("drift")
    retrain_raw = result.get("retrain")
    return DriftRunStatus(
        run_id=run.run_id,
        trigger=run.trigger,
        started_at=run.started_at,
        ended_at=run.ended_at,
        state=run.state,
        stage=run.stage,
        percent=run.percent,
        log_count=len(run.logs),
        logs=list(run.logs) if include_logs else [],
        drift=DriftResponse(**drift_raw) if drift_raw else None,
        retrain=RetrainResponse(**retrain_raw) if retrain_raw else None,
        skipped=result.get("skipped"),
        skip_reason=result.get("skip_reason"),
        threshold=result.get("threshold"),
        error=run.error,
    )


@router.get(
    "/drift/runs/{run_id}",
    response_model=DriftRunStatus,
    summary="Snapshot of an in-flight or completed drift auto-retrain run",
)
def drift_run_status(
    run_id: str,
    _admin: Annotated[TokenPayload, Depends(require_admin)],
    tail: int = Query(200, ge=0, le=500),
) -> DriftRunStatus:
    run = DRIFT_RUN_STORE.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}")
    snapshot = _drift_run_to_status(run)
    if tail < len(snapshot.logs):
        snapshot.logs = snapshot.logs[-tail:]
    return snapshot


@router.get(
    "/drift/active",
    response_model=DriftRunStatus | None,
    summary="Currently in-flight drift run (if any), for reconnect-on-reload",
)
def drift_active_run(
    _admin: Annotated[TokenPayload, Depends(require_admin)],
) -> DriftRunStatus | None:
    run = DRIFT_RUN_STORE.active()
    if run is None:
        return None
    return _drift_run_to_status(run)


@router.get(
    "/drift/runs/{run_id}/logs",
    summary="Server-sent events stream of logs for a drift auto-retrain run",
)
async def stream_drift_logs(
    run_id: str,
    request: Request,
    _admin: Annotated[TokenPayload, Depends(require_admin)],
) -> StreamingResponse:
    run = DRIFT_RUN_STORE.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}")
    queue = DRIFT_RUN_STORE.subscribe(run_id)
    if queue is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}")

    async def event_generator() -> Any:
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                payload = json.dumps(event)
                yield f"event: {event['event']}\ndata: {payload}\n\n"
                latest = DRIFT_RUN_STORE.get(run_id)
                if (
                    latest is not None
                    and latest.state in ("succeeded", "failed")
                    and queue.empty()
                ):
                    break
        finally:
            DRIFT_RUN_STORE.unsubscribe(run_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
