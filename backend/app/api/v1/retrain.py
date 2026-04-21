"""Admin retrain endpoints — JWT-gated.

Two flavours are exposed:

* ``POST /retrain`` — synchronous convenience wrapper kept for backwards
  compatibility and the existing `drift/retrain` auto-trigger path.
* ``POST /retrain/start`` — non-blocking launcher that returns a run id
  the UI uses to stream logs and progress while the job runs in the
  background. The synchronous endpoint is a thin wrapper around the
  same code path so behaviour stays consistent.
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from backend.app.api.v1.model_registry import MODEL_STORE
from backend.app.api.v1.schemas import (
    RetrainAuditEntry,
    RetrainHistoryResponse,
    RetrainResponse,
    RetrainRunStart,
    RetrainRunStatus,
)
from backend.app.core.config import Settings, get_settings
from backend.app.core.metrics import RETRAIN_TOTAL
from backend.app.core.security import TokenPayload, require_admin
from backend.app.monitoring.audit import read_audit
from backend.app.monitoring.retrain_runs import RUN_STORE, RetrainRun
from backend.app.monitoring.retraining import RetrainOutcome, run_retraining

router = APIRouter(prefix="/retrain", tags=["admin"])


def _outcome_to_response(outcome: RetrainOutcome) -> RetrainResponse:
    audit = outcome.audit_entry
    return RetrainResponse(
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


def _execute_retrain(
    *,
    settings: Settings,
    trigger: str,
    log_cb: Any = None,
    stage_cb: Any = None,
) -> RetrainOutcome:
    """Shared body — validates prerequisites, runs the retrain cycle, hot-reloads."""
    test_path = Path("data/processed/test.parquet")
    if not test_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Holdout split missing. Run `dvc repro preprocess` first.",
        )
    challenger_dir = Path("models/staging")
    challenger_dir.mkdir(parents=True, exist_ok=True)

    try:
        outcome = run_retraining(
            test_path=test_path,
            champion_path=settings.model_path,
            challenger_out=challenger_dir,
            trigger=trigger,
            audit_log_path=settings.retrain_history_path,
            log_cb=log_cb,
            stage_cb=stage_cb,
        )
    except Exception as exc:
        RETRAIN_TOTAL.labels(outcome="failed").inc()
        raise HTTPException(status_code=500, detail=f"Retrain failed: {exc}") from exc

    if outcome.decision.promoted:
        MODEL_STORE.load(
            model_path=settings.model_path,
            metadata_path=settings.metadata_path,
            reference_path=settings.reference_data_path,
        )
    return outcome


@router.post(
    "",
    response_model=RetrainResponse,
    summary="Retrain a challenger and promote it if it beats production",
)
def trigger_retrain(
    settings: Annotated[Settings, Depends(get_settings)],
    _admin: Annotated[TokenPayload, Depends(require_admin)],
    trigger: str = Query("manual", description="Free-form provenance tag stored on the audit row"),
) -> RetrainResponse:
    outcome = _execute_retrain(settings=settings, trigger=trigger)
    return _outcome_to_response(outcome)


# ---------------------------------------------------------------------------
# Async flavour — background job + SSE log stream
# ---------------------------------------------------------------------------

def _run_in_thread(run: RetrainRun, settings: Settings, trigger: str) -> None:
    """Thread body used by POST /retrain/start.

    Runs the retrain pipeline synchronously but pushes every log line and
    stage transition into the shared :data:`RUN_STORE` so the SSE endpoint
    can relay them to the browser in real time.
    """
    try:
        outcome = _execute_retrain(
            settings=settings,
            trigger=trigger,
            log_cb=lambda line: RUN_STORE.append_log(run.run_id, line),
            stage_cb=lambda stage: RUN_STORE.set_stage(run.run_id, stage),
        )
    except HTTPException as exc:
        RUN_STORE.append_log(run.run_id, f"[error] {exc.detail}")
        RUN_STORE.finalise(run.run_id, state="failed", error=str(exc.detail))
        return
    except Exception as exc:  # noqa: BLE001 — capture everything so the UI can show it
        RUN_STORE.append_log(run.run_id, f"[error] {exc}")
        RUN_STORE.finalise(run.run_id, state="failed", error=str(exc))
        return

    response = _outcome_to_response(outcome)
    RUN_STORE.finalise(run.run_id, state="succeeded", result=response.model_dump())


@router.post(
    "/start",
    response_model=RetrainRunStart,
    summary="Launch a retrain in the background and return a run id",
)
def start_retrain(
    settings: Annotated[Settings, Depends(get_settings)],
    _admin: Annotated[TokenPayload, Depends(require_admin)],
    trigger: str = Query("manual", description="Provenance tag stored on the audit row"),
) -> RetrainRunStart:
    active = RUN_STORE.active()
    if active is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Retrain already running (run_id={active.run_id}); wait for it to finish.",
        )
    run = RUN_STORE.create(trigger=trigger)
    thread = threading.Thread(
        target=_run_in_thread,
        args=(run, settings, trigger),
        daemon=True,
        name=f"retrain-{run.run_id}",
    )
    thread.start()
    base = "/api/v1/retrain/runs"
    return RetrainRunStart(
        run_id=run.run_id,
        status_url=f"{base}/{run.run_id}",
        logs_url=f"{base}/{run.run_id}/logs",
    )


def _run_to_status(run: RetrainRun, *, include_logs: bool = True) -> RetrainRunStatus:
    result = run.result
    result_obj = RetrainResponse(**result) if result else None
    return RetrainRunStatus(
        run_id=run.run_id,
        trigger=run.trigger,
        started_at=run.started_at,
        ended_at=run.ended_at,
        state=run.state,
        stage=run.stage,
        percent=run.percent,
        log_count=len(run.logs),
        logs=list(run.logs) if include_logs else [],
        result=result_obj,
        error=run.error,
    )


@router.get(
    "/runs/{run_id}",
    response_model=RetrainRunStatus,
    summary="Snapshot of an in-flight or completed retrain run",
)
def run_status(
    run_id: str,
    _admin: Annotated[TokenPayload, Depends(require_admin)],
    tail: int = Query(200, ge=0, le=500, description="Number of recent log lines to include"),
) -> RetrainRunStatus:
    run = RUN_STORE.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}")
    snapshot = _run_to_status(run)
    if tail < len(snapshot.logs):
        snapshot.logs = snapshot.logs[-tail:]
    return snapshot


@router.get(
    "/runs",
    summary="List recent retrain runs (newest first)",
)
def list_runs(
    _admin: Annotated[TokenPayload, Depends(require_admin)],
) -> dict[str, Any]:
    runs = RUN_STORE.list()
    return {
        "n": len(runs),
        "runs": [_run_to_status(r, include_logs=False).model_dump() for r in reversed(runs)],
    }


@router.get(
    "/active",
    response_model=RetrainRunStatus | None,
    summary="The in-flight retrain run (if any) so the UI can reconnect on reload",
)
def active_run(
    _admin: Annotated[TokenPayload, Depends(require_admin)],
) -> RetrainRunStatus | None:
    run = RUN_STORE.active()
    if run is None:
        return None
    return _run_to_status(run)


@router.get(
    "/runs/{run_id}/logs",
    summary="Server-sent events stream of logs for a retrain run",
)
async def stream_logs(
    run_id: str,
    request: Request,
    _admin: Annotated[TokenPayload, Depends(require_admin)],
) -> StreamingResponse:
    run = RUN_STORE.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}")

    queue = RUN_STORE.subscribe(run_id)
    if queue is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}")

    async def event_generator() -> Any:
        try:
            # 5s keepalive — shorter than any default proxy idle timeout
            # we're likely to meet (nginx 60s, cloudflare 100s). Long silent
            # stretches during MLP training used to drop the connection.
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
                latest = RUN_STORE.get(run_id)
                if latest is not None and latest.state in ("succeeded", "failed") and queue.empty():
                    break
        finally:
            RUN_STORE.unsubscribe(run_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/history",
    response_model=RetrainHistoryResponse,
    summary="Newest-first audit log of retrain decisions",
)
def history(
    settings: Annotated[Settings, Depends(get_settings)],
    _admin: Annotated[TokenPayload, Depends(require_admin)],
    limit: int = Query(50, ge=1, le=500),
) -> RetrainHistoryResponse:
    rows = read_audit(settings.retrain_history_path, limit=limit)
    entries = [RetrainAuditEntry(**row) for row in rows]
    return RetrainHistoryResponse(n=len(entries), entries=entries)
