"""Single + batch prediction endpoints."""
from __future__ import annotations

import io
import time
from datetime import datetime, timezone
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.app.api.v1.model_registry import MODEL_STORE, predict_one
from backend.app.api.v1.schemas import (
    BatchPredictionResponse,
    BatchPredictionRow,
    InterventionRecommendation,
    PredictionResponse,
    ShapContribution,
    StudentFeatures,
)
from backend.app.core.metrics import (
    BATCH_PREDICTIONS,
    PREDICT_LATENCY,
    PREDICTION_TOTAL,
    record_prediction,
)
from backend.app.core.security import require_api_key
from backend.app.interventions.recommender import recommend
from backend.app.ml.schemas import PredictionFeaturesSchema, feature_columns

router = APIRouter(prefix="/predict", tags=["predictions"])


def _model_version(loaded: Any) -> str:
    run_id = str(loaded.metadata.get("champion_run_id", "") or "")
    if run_id:
        return run_id[:12]
    return loaded.model_name


def _shap_contributions(raw: list[dict[str, object]]) -> list[ShapContribution]:
    """Map explainer output to the frontend ShapContribution shape.

    Older explainer payloads used ``shap_value`` / ``direction``; the
    newer shape provides ``value`` / ``contribution`` directly. We accept
    both so a stale model bundle still renders.
    """
    out: list[ShapContribution] = []
    for f in raw:
        contribution = float(f.get("contribution", f.get("shap_value", 0.0)) or 0.0)
        value = float(f.get("value", contribution) or contribution)
        out.append(
            ShapContribution(
                feature=str(f.get("feature", "")),
                value=value,
                contribution=contribution,
            )
        )
    return out


def _build_response(
    loaded: Any, record: dict[str, Any], *, student_id: str | None = None,
) -> PredictionResponse:
    res = predict_one(loaded, record)
    interventions = recommend(risk_level=res["risk_level"], top_features=res["top_features"])
    return PredictionResponse(
        student_id=student_id,
        risk_level=res["risk_level"],
        predicted_class=res["predicted_class"],
        probabilities=res["probabilities"],
        top_shap_features=_shap_contributions(res["top_features"]),
        recommended_interventions=[
            InterventionRecommendation(
                code=i.code,
                title=i.title,
                description=i.description,
                owner=i.owner,
                priority=i.priority,
            )
            for i in interventions
        ],
        model_name=loaded.model_name,
        model_version=_model_version(loaded),
        scored_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predict dropout risk for a single student",
    dependencies=[Depends(require_api_key)],
)
def predict_single(payload: StudentFeatures) -> PredictionResponse:
    if not MODEL_STORE.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded — has the training pipeline run?",
        )
    loaded = MODEL_STORE.get()
    start = time.perf_counter()
    response = _build_response(loaded, payload.to_record())
    PREDICT_LATENCY.observe(time.perf_counter() - start)
    PREDICTION_TOTAL.labels(risk_level=response.risk_level).inc()
    record_prediction()
    return response


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Predict dropout risk for a CSV upload",
    dependencies=[Depends(require_api_key)],
)
async def predict_batch(file: Annotated[UploadFile, File(...)]) -> BatchPredictionResponse:
    if not MODEL_STORE.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    feat_cols = feature_columns()
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV missing required columns: {missing[:5]}{' ...' if len(missing) > 5 else ''}",
        )

    try:
        validated = PredictionFeaturesSchema.validate(df[feat_cols], lazy=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Schema validation failed: {exc}") from exc

    loaded = MODEL_STORE.get()
    rows: list[BatchPredictionRow] = []
    distribution: dict[str, int] = {"High": 0, "Medium": 0, "Low": 0}
    failed = 0
    for idx, row in validated.iterrows():
        try:
            response = _build_response(
                loaded, row.to_dict(), student_id=f"S-{int(idx):05d}",
            )
        except Exception:  # noqa: BLE001 — per-row resilience for large batches
            failed += 1
            continue
        payload = response.model_dump()
        payload.pop("row_index", None)  # base schema doesn't have it; add explicitly
        rows.append(BatchPredictionRow(row_index=int(idx), **payload))
        distribution[response.risk_level] = distribution.get(response.risk_level, 0) + 1
        PREDICTION_TOTAL.labels(risk_level=response.risk_level).inc()
        record_prediction()

    BATCH_PREDICTIONS.inc(len(rows))
    return BatchPredictionResponse(
        total_rows=len(validated),
        scored_rows=len(rows),
        failed_rows=failed,
        model_version=_model_version(loaded),
        predictions=rows,
        risk_distribution=distribution,
    )
