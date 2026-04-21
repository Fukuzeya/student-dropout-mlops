"""End-to-end /predict tests using a freshly-trained tiny model."""
from __future__ import annotations

import io
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from backend.app.api.v1.model_registry import MODEL_STORE
from backend.app.main import create_app
from backend.app.ml.features import build_preprocessor
from backend.app.ml.models import MODEL_FACTORIES
from backend.app.ml.schemas import TARGET_CLASSES, feature_columns


def _train_tiny_pipeline(df: pd.DataFrame) -> Pipeline:
    feat_cols = feature_columns()
    y = pd.Categorical(df["Target"], categories=TARGET_CLASSES).codes
    pipe = Pipeline([
        ("features", build_preprocessor()),
        ("model", MODEL_FACTORIES["xgboost"](
            {"n_estimators": 30, "max_depth": 3, "objective": "multi:softprob"}, 0,
        )),
    ])
    pipe.fit(df[feat_cols], y)
    return pipe


def _load_into_store(tmp_path: Path, df: pd.DataFrame) -> None:
    pipe = _train_tiny_pipeline(df)
    bundle_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "pipeline": pipe,
            "model_name": "xgboost",
            "classes": TARGET_CLASSES,
            "feature_columns": feature_columns(),
        },
        bundle_path,
    )
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text('{"metrics": {"macro_f1": 0.9}}')
    ref_path = tmp_path / "ref.parquet"
    df.to_parquet(ref_path, index=False)
    MODEL_STORE.load(bundle_path, metadata_path, ref_path)


def test_predict_single_returns_full_response(
    tmp_path: Path, synthetic_raw: pd.DataFrame, sample_student_record: dict,
) -> None:
    _load_into_store(tmp_path, synthetic_raw)
    client = TestClient(create_app())
    resp = client.post(
        "/api/v1/predict",
        json=sample_student_record,
        headers={"X-API-Key": "test-api-key"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert body["predicted_class"] in TARGET_CLASSES
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["recommended_interventions"]
    assert body["model_name"] == "xgboost"
    assert body["model_version"]  # either run_id[:12] or model_name fallback
    assert body["scored_at"]


def test_predict_batch_csv(
    tmp_path: Path, synthetic_raw: pd.DataFrame,
) -> None:
    _load_into_store(tmp_path, synthetic_raw)
    client = TestClient(create_app())
    csv_buf = io.StringIO()
    synthetic_raw.drop(columns=["Target"]).head(5).to_csv(csv_buf, index=False)
    resp = client.post(
        "/api/v1/predict/batch",
        headers={"X-API-Key": "test-api-key"},
        files={"file": ("students.csv", csv_buf.getvalue().encode("utf-8"), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_rows"] == 5
    assert body["scored_rows"] + body["failed_rows"] == 5
    assert len(body["predictions"]) == body["scored_rows"]
    assert sum(body["risk_distribution"].values()) == body["scored_rows"]
    assert set(body["risk_distribution"]) == {"High", "Medium", "Low"}


def test_health_reports_model_loaded(tmp_path: Path, synthetic_raw: pd.DataFrame) -> None:
    _load_into_store(tmp_path, synthetic_raw)
    client = TestClient(create_app())
    resp = client.get("/api/v1/monitoring/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_metrics_endpoint() -> None:
    client = TestClient(create_app())
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"predict_latency_seconds" in resp.content
