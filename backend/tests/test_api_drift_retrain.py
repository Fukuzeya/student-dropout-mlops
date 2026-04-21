"""End-to-end tests for the chained POST /monitoring/drift/auto-retrain endpoint.

The endpoint glues `compute_drift` to `run_retraining`. We stub both so the
test is fast and deterministic — what we're actually verifying is the gating
logic (skip vs. fire), the audit trail, and the response shape the UI relies on.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.api.v1 import monitoring as monitoring_api
from backend.app.main import create_app
from backend.app.ml.registry import PromotionDecision
from backend.app.ml.statistics import McNemarResult
from backend.app.monitoring.audit import AuditEntry, utc_now_iso
from backend.app.monitoring.drift import DriftResult
from backend.app.monitoring.retraining import RetrainOutcome


def _admin_jwt(client: TestClient) -> str:
    resp = client.post("/api/v1/auth/token", data={"username": "admin", "password": "admin"})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _seed_artifacts(tmp_path: Path, synthetic_raw: pd.DataFrame) -> Path:
    """Create reference + holdout parquets the endpoint expects to find."""
    ref_dir = tmp_path / "data" / "reference"
    ref_dir.mkdir(parents=True)
    ref_path = ref_dir / "reference.parquet"
    synthetic_raw.to_parquet(ref_path, index=False)

    test_dir = tmp_path / "data" / "processed"
    test_dir.mkdir(parents=True)
    synthetic_raw.to_parquet(test_dir / "test.parquet", index=False)
    return ref_path


def _stub_drift(detected: bool, share: float) -> DriftResult:
    return DriftResult(
        drift_share=share,
        n_drifted=int(round(share * 10)),
        n_total=10,
        report_path=Path("reports/drift/stub.html"),
        detected=detected,
    )


def _stub_outcome(promoted: bool) -> RetrainOutcome:
    decision = PromotionDecision(
        promoted=promoted,
        reason="stubbed",
        champion_macro_f1=0.80,
        challenger_macro_f1=0.86 if promoted else 0.79,
        per_class_deltas={"Dropout": 0.06, "Enrolled": 0.05, "Graduate": 0.07},
        mcnemar=McNemarResult(b=12, c=2, statistic=6.43, p_value=0.01, significant_at_05=True),
    )
    audit = AuditEntry(
        timestamp=utc_now_iso(),
        trigger="drift:share=0.500",
        promoted=promoted,
        reason="stubbed",
        champion_macro_f1=decision.champion_macro_f1,
        challenger_macro_f1=decision.challenger_macro_f1,
        macro_f1_delta=decision.challenger_macro_f1 - decision.champion_macro_f1,
        per_class_deltas=decision.per_class_deltas,
        mcnemar_p_value=0.01,
        mcnemar_b=12,
        mcnemar_c=2,
        mcnemar_significant=True,
        n_test=200,
    )
    return RetrainOutcome(
        decision=decision,
        challenger_metrics={"macro_f1": decision.challenger_macro_f1},
        champion_metrics={"macro_f1": decision.champion_macro_f1},
        audit_entry=audit,
    )


@pytest.fixture
def stubbed_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, synthetic_raw: pd.DataFrame,
) -> tuple[TestClient, dict[str, object]]:
    monkeypatch.chdir(tmp_path)
    ref_path = _seed_artifacts(tmp_path, synthetic_raw)
    monkeypatch.setenv("REFERENCE_DATA_PATH", str(ref_path))
    monkeypatch.setenv("DRIFT_REPORT_DIR", str(tmp_path / "reports" / "drift"))
    monkeypatch.setenv("RETRAIN_HISTORY_PATH", str(tmp_path / "reports" / "retraining" / "history.jsonl"))
    from backend.app.core.config import get_settings
    get_settings.cache_clear()

    state: dict[str, object] = {"drift_calls": 0, "retrain_calls": 0, "promote_next": False}

    def fake_compute_drift(**kwargs: object) -> DriftResult:
        state["drift_calls"] = int(state["drift_calls"]) + 1  # type: ignore[arg-type]
        share = float(state.get("drift_share", 0.10))  # type: ignore[arg-type]
        threshold = float(kwargs.get("threshold", 0.30))  # type: ignore[arg-type]
        return _stub_drift(detected=share >= threshold, share=share)

    def fake_run_retraining(**_: object) -> RetrainOutcome:
        state["retrain_calls"] = int(state["retrain_calls"]) + 1  # type: ignore[arg-type]
        return _stub_outcome(promoted=bool(state["promote_next"]))

    monkeypatch.setattr(monitoring_api, "compute_drift", fake_compute_drift)
    monkeypatch.setattr(monitoring_api, "run_retraining", fake_run_retraining)
    # Promotion path reloads the model store from disk; stub since we have no real bundle.
    monkeypatch.setattr(monitoring_api.MODEL_STORE, "load", lambda **_: None)
    return TestClient(create_app()), state


def _csv_payload(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.drop(columns=["Target"]).head(5).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def test_skips_retrain_when_drift_below_threshold(
    stubbed_endpoint: tuple[TestClient, dict[str, object]], synthetic_raw: pd.DataFrame,
) -> None:
    client, state = stubbed_endpoint
    state["drift_share"] = 0.10  # well under default 0.30
    token = _admin_jwt(client)
    resp = client.post(
        "/api/v1/monitoring/drift/auto-retrain",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("batch.csv", _csv_payload(synthetic_raw), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["skipped"] is True
    assert body["retrain"] is None
    assert body["drift"]["drift_share"] == pytest.approx(0.10)
    assert state["retrain_calls"] == 0


def test_force_overrides_threshold_and_runs_retrain(
    stubbed_endpoint: tuple[TestClient, dict[str, object]], synthetic_raw: pd.DataFrame,
) -> None:
    client, state = stubbed_endpoint
    state["drift_share"] = 0.05  # under threshold; only `force` should fire it
    state["promote_next"] = True
    token = _admin_jwt(client)
    resp = client.post(
        "/api/v1/monitoring/drift/auto-retrain",
        headers={"Authorization": f"Bearer {token}"},
        params={"force": "true"},
        files={"file": ("batch.csv", _csv_payload(synthetic_raw), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["skipped"] is False
    assert body["retrain"] is not None
    assert body["retrain"]["promoted"] is True
    assert body["retrain"]["mcnemar_p_value"] == pytest.approx(0.01)
    assert body["retrain"]["trigger"].startswith("drift:")
    assert state["retrain_calls"] == 1


def test_drift_above_threshold_triggers_retrain(
    stubbed_endpoint: tuple[TestClient, dict[str, object]], synthetic_raw: pd.DataFrame,
) -> None:
    client, state = stubbed_endpoint
    state["drift_share"] = 0.55
    state["promote_next"] = False  # challenger loses → not promoted, but retrain still ran
    token = _admin_jwt(client)
    resp = client.post(
        "/api/v1/monitoring/drift/auto-retrain",
        headers={"Authorization": f"Bearer {token}"},
        params={"threshold": "0.30"},
        files={"file": ("batch.csv", _csv_payload(synthetic_raw), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["skipped"] is False
    assert body["retrain"]["promoted"] is False
    assert body["drift"]["drift_share"] == pytest.approx(0.55)
    assert state["retrain_calls"] == 1


def test_requires_admin_jwt(
    stubbed_endpoint: tuple[TestClient, dict[str, object]], synthetic_raw: pd.DataFrame,
) -> None:
    client, _ = stubbed_endpoint
    resp = client.post(
        "/api/v1/monitoring/drift/auto-retrain",
        files={"file": ("batch.csv", _csv_payload(synthetic_raw), "text/csv")},
    )
    assert resp.status_code == 401


def test_503_when_reference_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, synthetic_raw: pd.DataFrame,
) -> None:
    monkeypatch.chdir(tmp_path)
    # Create only the holdout, NOT the reference snapshot.
    proc_dir = tmp_path / "data" / "processed"
    proc_dir.mkdir(parents=True)
    synthetic_raw.to_parquet(proc_dir / "test.parquet", index=False)
    monkeypatch.setenv("REFERENCE_DATA_PATH", str(tmp_path / "missing.parquet"))
    from backend.app.core.config import get_settings
    get_settings.cache_clear()
    client = TestClient(create_app())
    token = _admin_jwt(client)
    resp = client.post(
        "/api/v1/monitoring/drift/auto-retrain",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("batch.csv", _csv_payload(synthetic_raw), "text/csv")},
    )
    assert resp.status_code == 503
    assert "Reference snapshot missing" in resp.json()["detail"]


def test_503_when_holdout_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, synthetic_raw: pd.DataFrame,
) -> None:
    monkeypatch.chdir(tmp_path)
    ref_dir = tmp_path / "data" / "reference"
    ref_dir.mkdir(parents=True)
    ref_path = ref_dir / "reference.parquet"
    synthetic_raw.to_parquet(ref_path, index=False)
    monkeypatch.setenv("REFERENCE_DATA_PATH", str(ref_path))
    from backend.app.core.config import get_settings
    get_settings.cache_clear()
    client = TestClient(create_app())
    token = _admin_jwt(client)
    resp = client.post(
        "/api/v1/monitoring/drift/auto-retrain",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("batch.csv", _csv_payload(synthetic_raw), "text/csv")},
    )
    assert resp.status_code == 503
    assert "Holdout split missing" in resp.json()["detail"]
