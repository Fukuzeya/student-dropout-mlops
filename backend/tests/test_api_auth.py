"""Auth: API key gating + JWT issuance."""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_predict_requires_api_key() -> None:
    client = TestClient(create_app())
    resp = client.post("/api/v1/predict", json={})
    assert resp.status_code == 401


def test_invalid_api_key_rejected(sample_student_record: dict) -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/api/v1/predict", json=sample_student_record, headers={"X-API-Key": "wrong"},
    )
    assert resp.status_code == 401


def test_admin_token_flow_success() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_admin_token_flow_bad_password() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "nope"},
    )
    assert resp.status_code == 401


def test_retrain_requires_admin_jwt() -> None:
    client = TestClient(create_app())
    resp = client.post("/api/v1/retrain")
    assert resp.status_code == 401
