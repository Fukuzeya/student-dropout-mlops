"""Recommender tests — the rules must fire on the expected SHAP signals."""
from __future__ import annotations

from backend.app.interventions.recommender import CATALOGUE, recommend


def test_low_risk_returns_recognition() -> None:
    interventions = recommend(risk_level="Low", top_features=[])
    assert len(interventions) == 1
    assert interventions[0].code == "RECOGNITION"


def test_high_risk_with_financial_signal() -> None:
    interventions = recommend(
        risk_level="High",
        top_features=[{"feature": "binary__Tuition fees up to date", "shap_value": 0.5, "direction": "increases-risk"}],
    )
    codes = {i.code for i in interventions}
    assert "FIN_AID_REFERRAL" in codes


def test_high_risk_with_momentum_signal() -> None:
    interventions = recommend(
        risk_level="High",
        top_features=[
            {"feature": "momentum__delta_grade", "shap_value": 0.7, "direction": "increases-risk"},
        ],
    )
    codes = {i.code for i in interventions}
    assert "MOMENTUM_CHECKIN" in codes


def test_caps_at_four_interventions() -> None:
    interventions = recommend(
        risk_level="High",
        top_features=[
            {"feature": "binary__Tuition fees up to date", "shap_value": 1, "direction": "increases-risk"},
            {"feature": "momentum__delta_grade", "shap_value": 1, "direction": "increases-risk"},
            {"feature": "demo_cat__Marital status_2", "shap_value": 1, "direction": "increases-risk"},
            {"feature": "binary__Displaced", "shap_value": 1, "direction": "increases-risk"},
            {"feature": "fin_cat__Daytime/evening attendance_0", "shap_value": 1, "direction": "increases-risk"},
            {"feature": "fin_cat__Course_33", "shap_value": 1, "direction": "increases-risk"},
        ],
    )
    assert len(interventions) <= 4


def test_catalogue_complete() -> None:
    # All catalogue entries have non-empty fields
    for code, item in CATALOGUE.items():
        assert item.code == code
        assert item.title and item.description and item.owner and item.priority
