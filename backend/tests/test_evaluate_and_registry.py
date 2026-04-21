"""Tests for evaluate.py + registry.py — methodology-critical code."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.app.ml.evaluate import evaluate_predictions
from backend.app.ml.registry import compare_for_promotion
from backend.app.ml.schemas import TARGET_CLASSES
from backend.app.monitoring.audit import AuditEntry, append_audit, read_audit, utc_now_iso


def test_evaluate_predictions_macro_f1_perfect() -> None:
    y = np.array(["Dropout", "Enrolled", "Graduate"] * 10)
    metrics = evaluate_predictions(y, y)
    assert metrics["macro_f1"] == 1.0
    assert metrics["dropout_recall"] == 1.0
    for cls in TARGET_CLASSES:
        assert metrics["per_class"][cls]["f1"] == 1.0


def test_promotion_rejects_smaller_gain() -> None:
    champion = {"macro_f1": 0.85, "per_class": {c: {"f1": 0.85} for c in TARGET_CLASSES}}
    challenger = {"macro_f1": 0.855, "per_class": {c: {"f1": 0.85} for c in TARGET_CLASSES}}
    decision = compare_for_promotion(champion, challenger)
    assert decision.promoted is False
    assert "macro-F1 gain" in decision.reason


def test_promotion_rejects_per_class_regression() -> None:
    champion = {"macro_f1": 0.85, "per_class": {c: {"f1": 0.85} for c in TARGET_CLASSES}}
    challenger = {
        "macro_f1": 0.87,
        "per_class": {"Dropout": {"f1": 0.78}, "Enrolled": {"f1": 0.90}, "Graduate": {"f1": 0.93}},
    }
    decision = compare_for_promotion(champion, challenger)
    assert decision.promoted is False
    assert "Dropout" in decision.reason


def test_promotion_accepts_clean_win() -> None:
    champion = {"macro_f1": 0.80, "per_class": {c: {"f1": 0.80} for c in TARGET_CLASSES}}
    challenger = {"macro_f1": 0.85, "per_class": {c: {"f1": 0.85} for c in TARGET_CLASSES}}
    decision = compare_for_promotion(champion, challenger)
    assert decision.promoted is True


# ---------------------------------------------------------------------------
# McNemar gate
# ---------------------------------------------------------------------------

def test_promotion_blocked_when_mcnemar_not_significant() -> None:
    """Effect size passes but the paired test cannot reject the null."""
    champion = {"macro_f1": 0.80, "per_class": {c: {"f1": 0.80} for c in TARGET_CLASSES}}
    challenger = {"macro_f1": 0.85, "per_class": {c: {"f1": 0.85} for c in TARGET_CLASSES}}
    # Identical predictions ⇒ McNemar's b = c = 0 ⇒ p = 1.0 ⇒ block.
    y = np.array(["Dropout", "Enrolled", "Graduate"] * 10)
    decision = compare_for_promotion(
        champion, challenger,
        paired_predictions=(y, y, y),
    )
    assert decision.promoted is False
    assert "McNemar" in decision.reason
    assert decision.mcnemar is not None
    assert decision.mcnemar.p_value == 1.0


def test_promotion_passes_with_significant_mcnemar() -> None:
    champion_metrics = {"macro_f1": 0.80, "per_class": {c: {"f1": 0.80} for c in TARGET_CLASSES}}
    challenger_metrics = {"macro_f1": 0.85, "per_class": {c: {"f1": 0.85} for c in TARGET_CLASSES}}
    # Champion gets every row wrong; challenger gets every row right.
    y = np.array(["Dropout"] * 30)
    champ_pred = np.array(["Enrolled"] * 30)
    chal_pred = y.copy()
    decision = compare_for_promotion(
        champion_metrics, challenger_metrics,
        paired_predictions=(y, champ_pred, chal_pred),
    )
    assert decision.promoted is True
    assert decision.mcnemar is not None
    assert decision.mcnemar.p_value < 0.05
    assert "McNemar p=" in decision.reason


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def _entry(promoted: bool, delta: float) -> AuditEntry:
    return AuditEntry(
        timestamp=utc_now_iso(),
        trigger="unit-test",
        promoted=promoted,
        reason="test",
        champion_macro_f1=0.80,
        challenger_macro_f1=0.80 + delta,
        macro_f1_delta=delta,
        per_class_deltas={c: 0.0 for c in TARGET_CLASSES},
        mcnemar_p_value=0.01 if promoted else 0.5,
        mcnemar_b=10,
        mcnemar_c=2 if promoted else 8,
        mcnemar_significant=promoted,
        n_test=200,
    )


def test_audit_log_round_trips_and_returns_newest_first(tmp_path: Path) -> None:
    log = tmp_path / "history.jsonl"
    append_audit(_entry(False, -0.01), log)
    append_audit(_entry(True, 0.02), log)
    rows = read_audit(log)
    assert [r["promoted"] for r in rows] == [True, False]  # newest first
    assert all("timestamp" in r and "mcnemar_p_value" in r for r in rows)


def test_audit_log_limit_caps_response(tmp_path: Path) -> None:
    log = tmp_path / "history.jsonl"
    for i in range(5):
        append_audit(_entry(i % 2 == 0, 0.01 * i), log)
    rows = read_audit(log, limit=2)
    assert len(rows) == 2


def test_audit_log_skips_corrupt_lines(tmp_path: Path) -> None:
    log = tmp_path / "history.jsonl"
    append_audit(_entry(True, 0.02), log)
    log.open("a", encoding="utf-8").write("{not json}\n")
    append_audit(_entry(False, -0.01), log)
    rows = read_audit(log)
    assert len(rows) == 2  # corrupt line dropped, both valid entries kept
