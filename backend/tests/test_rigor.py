"""Rigor-pipeline unit tests.

Cover the five modules wired into the DVC `evaluate` stage:

* :mod:`backend.app.ml.statistics`     — bootstrap CIs + McNemar.
* :mod:`backend.app.ml.calibration`    — Brier, ECE, temperature scaling.
* :mod:`backend.app.ml.threshold`      — Dropout-recall-targeted threshold.
* :mod:`backend.app.ml.cost_sensitive` — cost-matrix evaluation.
* :mod:`backend.app.ml.fairness`       — subgroup parity gaps.

The tests stay deterministic (numpy seeded) and run in milliseconds so they
fit comfortably under the 80% coverage gate without slowing CI down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.ml.calibration import (
    apply_temperature,
    compute_calibration,
    expected_calibration_error,
    fit_temperature_scaler,
)
from backend.app.ml.cost_sensitive import DEFAULT_COST_MATRIX, evaluate_cost
from backend.app.ml.fairness import add_age_band, evaluate_fairness
from backend.app.ml.schemas import TARGET_CLASSES
from backend.app.ml.statistics import bootstrap_ci, mcnemar_test
from backend.app.ml.threshold import (
    choose_threshold,
    reclassify_with_threshold,
    sweep_thresholds,
)

CLASSES = TARGET_CLASSES


# ---------------------------------------------------------------------------
# statistics.py
# ---------------------------------------------------------------------------

def test_bootstrap_ci_perfect_predictions_is_degenerate() -> None:
    """A perfect classifier has zero variance — CI must collapse to the point."""
    y = np.array(["Dropout", "Enrolled", "Graduate"] * 30)
    res = bootstrap_ci(y, y, metric="macro_f1", n_resamples=200, random_state=0)
    assert res.point == pytest.approx(1.0)
    assert res.lower == pytest.approx(1.0)
    assert res.upper == pytest.approx(1.0)


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.choice(CLASSES, size=200)
    y_pred = y_true.copy()
    flip = rng.integers(0, len(y_true), size=40)
    y_pred[flip] = rng.choice(CLASSES, size=40)
    res = bootstrap_ci(
        y_true, y_pred, metric="macro_f1", n_resamples=400, random_state=1
    )
    assert res.lower <= res.point <= res.upper
    assert 0.0 < res.lower < res.upper < 1.0


def test_bootstrap_ci_unknown_metric_raises() -> None:
    y = np.array(["Dropout"] * 5)
    with pytest.raises(KeyError):
        bootstrap_ci(y, y, metric="nonexistent")


def test_mcnemar_returns_neutral_when_no_disagreement() -> None:
    y = np.array(["Dropout", "Enrolled", "Graduate"] * 5)
    res = mcnemar_test(y, y, y)
    assert res.b == 0 and res.c == 0
    assert res.p_value == 1.0
    assert res.significant_at_05 is False


def test_mcnemar_detects_strong_imbalance() -> None:
    y = np.array(["Dropout"] * 30)
    champ = y.copy()
    chall = np.array(["Enrolled"] * 30)  # always wrong
    res = mcnemar_test(y, champ, chall)
    assert res.b == 30 and res.c == 0
    assert res.significant_at_05 is True


# ---------------------------------------------------------------------------
# calibration.py
# ---------------------------------------------------------------------------

def _well_calibrated_proba(y_true: np.ndarray, classes: list[str]) -> np.ndarray:
    """Return the one-hot encoding (perfectly calibrated, ECE = 0)."""
    idx = {c: i for i, c in enumerate(classes)}
    p = np.zeros((len(y_true), len(classes)))
    for r, label in enumerate(y_true):
        p[r, idx[label]] = 1.0
    return p


def test_compute_calibration_perfect_confidence_zero_ece() -> None:
    y = np.array(["Dropout", "Enrolled", "Graduate"] * 20)
    p = _well_calibrated_proba(y, CLASSES)
    report = compute_calibration(y, p, CLASSES, n_bins=10)
    assert report.ece_macro == pytest.approx(0.0, abs=1e-9)
    assert report.brier_macro == pytest.approx(0.0, abs=1e-9)
    assert set(report.brier_per_class) == set(CLASSES)


def test_expected_calibration_error_equal_mass_bins() -> None:
    rng = np.random.default_rng(0)
    n = 300
    p = rng.uniform(0, 1, n)
    # Make accuracy track confidence: y ~ Bernoulli(p)
    y = (rng.uniform(0, 1, n) < p).astype(float)
    ece, conf, acc, count = expected_calibration_error(y, p, n_bins=10)
    assert 0.0 <= ece < 0.2
    assert count.sum() == n  # every sample landed in some bin


def test_temperature_scaler_recovers_sane_value() -> None:
    rng = np.random.default_rng(0)
    n = 500
    y = rng.choice(CLASSES, n)
    p = _well_calibrated_proba(y, CLASSES)
    # Soften the predictions (simulate over-confidence)
    soft = p * 0.7 + 0.1
    soft = soft / soft.sum(axis=1, keepdims=True)
    t = fit_temperature_scaler(y, soft, CLASSES)
    assert t > 0.0
    rescaled = apply_temperature(soft, t)
    assert rescaled.shape == soft.shape
    assert np.allclose(rescaled.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# threshold.py
# ---------------------------------------------------------------------------

def test_reclassify_with_threshold_promotes_dropout() -> None:
    classes = CLASSES
    proba = np.array([
        [0.40, 0.35, 0.25],   # argmax=Dropout, also above any low threshold
        [0.20, 0.45, 0.35],   # argmax=Enrolled, T=0.10 should flip to Dropout
        [0.05, 0.50, 0.45],   # argmax=Enrolled, T=0.10 keeps Enrolled (Dropout < T)
    ])
    out = reclassify_with_threshold(proba, classes, threshold=0.10)
    assert out[0] == "Dropout"
    assert out[1] == "Dropout"  # promoted
    assert out[2] == "Enrolled"


def test_reclassify_with_threshold_unknown_class_raises() -> None:
    proba = np.zeros((1, 3))
    with pytest.raises(ValueError):
        reclassify_with_threshold(proba, CLASSES, 0.5, target_class="Missing")


def test_sweep_thresholds_emits_one_row_per_grid_point() -> None:
    rng = np.random.default_rng(0)
    y = rng.choice(CLASSES, 80)
    proba = rng.dirichlet(alpha=[1, 1, 1], size=80)
    sweep = sweep_thresholds(y, proba, CLASSES)
    assert len(sweep) > 0
    assert all(0.0 <= row.threshold <= 1.0 for row in sweep)
    assert all(0.0 <= row.macro_f1 <= 1.0 for row in sweep)


def test_choose_threshold_falls_back_when_target_unreachable() -> None:
    """If recall target can never be met, fall back to the most-sensitive op-point."""
    y = np.array(["Dropout", "Enrolled", "Graduate"] * 5)
    # Very low Dropout probabilities — no threshold can hit recall ≥ 0.99.
    proba = np.tile(np.array([0.05, 0.5, 0.45]), (15, 1))
    decision = choose_threshold(y, proba, CLASSES, target_recall=0.99)
    assert "No threshold met" in decision.rationale


def test_choose_threshold_picks_recall_compliant_threshold() -> None:
    rng = np.random.default_rng(2)
    n = 200
    y = rng.choice(CLASSES, n)
    p = _well_calibrated_proba(y, CLASSES) * 0.85
    # Add noise so the sweep is non-degenerate
    p += rng.uniform(0, 0.15, p.shape)
    p = p / p.sum(axis=1, keepdims=True)
    decision = choose_threshold(y, p, CLASSES, target_recall=0.5)
    assert 0.10 <= decision.chosen_threshold <= 0.90


# ---------------------------------------------------------------------------
# cost_sensitive.py
# ---------------------------------------------------------------------------

def test_evaluate_cost_perfect_predictions_zero_cost() -> None:
    y = np.array(["Dropout", "Enrolled", "Graduate"] * 10)
    report = evaluate_cost(y, y)
    assert report.total_cost == 0.0
    assert report.cost_per_sample == 0.0
    assert report.expected_utility == 0.0


def test_evaluate_cost_uses_default_matrix_for_misses() -> None:
    y = np.array(["Dropout"] * 4)
    pred = np.array(["Graduate"] * 4)
    report = evaluate_cost(y, pred)
    expected = DEFAULT_COST_MATRIX["Dropout"]["Graduate"] * 4
    assert report.total_cost == pytest.approx(expected)
    assert report.expected_utility == pytest.approx(-expected / 4)


def test_evaluate_cost_custom_matrix_overrides_defaults() -> None:
    y = np.array(["Dropout"] * 2)
    pred = np.array(["Enrolled"] * 2)
    matrix = {
        "Dropout": {"Dropout": 0.0, "Enrolled": 99.0, "Graduate": 0.0},
        "Enrolled": {"Dropout": 0.0, "Enrolled": 0.0, "Graduate": 0.0},
        "Graduate": {"Dropout": 0.0, "Enrolled": 0.0, "Graduate": 0.0},
    }
    report = evaluate_cost(y, pred, cost_matrix=matrix)
    assert report.total_cost == pytest.approx(198.0)


# ---------------------------------------------------------------------------
# fairness.py
# ---------------------------------------------------------------------------

def test_add_age_band_creates_three_buckets(synthetic_raw: pd.DataFrame) -> None:
    df = add_age_band(synthetic_raw)
    assert "age_band" in df.columns
    assert set(df["age_band"].unique()).issubset({"<22", "22-29", "30+"})


def test_evaluate_fairness_returns_attribute_per_sensitive_column(
    synthetic_raw: pd.DataFrame,
) -> None:
    df = synthetic_raw.assign(
        y_true=synthetic_raw["Target"].astype(str),
        y_pred=synthetic_raw["Target"].astype(str),  # perfect classifier
    )
    report = evaluate_fairness(df, min_group_size=1)
    # All five sensitive attributes should appear (Gender, Age band, Scholarship,
    # Debtor, International)
    assert {a.attribute for a in report.attributes} >= {
        "Gender", "Age band", "Scholarship holder", "Debtor", "International",
    }
    # Perfect classifier ⇒ EO/PE gaps zero across every attribute.
    for attr in report.attributes:
        assert attr.equal_opportunity_gap == pytest.approx(0.0, abs=1e-9)
        assert attr.predictive_equality_gap == pytest.approx(0.0, abs=1e-9)
    assert report.summary_max_gap == pytest.approx(0.0, abs=1e-9)
    assert report.n == len(df)


def test_evaluate_fairness_detects_group_disparity(
    synthetic_raw: pd.DataFrame,
) -> None:
    """A classifier that fails for Gender=0 only must surface a gender gap."""
    yt = synthetic_raw["Target"].astype(str).to_numpy()
    yp = yt.copy()
    # Force Gender=0 rows to predict "Graduate" — destroys their Dropout recall
    mask_zero = synthetic_raw["Gender"].to_numpy() == 0
    yp[mask_zero] = "Graduate"
    df = synthetic_raw.assign(y_true=yt, y_pred=yp)
    report = evaluate_fairness(df, min_group_size=1)
    gender = next(a for a in report.attributes if a.attribute == "Gender")
    assert gender.equal_opportunity_gap > 0.0
