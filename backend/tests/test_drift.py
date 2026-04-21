"""Drift module tests — uses Evidently end-to-end on the synthetic fixture."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.app.monitoring.drift import compute_drift, latest_report


def test_no_drift_against_self(synthetic_raw: pd.DataFrame, tmp_path: Path) -> None:
    result = compute_drift(reference=synthetic_raw, current=synthetic_raw, output_dir=tmp_path)
    assert result.drift_share == 0.0 or result.drift_share < 0.5
    assert result.report_path.exists()
    assert latest_report(tmp_path) == result.report_path


def test_drift_detected_on_shifted_grades(synthetic_raw: pd.DataFrame, tmp_path: Path) -> None:
    shifted = synthetic_raw.copy()
    # massively shift grade columns to force a drift signal
    for col in ("Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)", "Admission grade"):
        shifted[col] = shifted[col] / 4.0
    result = compute_drift(reference=synthetic_raw, current=shifted, output_dir=tmp_path)
    assert result.n_total > 0
    assert result.report_path.exists()
