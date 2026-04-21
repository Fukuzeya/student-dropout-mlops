"""Evidently-backed data and target drift checks.

Two surfaces:

* `compute_drift` — pure function used by the API and the retraining loop.
  Returns the dataset drift share (0..1) and the path to a self-contained
  HTML report on disk. The HTML is what the Angular dashboard embeds.
* `cmd_drift`     — CLI entry for ad-hoc runs / cron jobs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(slots=True)
class DriftResult:
    drift_share: float
    n_drifted: int
    n_total: int
    report_path: Path
    detected: bool


def compute_drift(
    *,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path,
    threshold: float = 0.30,
) -> DriftResult:
    """Run an Evidently DataDriftPreset against the reference snapshot.

    Args:
        reference: snapshot of the training distribution (saved during
            `dvc preprocess`).
        current: production batch we want to score for drift.
        output_dir: directory where the HTML report is persisted.
        threshold: dataset-level drift share above which we declare drift.
    """
    # Imported lazily so unit tests can stub Evidently without installing it.
    # Evidently 0.7 moved the public surface under `evidently.legacy.*`; we
    # keep the Report/Preset workflow because the new metric API in 0.7
    # is not backwards-compatible and our HTML report template relies on
    # the legacy `DataDriftPreset` layout.
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.pipeline.column_mapping import ColumnMapping
    from evidently.legacy.report import Report

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = output_dir / f"drift_{timestamp}.html"

    common = [c for c in reference.columns if c in current.columns and c != "Target"]
    column_mapping = ColumnMapping(
        target=None,
        prediction=None,
        numerical_features=[c for c in common if pd.api.types.is_numeric_dtype(reference[c])],
        categorical_features=[c for c in common if not pd.api.types.is_numeric_dtype(reference[c])],
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference[common],
        current_data=current[common],
        column_mapping=column_mapping,
    )
    report.save_html(str(report_path))

    summary = report.as_dict()
    metric = summary["metrics"][0]["result"]
    n_drifted = int(metric.get("number_of_drifted_columns", 0))
    n_total = int(metric.get("number_of_columns", len(common)))
    share = float(metric.get("share_of_drifted_columns", n_drifted / max(n_total, 1)))

    return DriftResult(
        drift_share=share,
        n_drifted=n_drifted,
        n_total=n_total,
        report_path=report_path,
        detected=share >= threshold,
    )


def latest_report(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    reports = sorted(output_dir.glob("drift_*.html"))
    return reports[-1] if reports else None
