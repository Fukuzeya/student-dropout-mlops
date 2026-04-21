"""Build a drift-demo CSV that reliably trips the 30% drift threshold.

Starts from the existing realistic-looking 100-student sample and mutates
the strong academic, financial, demographic, and macro-economic signals so
the distribution diverges sharply from the training reference. Every
mutation stays inside the valid Pandera ranges so the API accepts the
payload — we're demonstrating drift, not invalid input.

Run with::

    docker compose exec api python /app/scripts/build_drift_sample.py

The output lands at ``data/samples/drifted_100_students.csv`` and is
validated against ``PredictionFeaturesSchema`` before being written.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.schemas import PredictionFeaturesSchema


SOURCE = Path("data/samples/sample_100_students.csv")
TARGET = Path("data/samples/drifted_100_students.csv")


def build_drifted_frame(src: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = len(src)
    df = src.copy()

    # --- Academic momentum: simulate an at-risk cohort --------------------
    df["Curricular units 1st sem (grade)"] = rng.uniform(0.0, 7.5, n).round(2)
    df["Curricular units 2nd sem (grade)"] = rng.uniform(0.0, 6.0, n).round(2)
    df["Curricular units 1st sem (approved)"] = rng.integers(0, 3, n)
    df["Curricular units 2nd sem (approved)"] = rng.integers(0, 2, n)
    df["Curricular units 1st sem (without evaluations)"] = rng.integers(2, 6, n)
    df["Curricular units 2nd sem (without evaluations)"] = rng.integers(3, 7, n)
    df["Curricular units 1st sem (evaluations)"] = rng.integers(0, 3, n)
    df["Curricular units 2nd sem (evaluations)"] = rng.integers(0, 2, n)

    # --- Financial distress signal ----------------------------------------
    df["Tuition fees up to date"] = 0                  # everyone behind
    df["Debtor"] = 1                                    # everyone in arrears
    df["Scholarship holder"] = 0                        # no scholarships

    # --- Demographic shift -----------------------------------------------
    df["Age at enrollment"] = rng.integers(28, 46, n)   # mature students
    df["Displaced"] = 1

    # --- Admission / prior grades way lower -------------------------------
    df["Admission grade"] = rng.uniform(60.0, 95.0, n).round(1)
    df["Previous qualification (grade)"] = rng.uniform(50.0, 90.0, n).round(1)

    # --- Macro-economic shock --------------------------------------------
    df["Unemployment rate"] = round(float(rng.uniform(35.0, 45.0)), 1)
    df["Inflation rate"] = round(float(rng.uniform(20.0, 30.0)), 1)
    df["GDP"] = round(float(rng.uniform(-8.0, -3.0)), 2)

    return df


def main() -> None:
    src = pd.read_csv(SOURCE)
    drifted = build_drifted_frame(src)

    # The API coerces int-like ints, but Pandera with coerce=True handles
    # stray floats too. Validate before writing so bad mutations fail loudly.
    PredictionFeaturesSchema.validate(drifted, lazy=True)

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    drifted.to_csv(TARGET, index=False)
    print(f"Wrote {TARGET} ({len(drifted)} rows, {len(drifted.columns)} cols)")

    # Quick sanity print so the reviewer can see how far we moved.
    numeric = drifted.select_dtypes(include="number")
    print("\nFeature shift summary (mean delta vs source):")
    delta = (numeric.mean() - src[numeric.columns].mean()).round(2)
    for col in delta.index:
        src_mean = float(src[col].mean())
        new_mean = float(numeric[col].mean())
        print(f"  {col:<55} {src_mean:>8.2f}  →  {new_mean:>8.2f}")


if __name__ == "__main__":
    main()
