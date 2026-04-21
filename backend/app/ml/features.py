"""Feature engineering pipeline.

The pipeline is a single `sklearn.compose.ColumnTransformer` that:

* one-hot encodes the high-cardinality UCI codes (course, parent occupation,
  application mode) so tree models and the MLP see consistent inputs,
* standardises continuous variables (grades, macro-economic indicators),
* injects a custom `AcademicMomentumTransformer` that engineers
  semester-over-semester momentum features. Curricular performance change
  between the 1st and 2nd semester is the strongest leading indicator of
  dropout in the African context — this is the project's "original insight"
  contribution and is what we ablate against in `ablation.py`.

The fitted pipeline is pickled together with the model so the API applies
identical transforms at inference time. There is no train/serve skew.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Column groups — the names match the UCI codebook exactly.
# ---------------------------------------------------------------------------

CATEGORICAL_COLS: Final = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

BINARY_COLS: Final = [
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

NUMERIC_COLS: Final = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]


class AcademicMomentumTransformer(BaseEstimator, TransformerMixin):
    """Engineer 1st-vs-2nd-semester momentum features.

    Why: a student whose performance collapses between semesters is far more
    at risk than one with consistently low marks (who may already have
    interventions in place). Encoding the *delta* explicitly gives every
    model — even linear ones — direct access to this signal.
    """

    feature_names_: list[str]

    def fit(self, x: pd.DataFrame, _y: pd.Series | None = None) -> "AcademicMomentumTransformer":
        self.feature_names_ = [
            "delta_grade",
            "delta_approved",
            "delta_evaluations",
            "approved_ratio_1st",
            "approved_ratio_2nd",
            "approved_ratio_total",
        ]
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        eps = 1e-6
        delta_grade = x["Curricular units 2nd sem (grade)"] - x["Curricular units 1st sem (grade)"]
        delta_approved = x["Curricular units 2nd sem (approved)"] - x["Curricular units 1st sem (approved)"]
        delta_eval = x["Curricular units 2nd sem (evaluations)"] - x["Curricular units 1st sem (evaluations)"]
        ratio_1 = x["Curricular units 1st sem (approved)"] / (x["Curricular units 1st sem (enrolled)"] + eps)
        ratio_2 = x["Curricular units 2nd sem (approved)"] / (x["Curricular units 2nd sem (enrolled)"] + eps)
        total_approved = x["Curricular units 1st sem (approved)"] + x["Curricular units 2nd sem (approved)"]
        total_enrolled = x["Curricular units 1st sem (enrolled)"] + x["Curricular units 2nd sem (enrolled)"]
        ratio_total = total_approved / (total_enrolled + eps)
        out = np.column_stack([delta_grade, delta_approved, delta_eval, ratio_1, ratio_2, ratio_total])
        return out.astype(np.float64)

    def get_feature_names_out(self, _input_features: list[str] | None = None) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)


def build_preprocessor(*, drop_groups: list[str] | None = None) -> ColumnTransformer:
    """Assemble the preprocessing ColumnTransformer.

    Args:
        drop_groups: Names of feature groups to exclude (used by ablation
            study). Recognised values: ``"academic"``, ``"demographic"``,
            ``"macroeconomic"``, ``"financial_aid"``, ``"momentum"``.
    """
    drop = set(drop_groups or [])

    transformers: list[tuple[str, object, list[str]]] = []

    if "demographic" not in drop:
        demo_cat = [c for c in CATEGORICAL_COLS if c in {
            "Marital status", "Nacionality", "Mother's qualification",
            "Father's qualification", "Mother's occupation", "Father's occupation",
        }]
        transformers.append(
            ("demo_cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), demo_cat),
        )

    if "financial_aid" not in drop:
        fin_cat = [c for c in CATEGORICAL_COLS if c in {
            "Application mode", "Course", "Daytime/evening attendance",
            "Previous qualification",
        }]
        transformers.append(
            ("fin_cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), fin_cat),
        )
        transformers.append(("binary", "passthrough", BINARY_COLS))

    numeric_keep = list(NUMERIC_COLS)
    if "academic" in drop:
        numeric_keep = [c for c in numeric_keep if "Curricular" not in c and "grade" not in c.lower()]
    if "macroeconomic" in drop:
        numeric_keep = [c for c in numeric_keep if c not in {"Unemployment rate", "Inflation rate", "GDP"}]
    if numeric_keep:
        transformers.append(("numeric", StandardScaler(), numeric_keep))

    if "momentum" not in drop and "academic" not in drop:
        transformers.append((
            "momentum",
            Pipeline([("eng", AcademicMomentumTransformer()), ("scale", StandardScaler())]),
            [
                "Curricular units 1st sem (grade)",
                "Curricular units 2nd sem (grade)",
                "Curricular units 1st sem (approved)",
                "Curricular units 2nd sem (approved)",
                "Curricular units 1st sem (enrolled)",
                "Curricular units 2nd sem (enrolled)",
                "Curricular units 1st sem (evaluations)",
                "Curricular units 2nd sem (evaluations)",
            ],
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def fitted_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the post-transform column names (handy for SHAP plots)."""
    return [str(n) for n in preprocessor.get_feature_names_out()]
