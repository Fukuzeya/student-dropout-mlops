"""Pandera contract tests — the data layer's tripwires."""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest

from backend.app.ml.schemas import (
    PredictionFeaturesSchema,
    feature_columns,
    validate_raw,
)


def test_validates_clean_synthetic(synthetic_raw: pd.DataFrame) -> None:
    validated = validate_raw(synthetic_raw)
    assert len(validated) == len(synthetic_raw)
    assert "Target" in validated.columns


def test_rejects_invalid_target(synthetic_raw: pd.DataFrame) -> None:
    bad = synthetic_raw.copy()
    bad.loc[0, "Target"] = "WithdrawnByUniversity"
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(bad)


def test_rejects_out_of_range_grade(synthetic_raw: pd.DataFrame) -> None:
    bad = synthetic_raw.copy()
    bad.loc[0, "Curricular units 1st sem (grade)"] = 99.0
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(bad)


def test_inference_schema_drops_target(synthetic_raw: pd.DataFrame) -> None:
    feats = synthetic_raw[feature_columns()]
    validated = PredictionFeaturesSchema.validate(feats, lazy=True)
    assert "Target" not in validated.columns


def test_feature_columns_count() -> None:
    cols = feature_columns()
    # 35 features (36 columns – 1 target)
    assert len(cols) == 35
