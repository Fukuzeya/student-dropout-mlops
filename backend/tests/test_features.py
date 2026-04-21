"""Feature-pipeline tests — checks the preprocessor produces a finite
matrix and that the engineered momentum features have the right shape."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.features import (
    AcademicMomentumTransformer,
    build_preprocessor,
    fitted_feature_names,
)
from backend.app.ml.schemas import feature_columns


def test_momentum_transformer_shape(synthetic_raw: pd.DataFrame) -> None:
    transformer = AcademicMomentumTransformer().fit(synthetic_raw)
    out = transformer.transform(synthetic_raw)
    assert out.shape == (len(synthetic_raw), 6)
    assert np.isfinite(out).all()


def test_preprocessor_produces_dense_matrix(synthetic_raw: pd.DataFrame) -> None:
    pre = build_preprocessor()
    x = pre.fit_transform(synthetic_raw[feature_columns()])
    assert isinstance(x, np.ndarray)
    assert x.shape[0] == len(synthetic_raw)
    assert np.isfinite(x).all()
    assert len(fitted_feature_names(pre)) == x.shape[1]


def test_ablation_drops_macro(synthetic_raw: pd.DataFrame) -> None:
    full = build_preprocessor().fit_transform(synthetic_raw[feature_columns()])
    no_macro = build_preprocessor(drop_groups=["macroeconomic"]).fit_transform(
        synthetic_raw[feature_columns()]
    )
    assert no_macro.shape[1] < full.shape[1]
