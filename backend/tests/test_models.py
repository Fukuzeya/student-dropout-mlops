"""Smoke-train every model on the tiny fixture.

If any model fails to fit / predict here, training will explode in CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from backend.app.ml.features import build_preprocessor
from backend.app.ml.models import MODEL_FACTORIES
from backend.app.ml.schemas import TARGET_CLASSES, feature_columns

DEFAULT_HYPERPARAMS: dict[str, dict[str, object]] = {
    "logreg": {"C": 1.0, "max_iter": 200, "class_weight": "balanced"},
    "random_forest": {"n_estimators": 20, "max_depth": 4},
    "xgboost": {"n_estimators": 30, "max_depth": 3, "objective": "multi:softprob"},
    "lightgbm": {"n_estimators": 30, "num_leaves": 7, "objective": "multiclass"},
    "mlp": {"hidden_dims": [16], "max_epochs": 5, "patience": 3, "batch_size": 16},
}


@pytest.mark.parametrize("name", list(MODEL_FACTORIES.keys()))
def test_model_fit_predict(name: str, synthetic_raw: pd.DataFrame) -> None:
    feat_cols = feature_columns()
    y = pd.Categorical(synthetic_raw["Target"], categories=TARGET_CLASSES).codes
    pipe = Pipeline([
        ("features", build_preprocessor()),
        ("model", MODEL_FACTORIES[name](DEFAULT_HYPERPARAMS[name], 0)),
    ])
    pipe.fit(synthetic_raw[feat_cols], y)
    proba = pipe.predict_proba(synthetic_raw[feat_cols])
    assert proba.shape == (len(synthetic_raw), len(TARGET_CLASSES))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    pred = pipe.predict(synthetic_raw[feat_cols])
    assert pred.shape == (len(synthetic_raw),)
    assert set(np.unique(pred)).issubset(set(range(len(TARGET_CLASSES))))
