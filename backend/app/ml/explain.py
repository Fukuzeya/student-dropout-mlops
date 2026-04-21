"""SHAP wrapper used by the API and by the training pipeline.

Strategy:
* For tree models (XGBoost, LightGBM, RandomForest) we use the fast
  TreeExplainer which gives exact Shapley values.
* For LogisticRegression we use LinearExplainer.
* For the PyTorch MLP we fall back to KernelExplainer with a small
  background sample (slower but model-agnostic).

The class returns the **raw** Shapley contributions plus the post-transform
feature names so the API can map them back to user-friendly labels.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.app.ml.features import fitted_feature_names

log = logging.getLogger(__name__)


class ShapExplainer:
    """Lazy SHAP explainer bound to a fitted sklearn Pipeline."""

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame, *, classes: list[str]):
        self._pipeline = pipeline
        self._classes = classes
        self._preprocessor: ColumnTransformer = pipeline.named_steps["features"]
        self._model = pipeline.named_steps["model"]
        self._feature_names = fitted_feature_names(self._preprocessor)

        bg_transformed = self._preprocessor.transform(background.head(100))
        self._explainer = self._build_explainer(bg_transformed)

    def _build_explainer(self, background: np.ndarray) -> Any:
        model = self._model
        # Tree models — exact + fast
        if model.__class__.__name__ in {"XGBClassifier", "LGBMClassifier", "RandomForestClassifier"}:
            return shap.TreeExplainer(model)
        if isinstance(model, LogisticRegression):
            return shap.LinearExplainer(model, background)
        # MLP / anything else — model-agnostic
        log.info("Falling back to KernelExplainer for model=%s", type(model).__name__)
        return shap.KernelExplainer(model.predict_proba, shap.sample(background, 50))

    def shap_values(self, x: pd.DataFrame) -> np.ndarray:
        x_t = self._preprocessor.transform(x)
        values = self._explainer.shap_values(x_t)
        # Newer SHAP returns an Explanation; older returns list-per-class.
        if isinstance(values, list):
            return np.stack(values, axis=-1)  # shape (n, features, classes)
        if values.ndim == 2:
            return values[..., None]
        return values

    def top_features(
        self, row: pd.DataFrame, *, predicted_class_idx: int, top_n: int = 8,
    ) -> list[dict[str, float | str]]:
        """Return the top contributing features for a single prediction.

        Shape matches the frontend ShapContribution contract: ``feature`` is
        the fitted post-transform feature label, ``contribution`` is the
        signed Shapley value (positive pushes toward the predicted class),
        and ``value`` is the numeric contribution magnitude surfaced next
        to the bar. ``direction`` is retained for the interventions rules.
        """
        values = self.shap_values(row)[0]  # (features, classes)
        contributions = values[:, predicted_class_idx]
        order = np.argsort(np.abs(contributions))[::-1][:top_n]
        return [
            {
                "feature": self._feature_names[i],
                "value": float(contributions[i]),
                "contribution": float(contributions[i]),
                "direction": "increases-risk" if contributions[i] > 0 else "decreases-risk",
            }
            for i in order
        ]

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)
