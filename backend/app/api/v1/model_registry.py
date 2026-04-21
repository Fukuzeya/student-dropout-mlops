"""Process-local cache of the loaded champion bundle.

Loaded on FastAPI startup and refreshed when the retrain endpoint promotes
a new model. Keeping this in one place means request handlers never touch
disk or worry about thread-safety on swap.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.explain import ShapExplainer
from backend.app.ml.schemas import RISK_LEVELS, TARGET_CLASSES


@dataclass(slots=True)
class LoadedModel:
    pipeline: Any
    model_name: str
    classes: list[str]
    feature_columns: list[str]
    metadata: dict[str, Any]
    explainer: ShapExplainer | None


class ModelStore:
    """Thread-safe holder for the currently-active model."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded: LoadedModel | None = None

    def load(self, model_path: Path, metadata_path: Path, reference_path: Path | None = None) -> LoadedModel:
        bundle = joblib.load(model_path)
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())

        explainer: ShapExplainer | None = None
        if reference_path is not None and reference_path.exists():
            try:
                ref = pd.read_parquet(reference_path)
                explainer = ShapExplainer(
                    bundle["pipeline"],
                    ref[bundle["feature_columns"]],
                    classes=bundle["classes"],
                )
            except Exception:  # noqa: BLE001 — fail open; SHAP is enrichment
                explainer = None

        loaded = LoadedModel(
            pipeline=bundle["pipeline"],
            model_name=bundle.get("model_name", "unknown"),
            classes=bundle["classes"],
            feature_columns=bundle["feature_columns"],
            metadata=metadata,
            explainer=explainer,
        )
        with self._lock:
            self._loaded = loaded
        return loaded

    def is_loaded(self) -> bool:
        with self._lock:
            return self._loaded is not None

    def get(self) -> LoadedModel:
        with self._lock:
            if self._loaded is None:
                raise RuntimeError("Model not loaded yet")
            return self._loaded


def predict_one(loaded: LoadedModel, record: dict[str, Any]) -> dict[str, Any]:
    df = pd.DataFrame([record])[loaded.feature_columns]
    proba = loaded.pipeline.predict_proba(df)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = loaded.classes[pred_idx]
    risk_level = _risk_from_class(pred_class)

    top_features: list[dict[str, object]] = []
    if loaded.explainer is not None:
        try:
            top_features = loaded.explainer.top_features(df, predicted_class_idx=pred_idx)
        except Exception:  # noqa: BLE001
            top_features = []

    return {
        "predicted_class": pred_class,
        "probabilities": {cls: float(p) for cls, p in zip(loaded.classes, proba)},
        "risk_level": risk_level,
        "top_features": top_features,
    }


def _risk_from_class(predicted_class: str) -> str:
    """Map class to UI-friendly risk band.

    Dropout → HIGH, Enrolled → MEDIUM (still on track but not graduated),
    Graduate → LOW. Mirrors the colour scheme of the Angular dashboard.
    """
    if predicted_class == "Dropout":
        return RISK_LEVELS[2]  # HIGH
    if predicted_class == "Enrolled":
        return RISK_LEVELS[1]  # MEDIUM
    if predicted_class == "Graduate":
        return RISK_LEVELS[0]  # LOW
    return RISK_LEVELS[1]


def assert_known_classes(loaded: LoadedModel) -> None:
    if not set(loaded.classes).issubset(set(TARGET_CLASSES)):
        raise RuntimeError(
            f"Loaded model classes {loaded.classes} do not match canonical {TARGET_CLASSES}"
        )


# Singleton instance used by FastAPI dependencies
MODEL_STORE = ModelStore()
