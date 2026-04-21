"""Model registry — single import point for the training script."""
from __future__ import annotations

from typing import Any, Callable

from backend.app.ml.models.lightgbm_model import build_lightgbm
from backend.app.ml.models.logreg import build_logreg
from backend.app.ml.models.mlp import build_mlp
from backend.app.ml.models.random_forest import build_random_forest
from backend.app.ml.models.xgboost_model import build_xgboost

ModelFactory = Callable[[dict[str, Any], int], Any]

MODEL_FACTORIES: dict[str, ModelFactory] = {
    "logreg": build_logreg,
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "mlp": build_mlp,
}

__all__ = ["MODEL_FACTORIES", "ModelFactory"]
