"""Common protocol every model wrapper conforms to."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SupportsPredictProba(Protocol):
    """Minimal interface the trainer needs from every estimator."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SupportsPredictProba": ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...
