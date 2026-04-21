"""LightGBM — second strong gradient-boosted baseline."""
from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier


def build_lightgbm(params: dict[str, Any], seed: int) -> LGBMClassifier:
    return LGBMClassifier(random_state=seed, n_jobs=-1, verbosity=-1, **params)
