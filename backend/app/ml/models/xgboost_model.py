"""XGBoost — the proposed champion. Compared against four baselines."""
from __future__ import annotations

from typing import Any

from xgboost import XGBClassifier


def build_xgboost(params: dict[str, Any], seed: int) -> XGBClassifier:
    # `num_class` and label encoding are handled by the trainer.
    return XGBClassifier(
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
        **params,
    )
