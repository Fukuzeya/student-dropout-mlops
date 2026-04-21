"""Random forest baseline."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(params: dict[str, Any], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=seed, **params)
