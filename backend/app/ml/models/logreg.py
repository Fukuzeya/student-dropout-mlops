"""Logistic regression baseline."""
from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression


def build_logreg(params: dict[str, Any], seed: int) -> LogisticRegression:
    # `n_jobs` was deprecated in sklearn 1.8 for LogisticRegression — it had
    # no effect once the multi-threaded lbfgs solver landed. Omit it to keep
    # the training output free of FutureWarnings.
    return LogisticRegression(random_state=seed, **params)
