"""Statistical-rigor utilities for the evaluation pipeline.

Two facilities are exposed:

* :func:`bootstrap_ci` — non-parametric bootstrap (BCa-lite, percentile method)
  for any classification metric, returning point estimate + 95% CI.
* :func:`mcnemar_test` — paired error test comparing two classifiers'
  predictions on the same holdout (champion vs. baseline).

Both are model-agnostic: callers pass arrays, not estimators, so the same
helpers are reused by training, evaluation, and ad-hoc analysis notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import binom
from sklearn.metrics import f1_score, recall_score


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    point: float
    lower: float
    upper: float
    n_resamples: int
    confidence: float

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "metric": self.metric,
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "n_resamples": self.n_resamples,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class McNemarResult:
    """Mid-p McNemar's test for two classifiers' paired predictions."""

    b: int  # champion correct, challenger wrong
    c: int  # champion wrong, challenger correct
    statistic: float
    p_value: float
    significant_at_05: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "b": self.b,
            "c": self.c,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant_at_05": self.significant_at_05,
        }


_MetricFn = Callable[[np.ndarray, np.ndarray], float]

_BUILTIN_METRICS: dict[str, _MetricFn] = {
    "macro_f1": lambda y, p: float(f1_score(y, p, average="macro", zero_division=0)),
    "weighted_f1": lambda y, p: float(f1_score(y, p, average="weighted", zero_division=0)),
    "dropout_recall": lambda y, p: float(
        recall_score(y, p, labels=["Dropout"], average="macro", zero_division=0)
    ),
}


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str = "macro_f1",
    metric_fn: _MetricFn | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: int | None = 42,
) -> BootstrapResult:
    """Compute a percentile-bootstrap confidence interval for `metric`.

    The estimator is treated as a black box; we resample the (y_true, y_pred)
    pairs with replacement and recompute the metric each time.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    fn = metric_fn or _BUILTIN_METRICS.get(metric)
    if fn is None:
        raise KeyError(f"Unknown metric '{metric}' — provide metric_fn explicitly.")
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    point = fn(y_true, y_pred)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = fn(y_true[idx], y_pred[idx])
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1.0 - alpha))
    return BootstrapResult(
        metric=metric,
        point=float(point),
        lower=lower,
        upper=upper,
        n_resamples=n_resamples,
        confidence=confidence,
    )


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_champion: np.ndarray,
    y_pred_challenger: np.ndarray,
) -> McNemarResult:
    """Exact-binomial mid-p McNemar's test on paired predictions.

    The `b` cell counts cases the champion got right and the challenger got
    wrong; `c` is the reverse. Under the null (both classifiers have equal
    error rates), b ~ Binomial(b + c, 0.5). We use the two-sided exact test
    with mid-p correction so it remains well-behaved for small disagreements.
    """
    if not (y_true.shape == y_pred_champion.shape == y_pred_challenger.shape):
        raise ValueError("All input arrays must share the same shape")

    champ_correct = y_pred_champion == y_true
    chall_correct = y_pred_challenger == y_true
    b = int(np.sum(champ_correct & ~chall_correct))
    c = int(np.sum(~champ_correct & chall_correct))
    n = b + c

    if n == 0:
        return McNemarResult(b=b, c=c, statistic=0.0, p_value=1.0, significant_at_05=False)

    k = min(b, c)
    # Two-sided exact binomial, with mid-p adjustment to reduce conservatism.
    cdf = binom.cdf(k, n, 0.5)
    pmf_k = binom.pmf(k, n, 0.5)
    p_value = min(1.0, 2.0 * cdf - pmf_k)
    statistic = ((abs(b - c) - 1) ** 2) / n if n > 0 else 0.0
    return McNemarResult(
        b=b,
        c=c,
        statistic=float(statistic),
        p_value=float(p_value),
        significant_at_05=bool(p_value < 0.05),
    )
