"""Cost-sensitive evaluation.

A flat macro-F1 hides the fact that, for a UZ early-warning system,
*missing a Dropout* costs far more than a false alarm: an undetected
high-risk student may leave university entirely, whereas a false alarm
costs at most one bursar consultation.

We make this explicit via a 3×3 cost matrix indexed
``cost[true_class][pred_class]``. The defaults reflect the
operationalisation we discuss in the paper; tweak via params.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.app.ml.schemas import TARGET_CLASSES

# Rows are TRUE class; columns are PREDICTED class. Diagonals are 0 (correct).
# Cost units are nominal "intervention cost units" (1 unit ≈ one bursar
# consultation). Calibrate with the institution's own cost study.
DEFAULT_COST_MATRIX: dict[str, dict[str, float]] = {
    # True Dropout — missing the student (predicting Graduate) is the worst case.
    "Dropout":  {"Dropout": 0.0,  "Enrolled": 5.0,  "Graduate": 10.0},
    # True Enrolled — small cost either way; falsely escalating is mild churn.
    "Enrolled": {"Dropout": 1.5,  "Enrolled": 0.0,  "Graduate": 2.5},
    # True Graduate — wrongly flagging is annoying but cheap.
    "Graduate": {"Dropout": 1.0,  "Enrolled": 0.5,  "Graduate": 0.0},
}


@dataclass(frozen=True)
class CostReport:
    classes: list[str]
    cost_matrix: dict[str, dict[str, float]]
    total_cost: float
    cost_per_sample: float
    expected_utility: float
    n: int

    def as_dict(self) -> dict[str, object]:
        return {
            "classes": self.classes,
            "cost_matrix": self.cost_matrix,
            "total_cost": self.total_cost,
            "cost_per_sample": self.cost_per_sample,
            "expected_utility": self.expected_utility,
            "n": self.n,
        }


def _matrix_to_array(
    cost_matrix: dict[str, dict[str, float]], classes: list[str]
) -> np.ndarray:
    arr = np.zeros((len(classes), len(classes)), dtype=np.float64)
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            arr[i, j] = float(cost_matrix.get(true_cls, {}).get(pred_cls, 0.0))
    return arr


def evaluate_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    classes: list[str] | None = None,
    cost_matrix: dict[str, dict[str, float]] | None = None,
) -> CostReport:
    classes = classes or TARGET_CLASSES
    matrix = cost_matrix or DEFAULT_COST_MATRIX
    cost_arr = _matrix_to_array(matrix, classes)
    idx = {c: i for i, c in enumerate(classes)}
    total = 0.0
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        total += cost_arr[idx[str(true_label)], idx[str(pred_label)]]
    n = int(len(y_true))
    avg = total / n if n else 0.0
    # Utility = -avg cost (higher is better; comparable across model variants).
    return CostReport(
        classes=classes,
        cost_matrix=matrix,
        total_cost=float(total),
        cost_per_sample=float(avg),
        expected_utility=float(-avg),
        n=n,
    )


def cost_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str],
    *,
    target_class: str = "Dropout",
    cost_matrix: dict[str, dict[str, float]] | None = None,
    grid: np.ndarray | None = None,
) -> tuple[float, list[dict[str, float]]]:
    """Pick the threshold that minimises expected cost (re-using threshold sweep)."""
    from backend.app.ml.threshold import reclassify_with_threshold  # local import to avoid cycle

    if grid is None:
        grid = np.round(np.arange(0.10, 0.91, 0.05), 2)

    history: list[dict[str, float]] = []
    best_t, best_cost = float(grid[0]), float("inf")
    for t in grid:
        y_pred = reclassify_with_threshold(y_proba, classes, float(t), target_class)
        report = evaluate_cost(y_true, y_pred, classes=classes, cost_matrix=cost_matrix)
        history.append({"threshold": float(t), "cost_per_sample": report.cost_per_sample})
        if report.cost_per_sample < best_cost:
            best_cost = report.cost_per_sample
            best_t = float(t)
    return best_t, history
