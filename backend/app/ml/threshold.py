"""Decision-threshold tuning for the Dropout class.

Default sklearn behaviour is `argmax(proba)`, which implicitly uses a
0.5-style threshold. For a costly-miss class like Dropout, the right
operating point usually sits *below* 0.5 so we trade some precision for
the recall the institution actually cares about.

We expose a small, explicit policy:

1. Sweep candidate thresholds in [0.10, 0.90].
2. At each threshold, **re-classify** by overriding the predicted class to
   "Dropout" whenever P(Dropout) ≥ T; otherwise keep the original argmax.
3. Pick the threshold that maximises the weighted objective:
       J(T) = α · F1_macro(T) + β · I(recall_dropout(T) ≥ R*)
   where R* is a target recall (default 0.85) and α, β are weights.
4. Persist the chosen threshold inside the model bundle so the API uses
   it identically to the offline evaluation — no train/serve skew.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ThresholdSweepRow:
    threshold: float
    macro_f1: float
    dropout_precision: float
    dropout_recall: float
    dropout_f1: float
    flagged_pct: float


@dataclass(frozen=True)
class ThresholdDecision:
    chosen_threshold: float
    target_recall: float
    objective: float
    sweep: list[ThresholdSweepRow]
    rationale: str

    def as_dict(self) -> dict[str, object]:
        return {
            "chosen_threshold": self.chosen_threshold,
            "target_recall": self.target_recall,
            "objective": self.objective,
            "rationale": self.rationale,
            "sweep": [row.__dict__ for row in self.sweep],
        }


def reclassify_with_threshold(
    y_proba: np.ndarray,
    classes: list[str],
    threshold: float,
    target_class: str = "Dropout",
) -> np.ndarray:
    """Override argmax to emit `target_class` whenever its prob ≥ threshold.

    For all other rows, returns the original argmax label. This keeps the
    semantics simple: we are *only* tuning sensitivity to the costly-miss
    class, not re-balancing the entire decision boundary.
    """
    if target_class not in classes:
        raise ValueError(f"target_class '{target_class}' not in {classes}")
    target_idx = classes.index(target_class)
    argmax = y_proba.argmax(axis=1)
    flag = y_proba[:, target_idx] >= threshold
    pred_idx = np.where(flag, target_idx, argmax)
    return np.asarray([classes[i] for i in pred_idx])


def sweep_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str],
    *,
    target_class: str = "Dropout",
    grid: np.ndarray | None = None,
) -> list[ThresholdSweepRow]:
    if grid is None:
        grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows: list[ThresholdSweepRow] = []
    for t in grid:
        y_pred = reclassify_with_threshold(y_proba, classes, float(t), target_class)
        rows.append(
            ThresholdSweepRow(
                threshold=float(t),
                macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                dropout_precision=float(
                    precision_score(y_true, y_pred, labels=[target_class],
                                    average="macro", zero_division=0)
                ),
                dropout_recall=float(
                    recall_score(y_true, y_pred, labels=[target_class],
                                 average="macro", zero_division=0)
                ),
                dropout_f1=float(
                    f1_score(y_true, y_pred, labels=[target_class],
                             average="macro", zero_division=0)
                ),
                flagged_pct=float(np.mean(y_pred == target_class)),
            )
        )
    return rows


def choose_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str],
    *,
    target_class: str = "Dropout",
    target_recall: float = 0.85,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> ThresholdDecision:
    """Pick the threshold that maximises α·macroF1 + β·I(recall ≥ target).

    If no threshold meets the recall target, fall back to the threshold
    with the highest Dropout recall (preferring the smallest threshold
    among ties, which gives the most defensive operating point).
    """
    sweep = sweep_thresholds(y_true, y_proba, classes, target_class=target_class)
    best: ThresholdSweepRow | None = None
    best_score = -np.inf
    for row in sweep:
        meets = row.dropout_recall >= target_recall
        score = alpha * row.macro_f1 + (beta if meets else 0.0)
        if score > best_score:
            best_score = score
            best = row
    if best is None:  # pragma: no cover — sweep always produces ≥ 1 row
        raise RuntimeError("Threshold sweep returned no rows")

    if best.dropout_recall < target_recall:
        # Safety net — promote whichever row maximises Dropout recall.
        best = max(sweep, key=lambda r: (r.dropout_recall, -r.threshold))
        rationale = (
            f"No threshold met target_recall={target_recall:.2f}; "
            f"selected the most-sensitive operating point with "
            f"recall={best.dropout_recall:.3f}."
        )
    else:
        rationale = (
            f"Maximises α·macro_F1 + β·I(recall≥{target_recall:.2f}) at "
            f"T={best.threshold:.2f} (macro_F1={best.macro_f1:.3f}, "
            f"recall={best.dropout_recall:.3f})."
        )
    return ThresholdDecision(
        chosen_threshold=best.threshold,
        target_recall=target_recall,
        objective=float(best_score),
        sweep=sweep,
        rationale=rationale,
    )
