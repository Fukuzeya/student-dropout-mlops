"""Probability calibration diagnostics + recalibration helpers.

We compute three things examiners look for in production-grade ML:

* **Brier score** — proper scoring rule, lower is better; computed per-class
  and macro-averaged.
* **Expected Calibration Error (ECE)** with adaptive (equal-mass) bins —
  the standard "are the predicted probabilities trustworthy" metric.
* **Reliability diagrams** — empirical-vs-predicted probability per bin;
  one curve per class. Saved as a PNG so MLflow can attach it to the run.

Optional :func:`fit_temperature_scaler` provides a single-parameter
post-hoc recalibration. We deliberately keep it light (no Isotonic /
Platt for the multi-class case) because temperature scaling has the best
empirical track record on tree-ensembles and adds minimal moving parts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless container friendly
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402


@dataclass(frozen=True)
class CalibrationReport:
    classes: list[str]
    brier_per_class: dict[str, float]
    brier_macro: float
    ece_per_class: dict[str, float]
    ece_macro: float
    n_bins: int
    n_samples: int

    def as_dict(self) -> dict[str, object]:
        return {
            "classes": self.classes,
            "brier_per_class": self.brier_per_class,
            "brier_macro": self.brier_macro,
            "ece_per_class": self.ece_per_class,
            "ece_macro": self.ece_macro,
            "n_bins": self.n_bins,
            "n_samples": self.n_samples,
        }


def _one_hot(y_true: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    classes = list(classes)
    out = np.zeros((len(y_true), len(classes)), dtype=np.float64)
    idx_lookup = {c: i for i, c in enumerate(classes)}
    for row, label in enumerate(y_true):
        out[row, idx_lookup[label]] = 1.0
    return out


def brier_per_class(y_true_oh: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
    return ((y_proba - y_true_oh) ** 2).mean(axis=0)


def expected_calibration_error(
    y_true_oh_col: np.ndarray, y_proba_col: np.ndarray, n_bins: int = 15
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive (equal-mass) ECE for one-vs-rest probabilities of a single class."""
    if y_true_oh_col.shape != y_proba_col.shape:
        raise ValueError("Shapes must match")
    n = len(y_proba_col)
    if n == 0:
        return 0.0, np.array([]), np.array([]), np.array([])
    order = np.argsort(y_proba_col)
    sorted_p = y_proba_col[order]
    sorted_y = y_true_oh_col[order]
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    bin_conf, bin_acc, bin_count = [], [], []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        if hi <= lo:
            continue
        slice_p = sorted_p[lo:hi]
        slice_y = sorted_y[lo:hi]
        confidence = float(slice_p.mean())
        accuracy = float(slice_y.mean())
        weight = (hi - lo) / n
        ece += weight * abs(confidence - accuracy)
        bin_conf.append(confidence)
        bin_acc.append(accuracy)
        bin_count.append(int(hi - lo))
    return ece, np.array(bin_conf), np.array(bin_acc), np.array(bin_count)


def compute_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: Sequence[str],
    n_bins: int = 15,
) -> CalibrationReport:
    if y_proba.ndim != 2 or y_proba.shape[1] != len(classes):
        raise ValueError("y_proba must be (n_samples, n_classes) and match classes length")
    y_oh = _one_hot(y_true, classes)
    brier = brier_per_class(y_oh, y_proba)
    eces: list[float] = []
    ece_per_class: dict[str, float] = {}
    for idx, cls in enumerate(classes):
        ece, *_ = expected_calibration_error(y_oh[:, idx], y_proba[:, idx], n_bins=n_bins)
        ece_per_class[cls] = float(ece)
        eces.append(ece)
    return CalibrationReport(
        classes=list(classes),
        brier_per_class={c: float(brier[i]) for i, c in enumerate(classes)},
        brier_macro=float(brier.mean()),
        ece_per_class=ece_per_class,
        ece_macro=float(np.mean(eces)),
        n_bins=n_bins,
        n_samples=int(len(y_true)),
    )


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: Sequence[str],
    out_path: Path,
    n_bins: int = 15,
    title: str = "Reliability diagram",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_oh = _one_hot(y_true, classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1, label="Perfect")
    palette = ["#f43f5e", "#f59e0b", "#10b981", "#4f46e5"]
    for idx, cls in enumerate(classes):
        ece, conf, acc, count = expected_calibration_error(
            y_oh[:, idx], y_proba[:, idx], n_bins=n_bins
        )
        if len(conf) == 0:
            continue
        ax.plot(
            conf,
            acc,
            marker="o",
            linewidth=1.5,
            color=palette[idx % len(palette)],
            label=f"{cls} (ECE={ece:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def fit_temperature_scaler(
    y_true: np.ndarray, y_proba: np.ndarray, classes: Sequence[str]
) -> float:
    """Fit a single scalar T > 0 by minimising NLL on (logit(p), one-hot(y))."""
    y_oh = _one_hot(y_true, classes)
    eps = 1e-9
    # Recover pseudo-logits by log; if we lose calibration here it's irrelevant
    # since temperature scaling is monotone in the logits.
    logits = np.log(np.clip(y_proba, eps, 1.0 - eps))

    def neg_log_likelihood(t_arr: np.ndarray) -> float:
        t = float(t_arr[0])
        if t <= 0:
            return 1e9
        scaled = logits / t
        scaled -= scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return float(-np.sum(y_oh * np.log(np.clip(probs, eps, 1.0))) / len(y_true))

    res = minimize(
        neg_log_likelihood,
        x0=np.array([1.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 200},
    )
    return float(res.x[0]) if res.success else 1.0


def apply_temperature(y_proba: np.ndarray, temperature: float) -> np.ndarray:
    eps = 1e-9
    if temperature <= 0:
        return y_proba
    logits = np.log(np.clip(y_proba, eps, 1.0 - eps)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)
