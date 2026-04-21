"""MLflow Model Registry helpers.

Encapsulates the champion-vs-challenger promotion rule. The rule is the
key methodological safeguard against silently shipping a worse model.

A challenger replaces production only when **all** of the following hold:

1. ``macro-F1`` improves by at least ``min_macro_f1_gain`` (effect-size gate).
2. No per-class F1 regresses by more than ``max_per_class_regression``
   (no-class-left-behind gate; protects the costly Dropout class).
3. *If* paired predictions are supplied, McNemar's mid-p test rejects
   the null that both models have equal error rates (statistical-
   significance gate, ``mcnemar_alpha`` default 0.05). Without paired
   inputs the McNemar gate is skipped and the decision falls back to
   the effect-size gate alone — useful when comparing against a
   metrics-only champion baseline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from backend.app.ml.schemas import TARGET_CLASSES
from backend.app.ml.statistics import McNemarResult, mcnemar_test

log = logging.getLogger(__name__)

PROD_STAGE = "Production"
STAGING_STAGE = "Staging"


@dataclass(slots=True)
class PromotionDecision:
    promoted: bool
    reason: str
    champion_macro_f1: float
    challenger_macro_f1: float
    per_class_deltas: dict[str, float]
    mcnemar: McNemarResult | None = None


def compare_for_promotion(
    champion_metrics: dict[str, Any],
    challenger_metrics: dict[str, Any],
    *,
    min_macro_f1_gain: float = 0.01,
    max_per_class_regression: float = 0.02,
    paired_predictions: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    mcnemar_alpha: float = 0.05,
) -> PromotionDecision:
    """Decide whether to promote a challenger over the current champion.

    ``paired_predictions`` is an optional ``(y_true, y_champion, y_challenger)``
    tuple. When supplied, we run McNemar's mid-p test on the paired errors
    and require ``p_value < mcnemar_alpha`` for promotion.
    """
    champ_f1 = float(champion_metrics["macro_f1"])
    chal_f1 = float(challenger_metrics["macro_f1"])
    gain = chal_f1 - champ_f1

    per_class_deltas: dict[str, float] = {}
    for cls in TARGET_CLASSES:
        c = float(champion_metrics.get("per_class", {}).get(cls, {}).get("f1", 0.0))
        ch = float(challenger_metrics.get("per_class", {}).get(cls, {}).get("f1", 0.0))
        per_class_deltas[cls] = ch - c

    mcnemar: McNemarResult | None = None
    if paired_predictions is not None:
        y_true, y_champ, y_chal = paired_predictions
        mcnemar = mcnemar_test(y_true, y_champ, y_chal)

    if gain < min_macro_f1_gain:
        return PromotionDecision(False, f"macro-F1 gain {gain:+.4f} < {min_macro_f1_gain}",
                                 champ_f1, chal_f1, per_class_deltas, mcnemar)

    worst_class, worst_delta = min(per_class_deltas.items(), key=lambda kv: kv[1])
    if -worst_delta > max_per_class_regression:
        return PromotionDecision(
            False,
            f"{worst_class} F1 regressed {worst_delta:+.4f} (>{max_per_class_regression})",
            champ_f1, chal_f1, per_class_deltas, mcnemar,
        )

    if mcnemar is not None and mcnemar.p_value >= mcnemar_alpha:
        return PromotionDecision(
            False,
            f"McNemar p={mcnemar.p_value:.4f} ≥ α={mcnemar_alpha} — gain not significant",
            champ_f1, chal_f1, per_class_deltas, mcnemar,
        )

    suffix = (
        f" (McNemar p={mcnemar.p_value:.4f})" if mcnemar is not None else ""
    )
    return PromotionDecision(True, "challenger meets all promotion criteria" + suffix,
                             champ_f1, chal_f1, per_class_deltas, mcnemar)


def get_production_metrics(client: MlflowClient, model_name: str) -> dict[str, Any] | None:
    """Return the metrics dict logged on the current production run, or None."""
    try:
        prod_versions = client.get_latest_versions(model_name, stages=[PROD_STAGE])
    except mlflow.exceptions.MlflowException:
        return None
    if not prod_versions:
        return None
    run = client.get_run(prod_versions[0].run_id)
    metrics = run.data.metrics
    return {
        "macro_f1": metrics.get("macro_f1", 0.0),
        "per_class": {
            cls: {"f1": metrics.get(f"{cls}_f1", 0.0)} for cls in TARGET_CLASSES
        },
    }


def promote(client: MlflowClient, model_name: str, version: str) -> None:
    """Move `version` to Production and archive the previous one."""
    log.info("Promoting %s v%s to %s", model_name, version, PROD_STAGE)
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=PROD_STAGE,
        archive_existing_versions=True,
    )


def register_and_promote(
    *,
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
    description: str | None = None,
    tags: dict[str, str] | None = None,
) -> str | None:
    """Register the artifact at ``runs:/{run_id}/{artifact_path}`` and promote it.

    Returns the new version number on success or ``None`` if the call fails.
    We swallow registry failures rather than failing the retrain — the local
    joblib swap has already happened and the UI still needs a response.
    """
    try:
        client = MlflowClient()
        # Create the registered model on first promotion so the UI
        # landing page on MLflow shows it under "Models".
        try:
            client.create_registered_model(model_name, description=description)
        except mlflow.exceptions.RestException:
            # Already exists — fine.
            pass

        source = f"runs:/{run_id}/{artifact_path}"
        version = client.create_model_version(
            name=model_name,
            source=source,
            run_id=run_id,
            description=description,
            tags=tags or {},
        )
        promote(client, model_name, version.version)
        return str(version.version)
    except Exception as exc:  # noqa: BLE001 — registry errors must not break retrain
        log.warning("MLflow registry promotion failed: %s", exc)
        return None
