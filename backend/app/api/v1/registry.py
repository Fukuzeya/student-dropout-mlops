"""Model registry listing endpoint.

The Angular admin page shows a leaderboard of every candidate model the
training pipeline has evaluated, so reviewers can see *why* a particular
champion was chosen. We prefer MLflow's Registry when it's reachable and
fall back to the training run's leaderboard in ``metadata.json`` when the
MLflow server is offline — the demo runs locally and the fallback keeps
the UI populated without requiring MLflow to be up.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from backend.app.api.v1.schemas import ModelRegistryEntry
from backend.app.core.config import Settings, get_settings

log = logging.getLogger(__name__)

router = APIRouter(prefix="/registry", tags=["registry"])


def _from_mlflow(settings: Settings) -> list[ModelRegistryEntry] | None:
    """Try to pull versions from the MLflow Registry. Returns None on failure."""
    try:
        import mlflow  # noqa: F401 — import-only side effect
        from mlflow.tracking import MlflowClient
    except Exception:  # noqa: BLE001
        return None
    try:
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        versions = client.search_model_versions(f"name='{settings.mlflow_registered_model_name}'")
    except Exception as exc:  # noqa: BLE001
        log.info("mlflow_registry_unavailable", extra={"error": str(exc)})
        return None

    entries: list[ModelRegistryEntry] = []
    for version in versions:
        try:
            run = client.get_run(version.run_id)
            metrics = run.data.metrics
        except Exception:  # noqa: BLE001
            metrics = {}
        entries.append(
            ModelRegistryEntry(
                name=version.name,
                version=str(version.version),
                stage=str(version.current_stage or "None"),
                macro_f1=float(metrics.get("macro_f1", 0.0) or 0.0),
                dropout_recall=float(metrics.get("dropout_recall", 0.0) or 0.0),
                registered_at=datetime.fromtimestamp(
                    version.creation_timestamp / 1000, tz=timezone.utc,
                ).isoformat(),
            )
        )
    entries.sort(key=lambda e: e.macro_f1, reverse=True)
    return entries


def _from_metadata(settings: Settings) -> list[ModelRegistryEntry]:
    """Fallback — derive a leaderboard view from the champion's metadata.json."""
    if not settings.metadata_path.exists():
        return []
    payload: dict[str, Any] = json.loads(settings.metadata_path.read_text())
    leaderboard = payload.get("leaderboard") or []
    champion_name = str(payload.get("champion_model", "")).lower()
    champion_run_id = str(payload.get("champion_run_id", "") or "")
    registered_at = datetime.fromtimestamp(
        settings.metadata_path.stat().st_mtime, tz=timezone.utc,
    ).isoformat()

    entries: list[ModelRegistryEntry] = []
    for row in leaderboard:
        name = str(row.get("name", "unknown"))
        is_champion = name.lower() == champion_name
        version = champion_run_id[:12] if is_champion and champion_run_id else name
        entries.append(
            ModelRegistryEntry(
                name=name,
                version=version,
                stage="Production" if is_champion else "Archived",
                macro_f1=float(row.get("macro_f1", 0.0) or 0.0),
                dropout_recall=float(row.get("dropout_recall", 0.0) or 0.0),
                registered_at=registered_at,
            )
        )
    return entries


@router.get(
    "/models",
    response_model=list[ModelRegistryEntry],
    summary="Registered model versions with their stage + headline metrics",
)
def list_models(
    settings: Annotated[Settings, Depends(get_settings)],
) -> list[ModelRegistryEntry]:
    # MLflow typically only registers the champion. For presentation we want
    # the full bake-off visible too, so we overlay leaderboard candidates
    # from metadata.json as Archived entries and let any MLflow-registered
    # stage (Production/Staging) take precedence on name collision.
    mlflow_entries = _from_mlflow(settings) or []
    metadata_entries = _from_metadata(settings)

    seen: dict[str, ModelRegistryEntry] = {}
    for entry in mlflow_entries:
        seen[entry.name.lower()] = entry
    for entry in metadata_entries:
        key = entry.name.lower()
        if key not in seen:
            seen[key] = entry

    merged = list(seen.values())
    merged.sort(key=lambda e: e.macro_f1, reverse=True)
    return merged
