"""Cohort listing endpoint — the dashboard's students page fetches this.

The route serves the holdout split as a pre-scored list so the UI can
render a filterable/searchable table without issuing N predict calls. The
heavy work (scoring every row + building SHAP contributions) is memoised
in `COHORT_CACHE` keyed on the holdout parquet's fingerprint, so repeated
requests are cheap.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.api.v1.cohort import COHORT_CACHE
from backend.app.api.v1.model_registry import MODEL_STORE
from backend.app.api.v1.schemas import ScoredStudent
from backend.app.core.security import require_api_key

router = APIRouter(prefix="/students", tags=["students"])


@router.get(
    "/scored",
    response_model=list[ScoredStudent],
    summary="Return the holdout cohort with dropout-risk predictions",
    dependencies=[Depends(require_api_key)],
)
def scored_cohort() -> list[ScoredStudent]:
    if not MODEL_STORE.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded — run the training pipeline first.",
        )
    test_path = Path("data/processed/test.parquet")
    if not test_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Holdout split missing — run `dvc repro preprocess` first.",
        )
    return COHORT_CACHE.get_or_build(test_path)
