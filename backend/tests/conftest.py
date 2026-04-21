"""Shared pytest fixtures.

We avoid hitting the real UCI server in CI by synthesising a small
schema-conformant fixture. This makes the test suite fast, deterministic,
and offline-friendly.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from backend.app.ml.schemas import (
    APPLICATION_MODE_CODES,
    COURSE_CODES,
    NATIONALITY_CODES,
    PARENT_OCCUPATION_CODES,
    PARENT_QUALIFICATION_CODES,
    PREVIOUS_QUALIFICATION_CODES,
    TARGET_CLASSES,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Isolated env for every test — no leaking state between cases."""
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret")
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "admin")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    # Reset the cached settings so the new env is picked up.
    from backend.app.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    """50-row UCI-shaped dataset that satisfies Pandera ranges + class codes."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "Marital status": rng.integers(1, 7, n),
        "Application mode": rng.choice(APPLICATION_MODE_CODES, n),
        "Application order": rng.integers(0, 10, n),
        "Course": rng.choice(COURSE_CODES, n),
        "Daytime/evening attendance": rng.integers(0, 2, n),
        "Previous qualification": rng.choice(PREVIOUS_QUALIFICATION_CODES, n),
        "Previous qualification (grade)": rng.uniform(50, 200, n),
        "Nacionality": rng.choice(NATIONALITY_CODES, n),
        "Mother's qualification": rng.choice(PARENT_QUALIFICATION_CODES, n),
        "Father's qualification": rng.choice(PARENT_QUALIFICATION_CODES, n),
        "Mother's occupation": rng.choice(PARENT_OCCUPATION_CODES, n),
        "Father's occupation": rng.choice(PARENT_OCCUPATION_CODES, n),
        "Admission grade": rng.uniform(80, 200, n),
        "Displaced": rng.integers(0, 2, n),
        "Educational special needs": rng.integers(0, 2, n),
        "Debtor": rng.integers(0, 2, n),
        "Tuition fees up to date": rng.integers(0, 2, n),
        "Gender": rng.integers(0, 2, n),
        "Scholarship holder": rng.integers(0, 2, n),
        "Age at enrollment": rng.integers(17, 60, n),
        "International": rng.integers(0, 2, n),
        "Curricular units 1st sem (credited)": rng.integers(0, 10, n),
        "Curricular units 1st sem (enrolled)": rng.integers(0, 10, n),
        "Curricular units 1st sem (evaluations)": rng.integers(0, 15, n),
        "Curricular units 1st sem (approved)": rng.integers(0, 10, n),
        "Curricular units 1st sem (grade)": rng.uniform(0, 20, n),
        "Curricular units 1st sem (without evaluations)": rng.integers(0, 5, n),
        "Curricular units 2nd sem (credited)": rng.integers(0, 10, n),
        "Curricular units 2nd sem (enrolled)": rng.integers(0, 10, n),
        "Curricular units 2nd sem (evaluations)": rng.integers(0, 15, n),
        "Curricular units 2nd sem (approved)": rng.integers(0, 10, n),
        "Curricular units 2nd sem (grade)": rng.uniform(0, 20, n),
        "Curricular units 2nd sem (without evaluations)": rng.integers(0, 5, n),
        "Unemployment rate": rng.uniform(5, 20, n),
        "Inflation rate": rng.uniform(-2, 5, n),
        "GDP": rng.uniform(-5, 5, n),
        "Target": rng.choice(TARGET_CLASSES, n),
    })
    # Guarantee at least one row per class so stratification works
    df.loc[0, "Target"] = "Dropout"
    df.loc[1, "Target"] = "Enrolled"
    df.loc[2, "Target"] = "Graduate"
    return df


@pytest.fixture
def sample_student_record(synthetic_raw: pd.DataFrame) -> dict[str, object]:
    """A single record matching the API payload shape."""
    return synthetic_raw.iloc[0].drop("Target").to_dict()
