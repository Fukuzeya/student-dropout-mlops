"""Shared helper: score the holdout cohort and cache the result.

The Angular dashboard pulls the same cohort-level view from two endpoints
(`/students/scored` for the list, `/monitoring/kpis` for the tiles) so we
compute predictions once per model-version and serve both from the cache.
"""
from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.api.v1.model_registry import MODEL_STORE, predict_one
from backend.app.api.v1.schemas import (
    InterventionRecommendation,
    PredictionResponse,
    ScoredStudent,
    ShapContribution,
    StudentFeatures,
    StudentRecord,
)
from backend.app.interventions.recommender import recommend
from backend.app.ml.schemas import feature_columns

# Map UCI column name → pythonic field name on StudentFeatures. Built once at
# import time so /students/scored can return snake_case keys the frontend can
# access as `student().age_at_enrollment` without TS4111 index-signature errors.
_ALIAS_TO_FIELD: dict[str, str] = {
    (info.alias or name): name for name, info in StudentFeatures.model_fields.items()
}

# The UCI dataset uses Portuguese polytechnic course codes. We remap each
# code to the closest equivalent University of Zimbabwe programme so the
# dashboard reads like a real UZ Early-Warning System rather than an
# academic import. The mapping is analogy-driven, not literal: e.g. "Oral
# Hygiene" → BDS Dental Therapy (UZ College of Health Sciences), because
# UZ's dental programme is the domestic equivalent of that foreign degree.
COURSE_NAMES: dict[int, str] = {
    33: "BSc Agriculture (Crop Science)",
    171: "BA Media Studies",
    8014: "BSc Social Work (Evening)",
    9003: "BSc Agriculture (Agronomy)",
    9070: "BA Graphic Design & Media",
    9085: "BVM Veterinary Nursing",
    9119: "BSc Computer Science",
    9130: "BSc Animal Science",
    9147: "BCom Management",
    9238: "BSc Social Work",
    9254: "BSc Tourism & Hospitality",
    9500: "BSc Nursing Science",
    9556: "BDS Dental Therapy",
    9670: "BCom Marketing",
    9773: "BA Journalism & Media Studies",
    9853: "BEd Primary Education",
    9991: "BCom Management (Evening)",
}


# Small pools of common Zimbabwean given + family names. Kept deliberately
# short — a realistic-looking cohort doesn't need thousands of names, and
# we'd rather not ship an encyclopaedic list. Names are deterministically
# assigned per student-id so the dashboard is stable across refreshes.
_FIRST_NAMES: tuple[str, ...] = (
    "Tendai", "Tatenda", "Tawanda", "Tapiwa", "Tinashe", "Takudzwa", "Tafadzwa",
    "Farai", "Kudzai", "Kundai", "Munyaradzi", "Panashe", "Rumbidzai", "Ropafadzo",
    "Rudo", "Chipo", "Chiedza", "Vimbai", "Nyasha", "Anesu", "Anotida", "Tariro",
    "Ngonidzashe", "Simbarashe", "Tonderai", "Tanaka", "Tadiwa", "Shamiso",
    "Mufaro", "Makanaka", "Nomatter", "Blessing", "Memory", "Precious",
    "Sibusisiwe", "Sibonginkosi", "Nkosilathi", "Nkosana", "Thulani", "Thandiwe",
    "Bongani", "Bukhosi", "Mthokozisi", "Mduduzi", "Lindiwe", "Nobuhle",
    "Nokuthula", "Ndumiso", "Qhubekani", "Sipho", "Sandile", "Sithembile",
)

_SURNAMES: tuple[str, ...] = (
    "Moyo", "Ncube", "Ndlovu", "Sibanda", "Dube", "Mpofu", "Nyathi", "Mhlanga",
    "Mukeredzi", "Mutasa", "Mangwiro", "Chigumba", "Chiwenga", "Chiwetu",
    "Chitiyo", "Chipato", "Chirwa", "Chikono", "Chivasa", "Zhou", "Zvobgo",
    "Mpofu", "Mushava", "Nyamande", "Matavire", "Mahachi", "Makoni", "Marufu",
    "Madzivire", "Mberi", "Madondo", "Mudzingwa", "Mugabe", "Muchena", "Murambadoro",
    "Nhongo", "Nkomo", "Ngoma", "Nyoni", "Shumba", "Sithole", "Tshuma", "Zulu",
    "Zondo", "Tavengwa", "Kanyemba", "Kaseke", "Kazembe", "Mvundura", "Rusere",
)


def _deterministic_name(student_id: str) -> str:
    """Return a stable full name derived from the student-id hash.

    Hashing makes the output idempotent — the same id always yields the same
    name — without any per-process random state. We use blake2b truncated to
    8 bytes because it's stdlib + fast, and the 64-bit digest gives us more
    than enough spread across the ~2600 possible first/last combinations.
    """
    digest = hashlib.blake2b(student_id.encode("utf-8"), digest_size=8).digest()
    first_idx = int.from_bytes(digest[:4], "big") % len(_FIRST_NAMES)
    last_idx = int.from_bytes(digest[4:], "big") % len(_SURNAMES)
    return f"{_FIRST_NAMES[first_idx]} {_SURNAMES[last_idx]}"


class _CohortCache:
    __slots__ = ("_lock", "_key", "_rows")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple[str, float, int] | None = None
        self._rows: list[ScoredStudent] = []

    def get_or_build(self, test_path: Path) -> list[ScoredStudent]:
        stat = test_path.stat()
        loaded = MODEL_STORE.get()
        key = (loaded.model_name, stat.st_mtime, int(stat.st_size))
        with self._lock:
            if self._key == key and self._rows:
                return self._rows
            rows = _score_cohort(loaded, test_path)
            self._key = key
            self._rows = rows
            return rows

    def clear(self) -> None:
        with self._lock:
            self._key = None
            self._rows = []


COHORT_CACHE = _CohortCache()


def _shap(raw: list[dict[str, object]]) -> list[ShapContribution]:
    out: list[ShapContribution] = []
    for f in raw:
        contribution = float(f.get("contribution", f.get("shap_value", 0.0)) or 0.0)
        value = float(f.get("value", contribution) or contribution)
        out.append(ShapContribution(feature=str(f.get("feature", "")), value=value, contribution=contribution))
    return out


def _score_cohort(loaded: Any, test_path: Path) -> list[ScoredStudent]:
    df = pd.read_parquet(test_path)
    feat_cols = feature_columns()
    scored_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = str(loaded.metadata.get("champion_run_id", "") or "")
    model_version = run_id[:12] or loaded.model_name

    rows: list[ScoredStudent] = []
    for idx, raw_row in df[feat_cols].iterrows():
        record = raw_row.to_dict()
        try:
            res = predict_one(loaded, record)
        except Exception:  # noqa: BLE001 — skip unscorable rows to keep list intact
            continue
        interventions = recommend(risk_level=res["risk_level"], top_features=res["top_features"])
        sid = f"UZ-{int(idx):05d}"
        course_code = int(record.get("Course", 0) or 0)
        programme = COURSE_NAMES.get(course_code, f"Course {course_code}")
        cohort = "2024/25"
        display = _deterministic_name(sid)
        student = StudentRecord(
            student_id=sid,
            display_name=display,
            programme=programme,
            cohort=cohort,
            **{_ALIAS_TO_FIELD.get(k, k): record[k] for k in feat_cols},
        )
        prediction = PredictionResponse(
            student_id=sid,
            predicted_class=res["predicted_class"],
            risk_level=res["risk_level"],
            probabilities=res["probabilities"],
            top_shap_features=_shap(res["top_features"]),
            recommended_interventions=[
                InterventionRecommendation(
                    code=i.code,
                    title=i.title,
                    description=i.description,
                    owner=i.owner,
                    priority=i.priority,
                )
                for i in interventions
            ],
            model_name=loaded.model_name,
            model_version=model_version,
            scored_at=scored_at,
        )
        rows.append(ScoredStudent(student=student, prediction=prediction))
    return rows
