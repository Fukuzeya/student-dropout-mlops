"""Maps a prediction (+ its SHAP rationale) to concrete interventions.

The mapping is deliberately tailored to the **University of Zimbabwe**
support catalogue (Bursar's office, Student Counselling Services, Faculty
academic mentors, etc.). This is the project's "African contextualisation"
contribution — it would not be useful to recommend a US student-success
service that doesn't exist in Harare.

The rules are intentionally simple and explainable: each rule fires when
a specific top-SHAP feature exceeds a threshold, and contributes one
intervention. A counsellor reading the dashboard can see exactly *why*
each recommendation appears.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True, frozen=True)
class Intervention:
    code: str
    title: str
    description: str
    owner: str        # which UZ office actions this
    priority: str     # "urgent" | "high" | "medium" | "low"


# Catalogue of available interventions.
CATALOGUE: dict[str, Intervention] = {
    "FIN_AID_REFERRAL": Intervention(
        code="FIN_AID_REFERRAL",
        title="Refer to Bursar's Office for financial-aid review",
        description=(
            "Student shows financial stress signals (outstanding fees, debtor flag, "
            "or non-scholarship status combined with academic decline). Schedule a "
            "Bursar's Office consultation and review eligibility for the Cadetship "
            "Scheme or Student Loan Scheme."
        ),
        owner="Bursar's Office",
        priority="high",
    ),
    "TUTOR_ASSIGNMENT": Intervention(
        code="TUTOR_ASSIGNMENT",
        title="Assign peer tutor for core curricular units",
        description=(
            "Curricular grades or approval ratios are below cohort norms. Pair the "
            "student with a senior peer tutor in the lowest-performing unit through "
            "the Faculty Academic Mentorship Programme."
        ),
        owner="Faculty Academic Office",
        priority="high",
    ),
    "COUNSELLING": Intervention(
        code="COUNSELLING",
        title="Refer to Student Counselling Services",
        description=(
            "Demographic + academic-momentum profile (e.g. recent semester collapse, "
            "displacement, or special educational needs) suggests psychosocial risk. "
            "Book an introductory session at Student Counselling Services."
        ),
        owner="Student Counselling Services",
        priority="urgent",
    ),
    "MOMENTUM_CHECKIN": Intervention(
        code="MOMENTUM_CHECKIN",
        title="Schedule academic-momentum check-in",
        description=(
            "Performance dropped sharply between the 1st and 2nd semester. Faculty "
            "advisor should hold a 30-minute check-in to identify root causes "
            "before mid-term assessments."
        ),
        owner="Faculty Advisor",
        priority="urgent",
    ),
    "ATTENDANCE_REVIEW": Intervention(
        code="ATTENDANCE_REVIEW",
        title="Attendance / engagement review",
        description=(
            "Evening-attendance students with declining engagement should be "
            "contacted by the Class Representative and Faculty Office to confirm "
            "ongoing enrolment intent."
        ),
        owner="Faculty Office",
        priority="medium",
    ),
    "INTERNATIONAL_SUPPORT": Intervention(
        code="INTERNATIONAL_SUPPORT",
        title="International / displaced-student liaison",
        description=(
            "Displaced or international student profile detected. Connect to the "
            "Dean of Students' liaison for housing, visa and welfare guidance."
        ),
        owner="Dean of Students Office",
        priority="high",
    ),
    "RECOGNITION": Intervention(
        code="RECOGNITION",
        title="Continue current support plan",
        description="Profile predicts on-track graduation; include in the recognition shortlist.",
        owner="Faculty Office",
        priority="low",
    ),
}

# Substrings that, when found in a top SHAP feature name, trigger an
# intervention. Order = priority for tie-breaking.
RULES: list[tuple[str, str]] = [
    ("Tuition fees up to date", "FIN_AID_REFERRAL"),
    ("Debtor", "FIN_AID_REFERRAL"),
    ("Scholarship holder", "FIN_AID_REFERRAL"),
    ("delta_grade", "MOMENTUM_CHECKIN"),
    ("delta_approved", "MOMENTUM_CHECKIN"),
    ("approved_ratio", "TUTOR_ASSIGNMENT"),
    ("Curricular units", "TUTOR_ASSIGNMENT"),
    ("Admission grade", "TUTOR_ASSIGNMENT"),
    ("Educational special needs", "COUNSELLING"),
    ("Marital status", "COUNSELLING"),
    ("Daytime/evening attendance", "ATTENDANCE_REVIEW"),
    ("Displaced", "INTERNATIONAL_SUPPORT"),
    ("International", "INTERNATIONAL_SUPPORT"),
    ("Nacionality", "INTERNATIONAL_SUPPORT"),
]


def recommend(
    *, risk_level: str, top_features: Iterable[dict[str, object]],
) -> list[Intervention]:
    """Return interventions for a single prediction.

    A `Graduate` (Low risk) prediction always returns the recognition card.
    Otherwise we walk the top SHAP features and accumulate interventions
    suggested by the rules above (deduplicated, capped at 4).
    """
    if risk_level == "Low":
        return [CATALOGUE["RECOGNITION"]]

    selected: dict[str, Intervention] = {}
    for feat in top_features:
        name = str(feat.get("feature", ""))
        direction = str(feat.get("direction", ""))
        if direction != "increases-risk":
            continue
        for needle, code in RULES:
            if needle in name and code not in selected:
                selected[code] = CATALOGUE[code]
                break
        if len(selected) >= 4:
            break

    if not selected:
        # generic fallback so HIGH/MEDIUM risk always gets at least one card
        selected["MOMENTUM_CHECKIN"] = CATALOGUE["MOMENTUM_CHECKIN"]

    return list(selected.values())
