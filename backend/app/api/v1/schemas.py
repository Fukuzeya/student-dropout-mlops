"""Pydantic request / response models for the v1 API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StudentFeatures(BaseModel):
    """Single-student payload accepted by /predict.

    Field aliases match the UCI dataset column names exactly so a CSV row
    can be turned into a request body with no key mangling.
    """
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    marital_status: int = Field(alias="Marital status")
    application_mode: int = Field(alias="Application mode")
    application_order: int = Field(alias="Application order")
    course: int = Field(alias="Course")
    attendance: int = Field(alias="Daytime/evening attendance")
    previous_qualification: int = Field(alias="Previous qualification")
    previous_qualification_grade: float = Field(alias="Previous qualification (grade)")
    nacionality: int = Field(alias="Nacionality")
    mothers_qualification: int = Field(alias="Mother's qualification")
    fathers_qualification: int = Field(alias="Father's qualification")
    mothers_occupation: int = Field(alias="Mother's occupation")
    fathers_occupation: int = Field(alias="Father's occupation")
    admission_grade: float = Field(alias="Admission grade")
    displaced: int = Field(alias="Displaced")
    educational_special_needs: int = Field(alias="Educational special needs")
    debtor: int = Field(alias="Debtor")
    tuition_fees_up_to_date: int = Field(alias="Tuition fees up to date")
    gender: int = Field(alias="Gender")
    scholarship_holder: int = Field(alias="Scholarship holder")
    age_at_enrollment: int = Field(alias="Age at enrollment")
    international: int = Field(alias="International")
    cu_1_credited: int = Field(alias="Curricular units 1st sem (credited)")
    cu_1_enrolled: int = Field(alias="Curricular units 1st sem (enrolled)")
    cu_1_evaluations: int = Field(alias="Curricular units 1st sem (evaluations)")
    cu_1_approved: int = Field(alias="Curricular units 1st sem (approved)")
    cu_1_grade: float = Field(alias="Curricular units 1st sem (grade)")
    cu_1_no_eval: int = Field(alias="Curricular units 1st sem (without evaluations)")
    cu_2_credited: int = Field(alias="Curricular units 2nd sem (credited)")
    cu_2_enrolled: int = Field(alias="Curricular units 2nd sem (enrolled)")
    cu_2_evaluations: int = Field(alias="Curricular units 2nd sem (evaluations)")
    cu_2_approved: int = Field(alias="Curricular units 2nd sem (approved)")
    cu_2_grade: float = Field(alias="Curricular units 2nd sem (grade)")
    cu_2_no_eval: int = Field(alias="Curricular units 2nd sem (without evaluations)")
    unemployment_rate: float = Field(alias="Unemployment rate")
    inflation_rate: float = Field(alias="Inflation rate")
    gdp: float = Field(alias="GDP")

    def to_record(self) -> dict[str, Any]:
        """Return a dict keyed by UCI column names (as the model expects)."""
        return self.model_dump(by_alias=True)


class ShapContribution(BaseModel):
    """Single SHAP contribution in frontend-friendly shape.

    ``contribution`` is the signed Shapley value (positive = pushes toward
    the predicted class). ``value`` is surfaced alongside the bar in the
    SHAP waterfall and matches ``contribution`` in the current rendering.
    """

    feature: str
    value: float
    contribution: float


class InterventionRecommendation(BaseModel):
    code: str
    title: str
    description: str
    owner: str
    priority: str = Field(description="urgent | high | medium | low")


class PredictionResponse(BaseModel):
    student_id: str | None = None
    predicted_class: str = Field(description="Dropout | Enrolled | Graduate")
    risk_level: str = Field(description="High | Medium | Low")
    probabilities: dict[str, float]
    top_shap_features: list[ShapContribution]
    recommended_interventions: list[InterventionRecommendation]
    model_name: str
    model_version: str
    scored_at: str


class BatchPredictionRow(PredictionResponse):
    row_index: int


class BatchPredictionResponse(BaseModel):
    total_rows: int
    scored_rows: int
    failed_rows: int
    model_version: str
    predictions: list[BatchPredictionRow]
    risk_distribution: dict[str, int]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class HealthResponse(BaseModel):
    """Service health + at-a-glance production state for the dashboard."""

    status: str = Field(description="ok | degraded | down")
    model_loaded: bool
    model_name: str | None = None
    uptime_seconds: int
    model_version: str
    champion_macro_f1: float
    predictions_last_hour: int


class DriftResponse(BaseModel):
    detected: bool
    drift_share: float
    n_drifted: int
    n_total: int
    report_url: str


class DriftReportSummary(BaseModel):
    """Compact view of the most recent drift report."""

    generated_at: str
    drift_score: float
    features_drifted: int
    features_total: int
    target_drift_detected: bool
    report_url: str


class DashboardKpis(BaseModel):
    """KPI tiles rendered on the home dashboard."""

    total_students: int
    high_risk_count: int
    high_risk_pct: float
    predictions_today: int
    active_interventions: int
    model_macro_f1: float
    model_version: str
    last_drift_check: str
    drift_score: float


class ModelRegistryEntry(BaseModel):
    name: str
    version: str
    stage: str = Field(description="Production | Staging | Archived | None")
    macro_f1: float
    dropout_recall: float
    registered_at: str


class StudentRecord(BaseModel):
    """Lightweight student descriptor for the cohort list."""

    model_config = ConfigDict(extra="allow")

    student_id: str
    display_name: str | None = None
    programme: str | None = None
    cohort: str | None = None


class ScoredStudent(BaseModel):
    student: StudentRecord
    prediction: PredictionResponse


class RetrainResponse(BaseModel):
    promoted: bool
    reason: str
    champion_macro_f1: float
    challenger_macro_f1: float
    per_class_deltas: dict[str, float]
    timestamp: str
    trigger: str
    mcnemar_p_value: float | None = None
    mcnemar_significant: bool | None = None
    n_test: int
    registered_model_name: str | None = None
    registered_model_version: str | None = None


class RetrainAuditEntry(BaseModel):
    timestamp: str
    trigger: str
    promoted: bool
    reason: str
    champion_macro_f1: float
    challenger_macro_f1: float
    macro_f1_delta: float
    per_class_deltas: dict[str, float]
    mcnemar_p_value: float | None = None
    mcnemar_b: int | None = None
    mcnemar_c: int | None = None
    mcnemar_significant: bool | None = None
    n_test: int
    registered_model_name: str | None = None
    registered_model_version: str | None = None


class RetrainHistoryResponse(BaseModel):
    n: int
    entries: list[RetrainAuditEntry]


class RetrainRunStart(BaseModel):
    """Response for POST /retrain/start — returns a handle to poll/stream."""

    run_id: str
    status_url: str
    logs_url: str


class RetrainRunStatus(BaseModel):
    """Snapshot of a background retrain job."""

    run_id: str
    trigger: str
    started_at: str
    ended_at: str | None = None
    state: str                         # running | succeeded | failed
    stage: str                         # logreg | random_forest | xgboost | lightgbm | mlp | evaluate | done | failed | queued
    percent: int
    log_count: int
    logs: list[str] = []               # recent log tail, capped at 500
    result: RetrainResponse | None = None
    error: str | None = None


class DriftAutoRetrainResponse(BaseModel):
    """Outcome of the chained drift→retrain endpoint.

    Shape: drift block always populated; retrain block populated only when
    drift cleared the configured threshold (or `force=true`). `skipped`
    explains the no-op case so the UI can show why no retrain happened.
    """

    drift: DriftResponse
    retrain: RetrainResponse | None
    skipped: bool
    skip_reason: str | None = None
    threshold: float


class DriftRunStart(BaseModel):
    """Response for POST /monitoring/drift/start — handle to poll/stream."""

    run_id: str
    status_url: str
    logs_url: str


class DriftRunStatus(BaseModel):
    """Snapshot of a background drift→auto-retrain job.

    Mirrors :class:`RetrainRunStatus` but the `result` block carries the
    drift numbers alongside the retrain outcome so the UI can show the
    drift decision even when no promotion happened.
    """

    run_id: str
    trigger: str
    started_at: str
    ended_at: str | None = None
    state: str
    stage: str
    percent: int
    log_count: int
    logs: list[str] = []
    drift: DriftResponse | None = None
    retrain: RetrainResponse | None = None
    skipped: bool | None = None
    skip_reason: str | None = None
    threshold: float | None = None
    error: str | None = None


class EvaluationSummary(BaseModel):
    """Compact view of `reports/evaluation.json` — what the UI consumes.

    The full report is also returned via `details` so downstream tooling
    (or curious analysts hitting Swagger) can inspect everything without
    needing repo access.
    """

    model_name: str
    n_test: int
    macro_f1: float
    macro_f1_lower: float
    macro_f1_upper: float
    dropout_recall_argmax: float
    dropout_recall_tuned: float
    chosen_threshold: float
    expected_utility_argmax: float
    expected_utility_tuned: float
    fairness_max_gap: float
    fairness_summary_attribute: str
    calibration_ece: float
    calibration_ece_post: float | None = None
    temperature: float | None = None
    figure_urls: dict[str, str] = Field(default_factory=dict)
    details: dict[str, Any]
