"""Centralised settings — loaded once via pydantic-settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    cors_allow_origins: str = "http://localhost:4200,http://localhost:8080"

    api_key: str = Field(default="dev-api-key", description="Static API key for /predict*")
    jwt_secret: str = Field(default="dev-jwt-secret", description="HS256 secret for admin JWTs")
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    admin_username: str = "admin"
    admin_password: str = "admin"

    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "student-dropout"
    mlflow_registered_model_name: str = "student-dropout-classifier"

    model_path: Path = Path("models/champion/model.joblib")
    metadata_path: Path = Path("models/champion/metadata.json")
    reference_data_path: Path = Path("data/reference/reference.parquet")
    drift_report_dir: Path = Path("reports/drift")
    evaluation_report_path: Path = Path("reports/evaluation.json")
    figures_dir: Path = Path("reports/figures")
    retrain_history_path: Path = Path("reports/retraining/history.jsonl")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
