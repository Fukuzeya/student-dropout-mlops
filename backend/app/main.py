"""FastAPI app factory + entry point.

The factory pattern keeps the unit tests fast (we instantiate without
booting the model) and lets us mount different middlewares per environment.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from backend.app import __version__
from backend.app.api.v1 import auth, monitoring, predict, registry, retrain, students
from backend.app.api.v1.model_registry import MODEL_STORE
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging, get_logger
from backend.app.core.metrics import REGISTRY

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger(__name__)
    if settings.model_path.exists():
        try:
            loaded = MODEL_STORE.load(
                model_path=settings.model_path,
                metadata_path=settings.metadata_path,
                reference_path=settings.reference_data_path,
            )
            log.info("model_loaded", model_name=loaded.model_name)
        except Exception as exc:  # noqa: BLE001
            log.warning("model_load_failed", error=str(exc))
    else:
        log.warning("model_missing", path=str(settings.model_path))
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Student Dropout MLOps API",
        description=(
            "Predicts student dropout risk for African higher-education "
            "institutions. Provides per-prediction SHAP explanations, "
            "drift monitoring, and a champion-vs-challenger retraining loop."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router, prefix=API_PREFIX)
    app.include_router(predict.router, prefix=API_PREFIX)
    app.include_router(monitoring.router, prefix=API_PREFIX)
    app.include_router(retrain.router, prefix=API_PREFIX)
    app.include_router(students.router, prefix=API_PREFIX)
    app.include_router(registry.router, prefix=API_PREFIX)

    @app.get("/", tags=["meta"], summary="Service banner")
    def root() -> dict[str, str]:
        return {
            "service": "student-dropout-mlops",
            "version": __version__,
            "docs": "/docs",
        }

    @app.get("/metrics", tags=["meta"], summary="Prometheus scrape endpoint")
    def metrics() -> Response:
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
