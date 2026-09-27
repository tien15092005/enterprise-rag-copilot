from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app import __version__
from app.api.dependencies import get_database, get_request_settings
from app.core.config import Settings
from app.db.session import Database
from app.models.health import DependencyHealth, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Annotated[Settings, Depends(get_request_settings)],
) -> HealthResponse:
    """Process liveness check; does not require external services."""

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=__version__,
        dependencies={"postgres": DependencyHealth(status="not_checked")},
    )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
)
async def readiness(
    settings: Annotated[Settings, Depends(get_request_settings)],
    database: Annotated[Database, Depends(get_database)],
) -> HealthResponse | JSONResponse:
    """Dependency readiness check backed by a real PostgreSQL round trip."""

    try:
        await database.ping()
    except (SQLAlchemyError, OSError) as exc:
        response = HealthResponse(
            status="degraded",
            service=settings.app_name,
            version=__version__,
            dependencies={
                "postgres": DependencyHealth(
                    status="unavailable",
                    detail=type(exc).__name__,
                )
            },
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(mode="json"),
        )

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=__version__,
        dependencies={"postgres": DependencyHealth(status="ready")},
    )
