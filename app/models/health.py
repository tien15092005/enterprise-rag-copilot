from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class DependencyHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ready", "unavailable", "not_checked"]
    detail: str | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "degraded"]
    service: str
    version: str
    dependencies: dict[str, DependencyHealth]
