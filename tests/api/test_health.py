from fastapi import FastAPI
from httpx import AsyncClient

from app.api.dependencies import get_database


class HealthyDatabase:
    async def ping(self) -> None:
        return None


class UnavailableDatabase:
    async def ping(self) -> None:
        raise OSError("database unavailable")


async def test_health_reports_liveness_without_contacting_dependencies(
    client: AsyncClient,
) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Enterprise Knowledge Copilot",
        "version": "0.1.0",
        "dependencies": {"postgres": {"status": "not_checked", "detail": None}},
    }
    assert response.headers["X-Request-ID"]


async def test_health_preserves_supplied_request_id(client: AsyncClient) -> None:
    response = await client.get("/health", headers={"X-Request-ID": "test-request-id"})

    assert response.headers["X-Request-ID"] == "test-request-id"


async def test_openapi_schema_includes_health_routes(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health" in paths
    assert "/health/ready" in paths


async def test_readiness_reports_ready_postgres(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    app.dependency_overrides[get_database] = HealthyDatabase

    response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["dependencies"]["postgres"]["status"] == "ready"
    app.dependency_overrides.clear()


async def test_readiness_reports_unavailable_postgres(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    app.dependency_overrides[get_database] = UnavailableDatabase

    response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["dependencies"]["postgres"] == {
        "status": "unavailable",
        "detail": "OSError",
    }
    app.dependency_overrides.clear()
