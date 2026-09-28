from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.error_handlers import register_error_handlers


async def test_unexpected_error_response_does_not_expose_internal_details() -> None:
    application = FastAPI()
    register_error_handlers(application)

    @application.get("/explode")
    async def explode() -> None:
        raise RuntimeError("secret internal detail")

    transport = ASGITransport(app=application, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/explode")

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "An internal error occurred.",
            "details": {},
            "request_id": None,
        }
    }
    assert "secret internal detail" not in response.text
