from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from pypdf import PdfWriter
from sqlalchemy.exc import IntegrityError

from app.db.models import DocumentVersion
from app.db.session import Database
from app.models.versioning import VersionBusinessStatus, VersionIndexingStatus


async def create_document(client: AsyncClient, stable_key: str = "hr-policy") -> dict[str, Any]:
    response = await client.post(
        "/api/documents",
        json={
            "stable_key": stable_key,
            "title": "HR Policy",
            "description": "Human resources policy",
            "owner": "People Operations",
            "domain": "HR",
        },
    )
    assert response.status_code == 201
    return response.json()


async def upload_markdown_version(
    client: AsyncClient,
    logical_document_id: str,
    *,
    label: str,
    ordinal: int,
    content: bytes,
    activate: bool = True,
) -> Any:
    return await client.post(
        f"/api/documents/{logical_document_id}/versions",
        data={
            "version_label": label,
            "version_ordinal": str(ordinal),
            "revision_number": "1",
            "language": "en",
            "activate": str(activate).lower(),
            "chunk_size": "8",
            "chunk_overlap": "2",
        },
        files={"file": (f"policy-{label}.md", content, "text/markdown")},
    )


async def test_logical_document_creation_listing_and_detail(client: AsyncClient) -> None:
    created = await create_document(client)

    listing = await client.get("/api/documents")
    detail = await client.get(f"/api/documents/{created['logical_document_id']}")

    assert listing.status_code == 200
    assert listing.json()[0]["stable_key"] == "hr-policy"
    assert detail.status_code == 200
    assert detail.json()["versions"] == []


async def test_duplicate_logical_document_key_is_rejected(client: AsyncClient) -> None:
    await create_document(client)

    duplicate = await client.post(
        "/api/documents",
        json={"stable_key": "hr-policy", "title": "Duplicate"},
    )

    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "duplicate_logical_document"


async def test_full_upload_parse_chunk_and_current_version_flow(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    created = await create_document(client)
    content = b"# Handbook\n\n## Remote Work\n\nEmployees may work remotely two days weekly."

    response = await upload_markdown_version(
        client,
        created["logical_document_id"],
        label="2026",
        ordinal=1,
        content=content,
    )

    assert response.status_code == 201, response.text
    version = response.json()["document_version"]
    assert version["indexing_status"] == "ready"
    assert version["business_status"] == "active"
    assert version["is_current"] is True
    assert version["chunk_count"] == 1
    assert version["ingestion_job"]["status"] == "succeeded"
    stored_files = list((tmp_path / "uploads").rglob("*.md"))
    assert len(stored_files) == 1
    assert stored_files[0].read_bytes() == content


async def test_activating_new_version_explicitly_supersedes_old_current(
    client: AsyncClient,
) -> None:
    created = await create_document(client)
    document_id = created["logical_document_id"]
    first = await upload_markdown_version(
        client,
        document_id,
        label="2025",
        ordinal=1,
        content=b"# Policy\n\nOld leave allowance.",
    )
    second = await upload_markdown_version(
        client,
        document_id,
        label="2026",
        ordinal=2,
        content=b"# Policy\n\nNew leave allowance.",
    )

    assert first.status_code == 201
    assert second.status_code == 201
    versions = (await client.get(f"/api/documents/{document_id}/versions")).json()
    assert [version["business_status"] for version in versions] == [
        "superseded",
        "active",
    ]
    assert [version["is_current"] for version in versions] == [False, True]


async def test_duplicate_source_content_is_rejected(client: AsyncClient) -> None:
    created = await create_document(client)
    document_id = created["logical_document_id"]
    content = b"# Policy\n\nIdentical source."
    first = await upload_markdown_version(
        client, document_id, label="v1", ordinal=1, content=content
    )
    duplicate = await upload_markdown_version(
        client, document_id, label="v2", ordinal=2, content=content
    )

    assert first.status_code == 201
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "duplicate_document_version"


async def test_failed_pdf_ingestion_persists_failed_state(client: AsyncClient) -> None:
    created = await create_document(client)
    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buffer)

    response = await client.post(
        f"/api/documents/{created['logical_document_id']}/versions",
        data={
            "version_label": "scan-1",
            "version_ordinal": "1",
            "language": "en",
        },
        files={"file": ("scanned.pdf", buffer.getvalue(), "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_document"
    versions = (
        await client.get(f"/api/documents/{created['logical_document_id']}/versions")
    ).json()
    assert versions[0]["indexing_status"] == "failed"
    assert versions[0]["business_status"] == "draft"


async def test_ready_version_source_fields_are_immutable(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    created = await create_document(client)
    uploaded = await upload_markdown_version(
        client,
        created["logical_document_id"],
        label="2026",
        ordinal=1,
        content=b"# Policy\n\nImmutable content.",
    )
    version_id = UUID(uploaded.json()["document_version"]["document_version_id"])
    database: Database = app.state.database

    async with database.session_factory() as session:
        version = await session.get(DocumentVersion, version_id)
        assert version is not None
        version.source_filename = "silently-replaced.md"
        with pytest.raises(ValueError, match="immutable"):
            await session.commit()
        await session.rollback()


async def test_api_rejects_invalid_document_key(client: AsyncClient) -> None:
    response = await client.post(
        "/api/documents",
        json={"stable_key": "not a stable key", "title": "Invalid"},
    )

    assert response.status_code == 422


async def test_api_returns_structured_error_for_invalid_chunk_configuration(
    client: AsyncClient,
) -> None:
    created = await create_document(client)
    response = await client.post(
        f"/api/documents/{created['logical_document_id']}/versions",
        data={
            "version_label": "bad-chunks",
            "version_ordinal": "1",
            "language": "en",
            "chunk_size": "10",
            "chunk_overlap": "10",
        },
        files={"file": ("policy.md", b"# Policy\n\nText", "text/markdown")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_ingestion_request"


async def test_database_rejects_second_active_version_without_explicit_transition(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    created = await create_document(client)
    document_id = created["logical_document_id"]
    first = await upload_markdown_version(
        client,
        document_id,
        label="v1",
        ordinal=1,
        content=b"# First\n\nFirst source.",
        activate=True,
    )
    second = await upload_markdown_version(
        client,
        document_id,
        label="v2",
        ordinal=2,
        content=b"# Second\n\nSecond source.",
        activate=False,
    )
    assert first.status_code == 201
    assert second.status_code == 201
    second_id = UUID(second.json()["document_version"]["document_version_id"])
    database: Database = app.state.database

    async with database.session_factory() as session:
        second_version = await session.get(DocumentVersion, second_id)
        assert second_version is not None
        assert second_version.indexing_status is VersionIndexingStatus.READY
        second_version.business_status = VersionBusinessStatus.ACTIVE
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()
