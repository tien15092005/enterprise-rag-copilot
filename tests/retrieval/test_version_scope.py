from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.db.models import DocumentVersion, LogicalDocument
from app.db.session import Database
from app.models.query import QueryVersionSelector
from app.models.versioning import (
    DocumentLanguage,
    VectorIndexStatus,
    VersionBusinessStatus,
    VersionIndexingStatus,
    VersionSelectorKind,
)
from app.retrieval.version_scope import VersionScopeError, VersionScopeResolver


def version(
    document: LogicalDocument,
    ordinal: int,
    *,
    active: bool = False,
    effective_from: datetime | None = None,
    effective_to: datetime | None = None,
    indexed: bool = True,
) -> DocumentVersion:
    return DocumentVersion(
        logical_document=document,
        version_label=f"v{ordinal}",
        revision_number=1,
        version_ordinal=ordinal,
        effective_from=effective_from,
        effective_to=effective_to,
        language=DocumentLanguage.ENGLISH,
        source_filename=f"v{ordinal}.txt",
        checksum=f"{ordinal:064x}",
        storage_key=f"scope/v{ordinal}.txt",
        business_status=(VersionBusinessStatus.ACTIVE if active else VersionBusinessStatus.DRAFT),
        indexing_status=VersionIndexingStatus.READY,
        vector_index_status=(
            VectorIndexStatus.INDEXED if indexed else VectorIndexStatus.NOT_INDEXED
        ),
        ingestion_config={},
    )


@pytest.mark.asyncio
async def test_exact_current_all_and_no_current_scope(app: Any) -> None:
    database: Database = app.state.database
    async with database.session_factory() as session:
        document = LogicalDocument(stable_key="scope-main", title="Scope")
        v1 = version(document, 1)
        v2 = version(document, 2, active=True)
        session.add_all([document, v1, v2])
        await session.commit()
        resolver = VersionScopeResolver(session)

        exact = await resolver.resolve(
            None,
            QueryVersionSelector(
                kind=VersionSelectorKind.EXACT,
                document_version_id=v1.document_version_id,
            ),
        )
        current = await resolver.resolve(
            document.logical_document_id,
            QueryVersionSelector(kind=VersionSelectorKind.CURRENT),
        )
        all_versions = await resolver.resolve(
            document.logical_document_id,
            QueryVersionSelector(kind=VersionSelectorKind.ALL),
        )

        assert [item.document_version_id for item in exact.versions] == [v1.document_version_id]
        assert [item.document_version_id for item in current.versions] == [v2.document_version_id]
        assert [item.document_version_id for item in all_versions.versions] == [
            v1.document_version_id,
            v2.document_version_id,
        ]

        v2.business_status = VersionBusinessStatus.SUPERSEDED
        await session.commit()
        with pytest.raises(VersionScopeError, match="no explicitly active"):
            await resolver.resolve(
                document.logical_document_id,
                QueryVersionSelector(kind=VersionSelectorKind.CURRENT),
            )


@pytest.mark.asyncio
async def test_as_of_unique_missing_and_ambiguous_scope(app: Any) -> None:
    database: Database = app.state.database
    async with database.session_factory() as session:
        document = LogicalDocument(stable_key="scope-as-of", title="As Of")
        v1 = version(
            document,
            1,
            effective_from=datetime(2025, 1, 1, tzinfo=UTC),
            effective_to=datetime(2026, 1, 1, tzinfo=UTC),
        )
        v2 = version(
            document,
            2,
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
        session.add_all([document, v1, v2])
        await session.commit()
        resolver = VersionScopeResolver(session)

        resolved = await resolver.resolve(
            document.logical_document_id,
            QueryVersionSelector(
                kind=VersionSelectorKind.AS_OF,
                as_of=datetime(2025, 6, 1, tzinfo=UTC),
            ),
        )
        assert [item.document_version_id for item in resolved.versions] == [v1.document_version_id]

        with pytest.raises(VersionScopeError, match="no version"):
            await resolver.resolve(
                document.logical_document_id,
                QueryVersionSelector(
                    kind=VersionSelectorKind.AS_OF,
                    as_of=datetime(2024, 1, 1, tzinfo=UTC),
                ),
            )

        overlapping = version(
            document,
            3,
            effective_from=datetime(2025, 5, 1, tzinfo=UTC),
            effective_to=datetime(2025, 8, 1, tzinfo=UTC),
        )
        session.add(overlapping)
        await session.commit()
        with pytest.raises(VersionScopeError, match="multiple versions"):
            await resolver.resolve(
                document.logical_document_id,
                QueryVersionSelector(
                    kind=VersionSelectorKind.AS_OF,
                    as_of=datetime(2025, 6, 1, tzinfo=UTC),
                ),
            )


def test_invalid_selector_arguments() -> None:
    with pytest.raises(ValueError, match="requires document_version_id"):
        QueryVersionSelector(kind=VersionSelectorKind.EXACT)
