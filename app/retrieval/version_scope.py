from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.core.exceptions import AppError
from app.db.models import DocumentVersion, LogicalDocument
from app.models.query import QueryVersionSelector, ResolvedScope, ResolvedVersion
from app.models.versioning import (
    VectorIndexStatus,
    VersionBusinessStatus,
    VersionIndexingStatus,
    VersionSelectorKind,
)


class VersionScopeError(AppError):
    def __init__(self, message: str, *, code: str = "version_scope_unresolvable") -> None:
        super().__init__(message, code=code, status_code=409)


class VersionScopeResolver:
    """Resolve explicit, business-defined retrieval scope without version heuristics."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def resolve(
        self,
        logical_document_id: UUID | None,
        selector: QueryVersionSelector,
    ) -> ResolvedScope:
        if selector.kind is VersionSelectorKind.EXACT:
            return await self._resolve_exact(logical_document_id, selector)
        if logical_document_id is None:
            raise VersionScopeError(
                f"logical_document_id is required for {selector.kind.value} scope",
                code="logical_document_scope_required",
            )
        if await self._session.get(LogicalDocument, logical_document_id) is None:
            raise VersionScopeError(
                "logical document was not found", code="logical_document_not_found"
            )
        if selector.kind is VersionSelectorKind.CURRENT:
            versions = await self._eligible_versions(
                logical_document_id,
                DocumentVersion.business_status == VersionBusinessStatus.ACTIVE,
            )
            if len(versions) != 1:
                raise VersionScopeError(
                    "no explicitly active, ready, indexed current version exists"
                )
        elif selector.kind is VersionSelectorKind.ALL:
            versions = await self._eligible_versions(logical_document_id)
            if not versions:
                raise VersionScopeError("no ready, indexed versions exist in the logical document")
        else:
            assert selector.as_of is not None
            versions = await self._resolve_as_of(logical_document_id, selector.as_of)
        return self._scope(selector.kind, logical_document_id, versions)

    async def _resolve_exact(
        self, logical_document_id: UUID | None, selector: QueryVersionSelector
    ) -> ResolvedScope:
        assert selector.document_version_id is not None
        version = await self._session.get(DocumentVersion, selector.document_version_id)
        if version is None:
            raise VersionScopeError(
                "exact document version was not found", code="version_not_found"
            )
        if logical_document_id is not None and version.logical_document_id != logical_document_id:
            raise VersionScopeError(
                "exact version does not belong to the requested logical document",
                code="version_scope_mismatch",
            )
        if not self._is_retrievable(version):
            raise VersionScopeError("exact document version is not ready and vector-indexed")
        return self._scope(VersionSelectorKind.EXACT, version.logical_document_id, [version])

    async def _resolve_as_of(
        self, logical_document_id: UUID, as_of: datetime
    ) -> list[DocumentVersion]:
        versions = await self._eligible_versions(
            logical_document_id,
            DocumentVersion.effective_from.is_not(None),
            DocumentVersion.effective_from <= as_of,
            or_(DocumentVersion.effective_to.is_(None), DocumentVersion.effective_to > as_of),
        )
        if not versions:
            raise VersionScopeError("no version is effective at the requested timestamp")
        if len(versions) > 1:
            raise VersionScopeError(
                "multiple versions are effective at the requested timestamp",
                code="ambiguous_as_of_scope",
            )
        return versions

    async def _eligible_versions(
        self, logical_document_id: UUID, *conditions: ColumnElement[bool]
    ) -> list[DocumentVersion]:
        result = await self._session.scalars(
            select(DocumentVersion)
            .where(
                DocumentVersion.logical_document_id == logical_document_id,
                DocumentVersion.indexing_status == VersionIndexingStatus.READY,
                DocumentVersion.vector_index_status == VectorIndexStatus.INDEXED,
                *conditions,
            )
            .order_by(DocumentVersion.version_ordinal)
        )
        return list(result)

    @staticmethod
    def _is_retrievable(version: DocumentVersion) -> bool:
        return (
            version.indexing_status is VersionIndexingStatus.READY
            and version.vector_index_status is VectorIndexStatus.INDEXED
        )

    @staticmethod
    def _scope(
        kind: VersionSelectorKind,
        logical_document_id: UUID,
        versions: list[DocumentVersion],
    ) -> ResolvedScope:
        return ResolvedScope(
            kind=kind,
            logical_document_id=logical_document_id,
            versions=[
                ResolvedVersion(
                    document_version_id=version.document_version_id,
                    version_label=version.version_label,
                    effective_from=version.effective_from,
                    effective_to=version.effective_to,
                )
                for version in versions
            ],
        )
