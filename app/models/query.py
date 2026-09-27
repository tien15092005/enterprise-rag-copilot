from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.models.versioning import DocumentLanguage, VersionSelectorKind


class QueryVersionSelector(BaseModel):
    kind: VersionSelectorKind = VersionSelectorKind.CURRENT
    document_version_id: UUID | None = None
    as_of: datetime | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> QueryVersionSelector:
        if self.kind is VersionSelectorKind.EXACT and self.document_version_id is None:
            raise ValueError("exact selector requires document_version_id")
        if self.kind is not VersionSelectorKind.EXACT and self.document_version_id is not None:
            raise ValueError("document_version_id is valid only for exact selector")
        if self.kind is VersionSelectorKind.AS_OF and self.as_of is None:
            raise ValueError("as_of selector requires as_of")
        if self.kind is not VersionSelectorKind.AS_OF and self.as_of is not None:
            raise ValueError("as_of is valid only for as_of selector")
        return self


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    logical_document_id: UUID | None = None
    version_selector: QueryVersionSelector = Field(default_factory=QueryVersionSelector)
    top_k: int = Field(default=5, ge=1, le=100)
    retrieval_only: bool = False
    debug: bool = False


class ResolvedVersion(BaseModel):
    document_version_id: UUID
    version_label: str
    effective_from: datetime | None
    effective_to: datetime | None


class ResolvedScope(BaseModel):
    kind: VersionSelectorKind
    logical_document_id: UUID
    versions: list[ResolvedVersion]


class Citation(BaseModel):
    citation_key: str
    logical_document_id: UUID
    document_version_id: UUID
    version_label: str
    filename: str
    chunk_id: UUID
    page: int | None
    section_path: list[str] | None
    excerpt: str


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class QueryTiming(BaseModel):
    scope_resolution_ms: float
    query_embedding_ms: float
    retrieval_ms: float
    context_construction_ms: float
    generation_ms: float
    total_ms: float


class DebugCandidate(BaseModel):
    citation_key: str
    chunk_id: UUID
    document_version_id: UUID
    score: float
    ordinal: int
    page: int | None
    section_path: list[str] | None
    language: DocumentLanguage


class QueryResponse(BaseModel):
    answer: str | None
    answer_language: DocumentLanguage
    citations: list[Citation]
    resolved_scope: ResolvedScope
    timing: QueryTiming
    token_usage: TokenUsage | None
    prompt_version: str
    insufficient_evidence: bool
    debug_candidates: list[DebugCandidate] | None = None
