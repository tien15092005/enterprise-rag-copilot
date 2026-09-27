from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from app.models.versioning import DocumentLanguage, VectorIndexStatus


class VectorIndexResponse(BaseModel):
    document_version_id: UUID
    status: VectorIndexStatus
    chunks_indexed: int
    embedding_model: str
    embedding_revision: str | None
    embedding_dimension: int
    embedding_latency_ms: float
    qdrant_latency_ms: float
    total_latency_ms: float


class DenseRetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    document_version_id: UUID
    top_k: int = Field(default=5, ge=1, le=100)


class DenseRetrievalCandidate(BaseModel):
    chunk_id: UUID
    document_version_id: UUID
    logical_document_id: UUID
    text: str
    score: float
    ordinal: int
    page: int | None
    section_path: list[str] | None
    language: DocumentLanguage


class DenseRetrievalResponse(BaseModel):
    document_version_id: UUID
    embedding_model: str
    candidates: list[DenseRetrievalCandidate]
    embedding_latency_ms: float
    retrieval_latency_ms: float
    total_latency_ms: float


class DenseRetrievalBatch(BaseModel):
    document_version_ids: list[UUID]
    embedding_model: str
    candidates: list[DenseRetrievalCandidate]
    embedding_latency_ms: float
    retrieval_latency_ms: float
    total_latency_ms: float
