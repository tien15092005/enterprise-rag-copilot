from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.versioning import DocumentLanguage, VectorIndexStatus


class RetrievalMode(StrEnum):
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


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


class LexicalRetrievalCandidate(BaseModel):
    chunk_id: UUID
    document_version_id: UUID
    logical_document_id: UUID
    text: str
    lexical_score: float
    ordinal: int
    page: int | None
    section_path: list[str] | None
    language: DocumentLanguage


class HybridRetrievalCandidate(BaseModel):
    chunk_id: UUID
    document_version_id: UUID
    logical_document_id: UUID
    text: str
    ordinal: int
    page: int | None
    section_path: list[str] | None
    language: DocumentLanguage
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    rrf_score: float
    rrf_rank: int | None = None


class RerankedRetrievalCandidate(HybridRetrievalCandidate):
    reranker_score: float
    reranker_rank: int = Field(ge=1)


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


class LexicalRetrievalResponse(BaseModel):
    document_version_id: UUID
    candidates: list[LexicalRetrievalCandidate]
    retrieval_latency_ms: float
    total_latency_ms: float


class LexicalRetrievalBatch(BaseModel):
    document_version_ids: list[UUID]
    candidates: list[LexicalRetrievalCandidate]
    retrieval_latency_ms: float
    total_latency_ms: float


class HybridRetrievalResponse(BaseModel):
    document_version_id: UUID
    candidates: list[HybridRetrievalCandidate]
    dense_latency_ms: float
    lexical_latency_ms: float
    fusion_latency_ms: float
    total_latency_ms: float
    dense_candidate_count: int
    lexical_candidate_count: int
    merged_duplicate_count: int


class HybridRetrievalBatch(BaseModel):
    document_version_ids: list[UUID]
    candidates: list[HybridRetrievalCandidate]
    embedding_latency_ms: float
    dense_latency_ms: float
    lexical_latency_ms: float
    fusion_latency_ms: float
    total_latency_ms: float
    dense_candidate_count: int
    lexical_candidate_count: int
    merged_duplicate_count: int


class RerankedRetrievalBatch(BaseModel):
    document_version_ids: list[UUID]
    candidates: list[RerankedRetrievalCandidate]
    embedding_latency_ms: float
    dense_latency_ms: float
    lexical_latency_ms: float
    fusion_latency_ms: float
    reranking_latency_ms: float
    total_latency_ms: float
    dense_candidate_count: int
    lexical_candidate_count: int
    merged_duplicate_count: int
    pre_rerank_candidate_count: int
    post_rerank_candidate_count: int
    reranker_model: str
    reranker_revision: str | None


RetrievalCandidate = DenseRetrievalCandidate | HybridRetrievalCandidate | RerankedRetrievalCandidate
RetrievalBatch = DenseRetrievalBatch | HybridRetrievalBatch | RerankedRetrievalBatch
