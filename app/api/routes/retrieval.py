from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from app.api.dependencies import (
    get_dense_retrieval_service,
    get_hybrid_retrieval_service,
    get_lexical_retrieval_service,
    get_vector_indexing_service,
)
from app.models.retrieval import (
    DenseRetrievalRequest,
    DenseRetrievalResponse,
    HybridRetrievalResponse,
    LexicalRetrievalResponse,
    VectorIndexResponse,
)
from app.retrieval.hybrid import HybridRetrievalService
from app.retrieval.lexical import LexicalRetrievalService
from app.retrieval.service import DenseRetrievalService, VectorIndexingService

router = APIRouter(tags=["retrieval"])
IndexingService = Annotated[VectorIndexingService, Depends(get_vector_indexing_service)]
RetrievalService = Annotated[DenseRetrievalService, Depends(get_dense_retrieval_service)]
LexicalService = Annotated[LexicalRetrievalService, Depends(get_lexical_retrieval_service)]
HybridService = Annotated[HybridRetrievalService, Depends(get_hybrid_retrieval_service)]


@router.post(
    "/document-versions/{document_version_id}/index",
    response_model=VectorIndexResponse,
    summary="Index an ingested document version in Qdrant",
)
async def index_document_version(
    document_version_id: UUID,
    service: IndexingService,
) -> VectorIndexResponse:
    return await service.index_version(document_version_id)


@router.post(
    "/retrieval/dense",
    response_model=DenseRetrievalResponse,
    summary="Debug exact-version dense retrieval",
    description="Engineering endpoint. It searches only the explicitly supplied version.",
)
async def dense_retrieval(
    request: DenseRetrievalRequest,
    service: RetrievalService,
) -> DenseRetrievalResponse:
    return await service.search(request.query, request.document_version_id, request.top_k)


@router.post(
    "/retrieval/lexical",
    response_model=LexicalRetrievalResponse,
    summary="Debug exact-version PostgreSQL lexical retrieval",
)
async def lexical_retrieval(
    request: DenseRetrievalRequest,
    service: LexicalService,
) -> LexicalRetrievalResponse:
    return await service.search(request.query, request.document_version_id, request.top_k)


@router.post(
    "/retrieval/hybrid",
    response_model=HybridRetrievalResponse,
    summary="Debug exact-version dense plus lexical RRF retrieval",
)
async def hybrid_retrieval(
    request: DenseRetrievalRequest,
    service: HybridService,
) -> HybridRetrievalResponse:
    return await service.search(request.query, request.document_version_id, request.top_k)
