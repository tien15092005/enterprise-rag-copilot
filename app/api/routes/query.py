from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_rag_service
from app.generation.service import RagAnswerService
from app.models.query import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])
RagService = Annotated[RagAnswerService, Depends(get_rag_service)]


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a question from version-scoped enterprise evidence",
)
async def query_knowledge(request: QueryRequest, service: RagService) -> QueryResponse:
    return await service.answer(request)
