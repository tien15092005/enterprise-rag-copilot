from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_comparison_service
from app.comparison.models import ComparisonRequest, ComparisonResponse
from app.comparison.service import DocumentComparisonService

router = APIRouter(tags=["comparison"])
ComparisonService = Annotated[DocumentComparisonService, Depends(get_comparison_service)]


@router.post("/compare", response_model=ComparisonResponse)
async def compare_versions(
    request: ComparisonRequest,
    service: ComparisonService,
) -> ComparisonResponse:
    return await service.compare(request)
