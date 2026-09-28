from fastapi import APIRouter

from app.api.routes.agent import router as agent_router
from app.api.routes.comparisons import router as comparisons_router
from app.api.routes.documents import router as documents_router
from app.api.routes.query import router as query_router
from app.api.routes.retrieval import router as retrieval_router

api_router = APIRouter()
api_router.include_router(agent_router)
api_router.include_router(comparisons_router)
api_router.include_router(documents_router)
api_router.include_router(query_router)
api_router.include_router(retrieval_router)
