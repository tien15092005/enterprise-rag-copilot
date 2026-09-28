from typing import Annotated

from fastapi import APIRouter, Depends

from app.agent.models import AgentRequest, AgentResponse
from app.agent.service import AgentService
from app.api.dependencies import get_agent_service

router = APIRouter(tags=["agent"])
AgentServiceDependency = Annotated[AgentService, Depends(get_agent_service)]


@router.post("/agent", response_model=AgentResponse)
async def run_agent(request: AgentRequest, service: AgentServiceDependency) -> AgentResponse:
    return await service.run(request)
