from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ValidationError

from app.agent.errors import AgentError, InvalidAgentToolArgumentsError, UnknownAgentToolError
from app.agent.models import (
    AgentToolResult,
    AgentVersion,
    CompareVersionsInput,
    GetDocumentVersionInput,
    ListDocumentVersionsInput,
    SearchKnowledgeInput,
)
from app.agent.provider import AgentToolDefinition
from app.comparison.models import ComparisonRequest
from app.comparison.service import DocumentComparisonService
from app.generation.service import RagAnswerService
from app.ingestion.service import IngestionService
from app.models.query import QueryRequest

ToolInput = (
    SearchKnowledgeInput
    | CompareVersionsInput
    | ListDocumentVersionsInput
    | GetDocumentVersionInput
)
ToolHandler = Callable[[ToolInput], Awaitable[AgentToolResult]]


class AgentToolRegistry:
    """Typed allowlist over existing application services; it contains no retrieval logic."""

    def __init__(
        self,
        rag: RagAnswerService,
        comparison: DocumentComparisonService,
        ingestion: IngestionService,
    ) -> None:
        self._rag = rag
        self._comparison = comparison
        self._ingestion = ingestion
        self._schemas: dict[str, type[BaseModel]] = {
            "search_knowledge": SearchKnowledgeInput,
            "compare_versions": CompareVersionsInput,
            "list_document_versions": ListDocumentVersionsInput,
            "get_document_version": GetDocumentVersionInput,
        }
        self._handlers: dict[str, ToolHandler] = {
            "search_knowledge": self._search,
            "compare_versions": self._compare,
            "list_document_versions": self._list_versions,
            "get_document_version": self._get_version,
        }

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._schemas)

    def definitions(self) -> list[AgentToolDefinition]:
        descriptions = {
            "search_knowledge": (
                "Answer a question using the existing version-scoped RAG pipeline. "
                "Use current, exact, all, or as_of selectors explicitly."
            ),
            "compare_versions": (
                "Deterministically compare two explicit immutable versions of one document."
            ),
            "list_document_versions": "List immutable versions of one logical document.",
            "get_document_version": "Get safe metadata for one explicit document version.",
        }
        return [
            AgentToolDefinition(name, descriptions[name], schema.model_json_schema())
            for name, schema in self._schemas.items()
        ]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        scope_logical_document_id: UUID | None,
    ) -> AgentToolResult:
        schema = self._schemas.get(name)
        if schema is None:
            raise UnknownAgentToolError(name)
        try:
            parsed = schema.model_validate(arguments)
        except ValidationError as exc:
            raise InvalidAgentToolArgumentsError(
                name,
                [dict(item) for item in exc.errors(include_url=False, include_context=False)],
            ) from exc
        self._validate_request_scope(parsed, scope_logical_document_id)
        result = await self._handlers[name](parsed)  # type: ignore[arg-type]
        if name == "get_document_version" and scope_logical_document_id is not None:
            if UUID(str(result.payload["logical_document_id"])) != scope_logical_document_id:
                raise AgentError(
                    "document version is outside the agent request scope",
                    code="agent_scope_violation",
                    status_code=409,
                )
        return result

    @staticmethod
    def _validate_request_scope(value: BaseModel, expected: UUID | None) -> None:
        supplied = getattr(value, "logical_document_id", None)
        if expected is not None and supplied is not None and supplied != expected:
            raise AgentError(
                "tool logical_document_id is outside the agent request scope",
                code="agent_scope_violation",
                status_code=409,
            )

    async def _search(self, value: ToolInput) -> AgentToolResult:
        if not isinstance(value, SearchKnowledgeInput):
            raise TypeError("search_knowledge received unexpected input type")
        response = await self._rag.answer(
            QueryRequest(
                question=value.question,
                logical_document_id=value.logical_document_id,
                version_selector=value.version_selector,
                top_k=value.top_k,
                retrieval_mode=value.retrieval_mode,
                retrieval_only=False,
                debug=False,
            )
        )
        return AgentToolResult(
            result_type="knowledge_answer",
            payload=response.model_dump(mode="json", exclude={"debug", "citations"}),
            citations=response.citations,
            insufficient_evidence=response.insufficient_evidence,
            token_usage=response.token_usage,
        )

    async def _compare(self, value: ToolInput) -> AgentToolResult:
        if not isinstance(value, CompareVersionsInput):
            raise TypeError("compare_versions received unexpected input type")
        response = await self._comparison.compare(
            ComparisonRequest(
                logical_document_id=value.logical_document_id,
                left_document_version_id=value.left_document_version_id,
                right_document_version_id=value.right_document_version_id,
                include_unchanged=value.include_unchanged,
                generate_summary=False,
                summary_language=value.summary_language,
                debug=False,
            )
        )
        return AgentToolResult(
            result_type="version_comparison",
            payload=response.model_dump(mode="json", exclude={"debug", "citations"}),
            citations=[item.citation for item in response.citations],
            token_usage=response.token_usage,
        )

    async def _list_versions(self, value: ToolInput) -> AgentToolResult:
        if not isinstance(value, ListDocumentVersionsInput):
            raise TypeError("list_document_versions received unexpected input type")
        versions = await self._ingestion.list_versions(value.logical_document_id)
        payload = [self._safe_version(version).model_dump(mode="json") for version in versions]
        return AgentToolResult(result_type="document_version_list", payload={"versions": payload})

    async def _get_version(self, value: ToolInput) -> AgentToolResult:
        if not isinstance(value, GetDocumentVersionInput):
            raise TypeError("get_document_version received unexpected input type")
        version = await self._ingestion.get_version(value.document_version_id)
        return AgentToolResult(
            result_type="document_version",
            payload=self._safe_version(version).model_dump(mode="json"),
        )

    @staticmethod
    def _safe_version(version: Any) -> AgentVersion:
        return AgentVersion(
            document_version_id=version.document_version_id,
            logical_document_id=version.logical_document_id,
            version_label=version.version_label,
            revision_number=version.revision_number,
            version_ordinal=version.version_ordinal,
            language=version.language,
            source_filename=version.source_filename,
            business_status=version.business_status,
            indexing_status=version.indexing_status,
            vector_index_status=version.vector_index_status,
            effective_from=version.effective_from,
            effective_to=version.effective_to,
            published_at=version.published_at,
            is_current=version.is_current,
        )
