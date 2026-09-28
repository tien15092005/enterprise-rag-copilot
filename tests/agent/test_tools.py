from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from app.agent.errors import InvalidAgentToolArgumentsError, UnknownAgentToolError
from app.agent.tools import AgentToolRegistry
from app.models.versioning import (
    DocumentLanguage,
    VectorIndexStatus,
    VersionBusinessStatus,
    VersionIndexingStatus,
)


class UnusedService:
    pass


def registry() -> AgentToolRegistry:
    unused = UnusedService()
    return AgentToolRegistry(unused, unused, unused)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_registry_is_a_four_tool_allowlist_with_typed_schemas() -> None:
    tools = registry()
    definitions = tools.definitions()

    assert tools.tool_names == (
        "search_knowledge",
        "compare_versions",
        "list_document_versions",
        "get_document_version",
    )
    assert all(definition.parameters["type"] == "object" for definition in definitions)

    with pytest.raises(UnknownAgentToolError):
        await tools.execute("shell", {"command": "dir"}, scope_logical_document_id=None)
    with pytest.raises(InvalidAgentToolArgumentsError):
        await tools.execute("compare_versions", {}, scope_logical_document_id=None)


class RecordingRag:
    def __init__(self) -> None:
        self.request: Any = None

    async def answer(self, request: Any) -> Any:
        self.request = request
        return SimpleNamespace(
            model_dump=lambda **_kwargs: {"answer": "grounded [S1]"},
            citations=[],
            insufficient_evidence=False,
            token_usage=None,
        )


class RecordingComparison:
    def __init__(self) -> None:
        self.request: Any = None

    async def compare(self, request: Any) -> Any:
        self.request = request
        return SimpleNamespace(
            model_dump=lambda **_kwargs: {"counts": {"modified": 1}},
            citations=[],
            token_usage=None,
        )


class RecordingIngestion:
    def __init__(self) -> None:
        self.document_id = uuid4()
        self.version = SimpleNamespace(
            document_version_id=uuid4(),
            logical_document_id=self.document_id,
            version_label="2026",
            revision_number=1,
            version_ordinal=2,
            language=DocumentLanguage.ENGLISH,
            source_filename="policy.md",
            business_status=VersionBusinessStatus.ACTIVE,
            indexing_status=VersionIndexingStatus.READY,
            vector_index_status=VectorIndexStatus.INDEXED,
            effective_from=None,
            effective_to=None,
            published_at=None,
            is_current=True,
        )

    async def list_versions(self, _document_id: Any) -> list[Any]:
        return [self.version]

    async def get_version(self, _version_id: Any) -> Any:
        return self.version


@pytest.mark.asyncio
async def test_registry_delegates_without_reimplementing_business_logic() -> None:
    rag = RecordingRag()
    comparison = RecordingComparison()
    ingestion = RecordingIngestion()
    tools = AgentToolRegistry(rag, comparison, ingestion)  # type: ignore[arg-type]
    left_id, right_id = uuid4(), uuid4()

    await tools.execute(
        "search_knowledge",
        {
            "question": "policy",
            "logical_document_id": str(ingestion.document_id),
            "version_selector": {"kind": "current"},
            "retrieval_mode": "hybrid_rerank",
        },
        scope_logical_document_id=ingestion.document_id,
    )
    await tools.execute(
        "compare_versions",
        {
            "logical_document_id": str(ingestion.document_id),
            "left_document_version_id": str(left_id),
            "right_document_version_id": str(right_id),
        },
        scope_logical_document_id=ingestion.document_id,
    )
    listed = await tools.execute(
        "list_document_versions",
        {"logical_document_id": str(ingestion.document_id)},
        scope_logical_document_id=ingestion.document_id,
    )
    fetched = await tools.execute(
        "get_document_version",
        {"document_version_id": str(ingestion.version.document_version_id)},
        scope_logical_document_id=None,
    )

    assert rag.request.retrieval_mode.value == "hybrid_rerank"
    assert rag.request.version_selector.kind.value == "current"
    assert comparison.request.left_document_version_id == left_id
    assert comparison.request.right_document_version_id == right_id
    assert comparison.request.generate_summary is False
    assert listed.payload["versions"][0]["is_current"] is True
    assert fetched.payload["version_label"] == "2026"
