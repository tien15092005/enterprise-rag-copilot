from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.evaluation.datasets import load_dataset
from app.evaluation.runner import EvaluationCatalog, RetrievalEvaluationRunner
from app.models.query import ResolvedScope, ResolvedVersion
from app.models.retrieval import DenseRetrievalBatch, DenseRetrievalCandidate, RetrievalMode
from app.models.versioning import DocumentLanguage, VersionSelectorKind


class FakeResolver:
    def __init__(self, document_id: UUID, allowed_version_id: UUID) -> None:
        self.document_id = document_id
        self.allowed_version_id = allowed_version_id

    async def resolve(self, _document_id: UUID, _selector: object) -> ResolvedScope:
        return ResolvedScope(
            kind=VersionSelectorKind.EXACT,
            logical_document_id=self.document_id,
            versions=[
                ResolvedVersion(
                    document_version_id=self.allowed_version_id,
                    version_label="2025",
                    effective_from=None,
                    effective_to=None,
                )
            ],
        )


class LeakyRetrieval:
    def __init__(self, candidate: DenseRetrievalCandidate) -> None:
        self.candidate = candidate

    async def search_versions(
        self,
        _query: str,
        document_version_ids: list[UUID],
        _top_k: int,
        _mode: RetrievalMode,
    ) -> DenseRetrievalBatch:
        return DenseRetrievalBatch(
            document_version_ids=document_version_ids,
            embedding_model="fake",
            candidates=[self.candidate],
            embedding_latency_ms=1,
            retrieval_latency_ms=2,
            total_latency_ms=3,
        )


@pytest.mark.asyncio
async def test_runner_scores_evidence_and_exposes_version_leakage() -> None:
    source = load_dataset(Path("data/evaluation/dataset.json"))
    dataset = source.model_copy(update={"cases": [source.cases[0]]})
    document_id, allowed_version_id, leaked_version_id, chunk_id = (
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
    )
    candidate = DenseRetrievalCandidate(
        chunk_id=chunk_id,
        document_version_id=leaked_version_id,
        logical_document_id=document_id,
        text="evidence",
        score=0.9,
        ordinal=0,
        page=None,
        section_path=None,
        language=DocumentLanguage.ENGLISH,
    )
    runner = RetrievalEvaluationRunner(
        dataset,
        EvaluationCatalog(
            document_ids={source.cases[0].document: document_id},
            version_ids={(source.cases[0].document, "2025"): allowed_version_id},
            evidence_chunk_ids={source.cases[0].relevant[0]: chunk_id},
        ),
        FakeResolver(document_id, allowed_version_id),  # type: ignore[arg-type]
        LeakyRetrieval(candidate),  # type: ignore[arg-type]
    )

    result = await runner.run(
        RetrievalMode.DENSE,
        profile="test",
        configuration={"top_k": 5},
        top_k=5,
    )

    assert result.metrics["combined"].hit_rate[1] == 1
    assert result.metrics["combined"].leakage_count == 1
    assert result.metrics["combined"].scope_violations["exact"] == 1
