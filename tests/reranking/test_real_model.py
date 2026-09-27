from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.models.retrieval import HybridRetrievalCandidate
from app.models.versioning import DocumentLanguage
from app.reranking.cross_encoder import CrossEncoderReranker


def resolved_cache(settings: Settings) -> Path:
    return Path(settings.reranker_cache_directory).resolve()


@pytest.mark.external
@pytest.mark.skipif(
    os.getenv("RUN_REAL_RERANKER_TESTS") != "1",
    reason="set RUN_REAL_RERANKER_TESTS=1 to download/use the pinned local reranker",
)
@pytest.mark.asyncio
async def test_pinned_real_model_scores_english_and_vietnamese() -> None:
    settings = Settings()
    cache = resolved_cache(settings)
    assert cache.drive.upper() == "D:"
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model_name,
        revision=settings.reranker_model_revision,
        device="cpu",
        batch_size=2,
        max_length=settings.reranker_max_length,
        cache_folder=str(cache),
    )
    version_id = uuid4()
    document_id = uuid4()
    candidates = [
        HybridRetrievalCandidate(
            chunk_id=uuid4(),
            document_version_id=version_id,
            logical_document_id=document_id,
            text=text,
            ordinal=rank,
            page=None,
            section_path=None,
            language=language,
            rrf_score=1 / (60 + rank),
            rrf_rank=rank,
        )
        for rank, (text, language) in enumerate(
            [
                ("Employees may work remotely two days per week.", DocumentLanguage.ENGLISH),
                ("Nhân viên có thể làm việc từ xa hai ngày mỗi tuần.", DocumentLanguage.VIETNAMESE),
            ],
            start=1,
        )
    ]

    result = await reranker.rerank("How many remote work days are allowed?", candidates)

    assert {item.chunk_id for item in result} == {item.chunk_id for item in candidates}
    assert all(isinstance(item.reranker_score, float) for item in result)
