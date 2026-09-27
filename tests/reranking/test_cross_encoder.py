from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from app.models.retrieval import HybridRetrievalCandidate
from app.models.versioning import DocumentLanguage
from app.reranking.cross_encoder import CrossEncoderReranker
from app.reranking.errors import RerankerError


def candidate(text: str, rank: int) -> HybridRetrievalCandidate:
    return HybridRetrievalCandidate(
        chunk_id=uuid4(),
        document_version_id=uuid4(),
        logical_document_id=uuid4(),
        text=text,
        ordinal=rank,
        page=None,
        section_path=None,
        language=DocumentLanguage.VIETNAMESE,
        dense_rank=rank,
        dense_score=1 / rank,
        lexical_rank=rank,
        lexical_score=1 / rank,
        rrf_score=1 / (60 + rank),
        rrf_rank=rank,
    )


class PredictModel:
    def __init__(self, scores: list[float], *, fail: bool = False) -> None:
        self.scores = scores
        self.fail = fail
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]], **_kwargs: Any) -> list[float]:
        self.calls.append(pairs)
        if self.fail:
            raise RuntimeError("controlled")
        return self.scores


@pytest.mark.asyncio
async def test_cross_encoder_is_lazy_and_deterministically_reorders() -> None:
    model = PredictModel([0.1, 0.9])
    load_calls: list[str] = []

    def loader(model_name: str, **_kwargs: Any) -> PredictModel:
        load_calls.append(model_name)
        return model

    reranker = CrossEncoderReranker(
        model_name="fake-multilingual",
        revision="pinned",
        device="cpu",
        batch_size=2,
        max_length=512,
        cache_folder="D:/cache",
        model_loader=loader,
    )
    first, second = candidate("English evidence", 1), candidate("Bằng chứng tiếng Việt", 2)
    assert load_calls == []

    result = await reranker.rerank("chính sách", [first, second])

    assert load_calls == ["fake-multilingual"]
    assert [item.chunk_id for item in result] == [second.chunk_id, first.chunk_id]
    assert [item.reranker_score for item in result] == [0.9, 0.1]
    assert [item.reranker_rank for item in result] == [1, 2]
    assert result[0].rrf_rank == 2
    model.scores = [0.2]
    await reranker.rerank("policy", [first])
    assert load_calls == ["fake-multilingual"]


@pytest.mark.asyncio
async def test_cross_encoder_wraps_provider_failure() -> None:
    reranker = CrossEncoderReranker(
        model_name="fake",
        revision="pinned",
        device="cpu",
        batch_size=1,
        max_length=512,
        cache_folder=None,
        model_loader=lambda *_args, **_kwargs: PredictModel([], fail=True),
    )
    with pytest.raises(RerankerError):
        await reranker.rerank("query", [candidate("text", 1)])
