from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from app.embeddings.sentence_transformers import SentenceTransformerEmbeddingProvider


class RecordingModel:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.inputs: list[list[str]] = []
        self.encode_calls = 0

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, inputs: list[str], **_kwargs: Any) -> np.ndarray[Any, Any]:
        self.inputs.append(inputs)
        self.encode_calls += 1
        return np.array([[3.0, 4.0, 0.0] for _ in inputs])


@pytest.mark.asyncio
async def test_e5_prefixes_normalizes_and_loads_model_once() -> None:
    model = RecordingModel()
    load_calls = 0

    def loader(_name: str, **_kwargs: Any) -> RecordingModel:
        nonlocal load_calls
        load_calls += 1
        return model

    provider = SentenceTransformerEmbeddingProvider(
        model_name="test-e5",
        revision="fixed-revision",
        dimension=3,
        batch_size=8,
        device="cpu",
        model_loader=loader,
    )

    query = await provider.embed_query("annual leave")
    passages = await provider.embed_passages(["Leave is 20 days.", "Nghỉ phép là 20 ngày."])

    assert model.inputs == [
        ["query: annual leave"],
        ["passage: Leave is 20 days.", "passage: Nghỉ phép là 20 ngày."],
    ]
    assert load_calls == 1
    assert len(query) == provider.dimension == 3
    assert math.isclose(sum(value * value for value in query), 1.0)
    assert all(math.isclose(sum(value * value for value in item), 1.0) for item in passages)


@pytest.mark.asyncio
async def test_configured_dimension_must_match_model() -> None:
    provider = SentenceTransformerEmbeddingProvider(
        model_name="test-e5",
        revision=None,
        dimension=4,
        batch_size=8,
        device="cpu",
        model_loader=lambda *_args, **_kwargs: RecordingModel(dimension=3),
    )

    with pytest.raises(ValueError, match="dimension mismatch"):
        await provider.embed_query("test")
