from __future__ import annotations

import math
import os

import pytest

from app.core.config import Settings
from app.embeddings import create_embedding_provider


@pytest.mark.integration
@pytest.mark.external
async def test_real_multilingual_e5_dimension_and_normalization() -> None:
    if os.getenv("RUN_REAL_EMBEDDING_TEST") != "1":
        pytest.skip("RUN_REAL_EMBEDDING_TEST=1 is not configured")

    provider = create_embedding_provider(Settings())
    query = await provider.embed_query("Chính sách nghỉ phép")
    passage = (await provider.embed_passages(["Employees may work remotely."]))[0]

    assert len(query) == len(passage) == 384
    assert math.isclose(sum(value * value for value in query), 1.0, rel_tol=1e-5)
    assert math.isclose(sum(value * value for value in passage), 1.0, rel_tol=1e-5)
