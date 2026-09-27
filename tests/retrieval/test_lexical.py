from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.errors import InvalidRetrievalScopeError
from app.retrieval.lexical import LexicalRetrievalService


@pytest.mark.asyncio
async def test_lexical_retrieval_rejects_empty_version_scope() -> None:
    session = AsyncMock(spec=AsyncSession)

    with pytest.raises(InvalidRetrievalScopeError):
        await LexicalRetrievalService(session).search_versions("policy", [], 5)

    session.scalars.assert_not_awaited()
