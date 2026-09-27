from uuid import uuid4

import pytest

from app.ingestion.chunker import ChunkingConfig, TokenWindowChunker
from app.ingestion.parsers.base import ParsedBlock, ParsedDocument
from app.models.versioning import DocumentLanguage


def test_chunking_is_configurable_and_preserves_metadata() -> None:
    version_id = uuid4()
    parsed = ParsedDocument(
        blocks=(
            ParsedBlock(
                text="one two three four five six seven",
                page=3,
                section_path=("Policy", "Leave"),
            ),
        )
    )

    chunks = TokenWindowChunker().chunk(
        parsed,
        document_version_id=version_id,
        language=DocumentLanguage.ENGLISH,
        config=ChunkingConfig(chunk_size=4, chunk_overlap=1),
    )

    assert [chunk.text for chunk in chunks] == ["one two three four", "four five six seven"]
    assert all(chunk.document_version_id == version_id for chunk in chunks)
    assert all(chunk.page == 3 for chunk in chunks)
    assert all(chunk.section_path == ("Policy", "Leave") for chunk in chunks)
    assert chunks[0].section_fingerprint == chunks[1].section_fingerprint
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError, match="smaller"):
        ChunkingConfig(chunk_size=5, chunk_overlap=5)
