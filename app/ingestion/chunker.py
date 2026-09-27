from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from uuid import UUID, uuid5

from app.ingestion.parsers.base import ParsedDocument
from app.models.versioning import DocumentLanguage

_TOKEN = re.compile(r"\S+")
_CHUNK_NAMESPACE = UUID("c76fead1-e3eb-44b6-a815-9fba8c8fc26f")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: UUID
    document_version_id: UUID
    ordinal: int
    text: str
    page: int | None
    section_path: tuple[str, ...]
    language: DocumentLanguage
    section_fingerprint: str | None


class TokenWindowChunker:
    """Deterministic whitespace-token windows that never cross source metadata boundaries."""

    def chunk(
        self,
        document: ParsedDocument,
        *,
        document_version_id: UUID,
        language: DocumentLanguage,
        config: ChunkingConfig,
    ) -> tuple[DocumentChunk, ...]:
        chunks: list[DocumentChunk] = []
        step = config.chunk_size - config.chunk_overlap

        for block in document.blocks:
            matches = list(_TOKEN.finditer(block.text))
            for start in range(0, len(matches), step):
                window = matches[start : start + config.chunk_size]
                if not window:
                    break
                text = block.text[window[0].start() : window[-1].end()]
                ordinal = len(chunks)
                section_fingerprint = self._section_fingerprint(block.section_path)
                chunks.append(
                    DocumentChunk(
                        chunk_id=uuid5(
                            _CHUNK_NAMESPACE,
                            f"{document_version_id}:{ordinal}",
                        ),
                        document_version_id=document_version_id,
                        ordinal=ordinal,
                        text=text,
                        page=block.page,
                        section_path=block.section_path,
                        language=language,
                        section_fingerprint=section_fingerprint,
                    )
                )
                if start + config.chunk_size >= len(matches):
                    break

        if not chunks:
            raise ValueError("parsed document produced no chunks")
        return tuple(chunks)

    @staticmethod
    def _section_fingerprint(section_path: tuple[str, ...]) -> str | None:
        if not section_path:
            return None
        normalized = "/".join(part.strip().casefold() for part in section_path)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
