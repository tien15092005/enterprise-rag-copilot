from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.generation.citations import (
    CitationFactory,
    CitationRecord,
    CitationSource,
    citation_key_for_rank,
)


@dataclass(frozen=True, slots=True)
class ContextEvidence:
    citation: CitationRecord
    source: CitationSource
    rendered_text: str


@dataclass(frozen=True, slots=True)
class BuiltContext:
    text: str
    evidence: tuple[ContextEvidence, ...]
    truncated: bool


class EvidenceContextBuilder:
    def __init__(self, max_characters: int, excerpt_max_characters: int = 500) -> None:
        if max_characters < 1:
            raise ValueError("max_characters must be positive")
        self._max_characters = max_characters
        self._citation_factory = CitationFactory(excerpt_max_characters)

    def build(
        self,
        ranked_evidence: list[CitationSource],
        *,
        allowed_version_ids: set[UUID] | None = None,
    ) -> BuiltContext:
        blocks: list[str] = []
        included: list[ContextEvidence] = []
        used = 0
        truncated = False
        seen_chunk_ids: set[UUID] = set()

        for item in ranked_evidence:
            if item.candidate.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(item.candidate.chunk_id)
            text = item.candidate.text.strip()
            if not text:
                continue
            rank = len(included) + 1
            key = citation_key_for_rank(rank)
            prefix = self._prefix(key, item)
            separator_size = 2 if blocks else 0
            available = self._max_characters - used - separator_size
            if available <= len(prefix):
                truncated = True
                break
            rendered_text = text[: available - len(prefix)]
            if len(rendered_text) < len(text):
                truncated = True
            block = prefix + rendered_text
            blocks.append(block)
            used += separator_size + len(block)
            citation = self._citation_factory.create(
                item,
                rank=rank,
                supplied_text=rendered_text,
                allowed_version_ids=allowed_version_ids,
            )
            included.append(ContextEvidence(citation, item, rendered_text))
            if truncated:
                break

        return BuiltContext("\n\n".join(blocks), tuple(included), truncated)

    @staticmethod
    def _prefix(key: str, item: CitationSource) -> str:
        page = str(item.candidate.page) if item.candidate.page is not None else "not available"
        section = (
            " > ".join(item.candidate.section_path)
            if item.candidate.section_path
            else "not available"
        )
        return (
            f"[{key}]\n"
            f"Document: {item.document_title}\n"
            f"Version: {item.version_label} ({item.candidate.document_version_id})\n"
            f"File: {item.source_filename}\n"
            f"Page: {page}\n"
            f"Section: {section}\n"
            "Evidence: "
        )
