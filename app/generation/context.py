from __future__ import annotations

from dataclasses import dataclass

from app.models.query import Citation
from app.models.retrieval import DenseRetrievalCandidate


@dataclass(frozen=True, slots=True)
class EvidenceMetadata:
    candidate: DenseRetrievalCandidate
    document_title: str
    version_label: str
    filename: str


@dataclass(frozen=True, slots=True)
class ContextEvidence:
    citation: Citation
    candidate: DenseRetrievalCandidate
    rendered_text: str


@dataclass(frozen=True, slots=True)
class BuiltContext:
    text: str
    evidence: tuple[ContextEvidence, ...]
    truncated: bool


class EvidenceContextBuilder:
    def __init__(self, max_characters: int) -> None:
        if max_characters < 1:
            raise ValueError("max_characters must be positive")
        self._max_characters = max_characters

    def build(self, ranked_evidence: list[EvidenceMetadata]) -> BuiltContext:
        blocks: list[str] = []
        included: list[ContextEvidence] = []
        used = 0
        truncated = False

        for item in ranked_evidence:
            text = item.candidate.text.strip()
            if not text:
                continue
            key = f"S{len(included) + 1}"
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
            citation = Citation(
                citation_key=key,
                logical_document_id=item.candidate.logical_document_id,
                document_version_id=item.candidate.document_version_id,
                version_label=item.version_label,
                filename=item.filename,
                chunk_id=item.candidate.chunk_id,
                page=item.candidate.page,
                section_path=item.candidate.section_path,
                excerpt=rendered_text,
            )
            included.append(ContextEvidence(citation, item.candidate, rendered_text))
            if truncated:
                break

        return BuiltContext("\n\n".join(blocks), tuple(included), truncated)

    @staticmethod
    def _prefix(key: str, item: EvidenceMetadata) -> str:
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
            f"File: {item.filename}\n"
            f"Page: {page}\n"
            f"Section: {section}\n"
            "Evidence: "
        )
