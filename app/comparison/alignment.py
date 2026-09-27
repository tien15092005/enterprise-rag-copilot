from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import sqrt
from time import perf_counter
from uuid import UUID

from app.comparison.models import AlignmentMethod
from app.embeddings.base import EmbeddingProvider
from app.models.versioning import DocumentLanguage

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class SectionSnapshot:
    document_version_id: UUID
    logical_document_id: UUID
    chunk_ids: tuple[UUID, ...]
    citation_chunk_id: UUID
    citation_text: str
    text: str
    section_path: tuple[str, ...] | None
    section_fingerprint: str | None
    ordinal: int
    page: int | None
    language: DocumentLanguage


@dataclass(frozen=True, slots=True)
class AlignedSection:
    left: SectionSnapshot | None
    right: SectionSnapshot | None
    method: AlignmentMethod
    score: float | None


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    pairs: tuple[AlignedSection, ...]
    semantic_latency_ms: float


class SectionAligner:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        *,
        content_threshold: float,
        semantic_threshold: float,
    ) -> None:
        self._embeddings = embeddings
        self._content_threshold = content_threshold
        self._semantic_threshold = semantic_threshold

    async def align(
        self, left: list[SectionSnapshot], right: list[SectionSnapshot]
    ) -> AlignmentResult:
        matches: dict[int, tuple[int, AlignmentMethod, float]] = {}
        available_right = set(range(len(right)))
        self._match_exact_paths(left, right, matches, available_right)
        self._match_normalized_paths(left, right, matches, available_right)
        self._match_exact_headings(left, right, matches, available_right)
        self._match_normalized_headings(left, right, matches, available_right)
        self._match_fingerprints(left, right, matches, available_right)
        self._match_content(left, right, matches, available_right)
        semantic_latency = await self._match_semantic(left, right, matches, available_right)
        pairs = [
            AlignedSection(
                left=section,
                right=right[matches[index][0]] if index in matches else None,
                method=matches[index][1] if index in matches else AlignmentMethod.UNALIGNED,
                score=matches[index][2] if index in matches else None,
            )
            for index, section in enumerate(left)
        ]
        pairs.extend(
            AlignedSection(None, right[index], AlignmentMethod.UNALIGNED, None)
            for index in sorted(available_right, key=lambda item: right[item].ordinal)
        )
        return AlignmentResult(tuple(pairs), semantic_latency)

    @staticmethod
    def _match_exact_paths(
        left: list[SectionSnapshot],
        right: list[SectionSnapshot],
        matches: dict[int, tuple[int, AlignmentMethod, float]],
        available_right: set[int],
    ) -> None:
        for left_index, left_section in enumerate(left):
            candidates = [
                index
                for index in available_right
                if left_section.section_path is not None
                and left_section.section_path == right[index].section_path
            ]
            if candidates:
                right_index = min(candidates, key=lambda index: right[index].ordinal)
                matches[left_index] = (right_index, AlignmentMethod.EXACT_PATH, 1.0)
                available_right.remove(right_index)

    @staticmethod
    def _match_normalized_paths(
        left: list[SectionSnapshot],
        right: list[SectionSnapshot],
        matches: dict[int, tuple[int, AlignmentMethod, float]],
        available_right: set[int],
    ) -> None:
        for left_index, left_section in enumerate(left):
            if left_index in matches:
                continue
            left_key = normalize_path(left_section.section_path)
            candidates = [
                index
                for index in available_right
                if left_key and left_key == normalize_path(right[index].section_path)
            ]
            if candidates:
                right_index = min(candidates, key=lambda index: right[index].ordinal)
                matches[left_index] = (right_index, AlignmentMethod.NORMALIZED_PATH, 1.0)
                available_right.remove(right_index)

    @staticmethod
    def _match_fingerprints(
        left: list[SectionSnapshot],
        right: list[SectionSnapshot],
        matches: dict[int, tuple[int, AlignmentMethod, float]],
        available_right: set[int],
    ) -> None:
        for left_index, left_section in enumerate(left):
            if left_index in matches or left_section.section_fingerprint is None:
                continue
            candidates = [
                index
                for index in available_right
                if left_section.section_fingerprint == right[index].section_fingerprint
            ]
            if candidates:
                right_index = min(candidates, key=lambda index: right[index].ordinal)
                matches[left_index] = (right_index, AlignmentMethod.SECTION_FINGERPRINT, 1.0)
                available_right.remove(right_index)

    @staticmethod
    def _match_exact_headings(
        left: list[SectionSnapshot],
        right: list[SectionSnapshot],
        matches: dict[int, tuple[int, AlignmentMethod, float]],
        available_right: set[int],
    ) -> None:
        for left_index, left_section in enumerate(left):
            if left_index in matches:
                continue
            left_heading = _heading(left_section.section_path)
            candidates = [
                index
                for index in available_right
                if left_heading is not None and left_heading == _heading(right[index].section_path)
            ]
            if candidates:
                right_index = min(candidates, key=lambda index: right[index].ordinal)
                matches[left_index] = (right_index, AlignmentMethod.EXACT_HEADING, 1.0)
                available_right.remove(right_index)

    @staticmethod
    def _match_normalized_headings(
        left: list[SectionSnapshot],
        right: list[SectionSnapshot],
        matches: dict[int, tuple[int, AlignmentMethod, float]],
        available_right: set[int],
    ) -> None:
        for left_index, left_section in enumerate(left):
            if left_index in matches:
                continue
            left_heading = normalize_heading(left_section.section_path)
            candidates = [
                index
                for index in available_right
                if left_heading and left_heading == normalize_heading(right[index].section_path)
            ]
            if candidates:
                right_index = min(candidates, key=lambda index: right[index].ordinal)
                matches[left_index] = (right_index, AlignmentMethod.NORMALIZED_HEADING, 1.0)
                available_right.remove(right_index)

    def _match_content(
        self,
        left: list[SectionSnapshot],
        right: list[SectionSnapshot],
        matches: dict[int, tuple[int, AlignmentMethod, float]],
        available_right: set[int],
    ) -> None:
        candidates = [
            (
                content_similarity(left_section.text, right[right_index].text),
                left_index,
                right_index,
            )
            for left_index, left_section in enumerate(left)
            if left_index not in matches
            for right_index in available_right
        ]
        for score, left_index, right_index in sorted(
            candidates, key=lambda item: (-item[0], left[item[1]].ordinal, right[item[2]].ordinal)
        ):
            if score < self._content_threshold:
                break
            if left_index in matches or right_index not in available_right:
                continue
            matches[left_index] = (right_index, AlignmentMethod.CONTENT_SIMILARITY, score)
            available_right.remove(right_index)

    async def _match_semantic(
        self,
        left: list[SectionSnapshot],
        right: list[SectionSnapshot],
        matches: dict[int, tuple[int, AlignmentMethod, float]],
        available_right: set[int],
    ) -> float:
        left_indexes = [index for index in range(len(left)) if index not in matches]
        right_indexes = sorted(available_right)
        if not left_indexes or not right_indexes:
            return 0
        started = perf_counter()
        left_vectors = await self._embeddings.embed_passages(
            [left[index].text for index in left_indexes]
        )
        right_vectors = await self._embeddings.embed_passages(
            [right[index].text for index in right_indexes]
        )
        candidates = [
            (_cosine(left_vectors[li], right_vectors[ri]), left_index, right_index)
            for li, left_index in enumerate(left_indexes)
            for ri, right_index in enumerate(right_indexes)
        ]
        for score, left_index, right_index in sorted(
            candidates, key=lambda item: (-item[0], left[item[1]].ordinal, right[item[2]].ordinal)
        ):
            if score < self._semantic_threshold:
                break
            if left_index in matches or right_index not in available_right:
                continue
            matches[left_index] = (right_index, AlignmentMethod.SEMANTIC, score)
            available_right.remove(right_index)
        return (perf_counter() - started) * 1000


def normalize_path(path: tuple[str, ...] | None) -> str:
    if not path:
        return ""
    normalized = " / ".join(unicodedata.normalize("NFKC", item).casefold() for item in path)
    return _NON_WORD.sub(" ", normalized).strip()


def normalize_heading(path: tuple[str, ...] | None) -> str:
    heading = _heading(path)
    if heading is None:
        return ""
    return _NON_WORD.sub(" ", unicodedata.normalize("NFKC", heading).casefold()).strip()


def _heading(path: tuple[str, ...] | None) -> str | None:
    return path[-1] if path else None


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def content_similarity(left: str, right: str) -> float:
    return SequenceMatcher(
        None, normalize_text(left), normalize_text(right), autojunk=False
    ).ratio()


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    denominator = sqrt(sum(value * value for value in left)) * sqrt(
        sum(value * value for value in right)
    )
    return dot / denominator if denominator else 0.0
