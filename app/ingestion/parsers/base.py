from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.ingestion.errors import UnsupportedDocumentError


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    text: str
    page: int | None = None
    section_path: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    blocks: tuple[ParsedBlock, ...]


class DocumentParser(Protocol):
    supported_extensions: frozenset[str]

    def parse(self, content: bytes) -> ParsedDocument: ...


class ParserRegistry:
    def __init__(self, parsers: tuple[DocumentParser, ...]) -> None:
        self._by_extension: dict[str, DocumentParser] = {}
        for parser in parsers:
            for extension in parser.supported_extensions:
                self._by_extension[extension] = parser

    def for_filename(self, filename: str) -> DocumentParser:
        extension = Path(filename).suffix.lower()
        try:
            return self._by_extension[extension]
        except KeyError as exc:
            supported = ", ".join(sorted(self._by_extension))
            raise UnsupportedDocumentError(
                f"unsupported file type '{extension or '<none>'}'; supported: {supported}"
            ) from exc
