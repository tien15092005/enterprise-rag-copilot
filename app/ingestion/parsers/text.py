from __future__ import annotations

from app.ingestion.errors import UnsupportedDocumentError
from app.ingestion.parsers.base import ParsedBlock, ParsedDocument


class TextParser:
    supported_extensions = frozenset({".txt"})

    def parse(self, content: bytes) -> ParsedDocument:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UnsupportedDocumentError("TXT files must be UTF-8 encoded") from exc

        blocks = tuple(
            ParsedBlock(text=paragraph.strip())
            for paragraph in text.replace("\r\n", "\n").split("\n\n")
            if paragraph.strip()
        )
        if not blocks:
            raise UnsupportedDocumentError("document contains no extractable text")
        return ParsedDocument(blocks=blocks)
