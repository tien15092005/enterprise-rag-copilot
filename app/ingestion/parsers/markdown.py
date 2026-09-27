from __future__ import annotations

import re

from app.ingestion.errors import UnsupportedDocumentError
from app.ingestion.parsers.base import ParsedBlock, ParsedDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class MarkdownParser:
    supported_extensions = frozenset({".md", ".markdown"})

    def parse(self, content: bytes) -> ParsedDocument:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UnsupportedDocumentError("Markdown files must be UTF-8 encoded") from exc

        section_stack: list[str] = []
        paragraph_lines: list[str] = []
        blocks: list[ParsedBlock] = []

        def flush_paragraph() -> None:
            paragraph = "\n".join(paragraph_lines).strip()
            if paragraph:
                blocks.append(ParsedBlock(paragraph, section_path=tuple(section_stack)))
            paragraph_lines.clear()

        for line in text.replace("\r\n", "\n").splitlines():
            heading = _HEADING.match(line)
            if heading:
                flush_paragraph()
                level = len(heading.group(1))
                title = heading.group(2).strip()
                section_stack[level - 1 :] = [title]
            elif line.strip():
                paragraph_lines.append(line)
            else:
                flush_paragraph()
        flush_paragraph()

        if not blocks:
            raise UnsupportedDocumentError("document contains no extractable text")
        return ParsedDocument(blocks=tuple(blocks))
