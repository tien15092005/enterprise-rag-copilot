from __future__ import annotations

from io import BytesIO

from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from app.ingestion.errors import UnsupportedDocumentError
from app.ingestion.parsers.base import ParsedBlock, ParsedDocument


class DocxParser:
    supported_extensions = frozenset({".docx"})

    def parse(self, content: bytes) -> ParsedDocument:
        try:
            document = Document(BytesIO(content))
        except (PackageNotFoundError, ValueError) as exc:
            raise UnsupportedDocumentError("invalid DOCX document") from exc

        section_stack: list[str] = []
        blocks: list[ParsedBlock] = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            style_name = paragraph.style.name if paragraph.style is not None else ""
            if style_name.startswith("Heading "):
                try:
                    level = int(style_name.removeprefix("Heading "))
                except ValueError:
                    level = 1
                section_stack[level - 1 :] = [text]
                continue
            blocks.append(ParsedBlock(text=text, section_path=tuple(section_stack)))

        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    blocks.append(ParsedBlock(" | ".join(cells), section_path=tuple(section_stack)))

        if not blocks:
            raise UnsupportedDocumentError("DOCX contains no extractable text")
        return ParsedDocument(blocks=tuple(blocks))
