from __future__ import annotations

import asyncio
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Protocol
from uuid import UUID

from app.ingestion.errors import StoredFileTooLargeError

_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class StoredObject:
    storage_key: str
    checksum: str
    size_bytes: int
    original_filename: str


class DocumentStorage(Protocol):
    async def save(
        self,
        source: BinaryIO,
        *,
        document_version_id: UUID,
        original_filename: str,
    ) -> StoredObject: ...

    async def read(self, storage_key: str) -> bytes: ...

    async def delete(self, storage_key: str) -> None: ...


class LocalFilesystemStorage:
    """Controlled local storage implementation using opaque relative keys."""

    def __init__(self, root: Path, max_bytes: int) -> None:
        self._root = root.resolve()
        self._max_bytes = max_bytes
        self._root.mkdir(parents=True, exist_ok=True)

    async def save(
        self,
        source: BinaryIO,
        *,
        document_version_id: UUID,
        original_filename: str,
    ) -> StoredObject:
        return await asyncio.to_thread(
            self._save_sync,
            source,
            document_version_id,
            original_filename,
        )

    def _save_sync(
        self,
        source: BinaryIO,
        document_version_id: UUID,
        original_filename: str,
    ) -> StoredObject:
        display_name = Path(original_filename).name
        sanitized = _SAFE_FILENAME.sub("_", display_name).strip("._") or "document"
        storage_key = str(PurePosixPath("versions", str(document_version_id), sanitized))
        target = self._resolve_key(storage_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.uploading")
        checksum = hashlib.sha256()
        size_bytes = 0
        source.seek(0)

        try:
            with temporary.open("wb") as destination:
                while block := source.read(1024 * 1024):
                    size_bytes += len(block)
                    if size_bytes > self._max_bytes:
                        raise StoredFileTooLargeError(self._max_bytes)
                    checksum.update(block)
                    destination.write(block)
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

        return StoredObject(
            storage_key=storage_key,
            checksum=checksum.hexdigest(),
            size_bytes=size_bytes,
            original_filename=display_name,
        )

    async def read(self, storage_key: str) -> bytes:
        return await asyncio.to_thread(self._resolve_key(storage_key).read_bytes)

    async def delete(self, storage_key: str) -> None:
        path = self._resolve_key(storage_key)
        await asyncio.to_thread(path.unlink, True)

    def _resolve_key(self, storage_key: str) -> Path:
        candidate = (self._root / Path(PurePosixPath(storage_key))).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise ValueError("storage key escapes the configured upload directory")
        return candidate
