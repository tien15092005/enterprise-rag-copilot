import hashlib
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from app.ingestion.storage import LocalFilesystemStorage


async def test_local_storage_calculates_checksum_and_controls_path(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(tmp_path / "uploads", max_bytes=1024)

    stored = await storage.save(
        BytesIO(b"same bytes"),
        document_version_id=uuid4(),
        original_filename="../../unsafe policy.txt",
    )

    assert stored.original_filename == "unsafe policy.txt"
    assert stored.checksum == hashlib.sha256(b"same bytes").hexdigest()
    assert ".." not in stored.storage_key
    assert await storage.read(stored.storage_key) == b"same bytes"
