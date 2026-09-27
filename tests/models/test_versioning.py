from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models.versioning import (
    DocumentLanguage,
    DocumentVersionSnapshot,
    VersionBusinessStatus,
    VersionIndexingStatus,
    VersionSelector,
    VersionSelectorKind,
)


def make_snapshot(**overrides: object) -> DocumentVersionSnapshot:
    values: dict[str, object] = {
        "document_version_id": uuid4(),
        "logical_document_id": uuid4(),
        "version_label": "2026",
        "version_ordinal": 2,
        "language": DocumentLanguage.VIETNAMESE,
        "source_filename": "HR_Policy_2026.pdf",
        "content_sha256": "a" * 64,
        "business_status": VersionBusinessStatus.DRAFT,
        "indexing_status": VersionIndexingStatus.PENDING,
    }
    values.update(overrides)
    return DocumentVersionSnapshot.model_validate(values)


def test_active_version_must_be_ready() -> None:
    with pytest.raises(ValidationError, match="must be ready"):
        make_snapshot(business_status="active", indexing_status="processing")


def test_active_ready_version_is_valid_and_frozen() -> None:
    snapshot = make_snapshot(business_status="active", indexing_status="ready")

    assert snapshot.business_status is VersionBusinessStatus.ACTIVE
    with pytest.raises(ValidationError):
        snapshot.version_label = "silently-replaced"  # type: ignore[misc]


def test_effective_period_must_increase() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="effective_to"):
        make_snapshot(effective_from=now, effective_to=now - timedelta(days=1))


def test_current_selector_contains_no_fallback_fields() -> None:
    selector = VersionSelector()

    assert selector.kind is VersionSelectorKind.CURRENT
    assert selector.document_version_ids == ()
    assert selector.as_of is None


def test_exact_selector_requires_version_ids() -> None:
    with pytest.raises(ValidationError, match="requires at least one"):
        VersionSelector(kind="exact")
