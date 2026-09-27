import json
import logging

from app.core.logging import JsonFormatter


def test_json_formatter_emits_structured_event_and_unicode() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="xử lý tài liệu",
        args=(),
        exc_info=None,
    )
    record.event = "document_processed"
    record.document_version_id = "version-1"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["event"] == "document_processed"
    assert payload["message"] == "xử lý tài liệu"
    assert payload["document_version_id"] == "version-1"
    assert payload["level"] == "INFO"
