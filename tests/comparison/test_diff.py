from app.comparison.service import sentence_diff


def test_sentence_diff_preserves_source_language() -> None:
    result = sentence_diff(
        "Nhân viên được nghỉ mười ngày. Quy định khác giữ nguyên.",
        "Nhân viên được nghỉ mười hai ngày. Quy định khác giữ nguyên.",
    )

    assert result.removed == ["Nhân viên được nghỉ mười ngày."]
    assert result.added == ["Nhân viên được nghỉ mười hai ngày."]
