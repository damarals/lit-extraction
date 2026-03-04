from __future__ import annotations

import base64
from pathlib import Path

import pytest


def test_build_pdf_attachment_content_for_valid_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_bytes = b"%PDF-1.4\n%mock\n%%EOF\n"
    pdf_path.write_bytes(pdf_bytes)

    from litreview.io.pdf_attachment import build_pdf_attachment_content

    content = build_pdf_attachment_content(pdf_path)

    assert content["type"] == "file"
    assert content["filename"] == "paper.pdf"
    assert content["file_data"].startswith("data:application/pdf;base64,")
    encoded = content["file_data"].split(",", maxsplit=1)[1]
    assert base64.b64decode(encoded) == pdf_bytes


def test_build_pdf_attachment_rejects_missing_file(tmp_path: Path) -> None:
    from litreview.io.pdf_attachment import AttachmentValidationError, build_pdf_attachment_content

    with pytest.raises(AttachmentValidationError):
        build_pdf_attachment_content(tmp_path / "missing.pdf")


def test_build_pdf_attachment_rejects_non_pdf(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello", encoding="utf-8")

    from litreview.io.pdf_attachment import AttachmentValidationError, build_pdf_attachment_content

    with pytest.raises(AttachmentValidationError):
        build_pdf_attachment_content(text_file)
