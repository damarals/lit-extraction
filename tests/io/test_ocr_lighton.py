from __future__ import annotations

from pathlib import Path

import pytest


def test_lighton_ocr_rejects_missing_pdf(tmp_path: Path) -> None:
    from litreview.io.ocr_lighton import LightOnOCRExtractor, OCRError

    extractor = LightOnOCRExtractor()
    with pytest.raises(OCRError):
        extractor.extract_text(tmp_path / "missing.pdf")


def test_lighton_ocr_rejects_non_pdf(tmp_path: Path) -> None:
    from litreview.io.ocr_lighton import LightOnOCRExtractor, OCRError

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello", encoding="utf-8")
    extractor = LightOnOCRExtractor()

    with pytest.raises(OCRError):
        extractor.extract_text(file_path)
