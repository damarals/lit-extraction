from __future__ import annotations

from pathlib import Path

import pytest


def test_load_metadata_validates_pdf_path(tmp_path: Path) -> None:
    metadata = tmp_path / "papers.json"
    metadata.write_text('[{"title":"A","pdf_path":"missing.pdf"}]', encoding="utf-8")

    from src.io.metadata import MetadataValidationError, load_metadata

    with pytest.raises(MetadataValidationError):
        load_metadata(metadata)


def test_load_metadata_generates_paper_id_and_absolute_path(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    metadata = tmp_path / "papers.json"
    metadata.write_text(
        '[{"title":"Paper A","pdf_path":"paper.pdf"}]',
        encoding="utf-8",
    )

    from src.io.metadata import load_metadata

    papers = load_metadata(metadata)
    assert len(papers) == 1
    assert papers[0].paper_id
    assert Path(papers[0].pdf_path).is_absolute()
