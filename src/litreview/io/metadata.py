from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class MetadataValidationError(ValueError):
    """Raised when metadata file structure or content is invalid."""


class PaperMetadata(BaseModel):
    paper_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    pdf_path: str = Field(min_length=1)


def _stable_paper_id(title: str, pdf_path: Path) -> str:
    raw = f"{title}|{pdf_path.as_posix()}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()
    return digest[:12]


def _normalize_entry(entry: dict[str, Any], base_dir: Path) -> PaperMetadata:
    title = str(entry.get("title", "")).strip()
    pdf_path_raw = str(entry.get("pdf_path", "")).strip()
    explicit_id = str(entry.get("paper_id", "")).strip()

    if not title:
        raise MetadataValidationError("Each metadata entry must include non-empty 'title'.")
    if not pdf_path_raw:
        raise MetadataValidationError(
            f"Metadata entry '{title}' must include non-empty 'pdf_path'."
        )

    pdf_path = Path(pdf_path_raw)
    if not pdf_path.is_absolute():
        pdf_path = (base_dir / pdf_path).resolve()

    if not pdf_path.exists() or not pdf_path.is_file():
        raise MetadataValidationError(
            f"PDF not found for '{title}': {pdf_path.as_posix()}"
        )

    paper_id = explicit_id or _stable_paper_id(title, pdf_path)
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        pdf_path=pdf_path.as_posix(),
    )


def load_metadata(metadata_path: str | Path) -> list[PaperMetadata]:
    path = Path(metadata_path).resolve()
    if not path.exists():
        raise MetadataValidationError(f"Metadata file not found: {path.as_posix()}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MetadataValidationError(
            f"Metadata file is not valid JSON: {path.as_posix()}"
        ) from exc

    if not isinstance(raw, list):
        raise MetadataValidationError("Metadata root must be a JSON array.")

    base_dir = path.parent
    return [_normalize_entry(entry, base_dir) for entry in raw]
