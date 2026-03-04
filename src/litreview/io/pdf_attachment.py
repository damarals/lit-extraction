from __future__ import annotations

import base64
from pathlib import Path


class AttachmentValidationError(ValueError):
    """Raised when a PDF attachment cannot be prepared for model input."""


def build_pdf_attachment_content(pdf_path: str | Path) -> dict[str, str]:
    path = Path(pdf_path).resolve()
    if not path.exists() or not path.is_file():
        raise AttachmentValidationError(f"PDF not found: {path.as_posix()}")
    if path.suffix.lower() != ".pdf":
        raise AttachmentValidationError(
            f"Attachment must be a .pdf file: {path.as_posix()}"
        )

    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "file",
        "filename": path.name,
        "file_data": f"data:application/pdf;base64,{payload}",
    }
