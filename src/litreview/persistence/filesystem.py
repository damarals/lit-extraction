from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_paper_outputs(
    *,
    output_root: str | Path,
    paper_id: str,
    metadata: dict[str, Any],
    raw_extraction: dict[str, dict[str, Any]],
    raw_qa: dict[str, dict[str, Any]],
    consolidated_extraction: dict[str, Any] | None,
    consolidated_qa: dict[str, Any] | None,
    status: str,
    partial: bool,
    errors: list[str] | None = None,
) -> Path:
    root = Path(output_root).resolve() / paper_id
    raw_dir = root / "raw"
    consolidated_dir = root / "consolidated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    consolidated_dir.mkdir(parents=True, exist_ok=True)

    _write_json(root / "input_metadata.json", metadata)

    for model, payload in raw_extraction.items():
        _write_json(raw_dir / f"{model}_extraction.json", payload)
    for model, payload in raw_qa.items():
        _write_json(raw_dir / f"{model}_qa.json", payload)

    if consolidated_extraction is not None:
        _write_json(consolidated_dir / "extraction.json", consolidated_extraction)
    if consolidated_qa is not None:
        _write_json(consolidated_dir / "qa.json", consolidated_qa)

    _write_json(
        root / "status.json",
        {
            "status": status,
            "partial": partial,
            "errors": errors or [],
        },
    )
    return root
