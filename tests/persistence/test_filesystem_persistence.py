from __future__ import annotations

import json
from pathlib import Path


def test_persistence_writes_raw_and_consolidated(tmp_path: Path) -> None:
    from litreview.persistence.filesystem import save_paper_outputs

    out_dir = save_paper_outputs(
        output_root=tmp_path,
        paper_id="paper-1",
        metadata={"title": "A", "pdf_path": "/tmp/a.pdf"},
        raw_extraction={
            "minimax": {"field": "v1"},
            "kimi": {"field": "v2"},
        },
        raw_qa={
            "minimax": {"criterion": 2},
            "kimi": {"criterion": 1},
        },
        consolidated_extraction={"field": "final"},
        consolidated_qa={"criterion": {"score_final": 2}},
        status="partial",
        partial=True,
        errors=["kimi timeout"],
    )

    assert (out_dir / "raw/minimax_extraction.json").exists()
    assert (out_dir / "raw/kimi_extraction.json").exists()
    assert (out_dir / "raw/minimax_qa.json").exists()
    assert (out_dir / "consolidated/extraction.json").exists()
    assert (out_dir / "consolidated/qa.json").exists()

    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "partial"
    assert status["partial"] is True
    assert "kimi timeout" in status["errors"]
