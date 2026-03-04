from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from src.cli import app
from src.config import RuntimeConfig


def test_cli_run_processes_metadata_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    metadata = tmp_path / "papers.json"
    metadata.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "outputs"

    captured: dict[str, object] = {}

    def fake_load_runtime_config() -> RuntimeConfig:
        return RuntimeConfig(
            openrouter_api_key="k",
            openrouter_base_url="https://openrouter.ai/api/v1",
            model_minimax="a/minimax",
            model_kimi="a/kimi",
            model_glm="a/glm",
            model_gemini="a/gemini",
            model_timeout_seconds=90.0,
            model_max_retries=2,
            model_temperature=0.2,
            model_top_p=0.9,
            ocr_model_id="lightonai/LightOnOCR-2-1B",
            ocr_max_new_tokens=2048,
            ocr_page_scale=2.77,
            ocr_max_pages=None,
        )

    def fake_run_pipeline(
        *,
        metadata_path: str,
        output_root: str,
        models,
        client_factory,
        ocr_extractor_factory,
        max_workers: int,
    ) -> dict[str, int]:
        captured["metadata_path"] = metadata_path
        captured["output_root"] = output_root
        captured["max_workers"] = max_workers
        captured["models"] = models
        return {
            "total": 1,
            "success": 1,
            "partial": 0,
            "failed": 0,
        }

    monkeypatch.setattr("src.cli.load_runtime_config", fake_load_runtime_config)
    monkeypatch.setattr("src.cli.run_pipeline", fake_run_pipeline)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--metadata",
            metadata.as_posix(),
            "--output",
            output_dir.as_posix(),
            "--max-workers",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "total=1 success=1 partial=0 failed=0" in result.output
    assert captured["metadata_path"] == metadata.as_posix()
    assert captured["output_root"] == output_dir.as_posix()
    assert captured["max_workers"] == 2
