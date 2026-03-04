from __future__ import annotations

import typer

from litreview.config import load_runtime_config
from litreview.io.ocr_lighton import LightOnOCRExtractor
from litreview.pipeline.stages import PipelineModels, run_pipeline
from litreview.providers.openrouter import OpenRouterClient

app = typer.Typer(help="LitReview multi-LLM extraction pipeline.")


@app.callback()
def main() -> None:
    """LitReview command group."""


@app.command("run")
def run(
    metadata: str = typer.Option(..., "--metadata", help="Path to metadata JSON."),
    output: str = typer.Option("outputs", "--output", help="Output directory."),
    max_workers: int = typer.Option(4, "--max-workers", min=1, help="Worker count."),
) -> None:
    """Run the full multi-model extraction pipeline."""
    try:
        config = load_runtime_config()
    except ValueError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    models = PipelineModels(
        minimax=config.model_minimax,
        kimi=config.model_kimi,
        glm=config.model_glm,
        gemini=config.model_gemini,
    )

    def client_factory() -> OpenRouterClient:
        return OpenRouterClient(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            timeout_seconds=config.model_timeout_seconds,
            max_retries=config.model_max_retries,
            temperature=config.model_temperature,
            top_p=config.model_top_p,
        )

    def ocr_extractor_factory() -> LightOnOCRExtractor:
        return LightOnOCRExtractor(
            model_id=config.ocr_model_id,
            max_new_tokens=config.ocr_max_new_tokens,
            page_scale=config.ocr_page_scale,
            max_pages=config.ocr_max_pages,
        )

    summary = run_pipeline(
        metadata_path=metadata,
        output_root=output,
        models=models,
        client_factory=client_factory,
        ocr_extractor_factory=ocr_extractor_factory,
        max_workers=max_workers,
    )
    typer.echo(
        " ".join(
            [
                f"total={summary['total']}",
                f"success={summary['success']}",
                f"partial={summary['partial']}",
                f"failed={summary['failed']}",
            ]
        )
    )
