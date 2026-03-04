from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.consolidation.gemini import (
    consolidate_extraction_form,
    consolidate_qa_form,
)
from src.io.metadata import PaperMetadata, load_metadata
from src.io.ocr_lighton import OCRError
from src.persistence.filesystem import save_paper_outputs
from src.prompts.extraction_prompt import build_extraction_prompt
from src.prompts.qa_prompt import build_qa_prompt
from src.schemas.extraction import ExtractionFormV1
from src.schemas.qa import QAFormV1


@dataclass(frozen=True)
class PipelineModels:
    minimax: str
    kimi: str
    glm: str
    gemini: str

    def extraction_models(self) -> dict[str, str]:
        return {
            "minimax": self.minimax,
            "kimi": self.kimi,
            "glm": self.glm,
        }


@dataclass(frozen=True)
class PaperRunResult:
    paper_id: str
    status: str
    partial: bool
    errors: list[str]
    output_dir: Path


def _validate_extraction(payload: dict[str, Any]) -> dict[str, Any]:
    return ExtractionFormV1.model_validate(payload).model_dump(mode="json")


def _validate_qa(payload: dict[str, Any]) -> dict[str, Any]:
    return QAFormV1.model_validate(payload).model_dump(mode="json")


def _run_form_models(
    *,
    client: Any,
    models: dict[str, str],
    prompt: str,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
    form_name: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    valid_outputs: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    def call_and_validate(alias: str, model_name: str) -> tuple[str, dict[str, Any]]:
        data = client.chat_json(
            model=model_name,
            prompt=prompt,
        )
        return alias, validator(data)

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(call_and_validate, alias, model_name): alias
            for alias, model_name in models.items()
        }
        for future in as_completed(futures):
            alias = futures[future]
            try:
                model_alias, validated = future.result()
                valid_outputs[model_alias] = validated
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{form_name}:{alias} failed: {exc}")

    return valid_outputs, errors


def process_paper(
    *,
    paper: PaperMetadata,
    output_root: str | Path,
    client: Any,
    ocr_extractor: Any,
    models: PipelineModels,
) -> PaperRunResult:
    errors: list[str] = []
    raw_extraction: dict[str, dict[str, Any]] = {}
    raw_qa: dict[str, dict[str, Any]] = {}
    consolidated_extraction: dict[str, Any] | None = None
    consolidated_qa: dict[str, Any] | None = None

    try:
        ocr_text = ocr_extractor.extract_text(paper.pdf_path)
    except OCRError as exc:
        errors.append(str(exc))
        output_dir = save_paper_outputs(
            output_root=output_root,
            paper_id=paper.paper_id,
            metadata=paper.model_dump(mode="json"),
            raw_extraction=raw_extraction,
            raw_qa=raw_qa,
            consolidated_extraction=consolidated_extraction,
            consolidated_qa=consolidated_qa,
            status="failed",
            partial=False,
            errors=errors,
        )
        return PaperRunResult(
            paper_id=paper.paper_id,
            status="failed",
            partial=False,
            errors=errors,
            output_dir=output_dir,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        output_dir = save_paper_outputs(
            output_root=output_root,
            paper_id=paper.paper_id,
            metadata=paper.model_dump(mode="json"),
            raw_extraction=raw_extraction,
            raw_qa=raw_qa,
            consolidated_extraction=consolidated_extraction,
            consolidated_qa=consolidated_qa,
            status="failed",
            partial=False,
            errors=errors,
        )
        return PaperRunResult(
            paper_id=paper.paper_id,
            status="failed",
            partial=False,
            errors=errors,
            output_dir=output_dir,
        )

    extraction_prompt = build_extraction_prompt(title=paper.title, ocr_text=ocr_text)
    raw_extraction, extraction_errors = _run_form_models(
        client=client,
        models=models.extraction_models(),
        prompt=extraction_prompt,
        validator=_validate_extraction,
        form_name="extraction",
    )
    errors.extend(extraction_errors)
    if not raw_extraction:
        output_dir = save_paper_outputs(
            output_root=output_root,
            paper_id=paper.paper_id,
            metadata=paper.model_dump(mode="json"),
            raw_extraction=raw_extraction,
            raw_qa=raw_qa,
            consolidated_extraction=consolidated_extraction,
            consolidated_qa=consolidated_qa,
            status="failed",
            partial=False,
            errors=errors,
        )
        return PaperRunResult(
            paper_id=paper.paper_id,
            status="failed",
            partial=False,
            errors=errors,
            output_dir=output_dir,
        )

    qa_prompt = build_qa_prompt(title=paper.title, ocr_text=ocr_text)
    raw_qa, qa_errors = _run_form_models(
        client=client,
        models=models.extraction_models(),
        prompt=qa_prompt,
        validator=_validate_qa,
        form_name="qa",
    )
    errors.extend(qa_errors)
    if not raw_qa:
        output_dir = save_paper_outputs(
            output_root=output_root,
            paper_id=paper.paper_id,
            metadata=paper.model_dump(mode="json"),
            raw_extraction=raw_extraction,
            raw_qa=raw_qa,
            consolidated_extraction=consolidated_extraction,
            consolidated_qa=consolidated_qa,
            status="failed",
            partial=False,
            errors=errors,
        )
        return PaperRunResult(
            paper_id=paper.paper_id,
            status="failed",
            partial=False,
            errors=errors,
            output_dir=output_dir,
        )

    try:
        consolidated_extraction = consolidate_extraction_form(
            client=client,
            gemini_model=models.gemini,
            model_outputs=raw_extraction,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"consolidation:extraction failed: {exc}")

    try:
        consolidated_qa = consolidate_qa_form(
            client=client,
            gemini_model=models.gemini,
            model_outputs=raw_qa,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"consolidation:qa failed: {exc}")

    failed = consolidated_extraction is None or consolidated_qa is None
    partial = bool(errors) and not failed
    status = "failed" if failed else ("partial" if partial else "success")

    output_dir = save_paper_outputs(
        output_root=output_root,
        paper_id=paper.paper_id,
        metadata=paper.model_dump(mode="json"),
        raw_extraction=raw_extraction,
        raw_qa=raw_qa,
        consolidated_extraction=consolidated_extraction,
        consolidated_qa=consolidated_qa,
        status=status,
        partial=partial,
        errors=errors,
    )
    return PaperRunResult(
        paper_id=paper.paper_id,
        status=status,
        partial=partial,
        errors=errors,
        output_dir=output_dir,
    )


def run_pipeline(
    *,
    metadata_path: str | Path,
    output_root: str | Path,
    models: PipelineModels,
    client_factory: Callable[[], Any],
    ocr_extractor_factory: Callable[[], Any],
    max_workers: int = 4,
) -> dict[str, Any]:
    papers = load_metadata(metadata_path)
    results: list[PaperRunResult] = []
    shared_ocr_extractor = ocr_extractor_factory()

    def run_single(paper: PaperMetadata) -> PaperRunResult:
        client = client_factory()
        try:
            return process_paper(
                paper=paper,
                output_root=output_root,
                client=client,
                ocr_extractor=shared_ocr_extractor,
                models=models,
            )
        finally:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                close_fn()

    if max_workers <= 1:
        for paper in papers:
            results.append(run_single(paper))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single, paper): paper.paper_id for paper in papers}
            for future in as_completed(futures):
                results.append(future.result())

    total = len(results)
    success = len([result for result in results if result.status == "success"])
    partial = len([result for result in results if result.status == "partial"])
    failed = len([result for result in results if result.status == "failed"])

    return {
        "total": total,
        "success": success,
        "partial": partial,
        "failed": failed,
        "papers": [
            {
                "paper_id": result.paper_id,
                "status": result.status,
                "partial": result.partial,
                "errors": result.errors,
                "output_dir": result.output_dir.as_posix(),
            }
            for result in results
        ],
    }
