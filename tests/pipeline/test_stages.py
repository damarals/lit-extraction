from __future__ import annotations

from pathlib import Path
from typing import Any

from src.io.metadata import PaperMetadata


def _valid_extraction_payload() -> dict[str, Any]:
    return {
        "bibliographic": {
            "id": "10.1000/example",
            "authors": ["A. One"],
            "year": 2024,
            "venue": "Example Journal",
            "quality_proxy": "Q1",
        },
        "context": {
            "study_area": "Sao Paulo",
            "area_size_km2": 1521.0,
            "pollutants": ["PM2.5"],
            "data_source": "Reference stations",
            "auxiliary_data": ["Meteorology"],
            "spatial_resolution_m": 500,
            "temporal_aspect": "Static",
        },
        "modeling": {
            "modeling_paradigm": "Gaussian Process",
            "aq_data_dependency": "Data-driven",
        },
        "optimization": {
            "algorithm_class": "Metaheuristic",
            "specific_method": "NSGA-II",
            "objective_function": ["Coverage"],
            "multi_objective": {
                "type": "Single",
                "objectives": ["Coverage"],
                "aggregation_method": "Weighted sum",
            },
            "complexity": "Not reported",
            "software_tools": ["Python"],
        },
        "network_design": {
            "design_objective": "Greenfield",
            "sensor_heterogeneity": "Homogeneous",
            "sensors_initial": "N/A",
            "sensors_final": 20,
            "equity_aware": "No",
            "equity_metric": None,
        },
        "constraints": {
            "budget_cost": {"modeled": "Yes", "description": "Sensor count"},
            "connectivity": {"modeled": "No", "description": None},
            "deployment": ["Accessibility"],
            "candidate_sites": "Grid-based",
        },
        "validation": {
            "validation_type": "Simulation (real AQ data)",
            "validation_strategy": "Cross-validation",
            "performance_metrics": ["RMSE"],
            "baseline_comparisons": ["Random"],
            "optimality_guarantees": "None reported",
        },
        "reproducibility": {
            "code_availability": "Not available",
            "data_availability": "Not available",
            "limitations": "Small area",
        },
    }


def _valid_qa_payload() -> dict[str, Any]:
    criterion = {
        "score_final": 2,
        "justification": "Sufficient detail.",
        "evidence_text": "Evidence excerpt.",
    }
    return {
        "problem_formalization": criterion,
        "reproducibility": criterion,
        "computational_analysis": criterion,
        "data_driven_validation": criterion,
        "theoretical_guarantees": criterion,
        "baseline_comparison": criterion,
        "constraint_modeling": criterion,
        "appropriate_metrics": criterion,
    }


class FakeRouterClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat_json(
        self,
        *,
        model: str,
        prompt: str,
        pdf_attachment: dict[str, str] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "has_attachment": pdf_attachment is not None,
            }
        )
        if "synthesize a single final ExtractionFormV1" in prompt:
            return _valid_extraction_payload()
        if "single QAFormV1 JSON where each criterion includes" in prompt:
            return _valid_qa_payload()

        if "Return a single JSON object matching ExtractionFormV1." in prompt:
            if "kimi" in model:
                raise RuntimeError("kimi timeout")
            return _valid_extraction_payload()

        if "Score each QA criterion with 0, 1, or 2." in prompt:
            return _valid_qa_payload()

        raise AssertionError("Unexpected prompt path")


class FakeOCR:
    def __init__(self, text: str | None = None, error: str | None = None) -> None:
        self._text = text
        self._error = error

    def extract_text(self, _: str) -> str:
        if self._error is not None:
            raise RuntimeError(self._error)
        assert self._text is not None
        return self._text


def test_pipeline_marks_partial_when_one_model_fails(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test\n%%EOF\n")
    paper = PaperMetadata(
        paper_id="paper-1",
        title="Urban placement",
        pdf_path=pdf_path.as_posix(),
    )

    from src.pipeline.stages import PipelineModels, process_paper

    client = FakeRouterClient()
    result = process_paper(
        paper=paper,
        output_root=tmp_path / "outputs",
        client=client,
        ocr_extractor=FakeOCR(text="OCR TEXT"),
        models=PipelineModels(
            minimax="provider/minimax",
            kimi="provider/kimi",
            glm="provider/glm",
            gemini="provider/gemini",
        ),
    )

    assert result.status == "partial"
    assert result.partial is True
    assert (result.output_dir / "raw/minimax_extraction.json").exists()
    assert (result.output_dir / "raw/glm_extraction.json").exists()
    assert (result.output_dir / "consolidated/extraction.json").exists()
    assert (result.output_dir / "consolidated/qa.json").exists()
    assert any("kimi" in error for error in result.errors)


def test_pipeline_fails_when_ocr_fails(tmp_path: Path) -> None:
    bad_path = tmp_path / "paper.pdf"
    bad_path.write_bytes(b"%PDF-1.4\n%mock\n%%EOF\n")
    paper = PaperMetadata(
        paper_id="paper-2",
        title="Bad OCR",
        pdf_path=bad_path.as_posix(),
    )

    from src.pipeline.stages import PipelineModels, process_paper

    client = FakeRouterClient()
    result = process_paper(
        paper=paper,
        output_root=tmp_path / "outputs",
        client=client,
        ocr_extractor=FakeOCR(error="ocr failed"),
        models=PipelineModels(
            minimax="provider/minimax",
            kimi="provider/kimi",
            glm="provider/glm",
            gemini="provider/gemini",
        ),
    )

    assert result.status == "failed"
    assert result.partial is False
    assert len(client.calls) == 0
    assert any("ocr failed" in error for error in result.errors)
