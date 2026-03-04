from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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


class MockClient:
    def chat_json(
        self,
        *,
        model: str,
        prompt: str,
        pdf_attachment: dict[str, str] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if "synthesize a single final ExtractionFormV1" in prompt:
            return _valid_extraction_payload()
        if "single QAFormV1 JSON where each criterion includes" in prompt:
            return _valid_qa_payload()
        if "Return a single JSON object matching ExtractionFormV1." in prompt:
            return _valid_extraction_payload()
        if "Score each QA criterion with 0, 1, or 2." in prompt:
            return _valid_qa_payload()
        raise AssertionError(f"Unexpected prompt: {prompt}")

    def close(self) -> None:
        return None


class MockOCR:
    def extract_text(self, _: str) -> str:
        return "OCR extracted content from all pages."


def test_e2e_mocked_run_creates_expected_output_tree(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%mock\n%%EOF\n")
    metadata_path = tmp_path / "papers.json"
    metadata_path.write_text(
        json.dumps(
            [
                {
                    "paper_id": "paper-1",
                    "title": "Urban placement",
                    "pdf_path": pdf.as_posix(),
                }
            ]
        ),
        encoding="utf-8",
    )

    from src.pipeline.stages import PipelineModels, run_pipeline

    summary = run_pipeline(
        metadata_path=metadata_path,
        output_root=tmp_path / "outputs",
        models=PipelineModels(
            minimax="provider/minimax",
            kimi="provider/kimi",
            glm="provider/glm",
            gemini="provider/gemini",
        ),
        client_factory=MockClient,
        ocr_extractor_factory=MockOCR,
        max_workers=1,
    )

    assert summary["total"] == 1
    assert summary["success"] == 1
    assert summary["partial"] == 0
    assert summary["failed"] == 0

    paper_dir = tmp_path / "outputs" / "paper-1"
    assert (paper_dir / "raw/minimax_extraction.json").exists()
    assert (paper_dir / "raw/kimi_extraction.json").exists()
    assert (paper_dir / "raw/glm_extraction.json").exists()
    assert (paper_dir / "raw/minimax_qa.json").exists()
    assert (paper_dir / "raw/kimi_qa.json").exists()
    assert (paper_dir / "raw/glm_qa.json").exists()
    assert (paper_dir / "consolidated/extraction.json").exists()
    assert (paper_dir / "consolidated/qa.json").exists()
    assert (paper_dir / "status.json").exists()
