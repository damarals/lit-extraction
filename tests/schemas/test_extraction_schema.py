from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_extraction_requires_bibliographic() -> None:
    from src.schemas.extraction import ExtractionFormV1

    with pytest.raises(ValidationError):
        ExtractionFormV1.model_validate({})


def test_extraction_accepts_minimal_valid_payload() -> None:
    from src.schemas.extraction import ExtractionFormV1

    payload = {
        "bibliographic": {
            "id": "10.1000/example",
            "authors": ["A. One", "B. Two"],
            "year": 2024,
            "venue": "Example Journal",
            "quality_proxy": "Q1"
        },
        "context": {
            "study_area": "Sao Paulo",
            "area_size_km2": 1521.0,
            "pollutants": ["PM2.5", "NO2"],
            "data_source": "Reference stations",
            "auxiliary_data": ["Meteorology"],
            "spatial_resolution_m": 500,
            "temporal_aspect": "Static"
        },
        "modeling": {
            "modeling_paradigm": "Gaussian Process",
            "aq_data_dependency": "Data-driven"
        },
        "optimization": {
            "algorithm_class": "Metaheuristic",
            "specific_method": "NSGA-II",
            "objective_function": [
                "Coverage",
                "Prediction Error"
            ],
            "multi_objective": {
                "type": "Multiple",
                "objectives": ["Coverage", "RMSE"],
                "aggregation_method": "Pareto"
            },
            "complexity": "Not reported",
            "software_tools": ["Python", "Gurobi"]
        },
        "network_design": {
            "design_objective": "Greenfield",
            "sensor_heterogeneity": "Homogeneous",
            "sensors_initial": "N/A",
            "sensors_final": 20,
            "equity_aware": "No",
            "equity_metric": None
        },
        "constraints": {
            "budget_cost": {
                "modeled": "Yes",
                "description": "Sensor count limit"
            },
            "connectivity": {
                "modeled": "No",
                "description": None
            },
            "deployment": ["Accessibility"],
            "candidate_sites": "Grid-based"
        },
        "validation": {
            "validation_type": "Simulation (real AQ data)",
            "validation_strategy": "Cross-validation",
            "performance_metrics": ["RMSE", "MAE"],
            "baseline_comparisons": ["Random"],
            "optimality_guarantees": "None reported"
        },
        "reproducibility": {
            "code_availability": "Not available",
            "data_availability": "Not available",
            "limitations": "Small study area."
        }
    }

    model = ExtractionFormV1.model_validate(payload)
    assert model.bibliographic.year == 2024
