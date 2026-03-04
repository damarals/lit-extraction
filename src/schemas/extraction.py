from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.common import YesNo


class Bibliographic(BaseModel):
    id: str
    authors: list[str] = Field(min_length=1)
    year: int
    venue: str
    quality_proxy: str | None = None


class Context(BaseModel):
    study_area: str
    area_size_km2: float | None = None
    pollutants: list[str] = Field(min_length=1)
    data_source: str
    auxiliary_data: list[str] = Field(default_factory=list)
    spatial_resolution_m: int | float | None = None
    temporal_aspect: Literal["Static", "Dynamic", "Adaptive"]


class Modeling(BaseModel):
    modeling_paradigm: str
    aq_data_dependency: Literal["Field-free", "Data-driven", "Model-based"]


class MultiObjective(BaseModel):
    type: Literal["Single", "Multiple"]
    objectives: list[str] = Field(default_factory=list)
    aggregation_method: str | None = None


class Optimization(BaseModel):
    algorithm_class: str
    specific_method: str
    objective_function: list[str] = Field(min_length=1)
    multi_objective: MultiObjective
    complexity: str | None = None
    software_tools: list[str] = Field(default_factory=list)


class NetworkDesign(BaseModel):
    design_objective: Literal[
        "Greenfield", "Expansion", "Redistribution", "Reduction/Pruning"
    ]
    sensor_heterogeneity: Literal["Homogeneous", "Heterogeneous", "Multi-pollutant"]
    sensors_initial: int | str | None = None
    sensors_final: int | None = None
    equity_aware: YesNo
    equity_metric: str | None = None


class ModeledConstraint(BaseModel):
    modeled: YesNo
    description: str | None = None


class Constraints(BaseModel):
    budget_cost: ModeledConstraint
    connectivity: ModeledConstraint
    deployment: list[str] = Field(default_factory=list)
    candidate_sites: str


class Validation(BaseModel):
    validation_type: str
    validation_strategy: str
    performance_metrics: list[str] = Field(default_factory=list)
    baseline_comparisons: list[str] = Field(default_factory=list)
    optimality_guarantees: str | None = None


class Reproducibility(BaseModel):
    code_availability: str
    data_availability: str
    limitations: str | None = None


class ExtractionFormV1(BaseModel):
    bibliographic: Bibliographic
    context: Context
    modeling: Modeling
    optimization: Optimization
    network_design: NetworkDesign
    constraints: Constraints
    validation: Validation
    reproducibility: Reproducibility
