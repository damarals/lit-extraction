from __future__ import annotations

from pydantic import BaseModel, Field, computed_field


class QACriterion(BaseModel):
    score_final: int = Field(ge=0, le=2)
    justification: str = Field(min_length=1)
    evidence_text: str = Field(min_length=1)


class QAFormV1(BaseModel):
    problem_formalization: QACriterion
    reproducibility: QACriterion
    computational_analysis: QACriterion
    data_driven_validation: QACriterion
    theoretical_guarantees: QACriterion
    baseline_comparison: QACriterion
    constraint_modeling: QACriterion
    appropriate_metrics: QACriterion

    @computed_field
    @property
    def total_score(self) -> int:
        return (
            self.problem_formalization.score_final
            + self.reproducibility.score_final
            + self.computational_analysis.score_final
            + self.data_driven_validation.score_final
            + self.theoretical_guarantees.score_final
            + self.baseline_comparison.score_final
            + self.constraint_modeling.score_final
            + self.appropriate_metrics.score_final
        )
