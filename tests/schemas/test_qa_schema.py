from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_qa_score_bounds() -> None:
    from litreview.schemas.qa import QACriterion

    with pytest.raises(ValidationError):
        QACriterion(score_final=3, justification="x", evidence_text="y")


def test_qa_total_score_computed() -> None:
    from litreview.schemas.qa import QAFormV1

    payload = {
        "problem_formalization": {
            "score_final": 2,
            "justification": "Defined.",
            "evidence_text": "Formal objective."
        },
        "reproducibility": {
            "score_final": 1,
            "justification": "Partially described.",
            "evidence_text": "Pseudo-code only."
        },
        "computational_analysis": {
            "score_final": 2,
            "justification": "Runtime reported.",
            "evidence_text": "O(n log n)."
        },
        "data_driven_validation": {
            "score_final": 2,
            "justification": "Real data.",
            "evidence_text": "Urban AQ dataset."
        },
        "theoretical_guarantees": {
            "score_final": 0,
            "justification": "None.",
            "evidence_text": "No proof."
        },
        "baseline_comparison": {
            "score_final": 2,
            "justification": "Compared baselines.",
            "evidence_text": "Random/Grid."
        },
        "constraint_modeling": {
            "score_final": 1,
            "justification": "Basic constraints.",
            "evidence_text": "Budget only."
        },
        "appropriate_metrics": {
            "score_final": 2,
            "justification": "Good metrics.",
            "evidence_text": "RMSE, MAE."
        }
    }

    qa = QAFormV1.model_validate(payload)
    assert qa.total_score == 12
