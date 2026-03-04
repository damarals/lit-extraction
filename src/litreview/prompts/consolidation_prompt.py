from __future__ import annotations

import json
from typing import Any


def build_extraction_consolidation_prompt(model_outputs: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(model_outputs, indent=2, ensure_ascii=True, sort_keys=True)
    return (
        "You are consolidating extraction outputs from multiple models.\n"
        "Given the JSON outputs below, synthesize a single final ExtractionFormV1.\n"
        "Return JSON only, no markdown.\n"
        f"Model outputs:\n{payload}"
    )


def build_qa_consolidation_prompt(model_outputs: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(model_outputs, indent=2, ensure_ascii=True, sort_keys=True)
    return (
        "You are consolidating QA outputs from multiple models.\n"
        "Return a single QAFormV1 JSON where each criterion includes:\n"
        "- score_final\n"
        "- justification\n"
        "- evidence_text\n"
        "Return JSON only, no markdown.\n"
        f"Model outputs:\n{payload}"
    )
