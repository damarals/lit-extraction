from __future__ import annotations

from typing import Any

from src.prompts.consolidation_prompt import (
    build_extraction_consolidation_prompt,
    build_qa_consolidation_prompt,
)
from src.schemas.extraction import ExtractionFormV1
from src.schemas.qa import QAFormV1


def consolidate_extraction_form(
    *,
    client: Any,
    gemini_model: str,
    model_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    prompt = build_extraction_consolidation_prompt(model_outputs)
    payload = client.chat_json(model=gemini_model, prompt=prompt)
    return ExtractionFormV1.model_validate(payload).model_dump(mode="json")


def consolidate_qa_form(
    *,
    client: Any,
    gemini_model: str,
    model_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    prompt = build_qa_consolidation_prompt(model_outputs)
    payload = client.chat_json(model=gemini_model, prompt=prompt)
    return QAFormV1.model_validate(payload).model_dump(mode="json")
