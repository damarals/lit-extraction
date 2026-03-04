from __future__ import annotations


def test_extraction_prompt_mentions_pdf_attachment_and_json_only() -> None:
    from src.prompts.extraction_prompt import build_extraction_prompt

    prompt = build_extraction_prompt(
        title="Urban sensor placement",
        ocr_text="This is the extracted OCR text.",
    )
    assert "OCR text" in prompt
    assert "This is the extracted OCR text." in prompt
    assert "JSON" in prompt
    assert "no markdown" in prompt.lower()


def test_qa_prompt_mentions_score_range_and_evidence() -> None:
    from src.prompts.qa_prompt import build_qa_prompt

    prompt = build_qa_prompt(
        title="Urban sensor placement",
        ocr_text="Extracted text from OCR",
    )
    assert "0, 1, or 2" in prompt
    assert "evidence_text" in prompt


def test_consolidation_prompts_embed_model_outputs() -> None:
    from src.prompts.consolidation_prompt import (
        build_extraction_consolidation_prompt,
        build_qa_consolidation_prompt,
    )

    model_outputs = {
        "minimax": {"a": 1},
        "kimi": {"a": 2},
    }
    extraction_prompt = build_extraction_consolidation_prompt(model_outputs)
    qa_prompt = build_qa_consolidation_prompt(model_outputs)

    assert '"minimax": {' in extraction_prompt
    assert "synthesize a single final ExtractionFormV1" in extraction_prompt
    assert "score_final" in qa_prompt
