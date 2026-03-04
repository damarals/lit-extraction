from __future__ import annotations


def build_qa_prompt(*, title: str, ocr_text: str) -> str:
    return (
        "You are scoring study quality from OCR text.\n"
        f"Paper title: {title}\n"
        "Use only the OCR text below as source of truth.\n"
        "OCR text:\n"
        f"{ocr_text}\n"
        "Score each QA criterion with 0, 1, or 2.\n"
        "For each criterion, include score_final, justification, and evidence_text.\n"
        "Return JSON only, no markdown, no extra prose."
    )
