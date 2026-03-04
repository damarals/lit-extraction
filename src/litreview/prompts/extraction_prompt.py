from __future__ import annotations


def build_extraction_prompt(*, title: str, ocr_text: str) -> str:
    return (
        "You are extracting structured study information from OCR text.\n"
        f"Paper title: {title}\n"
        "Use only the OCR text below as source of truth.\n"
        "OCR text:\n"
        f"{ocr_text}\n"
        "Return a single JSON object matching ExtractionFormV1.\n"
        "If a field is not reported, use null or 'Not reported' where appropriate.\n"
        "Output JSON only, no markdown, no commentary."
    )
