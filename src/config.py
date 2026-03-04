from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Load environment variables from .env file if it exists."""
    env_path = Path(".env")
    if not env_path.exists():
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Don't override if already set in environment
                if key not in os.environ:
                    os.environ[key] = value


# Load .env on module import
_load_dotenv()


@dataclass(frozen=True)
class RuntimeConfig:
    openrouter_api_key: str
    openrouter_base_url: str
    model_minimax: str
    model_kimi: str
    model_glm: str
    model_gemini: str
    model_timeout_seconds: float
    model_max_retries: int
    model_temperature: float
    model_top_p: float
    ocr_model_id: str
    ocr_max_new_tokens: int
    ocr_page_scale: float
    ocr_max_pages: int | None


def load_runtime_config() -> RuntimeConfig:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required.")

    return RuntimeConfig(
        openrouter_api_key=api_key,
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        model_minimax=os.getenv("MODEL_MINIMAX", "minimax/minimax-m2.5"),
        model_kimi=os.getenv("MODEL_KIMI", "moonshotai/kimi-k2"),
        model_glm=os.getenv("MODEL_GLM", "zhipuai/glm-4.5"),
        model_gemini=os.getenv("MODEL_GEMINI", "google/gemini-2.5-pro"),
        model_timeout_seconds=float(os.getenv("MODEL_TIMEOUT_SECONDS", "90")),
        model_max_retries=int(os.getenv("MODEL_MAX_RETRIES", "2")),
        model_temperature=float(os.getenv("MODEL_TEMPERATURE", "0.2")),
        model_top_p=float(os.getenv("MODEL_TOP_P", "0.9")),
        ocr_model_id=os.getenv("OCR_MODEL_ID", "lightonai/LightOnOCR-2-1B"),
        ocr_max_new_tokens=int(os.getenv("OCR_MAX_NEW_TOKENS", "2048")),
        ocr_page_scale=float(os.getenv("OCR_PAGE_SCALE", "2.77")),
        ocr_max_pages=(
            int(os.getenv("OCR_MAX_PAGES", "0"))
            if os.getenv("OCR_MAX_PAGES", "0") != "0"
            else None
        ),
    )
