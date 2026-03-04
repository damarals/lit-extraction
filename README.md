# LitReview

CLI pipeline for extracting structured study data and quality-assessment scores
from PDFs using multiple LLMs via OpenRouter, then consolidating with Gemini.

The pipeline first runs OCR with `lightonai/LightOnOCR-2-1B`, then sends OCR text
to the LLMs for extraction and QA.

## Configuration

Create a `.env` (or export environment variables):

```bash
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_MINIMAX=minimax/minimax-m2.5
MODEL_KIMI=moonshotai/kimi-k2.5
MODEL_GLM=z-ai/glm-5
MODEL_GEMINI=google/gemini-2.5-flash
MODEL_TEMPERATURE=0.2
MODEL_TOP_P=0.9
MODEL_TIMEOUT_SECONDS=90
MODEL_MAX_RETRIES=2
OCR_MODEL_ID=lightonai/LightOnOCR-2-1B
OCR_MAX_NEW_TOKENS=2048
OCR_PAGE_SCALE=2.77
OCR_MAX_PAGES=0
```

## CLI

```bash
litreview run --metadata data/papers.json --output outputs --max-workers 4
```

### Metadata format

```json
[
  {
    "paper_id": "optional-id",
    "title": "Paper title",
    "pdf_path": "/absolute/or/relative/path/to/paper.pdf"
  }
]
```

Output per paper is written under `outputs/<paper_id>/` with `raw/`, `consolidated/`, and `status.json`.
