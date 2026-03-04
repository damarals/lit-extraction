from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


class OCRError(RuntimeError):
    """Raised when OCR extraction cannot be completed."""


@dataclass
class LightOnOCRExtractor:
    model_id: str = "lightonai/LightOnOCR-2-1B"
    max_new_tokens: int = 2048
    page_scale: float = 2.77
    max_pages: int | None = None
    device: str | None = None

    _processor: Any = None
    _model: Any = None
    _torch: Any = None
    _load_lock: Lock = field(default_factory=Lock)
    _infer_lock: Lock = field(default_factory=Lock)

    def extract_text(self, pdf_path: str | Path) -> str:
        path = Path(pdf_path).resolve()
        if not path.exists() or not path.is_file():
            raise OCRError(f"PDF not found: {path.as_posix()}")
        if path.suffix.lower() != ".pdf":
            raise OCRError(f"OCR input must be a .pdf file: {path.as_posix()}")

        try:
            import pypdfium2 as pdfium  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise OCRError(
                "Missing dependency 'pypdfium2'. Install OCR dependencies first."
            ) from exc

        try:
            pdf = pdfium.PdfDocument(path.as_posix())
        except Exception as exc:  # noqa: BLE001
            raise OCRError(f"Unable to open PDF for OCR: {path.as_posix()}") from exc

        page_count = len(pdf)
        if page_count == 0:
            raise OCRError(f"PDF has no pages: {path.as_posix()}")

        limit = page_count if not self.max_pages else min(page_count, self.max_pages)
        chunks: list[str] = []
        for index in range(limit):
            page = pdf[index]
            image = page.render(scale=self.page_scale).to_pil()
            text = self._ocr_image(image).strip()
            if text:
                chunks.append(text)

        if not chunks:
            raise OCRError(f"OCR produced no text: {path.as_posix()}")
        return "\n\n".join(chunks)

    def _load_model(self) -> None:
        if (
            self._processor is not None
            and self._model is not None
            and self._torch is not None
        ):
            return

        with self._load_lock:
            if (
                self._processor is not None
                and self._model is not None
                and self._torch is not None
            ):
                return

            try:
                import torch  # type: ignore
                from transformers import (  # type: ignore
                    LightOnOcrForConditionalGeneration,
                    LightOnOcrProcessor,
                )
            except Exception as exc:  # noqa: BLE001
                raise OCRError(
                    "Missing or incompatible OCR dependencies. "
                    "Install torch + transformers>=5 for LightOnOCR."
                ) from exc

            chosen_device = self.device
            if chosen_device is None:
                chosen_device = "cuda" if torch.cuda.is_available() else "cpu"

            dtype = torch.bfloat16 if chosen_device == "cuda" else torch.float32
            processor = LightOnOcrProcessor.from_pretrained(
                self.model_id,
                use_fast=False,
            )
            model = LightOnOcrForConditionalGeneration.from_pretrained(
                self.model_id,
                dtype=dtype,
            )
            if chosen_device != "cpu":
                model = model.to(chosen_device)
            model.eval()

            self._processor = processor
            self._model = model
            self._torch = torch

    def _ocr_image(self, pil_image: Any) -> str:
        self._load_model()
        assert self._processor is not None
        assert self._model is not None

        image_url = _pil_image_to_data_url(pil_image)
        conversation = [{"role": "user", "content": [{"type": "image", "url": image_url}]}]
        with self._infer_lock:
            inputs = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            for key, tensor in inputs.items():
                if key == "pixel_values":
                    tensor = tensor.to(self._model.dtype)
                inputs[key] = tensor.to(self._model.device)

            with self._torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return self._processor.decode(generated_ids, skip_special_tokens=True)


def _pil_image_to_data_url(pil_image: Any) -> str:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
