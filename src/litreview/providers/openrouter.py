from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter request or response parsing fails."""


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 90.0,
        max_retries: int = 2,
        temperature: float | None = None,
        top_p: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._max_retries = max_retries
        self._temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("MODEL_TEMPERATURE", "0.2"))
        )
        self._top_p = (
            top_p if top_p is not None else float(os.getenv("MODEL_TOP_P", "0.9"))
        )
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self._client.close()

    def chat_json(
        self,
        *,
        model: str,
        prompt: str,
        pdf_attachment: dict[str, str] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, str]] = [{"type": "text", "text": prompt}]
        if pdf_attachment is not None:
            content.append(pdf_attachment)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "response_format": response_format or {"type": "json_object"},
            "temperature": self._temperature,
            "top_p": self._top_p,
        }

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                if attempt >= self._max_retries:
                    raise OpenRouterError(f"Request failed: {exc}") from exc
                time.sleep(2**attempt * 0.1)
                continue

            if response.status_code >= 500:
                if attempt >= self._max_retries:
                    raise OpenRouterError(
                        f"Server error {response.status_code}: {response.text}"
                    )
                time.sleep(2**attempt * 0.1)
                continue

            if response.status_code >= 400:
                raise OpenRouterError(
                    f"Request rejected {response.status_code}: {response.text}"
                )

            try:
                payload_json = response.json()
                message_content = payload_json["choices"][0]["message"]["content"]
                text = _message_content_to_text(message_content)
                return json.loads(text)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise OpenRouterError(
                    f"Unexpected or invalid JSON response: {response.text}"
                ) from exc

        raise OpenRouterError("Unreachable retry state")


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                value = part.get("text")
                if isinstance(value, str):
                    text_chunks.append(value)
        if text_chunks:
            return "\n".join(text_chunks)
    raise OpenRouterError(f"Unsupported message content shape: {type(content)}")
