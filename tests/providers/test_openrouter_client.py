from __future__ import annotations

import json
from typing import Any

import httpx


def _json_response(content: dict[str, Any], status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=content,
        headers={"Content-Type": "application/json"},
    )


def test_openrouter_client_uses_auth_header_and_attachment_payload() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("Authorization")
        body = json.loads(request.content.decode("utf-8"))
        captured["body"] = body
        return _json_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"ok": true}'
                        }
                    }
                ]
            }
        )

    transport = httpx.MockTransport(handler)

    from litreview.providers.openrouter import OpenRouterClient

    client = OpenRouterClient(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        transport=transport,
    )
    result = client.chat_json(
        model="google/gemini-2.5-pro",
        prompt="Extract this paper",
        pdf_attachment={
            "type": "file",
            "filename": "paper.pdf",
            "file_data": "data:application/pdf;base64,Zm9v",
        },
    )

    assert captured["authorization"] == "Bearer test-key"
    content = captured["body"]["messages"][0]["content"]
    assert {"type": "text", "text": "Extract this paper"} in content
    assert {
        "type": "file",
        "filename": "paper.pdf",
        "file_data": "data:application/pdf;base64,Zm9v",
    } in content
    assert result == {"ok": True}


def test_openrouter_client_retries_on_server_error() -> None:
    attempts = {"count": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 2:
            return _json_response({"error": "temporary"}, status_code=500)
        return _json_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"status": "recovered"}'
                        }
                    }
                ]
            }
        )

    transport = httpx.MockTransport(handler)

    from litreview.providers.openrouter import OpenRouterClient

    client = OpenRouterClient(
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        transport=transport,
        max_retries=2,
    )
    data = client.chat_json(model="m", prompt="p")
    assert data["status"] == "recovered"
    assert attempts["count"] == 2


def test_openrouter_client_sends_default_sampling_params() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _json_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"ok": true}'
                        }
                    }
                ]
            }
        )

    transport = httpx.MockTransport(handler)

    from litreview.providers.openrouter import OpenRouterClient

    client = OpenRouterClient(api_key="x", transport=transport)
    client.chat_json(model="google/gemini-2.5-pro", prompt="p")

    assert captured["body"]["temperature"] == 0.2
    assert captured["body"]["top_p"] == 0.9
