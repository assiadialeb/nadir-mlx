"""Ollama HTTP API ↔ OpenAI-compatible translation helpers."""

from __future__ import annotations

from typing import Any

from orchestrator.gateway.router import GatewayRouteError


def ollama_chat_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Translate Ollama ``/api/chat`` body to chat.completions."""
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise GatewayRouteError(
            status_code=400,
            code="invalid_model",
            message="The model field is required.",
            param="model",
        )
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise GatewayRouteError(
            status_code=400,
            code="invalid_request",
            message="messages must be a non-empty array.",
            param="messages",
        )
    chat: dict[str, Any] = {
        "model": model.strip(),
        "messages": messages,
        "stream": bool(body.get("stream", False)),
    }
    options = body.get("options") if isinstance(body.get("options"), dict) else {}
    if "temperature" in options:
        chat["temperature"] = options["temperature"]
    if "top_p" in options:
        chat["top_p"] = options["top_p"]
    if "num_predict" in options:
        chat["max_tokens"] = options["num_predict"]
    return chat


def openai_chat_to_ollama(
    chat_payload: dict[str, Any],
    *,
    client_model_alias: str,
) -> dict[str, Any]:
    """Translate chat.completion JSON into Ollama chat response."""
    content = ""
    choices = chat_payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            content = message["content"]
    return {
        "model": client_model_alias,
        "message": {"role": "assistant", "content": content},
        "done": True,
    }


def ollama_embed_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Translate Ollama ``/api/embed`` body to embeddings."""
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise GatewayRouteError(
            status_code=400,
            code="invalid_model",
            message="The model field is required.",
            param="model",
        )
    if "input" in body:
        inputs = body["input"]
    elif "prompt" in body:
        inputs = body["prompt"]
    else:
        raise GatewayRouteError(
            status_code=400,
            code="invalid_request",
            message="input or prompt is required.",
            param="input",
        )
    return {"model": model.strip(), "input": inputs}


def openai_embeddings_to_ollama(
    embeddings_payload: dict[str, Any],
    *,
    client_model_alias: str,
) -> dict[str, Any]:
    """Translate embeddings JSON into Ollama embed response."""
    vectors: list[list[float]] = []
    data = embeddings_payload.get("data")
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict) and isinstance(row.get("embedding"), list):
                vectors.append(row["embedding"])
    if len(vectors) == 1:
        return {"model": client_model_alias, "embeddings": vectors}
    return {"model": client_model_alias, "embeddings": vectors}


def models_list_to_ollama_tags(models_payload: dict[str, Any]) -> dict[str, Any]:
    """Translate ``GET /v1/models`` payload into Ollama ``/api/tags``."""
    models: list[dict[str, Any]] = []
    data = models_payload.get("data")
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if not isinstance(model_id, str):
                continue
            models.append(
                {
                    "name": model_id,
                    "model": model_id,
                    "modified_at": "",
                    "size": 0,
                    "digest": "",
                    "details": {},
                }
            )
    return {"models": models}
