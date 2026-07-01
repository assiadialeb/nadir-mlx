"""OpenAI-compatible embedding server powered by mlx-embeddings."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Literal, Optional, Union

import mlx.core as mx
import uvicorn
from fastapi import Depends, FastAPI
from pydantic import BaseModel

from mlx_embeddings import generate, load

from orchestrator.embedding_response import (
    EmbeddingDimensionError,
    EmbeddingFormatError,
    build_embedding_data_entries,
    normalize_encoding_format,
)
from orchestrator.fastapi_openapi import OPENAPI_INFERENCE_ERRORS, InferenceApiError, to_http_exception
from orchestrator.inference_auth import require_inference_api_key
from orchestrator.security_utils import public_error_message

app = FastAPI(title="MLX Embedding Server")
_state: dict[str, Any] = {}


class EmbeddingsRequest(BaseModel):
    model: str = "default_model"
    input: Union[str, list[str]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None


def _extract_embeddings(output: Any) -> mx.array:
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    raise ValueError("Model output does not contain embeddings.")


def _estimate_tokens(texts: list[str]) -> int:
    return sum(max(1, len(text.split())) for text in texts)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "model": str(_state.get("model_id", ""))}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": _state.get("model_id", "default_model"),
                "object": "model",
                "created": int(_state.get("created", time.time())),
            }
        ],
    }


@app.post(
    "/v1/embeddings",
    dependencies=[Depends(require_inference_api_key)],
    responses=OPENAPI_INFERENCE_ERRORS,
)
def create_embeddings(body: EmbeddingsRequest) -> dict[str, Any]:
    try:
        return _create_embeddings(body)
    except InferenceApiError as exc:
        raise to_http_exception(exc) from exc


def _create_embeddings(body: EmbeddingsRequest) -> dict[str, Any]:
    model = _state.get("model")
    processor = _state.get("processor")
    if model is None or processor is None:
        raise InferenceApiError(503, "Model not loaded.")

    texts = [body.input] if isinstance(body.input, str) else list(body.input)
    if not texts:
        raise InferenceApiError(400, "input must not be empty.")

    try:
        encoding_format = normalize_encoding_format(body.encoding_format)
    except EmbeddingFormatError as exc:
        raise InferenceApiError(400, str(exc)) from exc

    try:
        output = generate(model, processor, texts)
        vectors = _extract_embeddings(output).tolist()
    except Exception as exc:
        raise InferenceApiError(
            500,
            public_error_message(exc, fallback="Embedding generation failed."),
        ) from exc

    if isinstance(body.input, str):
        rows = [vectors] if vectors and not isinstance(vectors[0], list) else vectors
    else:
        rows = vectors

    try:
        data = build_embedding_data_entries(
            rows,
            encoding_format=encoding_format,
            dimensions=body.dimensions,
        )
    except EmbeddingDimensionError as exc:
        raise InferenceApiError(400, str(exc)) from exc

    token_count = _estimate_tokens(texts)
    return {
        "object": "list",
        "data": data,
        "model": body.model,
        "usage": {
            "prompt_tokens": token_count,
            "total_tokens": token_count,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLX embedding server")
    parser.add_argument("--model", required=True, help="Local path to the embedding model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11400)
    parser.add_argument("--model-id", default=None, help="Model ID exposed via /v1/models")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Model path not found: {model_path}")

    print(f"Loading embedding model from {model_path} ...")
    model, processor = load(str(model_path))

    _state["model"] = model
    _state["processor"] = processor
    _state["model_id"] = args.model_id or model_path.name
    _state["created"] = time.time()

    print(f"MLX embedding server ready on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
