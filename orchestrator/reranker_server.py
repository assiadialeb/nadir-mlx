"""Jina /v1/rerank server for mlx-community JinaForRanking models (bundled projector)."""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from mlx_lm.utils import load_model, load_tokenizer
from pydantic import BaseModel

from orchestrator.tokenizer_compat import install_auto_fix_mistral_regex
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

DOC_EMBED_TOKEN_ID = 151670
QUERY_EMBED_TOKEN_ID = 151671
SPECIAL_TOKENS = {
    "query_embed_token": "<|rerank_token|>",
    "doc_embed_token": "<|embed_token|>",
}


class RerankRequest(BaseModel):
    model: str | None = None
    query: str
    documents: list[str | dict[str, Any]]
    top_n: int | None = None
    return_documents: bool | None = False


class RerankDocument(BaseModel):
    text: str


class RerankResult(BaseModel):
    document: RerankDocument | None = None
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    id: str | None = None
    results: list[RerankResult]


app = FastAPI(title="MLX Jina Reranker Server")
_state: dict[str, Any] = {}


class TorchProjector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(1024, 512, bias=False)
        self.linear2 = torch.nn.Linear(512, 512, bias=False)
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))


def _load_prompt_formatter(model_path: Path):
    modeling_file = model_path / "modeling.py"
    if not modeling_file.is_file():
        raise FileNotFoundError(f"Missing modeling.py in {model_path}")

    spec = importlib.util.spec_from_file_location("jina_ranking_modeling", modeling_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {modeling_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    formatter = getattr(module, "format_docs_prompts_func", None)
    if formatter is None:
        raise AttributeError("format_docs_prompts_func not found in modeling.py")
    return formatter


def _materialize_projector_safetensors(model_path: Path) -> Path:
    target = model_path / "projector.safetensors"
    if target.is_file():
        return target

    weights: dict[str, mx.array] = {}
    for weight_file in glob.glob(str(model_path / "model*.safetensors")):
        weights.update(mx.load(weight_file))

    required = (
        "projector.layers.0.weight",
        "projector.layers.0.scales",
        "projector.layers.2.weight",
        "projector.layers.2.scales",
    )
    if not all(key in weights for key in required):
        raise FileNotFoundError(
            f"No standalone or bundled projector weights found in {model_path}"
        )

    linear1 = mx.dequantize(
        weights["projector.layers.0.weight"],
        weights["projector.layers.0.scales"],
        group_size=32,
        bits=4,
        mode="mxfp4",
    )
    linear2 = mx.dequantize(
        weights["projector.layers.2.weight"],
        weights["projector.layers.2.scales"],
        group_size=32,
        bits=4,
        mode="mxfp4",
    )
    save_file(
        {
            "linear1.weight": torch.from_numpy(np.array(linear1.astype(mx.float32))),
            "linear2.weight": torch.from_numpy(np.array(linear2.astype(mx.float32))),
        },
        str(target),
    )
    return target


def _load_projector(projector_path: Path) -> TorchProjector:
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(projector_path), framework="pt") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)

    projector = TorchProjector()
    projector.load_state_dict(tensors)
    projector.eval()
    return projector


def _extract_document_text(document: str | dict[str, Any]) -> str:
    if isinstance(document, str):
        return document
    for key in ("text", "content", "document"):
        value = document.get(key)
        if isinstance(value, str):
            return value
    raise ValueError("Document dict must include a text field.")


def _compute_scores(query: str, documents: list[str]) -> list[float]:
    formatter = _state["prompt_formatter"]
    model = _state["model"]
    tokenizer = _state["tokenizer"]
    projector = _state["projector"]

    prompt = formatter(
        query,
        documents,
        instruction=None,
        special_tokens=SPECIAL_TOKENS,
        no_thinking=True,
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = mx.array([token_ids], dtype=mx.int32)
    hidden_states = model.model(tokens)[0]
    mx.eval(hidden_states)

    hidden_np = np.array(hidden_states.astype(mx.float32))
    doc_positions = [index for index, token in enumerate(token_ids) if token == DOC_EMBED_TOKEN_ID]
    query_position = next(
        (index for index, token in enumerate(token_ids) if token == QUERY_EMBED_TOKEN_ID),
        None,
    )
    if query_position is None:
        raise ValueError("Query embed token not found in tokenized prompt.")
    if len(doc_positions) != len(documents):
        raise ValueError("Document embed token count does not match input documents.")

    query_embed = projector(torch.from_numpy(hidden_np[query_position]))
    doc_embeds = torch.stack(
        [projector(torch.from_numpy(hidden_np[position])) for position in doc_positions]
    )

    query_norm = torch.linalg.norm(query_embed) + 1e-12
    doc_norms = torch.linalg.norm(doc_embeds, dim=1) + 1e-12
    scores = torch.sum(doc_embeds * query_embed, dim=1) / (doc_norms * query_norm)
    return [float(score) for score in scores.tolist()]


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "model": str(_state.get("model_id", ""))}


@app.post("/v1/rerank", response_model=RerankResponse)
def rerank_endpoint(request: RerankRequest) -> RerankResponse:
    if _state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    documents = [_extract_document_text(item) for item in request.documents]
    if not documents:
        return RerankResponse(id=str(uuid.uuid4()), results=[])

    try:
        scores = _compute_scores(request.query, documents)
    except Exception as exc:
        logger.exception("Reranking failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal Server Error during reranking.") from exc

    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda item: item[1], reverse=True)
    if request.top_n is not None:
        indexed_scores = indexed_scores[: request.top_n]

    results: list[RerankResult] = []
    for index, score in indexed_scores:
        document = None
        if request.return_documents:
            document = RerankDocument(text=documents[index])
        results.append(
            RerankResult(document=document, index=index, relevance_score=score)
        )

    return RerankResponse(id=str(uuid.uuid4()), results=results)


def _is_jina_for_ranking_model(model_path: Path) -> bool:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    return "JinaForRanking" in architectures


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX Jina reranker server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Model path not found: {model_path}")
    if not _is_jina_for_ranking_model(model_path):
        raise SystemExit(
            "This server only supports JinaForRanking models. "
            "Use local-reranker for jinaai/jina-reranker-v3-mlx."
        )

    install_auto_fix_mistral_regex()

    logger.info("Loading JinaForRanking model from %s", model_path)
    model, config = load_model(model_path, lazy=False, strict=False)
    tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id"))
    projector_path = _materialize_projector_safetensors(model_path)
    projector = _load_projector(projector_path)
    prompt_formatter = _load_prompt_formatter(model_path)

    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["projector"] = projector
    _state["prompt_formatter"] = prompt_formatter
    _state["model_id"] = args.model_id or model_path.name
    _state["created"] = time.time()

    logger.info("MLX Jina reranker ready on http://%s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
