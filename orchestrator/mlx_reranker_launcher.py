"""Launch reranker services with Jina-compatible /v1/rerank API."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def _is_jina_for_ranking_model(model_path: Path) -> bool:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    return "JinaForRanking" in architectures


def _patch_local_reranker_for_local_paths() -> None:
    """Patch local-reranker MLX backend for on-disk models (jinaai/jina-reranker-v3-mlx)."""
    from local_reranker import reranker_mlx

    if getattr(reranker_mlx.Reranker, "_mlx_local_path_patch_applied", False):
        return

    original_prepare = reranker_mlx.Reranker._prepare_model_files

    def _prepare_model_files(self, model_name: str) -> str:
        local_path = Path(model_name).expanduser()
        if local_path.is_dir():
            return str(local_path.resolve())
        return original_prepare(self, model_name)

    reranker_mlx.Reranker._prepare_model_files = _prepare_model_files
    reranker_mlx.Reranker._mlx_local_path_patch_applied = True


def _run_local_reranker(model_path: Path, host: str, port: int) -> None:
    _patch_local_reranker_for_local_paths()

    os.environ["RERANKER_BACKEND_TYPE"] = "mlx"
    os.environ["RERANKER_MODEL_NAME"] = str(model_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from local_reranker.cli import run_server
    from local_reranker.config import Settings

    settings = Settings(
        backend_type="mlx",
        model_name=str(model_path),
        host=host,
        port=port,
        log_level="info",
        reload=False,
        disable_batching=False,
    )
    run_server(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX reranker launcher")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Model path not found: {model_path}")

    if _is_jina_for_ranking_model(model_path):
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-m",
                "orchestrator.reranker_server",
                "--model",
                str(model_path),
                "--host",
                args.host,
                "--port",
                str(args.port),
            ],
        )

    _run_local_reranker(model_path, args.host, args.port)


if __name__ == "__main__":
    main()
