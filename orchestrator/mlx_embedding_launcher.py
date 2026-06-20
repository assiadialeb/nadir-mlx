"""Launch the local OpenAI-compatible embedding server."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX embedding launcher")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    from orchestrator.embedding_server import main as server_main

    import sys

    sys.argv = [
        "embedding_server",
        "--model",
        str(model_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model-id",
        model_path.name,
    ]
    server_main()


if __name__ == "__main__":
    main()
