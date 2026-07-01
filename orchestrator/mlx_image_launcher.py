"""Launch the local OpenAI-compatible image generation server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX image launcher")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--quantize-override", type=int, default=None)
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    from orchestrator.image_server import main as server_main

    sys.argv = [
        "image_server",
        "--model",
        str(model_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model-id",
        args.model_id or model_path.name,
    ]
    if args.quantize_override is not None:
        sys.argv.extend(["--quantize-override", str(args.quantize_override)])
    server_main()


if __name__ == "__main__":
    main()
