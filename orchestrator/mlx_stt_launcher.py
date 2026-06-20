"""Launch the local Whisper STT server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX STT launcher")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--default-language", default="")
    parser.add_argument("--default-chunk-duration", type=float, default=30.0)
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    from orchestrator.stt_server import main as server_main

    argv = [
        "stt_server",
        "--model",
        str(model_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model-id",
        args.model_id or model_path.name,
        "--default-chunk-duration",
        str(args.default_chunk_duration),
    ]
    if args.default_language:
        argv.extend(["--default-language", args.default_language])
    sys.argv = argv
    server_main()


if __name__ == "__main__":
    main()
