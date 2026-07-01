"""Launch the local Kokoro TTS server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX TTS launcher")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--default-voice", default="af_heart")
    parser.add_argument("--default-speed", type=float, default=1.0)
    parser.add_argument("--default-lang-code", default="a")
    parser.add_argument("--default-response-format", default=None)
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    from orchestrator.tts_server import main as server_main

    sys.argv = [
        "tts_server",
        "--model",
        str(model_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model-id",
        args.model_id or model_path.name,
        "--default-voice",
        args.default_voice,
        "--default-speed",
        str(args.default_speed),
        "--default-lang-code",
        args.default_lang_code,
    ]
    if args.default_response_format is not None:
        sys.argv.extend(["--default-response-format", args.default_response_format])
    server_main()


if __name__ == "__main__":
    main()
