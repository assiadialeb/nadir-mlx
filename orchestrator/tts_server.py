"""OpenAI-compatible TTS server (Kokoro via mlx-audio)."""

from __future__ import annotations

import argparse
import io
import os
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

app = FastAPI(title="MLX TTS Server")
_state: dict[str, object] = {}


class SpeechRequest(BaseModel):
    model: str = "default_model"
    input: str
    voice: Optional[str] = None
    speed: Optional[float] = Field(default=None, ge=0.25, le=4.0)
    lang_code: Optional[str] = None
    response_format: Literal["wav", "mp3"] = "wav"


def _defaults() -> dict[str, object]:
    defaults = _state.get("defaults")
    if isinstance(defaults, dict):
        return defaults
    return {}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "model": str(_state.get("model_id", ""))}


@app.get("/v1/models")
def list_models() -> dict[str, object]:
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


@app.get("/v1/audio/voices")
def list_voices() -> dict[str, object]:
    model_path = _state.get("model_path")
    voices: list[dict[str, str]] = []
    if isinstance(model_path, Path):
        voices_dir = model_path / "voices"
        if voices_dir.is_dir():
            voices = [
                {"id": voice_file.stem, "name": voice_file.stem}
                for voice_file in sorted(voices_dir.glob("*.safetensors"))
            ]
    default_voice = str(_defaults().get("voice_id", "af_heart"))
    return {
        "object": "list",
        "model": str(_state.get("model_id", "")),
        "default_voice": default_voice,
        "data": voices,
    }


@app.post("/v1/audio/speech")
def create_speech(body: SpeechRequest) -> Response:
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded.")

    text = body.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="input must not be empty.")

    defaults = _defaults()
    voice = body.voice or defaults.get("voice_id") or "af_heart"
    speed = body.speed if body.speed is not None else defaults.get("speaking_rate", 1.0)
    lang_code = body.lang_code or defaults.get("lang_code") or "a"

    audio_chunks: list[np.ndarray] = []
    sample_rate: int | None = None
    try:
        for result in model.generate(
            text,
            voice=str(voice),
            speed=float(speed),
            lang_code=str(lang_code),
        ):
            audio_chunks.append(np.asarray(result.audio))
            if sample_rate is None:
                sample_rate = int(result.sample_rate)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not audio_chunks or sample_rate is None:
        raise HTTPException(status_code=400, detail="No audio generated.")

    concatenated = np.concatenate(audio_chunks)
    buffer = io.BytesIO()
    from mlx_audio.audio_io import write as audio_write

    audio_write(buffer, concatenated, sample_rate, format=body.response_format)
    media_type = "audio/wav" if body.response_format == "wav" else "audio/mpeg"
    return Response(
        content=buffer.getvalue(),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename=speech.{body.response_format}"},
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLX Kokoro TTS server")
    parser.add_argument("--model", required=True, help="Local path to Kokoro checkpoint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11400)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--default-voice", default="af_heart")
    parser.add_argument("--default-speed", type=float, default=1.0)
    parser.add_argument("--default-lang-code", default="a")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TQDM_DISABLE", "1")
    args = _parse_args()
    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Model path not found: {model_path}")

    print(f"Loading Kokoro TTS model from {model_path} ...")
    from mlx_audio.utils import load_model

    model = load_model(str(model_path))
    _state["model"] = model
    _state["model_path"] = model_path
    _state["model_id"] = args.model_id or model_path.name
    _state["created"] = time.time()
    _state["defaults"] = {
        "voice_id": args.default_voice,
        "speaking_rate": args.default_speed,
        "lang_code": args.default_lang_code,
    }

    print(f"MLX TTS server ready on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
