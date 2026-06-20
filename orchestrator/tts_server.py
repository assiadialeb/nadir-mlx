"""OpenAI-compatible TTS server (Kokoro via mlx-audio)."""

from __future__ import annotations

import argparse
import io
import logging
import os
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from orchestrator.kokoro_tts_utils import (
    OPENAI_TTS_VOICES,
    is_kokoro_voice_id,
    resolve_kokoro_voice,
    resolve_lang_code,
)
from orchestrator.kokoro_voices import KOKORO_VOICES

logger = logging.getLogger(__name__)

app = FastAPI(title="MLX TTS Server")
_state: dict[str, object] = {}


class SpeechRequest(BaseModel):
    model: str = "default_model"
    input: str
    voice: Optional[str] = None
    speed: Optional[float] = Field(default=None, ge=0.25, le=4.0)
    lang_code: Optional[str] = None
    language: Optional[str] = None
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
    installed: dict[str, str] = {}
    if isinstance(model_path, Path):
        voices_dir = model_path / "voices"
        if voices_dir.is_dir():
            for voice_file in sorted(voices_dir.glob("*.safetensors")):
                installed[voice_file.stem] = voice_file.stem

    default_voice = str(_defaults().get("voice_id", "ff_siwis"))
    default_lang = str(_defaults().get("lang_code", "f"))
    catalog = [
        {
            "id": voice_id,
            "name": label,
            "installed": voice_id in installed,
        }
        for voice_id, label in KOKORO_VOICES
    ]
    return {
        "object": "list",
        "model": str(_state.get("model_id", "")),
        "default_voice": default_voice,
        "default_lang_code": default_lang,
        "openai_voice_mapping": True,
        "data": catalog or [
            {"id": voice_id, "name": voice_id, "installed": True}
            for voice_id in sorted(installed)
        ],
    }


def _validate_speech_voice(raw_voice: str) -> None:
    if is_kokoro_voice_id(raw_voice):
        return
    if raw_voice.lower() in OPENAI_TTS_VOICES:
        return
    raise HTTPException(
        status_code=400,
        detail=(
            f"Unknown voice '{raw_voice}'. Use a Kokoro id (e.g. ff_siwis) "
            "or an OpenAI voice name (alloy, nova, …)."
        ),
    )


def _synthesize_speech(
    model: object,
    *,
    text: str,
    voice: str,
    speed: float,
    lang_code: str,
) -> tuple[np.ndarray, int]:
    audio_chunks: list[np.ndarray] = []
    sample_rate: int | None = None
    try:
        for result in model.generate(
            text,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
        ):
            audio_chunks.append(np.asarray(result.audio))
            if sample_rate is None:
                sample_rate = int(result.sample_rate)
    except Exception as exc:
        message = str(exc)
        if "Failed to open file" in message and ".safetensors" in message:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Kokoro voice '{voice}' is not available for lang '{lang_code}'. "
                    "Pick a voice from GET /v1/audio/voices (e.g. ff_siwis for French)."
                ),
            ) from exc
        raise HTTPException(status_code=500, detail=message) from exc

    if not audio_chunks or sample_rate is None:
        raise HTTPException(status_code=400, detail="No audio generated.")
    return np.concatenate(audio_chunks), sample_rate


def _speech_audio_response(
    audio: np.ndarray,
    sample_rate: int,
    response_format: Literal["wav", "mp3"],
) -> Response:
    buffer = io.BytesIO()
    from mlx_audio.audio_io import write as audio_write

    audio_write(buffer, audio, sample_rate, format=response_format)
    media_type = "audio/wav" if response_format == "wav" else "audio/mpeg"
    return Response(
        content=buffer.getvalue(),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename=speech.{response_format}"},
    )


@app.post("/v1/audio/speech")
def create_speech(body: SpeechRequest) -> Response:
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded.")

    text = body.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="input must not be empty.")

    if body.voice:
        _validate_speech_voice(body.voice.strip())

    defaults = _defaults()
    lang_code = resolve_lang_code(
        body.lang_code,
        body.language,
        str(defaults.get("lang_code") or "f"),
    )
    voice, remap_note = resolve_kokoro_voice(
        body.voice,
        lang_code,
        str(defaults.get("voice_id") or "") or None,
    )
    if remap_note:
        logger.info(remap_note)

    speed = body.speed if body.speed is not None else defaults.get("speaking_rate", 1.0)
    audio, sample_rate = _synthesize_speech(
        model,
        text=text,
        voice=voice,
        speed=float(speed),
        lang_code=lang_code,
    )
    return _speech_audio_response(audio, sample_rate, body.response_format)


def _verify_kokoro_dependencies() -> None:
    """Fail fast when misaki extras required for Kokoro G2P are missing."""
    try:
        import misaki.espeak  # noqa: F401
        import misaki.en  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Kokoro TTS requires misaki with English/espeak extras. "
            "Install in the mlx-server venv: pip install 'misaki[en]==0.9.4' "
            "(or pip install -r requirements.txt), then restart the TTS server."
        ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLX Kokoro TTS server")
    parser.add_argument("--model", required=True, help="Local path to Kokoro checkpoint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11400)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--default-voice", default="ff_siwis")
    parser.add_argument("--default-speed", type=float, default=1.0)
    parser.add_argument("--default-lang-code", default="f")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TQDM_DISABLE", "1")
    args = _parse_args()
    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Model path not found: {model_path}")

    _verify_kokoro_dependencies()
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
