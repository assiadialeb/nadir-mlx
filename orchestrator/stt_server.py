"""OpenAI-compatible STT server (Whisper via mlx-audio)."""

from __future__ import annotations

import argparse
import io
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from orchestrator.stt_response_formats import (
    SttFormatError,
    normalize_stt_response_format,
    render_stt_response_body,
    segments_duration_seconds,
)

app = FastAPI(title="MLX STT Server")
_state: dict[str, object] = {}


class TranscriptionDefaults(BaseModel):
    language: Optional[str] = None
    chunk_duration: float = 30.0


def _defaults() -> TranscriptionDefaults:
    raw = _state.get("defaults")
    if isinstance(raw, TranscriptionDefaults):
        return raw
    if isinstance(raw, dict):
        return TranscriptionDefaults(**raw)
    return TranscriptionDefaults()


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


def _decode_uploaded_audio(audio_bytes: bytes) -> np.ndarray:
    """Decode uploaded audio to mono 16 kHz float32 for Whisper."""
    from mlx_audio.audio_io import read as audio_read
    from mlx_audio.stt.utils import SAMPLE_RATE, resample_audio

    try:
        audio, sample_rate = audio_read(io.BytesIO(audio_bytes), always_2d=False)
    except Exception as exc:
        message = str(exc)
        if "ffmpeg" in message.lower():
            raise HTTPException(
                status_code=400,
                detail=(
                    "This audio format requires ffmpeg. Install ffmpeg "
                    "(brew install ffmpeg) or upload WAV/MP3 decodable by miniaudio."
                ),
            ) from exc
        raise HTTPException(status_code=400, detail=f"Audio decode failed: {message}") from exc

    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        waveform = resample_audio(waveform, sample_rate, SAMPLE_RATE)
    return np.asarray(waveform, dtype=np.float32)


def _transcription_payload_from_result(result: object, *, task: str) -> dict[str, Any]:
    if hasattr(result, "text"):
        segments = list(getattr(result, "segments", None) or [])
        language = getattr(result, "language", None)
        text = str(result.text).strip()
        return {
            "text": text,
            "language": language,
            "task": task,
            "duration": segments_duration_seconds(segments),
            "segments": segments,
        }
    if isinstance(result, dict):
        payload = dict(result)
        payload.setdefault("task", task)
        payload.setdefault("segments", [])
        payload.setdefault("duration", segments_duration_seconds(list(payload["segments"])))
        return payload
    if hasattr(result, "__iter__") and not isinstance(result, str):
        chunks = list(result)
        if chunks and hasattr(chunks[-1], "text"):
            return _transcription_payload_from_result(chunks[-1], task=task)
        text = "".join(str(chunk) for chunk in chunks)
        return {"text": text, "task": task, "segments": [], "duration": 0.0}
    return {"text": str(result), "task": task, "segments": [], "duration": 0.0}


def _parse_optional_float(raw: Optional[str], field_name: str) -> Optional[float]:
    if raw is None or not str(raw).strip():
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name}: expected a number.",
        ) from exc


def _parse_word_timestamps(raw: Optional[str]) -> bool:
    if raw is None or not str(raw).strip():
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _run_transcription(
    stt_model: object,
    waveform: np.ndarray,
    *,
    task: str,
    effective_language: Optional[str],
    effective_chunk: float,
    word_timestamps: bool,
    prompt: Optional[str],
    temperature: Optional[float],
) -> dict[str, Any]:
    generate_kwargs: dict[str, object] = {
        "chunk_duration": effective_chunk,
        "stream": False,
        "task": task,
        "word_timestamps": word_timestamps,
        "return_timestamps": True,
    }
    if effective_language:
        generate_kwargs["language"] = effective_language
    if prompt:
        generate_kwargs["initial_prompt"] = prompt
    if temperature is not None:
        generate_kwargs["temperature"] = temperature

    result = stt_model.generate(waveform, **generate_kwargs)
    return _transcription_payload_from_result(result, task=task)


def _build_stt_response(payload: dict[str, Any], response_format: str) -> Response:
    try:
        body, media_type = render_stt_response_body(payload, response_format)
    except SttFormatError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if isinstance(body, dict):
        return JSONResponse(body, media_type=media_type)
    if media_type.startswith("text/plain"):
        return PlainTextResponse(str(body), media_type=media_type)
    return Response(content=str(body), media_type=media_type)


async def _create_stt_response(
    *,
    file: UploadFile,
    task: str,
    language: Optional[str],
    response_format: str,
    chunk_duration: Optional[float],
    word_timestamps: bool,
    prompt: Optional[str],
    temperature: Optional[float],
) -> Response:
    stt_model = _state.get("model")
    if stt_model is None:
        raise HTTPException(status_code=503, detail="STT model not loaded.")

    try:
        normalized_format = normalize_stt_response_format(response_format)
    except SttFormatError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    defaults = _defaults()
    effective_language = language or defaults.language
    effective_chunk = chunk_duration if chunk_duration is not None else defaults.chunk_duration

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    try:
        waveform = _decode_uploaded_audio(audio_bytes)
        payload = _run_transcription(
            stt_model,
            waveform,
            task=task,
            effective_language=effective_language,
            effective_chunk=effective_chunk,
            word_timestamps=word_timestamps,
            prompt=prompt,
            temperature=temperature,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _build_stt_response(payload, normalized_format)


@app.post("/v1/audio/transcriptions", response_model=None)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("default_model"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
    chunk_duration: Optional[float] = Form(None),
    word_timestamps: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),
) -> Response:
    return await _create_stt_response(
        file=file,
        task="transcribe",
        language=language,
        response_format=response_format,
        chunk_duration=chunk_duration,
        word_timestamps=_parse_word_timestamps(word_timestamps),
        prompt=prompt,
        temperature=_parse_optional_float(temperature, "temperature"),
    )


@app.post("/v1/audio/translations", response_model=None)
async def create_translation(
    file: UploadFile = File(...),
    model: str = Form("default_model"),
    response_format: str = Form("json"),
    chunk_duration: Optional[float] = Form(None),
    word_timestamps: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),
) -> Response:
    return await _create_stt_response(
        file=file,
        task="translate",
        language=None,
        response_format=response_format,
        chunk_duration=chunk_duration,
        word_timestamps=_parse_word_timestamps(word_timestamps),
        prompt=prompt,
        temperature=_parse_optional_float(temperature, "temperature"),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLX Whisper STT server")
    parser.add_argument("--model", required=True, help="Local path to Whisper checkpoint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11400)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--default-language", default=None)
    parser.add_argument("--default-chunk-duration", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TQDM_DISABLE", "1")
    args = _parse_args()
    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Model path not found: {model_path}")

    print(f"Loading Whisper STT model from {model_path} ...")
    from orchestrator.whisper_assets import ensure_whisper_hf_assets
    from mlx_audio.utils import load_model

    ensure_whisper_hf_assets(model_path)
    model = load_model(str(model_path))
    _state["model"] = model
    _state["model_id"] = args.model_id or model_path.name
    _state["created"] = time.time()
    _state["defaults"] = TranscriptionDefaults(
        language=args.default_language,
        chunk_duration=args.default_chunk_duration,
    )

    print(f"MLX STT server ready on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
