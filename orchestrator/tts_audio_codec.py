"""TTS response format normalization and audio encoding helpers."""

from __future__ import annotations

import io
import shutil
import subprocess
from typing import Final, Iterator

import numpy as np

SUPPORTED_TTS_RESPONSE_FORMATS: Final[frozenset[str]] = frozenset({
    "wav",
    "mp3",
    "opus",
    "aac",
    "flac",
    "pcm",
})

TTS_RESPONSE_MEDIA_TYPES: Final[dict[str, str]] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/L16",
}


class TtsFormatError(ValueError):
    """Raised when the client requests an unsupported TTS response format."""

    def __init__(self, response_format: str) -> None:
        self.response_format = response_format
        supported = ", ".join(sorted(SUPPORTED_TTS_RESPONSE_FORMATS))
        super().__init__(
            f"Unsupported response_format '{response_format}'. "
            f"Supported formats: {supported}."
        )


def normalize_tts_response_format(raw: str | None) -> str:
    """Normalize and validate an OpenAI-style TTS response_format value."""
    normalized = (raw or "wav").strip().lower()
    if normalized not in SUPPORTED_TTS_RESPONSE_FORMATS:
        raise TtsFormatError(normalized)
    return normalized


def resolve_tts_response_format(
    request_format: str | None,
    server_default: str | None,
) -> str:
    """Pick request format when set, otherwise server_config advanced default."""
    return normalize_tts_response_format(request_format or server_default)


def tts_media_type(response_format: str) -> str:
    """Return the HTTP media type for a TTS response format."""
    return TTS_RESPONSE_MEDIA_TYPES[normalize_tts_response_format(response_format)]


def _audio_to_int16_mono(audio: np.ndarray) -> tuple[np.ndarray, int]:
    """Convert synthesized audio to mono int16 samples."""
    samples = np.asarray(audio)
    if samples.dtype in (np.float32, np.float64):
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767).astype(np.int16)
    elif samples.dtype != np.int16:
        samples = samples.astype(np.int16)

    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.int16)
    return samples, 1


def _encode_aac_with_ffmpeg(
    buffer: io.BytesIO,
    audio: np.ndarray,
    sample_rate: int,
) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required for AAC encoding. Install with: brew install ffmpeg"
        )

    samples, nchannels = _audio_to_int16_mono(audio)
    pcm_bytes = samples.tobytes()
    command = [
        ffmpeg_path,
        "-y",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(nchannels),
        "-i",
        "pipe:0",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-f",
        "adts",
        "pipe:1",
    ]
    result = subprocess.run(command, input=pcm_bytes, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(f"ffmpeg AAC encoding failed: {stderr}")
    buffer.write(result.stdout)
    buffer.seek(0)


def encode_speech_audio(
    audio: np.ndarray,
    sample_rate: int,
    response_format: str,
) -> tuple[bytes, str]:
    """Encode synthesized speech audio for an OpenAI-compatible response."""
    normalized = normalize_tts_response_format(response_format)
    buffer = io.BytesIO()

    if normalized == "aac":
        _encode_aac_with_ffmpeg(buffer, audio, sample_rate)
    else:
        from mlx_audio.audio_io import write as audio_write

        audio_write(buffer, audio, sample_rate, format=normalized)

    return buffer.getvalue(), tts_media_type(normalized)


def iter_encoded_audio_chunks(
    encoded_audio: bytes,
    chunk_size: int = 8192,
) -> Iterator[bytes]:
    """Yield fixed-size chunks from encoded audio for streaming responses."""
    for offset in range(0, len(encoded_audio), chunk_size):
        yield encoded_audio[offset : offset + chunk_size]
