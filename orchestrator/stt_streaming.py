"""SSE helpers for partial STT transcripts (MLX-88)."""

from __future__ import annotations

import json
from typing import Any, Iterator

import numpy as np

SSE_MEDIA_TYPE = "text/event-stream"


def encode_sse_event(event: str, payload: dict[str, Any]) -> bytes:
    """Format one Server-Sent Events frame."""
    body = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n".encode("utf-8")


def build_generate_kwargs(
    *,
    task: str,
    effective_language: str | None,
    effective_chunk: float,
    word_timestamps: bool,
    prompt: str | None,
    temperature: float | None,
    stream: bool,
) -> dict[str, object]:
    """Shared mlx-audio Whisper generate() arguments."""
    generate_kwargs: dict[str, object] = {
        "chunk_duration": effective_chunk,
        "stream": stream,
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
    return generate_kwargs


def _partial_text(chunk: object) -> str:
    if hasattr(chunk, "text"):
        return str(getattr(chunk, "text", "")).strip()
    if isinstance(chunk, dict):
        return str(chunk.get("text") or "").strip()
    return str(chunk).strip()


def _chunk_is_final(chunk: object) -> bool:
    if hasattr(chunk, "is_final"):
        return bool(chunk.is_final)
    if isinstance(chunk, dict):
        return bool(chunk.get("is_final") or chunk.get("final"))
    return False


def iter_transcription_sse(
    stt_model: object,
    waveform: np.ndarray,
    *,
    task: str,
    effective_language: str | None,
    effective_chunk: float,
    word_timestamps: bool,
    prompt: str | None,
    temperature: float | None,
    payload_from_result: Any,
) -> Iterator[bytes]:
    """Yield SSE frames with partial and final transcript payloads."""
    generate_kwargs = build_generate_kwargs(
        task=task,
        effective_language=effective_language,
        effective_chunk=effective_chunk,
        word_timestamps=word_timestamps,
        prompt=prompt,
        temperature=temperature,
        stream=True,
    )

    stream_fn = getattr(stt_model, "generate_streaming", None)
    if callable(stream_fn):
        last_text = ""
        for chunk in stream_fn(waveform, **generate_kwargs):
            text = _partial_text(chunk)
            if not text or text == last_text:
                continue
            last_text = text
            yield encode_sse_event(
                "transcript",
                {
                    "object": "stt.transcript.delta",
                    "text": text,
                    "is_final": _chunk_is_final(chunk),
                    "task": task,
                },
            )
        if last_text:
            yield encode_sse_event(
                "transcript",
                {
                    "object": "stt.transcript.completed",
                    "text": last_text,
                    "is_final": True,
                    "task": task,
                },
            )
        yield encode_sse_event("done", {"object": "stt.transcript.done"})
        return

    payload = payload_from_result(
        stt_model.generate(
            waveform,
            **build_generate_kwargs(
                task=task,
                effective_language=effective_language,
                effective_chunk=effective_chunk,
                word_timestamps=word_timestamps,
                prompt=prompt,
                temperature=temperature,
                stream=False,
            ),
        ),
        task=task,
    )
    yield encode_sse_event(
        "transcript",
        {
            "object": "stt.transcript.completed",
            "text": payload.get("text", ""),
            "is_final": True,
            "task": task,
            "language": payload.get("language"),
            "duration": payload.get("duration"),
            "segments": payload.get("segments") or [],
        },
    )
    yield encode_sse_event("done", {"object": "stt.transcript.done"})
