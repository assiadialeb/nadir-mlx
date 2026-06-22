"""STT response formatting helpers for OpenAI-compatible Whisper APIs."""

from __future__ import annotations

from typing import Any, Final

SUPPORTED_STT_RESPONSE_FORMATS: Final[frozenset[str]] = frozenset({
    "json",
    "text",
    "verbose_json",
    "srt",
    "vtt",
})


class SttFormatError(ValueError):
    """Raised when the client requests an unsupported STT response format."""

    def __init__(self, response_format: str) -> None:
        self.response_format = response_format
        supported = ", ".join(sorted(SUPPORTED_STT_RESPONSE_FORMATS))
        super().__init__(
            f"Unsupported response_format '{response_format}'. "
            f"Supported formats: {supported}."
        )


def normalize_stt_response_format(raw: str | None) -> str:
    """Normalize and validate an OpenAI-style STT response_format value."""
    normalized = (raw or "json").strip().lower()
    if normalized not in SUPPORTED_STT_RESPONSE_FORMATS:
        raise SttFormatError(normalized)
    return normalized


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp (HH:MM:SS,mmm)."""
    clamped = max(0.0, float(seconds))
    total_ms = int(round(clamped * 1000.0))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    return f"{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    """Format seconds as a WebVTT timestamp (HH:MM:SS.mmm)."""
    clamped = max(0.0, float(seconds))
    total_ms = int(round(clamped * 1000.0))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def _iter_nonempty_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        normalized.append(
            {
                "id": segment.get("id"),
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": text,
                "words": segment.get("words"),
            }
        )
    return normalized


def segments_duration_seconds(segments: list[dict[str, Any]]) -> float:
    """Return the end time of the last non-empty segment."""
    nonempty = _iter_nonempty_segments(segments)
    if not nonempty:
        return 0.0
    return float(nonempty[-1]["end"])


def segments_to_srt(segments: list[dict[str, Any]]) -> str:
    """Render Whisper segments as SubRip (SRT) subtitles."""
    blocks: list[str] = []
    for index, segment in enumerate(_iter_nonempty_segments(segments), start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    (
                        f"{format_srt_timestamp(segment['start'])} --> "
                        f"{format_srt_timestamp(segment['end'])}"
                    ),
                    segment["text"],
                ]
            )
        )
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def segments_to_vtt(segments: list[dict[str, Any]]) -> str:
    """Render Whisper segments as WebVTT subtitles."""
    lines = ["WEBVTT", ""]
    for segment in _iter_nonempty_segments(segments):
        lines.append(
            f"{format_vtt_timestamp(segment['start'])} --> "
            f"{format_vtt_timestamp(segment['end'])}"
        )
        lines.append(segment["text"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_verbose_json_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Shape a verbose OpenAI-compatible transcription payload."""
    segments = payload.get("segments") or []
    verbose_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(_iter_nonempty_segments(list(segments))):
        entry: dict[str, Any] = {
            "id": segment.get("id", index),
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
        }
        if segment.get("words"):
            entry["words"] = segment["words"]
        verbose_segments.append(entry)

    return {
        "task": payload.get("task", "transcribe"),
        "language": payload.get("language"),
        "duration": payload.get("duration", segments_duration_seconds(list(segments))),
        "text": str(payload.get("text", "")).strip(),
        "segments": verbose_segments,
    }


def render_stt_response_body(
    payload: dict[str, Any],
    response_format: str,
) -> tuple[str | dict[str, Any], str]:
    """Return response body and media type for the requested format."""
    normalized = normalize_stt_response_format(response_format)
    text = str(payload.get("text", "")).strip()

    if normalized == "text":
        return text, "text/plain; charset=utf-8"
    if normalized == "verbose_json":
        return build_verbose_json_payload(payload), "application/json"
    if normalized == "srt":
        return segments_to_srt(list(payload.get("segments") or [])), "application/x-subrip"
    if normalized == "vtt":
        return segments_to_vtt(list(payload.get("segments") or [])), "text/vtt; charset=utf-8"
    return {"text": text}, "application/json"
