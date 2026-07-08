"""Tests for STT SSE streaming helpers (MLX-88)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from django.test import SimpleTestCase

from orchestrator.stt_streaming import encode_sse_event, iter_transcription_sse


def _parse_sse_events(raw: bytes) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in raw.decode("utf-8").strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data_line = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            if line.startswith("data:"):
                data_line = line.split(":", 1)[1].strip()
        events.append((event_name, json.loads(data_line)))
    return events


class SttStreamingTests(SimpleTestCase):
    def test_encode_sse_event_formats_frame(self) -> None:
        frame = encode_sse_event("transcript", {"text": "hi"})
        self.assertIn(b"event: transcript", frame)
        self.assertIn(b'"text": "hi"', frame)

    def test_iter_transcription_sse_uses_generate_streaming(self) -> None:
        mock_model = MagicMock()
        mock_model.generate_streaming.return_value = [
            SimpleNamespace(text="Hello", is_final=False),
            SimpleNamespace(text="Hello world", is_final=True),
        ]
        chunks = b"".join(
            iter_transcription_sse(
                mock_model,
                np.zeros(1600, dtype=np.float32),
                task="transcribe",
                effective_language=None,
                effective_chunk=30.0,
                word_timestamps=False,
                prompt=None,
                temperature=None,
                payload_from_result=MagicMock(),
            ),
        )
        events = _parse_sse_events(chunks)
        event_names = [name for name, _payload in events]
        self.assertIn("transcript", event_names)
        self.assertEqual(events[-1][0], "done")
        texts = [payload["text"] for _name, payload in events if payload.get("text")]
        self.assertIn("Hello world", texts)

    def test_iter_transcription_sse_falls_back_to_batch_generate(self) -> None:
        mock_model = MagicMock()
        del mock_model.generate_streaming
        mock_model.generate.return_value = SimpleNamespace(
            text="Batch only",
            language="english",
            segments=[],
        )

        def payload_from_result(result: object, *, task: str) -> dict:
            return {"text": "Batch only", "task": task, "segments": [], "duration": 1.0}

        chunks = b"".join(
            iter_transcription_sse(
                mock_model,
                np.zeros(1600, dtype=np.float32),
                task="transcribe",
                effective_language=None,
                effective_chunk=30.0,
                word_timestamps=False,
                prompt=None,
                temperature=None,
                payload_from_result=payload_from_result,
            ),
        )
        events = _parse_sse_events(chunks)
        completed = [payload for name, payload in events if name == "transcript"][-1]
        self.assertTrue(completed["is_final"])
        self.assertEqual(completed["text"], "Batch only")
        mock_model.generate.assert_called_once()
