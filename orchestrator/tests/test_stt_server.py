"""Tests for STT server transcription helpers."""

from __future__ import annotations

import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from django.test import SimpleTestCase

from orchestrator.stt_response_formats import segments_to_srt


class SttTranscriptionPayloadTests(SimpleTestCase):
    def test_payload_from_stt_output_includes_segments(self) -> None:
        from orchestrator.stt_server import _transcription_payload_from_result

        result = SimpleNamespace(
            text="Hello world",
            language="english",
            segments=[
                {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello world"},
            ],
        )
        payload = _transcription_payload_from_result(result, task="transcribe")
        self.assertEqual(payload["text"], "Hello world")
        self.assertEqual(payload["task"], "transcribe")
        self.assertEqual(len(payload["segments"]), 1)

    def test_run_transcription_passes_translate_task(self) -> None:
        from orchestrator.stt_server import _run_transcription

        mock_model = MagicMock()
        mock_model.generate.return_value = SimpleNamespace(
            text="Hello",
            language="french",
            segments=[{"start": 0.0, "end": 1.0, "text": "Hello"}],
        )
        waveform = np.zeros(16000, dtype=np.float32)

        payload = _run_transcription(
            mock_model,
            waveform,
            task="translate",
            effective_language=None,
            effective_chunk=30.0,
            word_timestamps=False,
            prompt="context",
            temperature=0.2,
        )

        self.assertEqual(payload["task"], "translate")
        generate_kwargs = mock_model.generate.call_args.kwargs
        self.assertEqual(generate_kwargs["task"], "translate")
        self.assertEqual(generate_kwargs["initial_prompt"], "context")
        self.assertEqual(generate_kwargs["temperature"], 0.2)

    def test_srt_render_from_segments(self) -> None:
        srt = segments_to_srt(
            [{"start": 0.0, "end": 2.0, "text": "Test phrase"}],
        )
        self.assertIn("Test phrase", srt)


class SttAudioDecodeTests(SimpleTestCase):
    @unittest.skipIf(
        os.environ.get("NADIR_CI_STUB_MLX", "").strip().lower() in ("1", "true", "yes", "on"),
        "requires mlx_audio runtime (skipped on Linux CI)",
    )
    def test_decode_wav_bytes_without_temp_mp3_file(self) -> None:
        from mlx_audio.audio_io import write as audio_write

        from orchestrator.stt_server import _decode_uploaded_audio

        sample_rate = 16000
        waveform = (np.sin(np.linspace(0, 800, sample_rate)) * 0.1).astype(np.float32)
        buffer = io.BytesIO()
        audio_write(buffer, waveform, sample_rate, format="wav")

        decoded = _decode_uploaded_audio(buffer.getvalue())
        self.assertEqual(decoded.dtype, np.float32)
        self.assertEqual(decoded.shape[0], sample_rate)

