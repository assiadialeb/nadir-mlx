"""Tests for TTS response format helpers."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
from django.test import SimpleTestCase

from orchestrator.tts_audio_codec import (
    TtsFormatError,
    encode_speech_audio,
    iter_encoded_audio_chunks,
    normalize_tts_response_format,
    tts_media_type,
)


class TtsAudioCodecTests(SimpleTestCase):
    def test_normalize_tts_response_format_accepts_openai_formats(self) -> None:
        self.assertEqual(normalize_tts_response_format("OPUS"), "opus")
        self.assertEqual(normalize_tts_response_format("aac"), "aac")

    def test_normalize_tts_response_format_rejects_unknown(self) -> None:
        with self.assertRaises(TtsFormatError):
            normalize_tts_response_format("webm")

    def test_tts_media_type_maps_opus_and_aac(self) -> None:
        self.assertEqual(tts_media_type("opus"), "audio/opus")
        self.assertEqual(tts_media_type("aac"), "audio/aac")

    @patch("orchestrator.tts_audio_codec._encode_aac_with_ffmpeg")
    def test_encode_speech_audio_uses_mlx_audio_for_opus(
        self,
        _mock_encode_aac: MagicMock,
    ) -> None:
        fake_audio_io = MagicMock()

        def _write(
            buffer: io.BytesIO,
            audio: np.ndarray,
            sample_rate: int,
            format: str | None = None,
        ) -> None:
            buffer.write(b"opus-bytes")
            buffer.seek(0)

        fake_audio_io.write = _write
        fake_mlx_audio = MagicMock()
        fake_mlx_audio.audio_io = fake_audio_io
        with patch.dict("sys.modules", {"mlx_audio": fake_mlx_audio, "mlx_audio.audio_io": fake_audio_io}):
            payload, media_type = encode_speech_audio(
                np.array([0.0, 0.1], dtype=np.float32),
                24000,
                "opus",
            )
        self.assertEqual(payload, b"opus-bytes")
        self.assertEqual(media_type, "audio/opus")

    @patch("orchestrator.tts_audio_codec._encode_aac_with_ffmpeg")
    def test_encode_speech_audio_uses_ffmpeg_for_aac(
        self,
        mock_encode_aac: MagicMock,
    ) -> None:
        def _write(buffer: io.BytesIO, audio: np.ndarray, sample_rate: int) -> None:
            buffer.write(b"aac-bytes")
            buffer.seek(0)

        mock_encode_aac.side_effect = _write
        payload, media_type = encode_speech_audio(
            np.array([0.0, 0.1], dtype=np.float32),
            24000,
            "aac",
        )
        self.assertEqual(payload, b"aac-bytes")
        self.assertEqual(media_type, "audio/aac")
        mock_encode_aac.assert_called_once()

    def test_iter_encoded_audio_chunks_splits_payload(self) -> None:
        chunks = list(iter_encoded_audio_chunks(b"abcdefghij", chunk_size=4))
        self.assertEqual(chunks, [b"abcd", b"efgh", b"ij"])
