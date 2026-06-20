"""Tests for STT server audio decoding."""

import io

import numpy as np
from django.test import SimpleTestCase
from fastapi import HTTPException
from mlx_audio.audio_io import write as audio_write

from orchestrator.stt_server import _decode_uploaded_audio


class SttAudioDecodeTests(SimpleTestCase):
    def test_decode_wav_bytes_without_temp_mp3_file(self) -> None:
        sample_rate = 16000
        waveform = (np.sin(np.linspace(0, 800, sample_rate)) * 0.1).astype(np.float32)
        buffer = io.BytesIO()
        audio_write(buffer, waveform, sample_rate, format="wav")

        decoded = _decode_uploaded_audio(buffer.getvalue())
        self.assertEqual(decoded.dtype, np.float32)
        self.assertEqual(decoded.shape[0], sample_rate)
