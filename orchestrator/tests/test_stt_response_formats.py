"""Tests for STT subtitle and response formatting."""

from django.test import SimpleTestCase

from orchestrator.stt_response_formats import (
    SttFormatError,
    build_verbose_json_payload,
    normalize_stt_response_format,
    render_stt_response_body,
    segments_to_srt,
    segments_to_vtt,
)


class SttResponseFormatTests(SimpleTestCase):
    def test_normalize_stt_response_format_accepts_srt_and_vtt(self) -> None:
        self.assertEqual(normalize_stt_response_format("SRT"), "srt")
        self.assertEqual(normalize_stt_response_format("vtt"), "vtt")

    def test_normalize_stt_response_format_rejects_unknown(self) -> None:
        with self.assertRaises(SttFormatError):
            normalize_stt_response_format("tsv")

    def test_segments_to_srt_renders_timestamped_blocks(self) -> None:
        payload = segments_to_srt(
            [
                {"start": 0.0, "end": 2.5, "text": "Hello world"},
                {"start": 2.5, "end": 5.0, "text": "Second line"},
            ]
        )
        self.assertIn("00:00,000 --> 00:02,500", payload)
        self.assertIn("Hello world", payload)
        self.assertIn("Second line", payload)

    def test_segments_to_vtt_includes_header(self) -> None:
        payload = segments_to_vtt(
            [{"start": 1.0, "end": 3.25, "text": "Bonjour"}],
        )
        self.assertTrue(payload.startswith("WEBVTT"))
        self.assertIn("00:01.000 --> 00:03.250", payload)
        self.assertIn("Bonjour", payload)

    def test_build_verbose_json_payload_includes_segments(self) -> None:
        payload = build_verbose_json_payload(
            {
                "task": "translate",
                "language": "french",
                "text": "Hello",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
            }
        )
        self.assertEqual(payload["task"], "translate")
        self.assertEqual(payload["segments"][0]["text"], "Hello")
        self.assertEqual(payload["duration"], 1.0)

    def test_render_stt_response_body_returns_json_by_default(self) -> None:
        body, media_type = render_stt_response_body({"text": "Hi"}, "json")
        self.assertEqual(body, {"text": "Hi"})
        self.assertEqual(media_type, "application/json")
