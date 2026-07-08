"""Tests for TTS/STT launch modes and model detection."""

from unittest import TestCase
from unittest.mock import patch

from django.test import override_settings

from orchestrator.model_utils import (
    get_model_capabilities,
    is_stt_focused_model,
    is_tts_focused_model,
)
from orchestrator.server_config_schema import (
    build_default_server_config,
    validate_and_normalize_server_config,
)
from orchestrator.server_manager import parse_launch_mode


class AudioModeTests(TestCase):
    def test_parse_launch_mode_accepts_tts_and_stt(self) -> None:
        self.assertEqual(parse_launch_mode("TTS"), "TTS")
        self.assertEqual(parse_launch_mode("STT"), "STT")

    def test_build_default_server_config_tts(self) -> None:
        config = build_default_server_config("TTS")
        assert config["voice_id"] == "ff_siwis"
        self.assertAlmostEqual(config["speaking_rate"], 1.0)
        assert config["lang_code"] == "f"

    def test_build_default_server_config_stt(self) -> None:
        config = build_default_server_config("STT")
        self.assertAlmostEqual(config["chunk_duration"], 30.0)

    def test_validate_tts_config(self) -> None:
        config = validate_and_normalize_server_config(
            "TTS",
            {"voice_id": "am_adam", "speaking_rate": 1.2, "lang_code": "b"},
            "Kokoro-82M-bf16",
        )
        assert config["voice_id"] == "am_adam"
        self.assertAlmostEqual(config["speaking_rate"], 1.2)
        assert config["model_id"] == "Kokoro-82M-bf16"

    def test_validate_stt_config_with_optional_language(self) -> None:
        config = validate_and_normalize_server_config(
            "STT",
            {"language": "fr", "chunk_duration": 15},
            "whisper-large-v3-turbo-asr-fp16",
        )
        assert config["language"] == "fr"
        assert config["chunk_duration"] == 15

    @override_settings(MODELS_DIR="/tmp")
    @patch("orchestrator.model_utils.is_model_complete", return_value=True)
    def test_tts_detection_from_folder_name(self, _mock_complete: object) -> None:
        self.assertTrue(is_tts_focused_model("/tmp/Kokoro-82M-bf16"))
        self.assertFalse(is_tts_focused_model("/tmp/Llama-3-8B"))

    @override_settings(MODELS_DIR="/tmp")
    @patch("orchestrator.model_utils.is_model_complete", return_value=True)
    def test_stt_detection_from_folder_name(self, _mock_complete: object) -> None:
        self.assertTrue(is_stt_focused_model("/tmp/whisper-large-v3-turbo-asr-fp16"))
        self.assertFalse(is_stt_focused_model("/tmp/Kokoro-82M-bf16"))

    def test_capabilities_flags_for_audio_models(self) -> None:
        caps = get_model_capabilities("Kokoro-82M-bf16")
        if caps["supports_tts"]:
            self.assertFalse(caps["supports_text"])
        caps = get_model_capabilities("whisper-large-v3-turbo-asr-fp16")
        if caps["supports_stt"]:
            self.assertFalse(caps["supports_text"])

    @override_settings(MODELS_DIR="/tmp")
    @patch("orchestrator.server_manager._get_python_bin", return_value="python")
    def test_build_launch_command_tts_passes_response_format(
        self,
        _mock_python_bin: object,
    ) -> None:
        from orchestrator.server_manager import _build_launch_command

        command = _build_launch_command(
            "/tmp/kokoro",
            11400,
            "TTS",
            {
                "host": "127.0.0.1",
                "advanced": {"response_format": "opus"},
            },
            "kokoro-model",
        )
        self.assertIn("--default-response-format", command)
        self.assertEqual(command[command.index("--default-response-format") + 1], "opus")
