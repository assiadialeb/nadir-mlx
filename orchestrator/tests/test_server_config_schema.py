"""Tests for server configuration schema validation."""

from unittest import TestCase

from orchestrator.server_config_schema import (
    _default_bind_host,
    build_default_server_config,
    parse_server_config_from_post,
    validate_and_normalize_server_config,
)


class ServerConfigSchemaTests(TestCase):
    def test_build_default_server_config_image_quality(self) -> None:
        config = build_default_server_config("IMAGE")
        self.assertEqual(config["default_quality"], "balanced")
        self.assertEqual(config["host"], _default_bind_host())

    def test_validate_sets_model_id_from_folder_name(self) -> None:
        config = validate_and_normalize_server_config("TEXT", {}, "Llama-3-8B")
        self.assertEqual(config["model_id"], "Llama-3-8B")
        self.assertEqual(config["advanced"], {})

    def test_validate_rejects_unknown_advanced_keys(self) -> None:
        with self.assertRaises(ValueError):
            validate_and_normalize_server_config(
                "TEXT",
                {"advanced": {"unknown_flag": True}},
                "model-a",
            )

    def test_parse_post_checkbox_and_numbers(self) -> None:
        config = parse_server_config_from_post(
            {
                "config_host": "127.0.0.1",
                "config_model_id": "my-chat",
                "config_max_tokens": "4096",
                "config_trust_remote_code": "on",
            },
            "TEXT",
            "folder-name",
        )
        self.assertEqual(config["host"], "127.0.0.1")
        self.assertEqual(config["model_id"], "my-chat")
        self.assertEqual(config["max_tokens"], 4096)
        self.assertTrue(config["trust_remote_code"])

    def test_validate_tts_config_defaults(self) -> None:
        config = validate_and_normalize_server_config("TTS", {}, "Kokoro-82M-bf16")
        self.assertEqual(config["voice_id"], "ff_siwis")
        self.assertEqual(config["lang_code"], "f")
        self.assertEqual(config["model_id"], "Kokoro-82M-bf16")

    def test_validate_stt_config_defaults(self) -> None:
        config = validate_and_normalize_server_config("STT", {}, "whisper-model")
        self.assertEqual(config["chunk_duration"], 30.0)
        self.assertNotIn("language", config)

    def test_validate_lifecycle_ops_defaults(self) -> None:
        config = validate_and_normalize_server_config("TEXT", {}, "model-a")
        self.assertEqual(config["ops"]["lifecycle_mode"], "always_on")
        self.assertEqual(config["ops"]["idle_minutes"], 30)

    def test_validate_lifecycle_on_demand(self) -> None:
        config = validate_and_normalize_server_config(
            "TEXT",
            {"ops": {"lifecycle_mode": "on_demand", "idle_minutes": 60}},
            "model-a",
        )
        self.assertEqual(config["ops"]["lifecycle_mode"], "on_demand")
        self.assertEqual(config["ops"]["idle_minutes"], 60)

    def test_validate_rejects_invalid_lifecycle_mode(self) -> None:
        with self.assertRaises(ValueError):
            validate_and_normalize_server_config(
                "TEXT",
                {"ops": {"lifecycle_mode": "sleepy"}},
                "model-a",
            )

    def test_validate_rejects_idle_minutes_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            validate_and_normalize_server_config(
                "TEXT",
                {"ops": {"lifecycle_mode": "on_demand", "idle_minutes": 2}},
                "model-a",
            )
