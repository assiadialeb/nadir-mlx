"""Tests for server configuration schema validation."""

import json
from unittest import TestCase
from unittest.mock import patch

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

    def test_validate_multimodal_advanced_keys(self) -> None:
        config = validate_and_normalize_server_config(
            "MULTIMODAL",
            {
                "advanced": {
                    "draft_kind": "mtp",
                    "enable_thinking": True,
                    "kv_bits": 4,
                },
            },
            "gemma-4-e2b",
        )
        self.assertEqual(config["advanced"]["draft_kind"], "mtp")
        self.assertTrue(config["advanced"]["enable_thinking"])

    def test_validate_normalizes_gateway_aliases_list(self) -> None:
        config = validate_and_normalize_server_config(
            "TEXT",
            {"model_id": "primary-chat", "gateway_aliases": "dev-chat, prod-chat"},
            "folder-name",
        )
        self.assertEqual(config["gateway_aliases"], ["dev-chat", "prod-chat"])

    def test_validate_image_quantize_override_advanced(self) -> None:
        config = validate_and_normalize_server_config(
            "IMAGE",
            {"advanced": {"quantize_override": 8}},
            "FLUX.1-schnell-4bit",
        )
        self.assertEqual(config["advanced"]["quantize_override"], 8)

    def test_validate_tts_response_format_advanced(self) -> None:
        config = validate_and_normalize_server_config(
            "TTS",
            {"advanced": {"response_format": "opus"}},
            "Kokoro-82M-bf16",
        )
        self.assertEqual(config["advanced"]["response_format"], "opus")

    def test_parse_post_rejects_invalid_advanced_json(self) -> None:
        with self.assertRaises(ValueError):
            parse_server_config_from_post(
                {"config_advanced": "not-json"},
                "TEXT",
                "model-a",
            )

    def test_config_fields_for_ui_json_includes_ops_section(self) -> None:
        from orchestrator.server_config_schema import config_fields_for_ui_json

        payload = json.loads(config_fields_for_ui_json())
        text_fields = {field["name"]: field for field in payload["TEXT"]}
        self.assertEqual(text_fields["lifecycle_mode"]["section"], "ops")

    def test_validate_rejects_invalid_draft_kind_enum(self) -> None:
        with self.assertRaisesRegex(ValueError, "advanced.draft_kind"):
            validate_and_normalize_server_config(
                "MULTIMODAL",
                {"advanced": {"draft_kind": "invalid-kind"}},
                "gemma-4-e2b",
            )

    def test_validate_rejects_num_draft_tokens_out_of_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "advanced.num_draft_tokens"):
            validate_and_normalize_server_config(
                "TEXT",
                {"advanced": {"num_draft_tokens": 99}},
                "llama-text",
            )

    def test_advanced_schema_for_ui_json_includes_version(self) -> None:
        from orchestrator.server_config_schema import advanced_schema_for_ui_json

        payload = json.loads(advanced_schema_for_ui_json())
        self.assertEqual(payload["version"], 1)
        self.assertIn("draft_kind", payload["fields"])

    def test_text_mtp_preview_extends_whitelist(self) -> None:
        from orchestrator.server_config_schema import advanced_whitelist_for_mode

        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("NADIR_TEXT_MTP_PREVIEW", None)
            base_keys = advanced_whitelist_for_mode("TEXT")
            self.assertNotIn("draft_kind", base_keys)
        with patch.dict("os.environ", {"NADIR_TEXT_MTP_PREVIEW": "1"}):
            preview_keys = advanced_whitelist_for_mode("TEXT")
            self.assertIn("draft_kind", preview_keys)
