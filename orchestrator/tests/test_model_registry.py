"""Tests for curated model registry defaults."""

from django.test import SimpleTestCase

from orchestrator.model_registry import (
    apply_registry_server_defaults,
    build_registry_defaults_for_folders,
    deep_merge_registry_defaults,
    get_registry_family,
    resolve_model_registry_profile,
    resolve_registry_entry,
    resolve_registry_family,
)
from orchestrator.server_config_schema import build_default_server_config


class ModelRegistryTests(SimpleTestCase):
    def test_resolve_flux_lite_by_exact_name(self) -> None:
        family = resolve_registry_family("Flux-1.lite-8B-MLX-Q4")
        self.assertEqual(family, "flux_lite")

    def test_resolve_whisper_by_pattern(self) -> None:
        family = resolve_registry_family("whisper-large-v3-turbo-asr-fp16")
        self.assertEqual(family, "whisper_stt")

    def test_resolve_kokoro_image_profile_for_tts_mode(self) -> None:
        profile = resolve_model_registry_profile("Kokoro-82M-6bit", "TTS")
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile["family_id"], "kokoro_tts")
        self.assertEqual(profile["server_config"]["voice_id"], "ff_siwis")
        self.assertEqual(profile["server_config"]["lang_code"], "f")

    def test_resolve_returns_none_for_wrong_launch_mode(self) -> None:
        profile = resolve_model_registry_profile("Kokoro-82M-6bit", "IMAGE")
        self.assertIsNone(profile)

    def test_apply_registry_server_defaults_merges_tts_fields(self) -> None:
        base = build_default_server_config("TTS")
        merged = apply_registry_server_defaults("TTS", "Kokoro-82M-6bit", base)
        self.assertEqual(merged["voice_id"], "ff_siwis")
        self.assertEqual(merged["registry"]["family_id"], "kokoro_tts")

    def test_build_defaults_for_installed_folders(self) -> None:
        defaults = build_registry_defaults_for_folders(
            ["Flux-1.lite-8B-MLX-Q4", "Kokoro-82M-6bit"],
        )
        self.assertEqual(defaults["IMAGE"]["Flux-1.lite-8B-MLX-Q4"]["default_quality"], "balanced")
        self.assertEqual(defaults["TTS"]["Kokoro-82M-6bit"]["voice_id"], "ff_siwis")

    def test_build_default_server_config_uses_registry(self) -> None:
        config = build_default_server_config("STT", "whisper-small-mlx")
        self.assertEqual(config["chunk_duration"], 30.0)
        self.assertEqual(config["registry"]["family_id"], "whisper_stt")

    def test_deep_merge_registry_defaults_merges_nested_advanced(self) -> None:
        base = {"advanced": {}, "max_tokens": 256}
        defaults = {"advanced": {"draft_kind": "mtp"}, "max_tokens": 512}
        merged = deep_merge_registry_defaults(base, defaults)
        self.assertEqual(merged["advanced"]["draft_kind"], "mtp")
        self.assertEqual(merged["max_tokens"], 256)

    def test_apply_registry_server_defaults_merges_gemma4_mtp_advanced(self) -> None:
        base = build_default_server_config("MULTIMODAL", "gemma-4-E4B-it-qat-4bit")
        merged = apply_registry_server_defaults(
            "MULTIMODAL",
            "gemma-4-E4B-it-qat-4bit",
            base,
        )
        self.assertEqual(merged["advanced"]["draft_kind"], "mtp")
        self.assertEqual(merged["registry"]["family_id"], "gemma4_vlm")

    def test_get_registry_family_returns_launch_modes(self) -> None:
        family = get_registry_family("flux_lite")
        self.assertIsNotNone(family)
        assert family is not None
        self.assertIn("IMAGE", family.get("launch_modes", []))

    def test_resolve_registry_entry_returns_explicit_model(self) -> None:
        entry = resolve_registry_entry("Flux-1.lite-8B-MLX-Q4")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry["folder_name"], "Flux-1.lite-8B-MLX-Q4")

    def test_resolve_registry_family_returns_none_for_unknown_folder(self) -> None:
        self.assertIsNone(resolve_registry_family("totally-unknown-model-folder-xyz"))
