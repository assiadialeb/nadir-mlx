"""Tests for curated model registry defaults."""

from django.test import SimpleTestCase

from orchestrator.model_registry import (
    apply_registry_server_defaults,
    build_registry_defaults_for_folders,
    resolve_model_registry_profile,
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
