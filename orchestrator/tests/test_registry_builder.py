"""Tests for offline registry builder helpers."""

from django.test import SimpleTestCase

from orchestrator.registry_builder import (
    build_model_entry,
    extract_upstream_repo,
    infer_family_id,
    merge_registry,
)


class RegistryBuilderTests(SimpleTestCase):
    def test_extract_upstream_from_readme_link(self) -> None:
        readme = (
            "> Original Model Link : "
            "[https://huggingface.co/Freepik/flux.1-lite-8B]"
            "(https://huggingface.co/Freepik/flux.1-lite-8B)\n"
        )
        upstream = extract_upstream_repo({}, readme)
        self.assertEqual(upstream, "Freepik/flux.1-lite-8B")

    def test_infer_family_for_flux_lite_folder(self) -> None:
        family = infer_family_id("Flux-1.lite-8B-MLX-Q4", "text-to-image")
        self.assertEqual(family, "flux_lite")

    def test_build_model_entry_includes_readme_hints(self) -> None:
        readme = (
            "mflux-generate --base-model dev --steps 50 --guidance 4.0 -q 4 "
            "--model mlx-community/flux.1-lite-8B-MLX-Q4 --prompt test\n"
        )
        card = {
            "pipeline_tag": "text-to-image",
            "cardData": {"base_model": "Freepik/flux.1-lite-8B"},
        }
        entry = build_model_entry("mlx-community/flux.1-lite-8B-MLX-Q4", card, readme)
        assert entry is not None
        self.assertEqual(entry["family"], "flux_lite")
        self.assertEqual(entry["upstream"], "Freepik/flux.1-lite-8B")
        self.assertEqual(entry["readme_hints"]["quality_steps"], 50)

    def test_merge_registry_preserves_families(self) -> None:
        existing = {
            "version": 1,
            "families": {"kokoro_tts": {"launch_modes": ["TTS"]}},
            "models": {},
            "patterns": [],
        }
        merged = merge_registry(existing, {"Kokoro-82M-6bit": {"family": "kokoro_tts"}})
        self.assertIn("kokoro_tts", merged["families"])
        self.assertIn("Kokoro-82M-6bit", merged["models"])
