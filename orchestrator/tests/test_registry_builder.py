"""Tests for offline registry builder helpers."""

from django.test import SimpleTestCase

from orchestrator.registry_builder import (
    build_model_entry,
    extract_upstream_repo,
    infer_family_id,
    merge_model_entry,
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

    def test_extract_upstream_prefers_converted_from_over_yaml_list(self) -> None:
        readme = (
            "---\n"
            "base_model:\n"
            "- yl4579/StyleTTS2-LJSpeech\n"
            "pipeline_tag: text-to-speech\n"
            "---\n"
            "converted to MLX format from [`hexagrad/Kokoro-82M`]() using mlx-audio\n"
        )
        upstream = extract_upstream_repo({}, readme)
        self.assertEqual(upstream, "hexagrad/Kokoro-82M")

    def test_extract_upstream_strips_yaml_list_prefix(self) -> None:
        readme = "base_model:\n- Qwen/Qwen3-0.6B\n"
        upstream = extract_upstream_repo({}, readme)
        self.assertEqual(upstream, "Qwen/Qwen3-0.6B")

    def test_normalize_repo_reference_rejects_absolute_paths(self) -> None:
        readme = (
            "converted to MLX format from "
            "[`/Volumes/T7/Models/hf-models/gemma-3-4b-it-qat-q4_0-unquantized`]() "
            "using mlx-vlm\n"
            "base_model:\n- google/gemma-3-4b-it-qat-q4_0-unquantized\n"
        )
        upstream = extract_upstream_repo(
            {},
            readme,
            folder_name="gemma-3-4b-it-qat-4bit",
            family_id="gemma3_vlm",
        )
        self.assertEqual(upstream, "google/gemma-3-4b-it-qat-q4_0-unquantized")

    def test_extract_upstream_rejects_internvl_for_gemma_folder(self) -> None:
        readme = (
            "---\n"
            "base_model:\n"
            "- OpenGVLab/InternVL3-1B-Instruct\n"
            "pipeline_tag: image-text-to-text\n"
            "---\n"
            "converted to MLX format from "
            "[`/Volumes/T7/Models/hf-models/gemma-3-4b-it-qat-q4_0-unquantized`]() "
            "using mlx-vlm\n"
        )
        upstream = extract_upstream_repo(
            {"cardData": {"base_model": "OpenGVLab/InternVL3-1B-Instruct"}},
            readme,
            folder_name="gemma-3-4b-it-qat-4bit",
            family_id="gemma3_vlm",
        )
        self.assertEqual(upstream, "google/gemma-3-4b-it-qat-q4_0-unquantized")

    def test_infer_family_for_flux_lite_folder(self) -> None:
        family = infer_family_id("Flux-1.lite-8B-MLX-Q4", "text-to-image")
        self.assertEqual(family, "flux_lite")

    def test_infer_family_for_parakeet(self) -> None:
        family = infer_family_id("parakeet-tdt-0.6b-v3", "automatic-speech-recognition")
        self.assertEqual(family, "parakeet_stt")

    def test_infer_family_for_llama(self) -> None:
        family = infer_family_id("Llama-3.2-3B-Instruct-4bit", "text-generation")
        self.assertEqual(family, "llama_text")

    def test_infer_family_for_gemma3_text(self) -> None:
        family = infer_family_id("gemma-3-1b-it-qat-4bit", "text-generation")
        self.assertEqual(family, "gemma3_text")

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

    def test_merge_model_entry_keeps_mlx_repo_id(self) -> None:
        merged = merge_model_entry(
            {"repo_id": "mlx-community/Kokoro-82M-6bit", "upstream": "hexagrad/Kokoro-82M"},
            {"repo_id": "local/Kokoro-82M-6bit", "readme_hints": {"source": "readme"}},
        )
        self.assertEqual(merged["repo_id"], "mlx-community/Kokoro-82M-6bit")
        self.assertEqual(merged["readme_hints"]["source"], "readme")

    def test_merge_registry_preserves_families(self) -> None:
        existing = {
            "version": 1,
            "families": {"kokoro_tts": {"launch_modes": ["TTS"]}},
            "models": {},
            "patterns": [],
        }
        merged = merge_registry(existing, {"Kokoro-82M-6bit": {"family": "kokoro_tts"}})
        self.assertIn("kokoro_tts", merged["families"])
        self.assertIn("llama_text", merged["families"])
        self.assertIn("Kokoro-82M-6bit", merged["models"])
