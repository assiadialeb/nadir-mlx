"""Unit tests for image model profiles and detection."""

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from orchestrator.image_model_loader import resolve_image_model_spec
from orchestrator.image_model_profiles import (
    apply_quantize_override,
    infer_quantize_from_name,
    parse_readme_inference_hints,
    resolve_image_profile,
)
from orchestrator.model_utils import is_image_focused_model, supports_image_mode
from orchestrator.server_manager import parse_launch_mode


class ImageModeTests(TestCase):
    def test_parse_launch_mode_accepts_image(self) -> None:
        self.assertEqual(parse_launch_mode("IMAGE"), "IMAGE")

    def test_infer_quantize_from_name_extracts_bits(self) -> None:
        self.assertEqual(infer_quantize_from_name("FLUX.1-schnell-4bit"), 4)
        self.assertEqual(infer_quantize_from_name("z-image-turbo-8bit"), 8)
        self.assertEqual(infer_quantize_from_name("Flux-1.lite-8B-MLX-Q4"), 4)
        self.assertIsNone(infer_quantize_from_name("flux-dev"))

    def test_parse_readme_inference_hints_for_flux_lite(self) -> None:
        model_dir = Path(self._temp_dir()) / "Flux-1.lite-8B-MLX-Q4"
        model_dir.mkdir()
        (model_dir / "README.md").write_text(
            "uvx --from mflux mflux-generate --base-model dev --steps 50 "
            "--guidance 4.0 --width 1024 --height 1024 -q 4 "
            "--model mlx-community/flux.1-lite-8B-MLX-Q4 --prompt 'Test'\n",
            encoding="utf-8",
        )

        hints = parse_readme_inference_hints(model_dir)
        self.assertEqual(hints["flux_base_model"], "dev")
        self.assertEqual(hints["quality_steps"], 50)
        self.assertEqual(hints["default_guidance"], 4.0)
        self.assertEqual(hints["quantize"], 4)

    def test_resolve_image_profile_flux_lite_uses_readme_and_from_name(self) -> None:
        model_dir = Path(self._temp_dir()) / "Flux-1.lite-8B-MLX-Q4"
        model_dir.mkdir()
        (model_dir / "README.md").write_text(
            "mflux-generate --base-model dev --steps 50 --guidance 4.0 -q 4 "
            "--model mlx-community/flux.1-lite-8B-MLX-Q4 --prompt test\n",
            encoding="utf-8",
        )

        profile = resolve_image_profile(model_dir)
        self.assertEqual(profile.profile_id, "flux_lite")
        self.assertEqual(profile.family, "flux1")
        self.assertEqual(profile.flux_base_model, "dev")
        self.assertEqual(profile.quantize, 4)
        self.assertEqual(profile.default_steps, 20)
        self.assertEqual(profile.fast_steps, 12)
        self.assertEqual(profile.quality_steps, 50)
        self.assertEqual(profile.default_guidance, 4.0)
        self.assertEqual(profile.source, "readme")
        self.assertEqual(profile.resolve_steps("balanced"), 20)
        self.assertEqual(profile.resolve_steps("quality"), 50)
        self.assertEqual(profile.resolve_steps("fast"), 12)

    def test_resolve_image_model_spec_flux_schnell(self) -> None:
        profile = resolve_image_model_spec(Path("FLUX.1-schnell-4bit"))
        self.assertEqual(profile.profile_id, "flux_schnell")
        self.assertEqual(profile.flux_base_model, "schnell")
        self.assertEqual(profile.default_steps, 4)
        self.assertFalse(profile.use_guidance)

    def test_resolve_image_model_spec_z_image(self) -> None:
        profile = resolve_image_model_spec(Path("Z-Image-Turbo-4bit"))
        self.assertEqual(profile.profile_id, "z_image_turbo")
        self.assertEqual(profile.default_steps, 9)

    def test_resolve_image_model_spec_flux2_klein(self) -> None:
        profile = resolve_image_model_spec(Path("flux2-klein-9b-4bit"))
        self.assertEqual(profile.profile_id, "flux2_klein_9b")
        self.assertEqual(profile.default_steps, 4)

    def test_supports_image_mode_requires_weights(self) -> None:
        model_dir = Path(self._temp_dir()) / "FLUX.1-schnell-4bit"
        model_dir.mkdir()
        self.assertFalse(supports_image_mode(model_dir))

        weight_file = model_dir / "model.safetensors"
        weight_file.write_bytes(b"\x00" * 128)
        self.assertTrue(supports_image_mode(model_dir))
        self.assertTrue(is_image_focused_model(model_dir))

    def test_resolve_image_profile_rejects_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            resolve_image_profile(Path("Qwen3-Embedding-0.6B-4bit"))

    def test_apply_quantize_override_replaces_inferred_bits(self) -> None:
        profile = resolve_image_model_spec(Path("FLUX.1-schnell-4bit"))
        self.assertEqual(profile.quantize, 4)

        overridden = apply_quantize_override(profile, 8)
        self.assertEqual(overridden.quantize, 8)
        self.assertEqual(profile.quantize, 4)

    def test_apply_quantize_override_rejects_non_positive(self) -> None:
        profile = resolve_image_model_spec(Path("FLUX.1-schnell-4bit"))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            apply_quantize_override(profile, 0)

    @patch("orchestrator.server_manager._get_python_bin", return_value="python")
    def test_build_launch_command_image_passes_quantize_override(
        self,
        _mock_python_bin: object,
    ) -> None:
        from orchestrator.server_manager import _build_launch_command

        command = _build_launch_command(
            "/tmp/flux",
            11400,
            "IMAGE",
            {
                "host": "127.0.0.1",
                "advanced": {"quantize_override": 8},
            },
            "flux-model",
        )
        self.assertIn("--quantize-override", command)
        self.assertEqual(command[command.index("--quantize-override") + 1], "8")

    def _temp_dir(self) -> str:
        import tempfile

        return tempfile.mkdtemp()
