"""Unit tests for image model detection and spec resolution."""

from pathlib import Path
from unittest import TestCase

from orchestrator.image_model_loader import infer_quantize_from_name, resolve_image_model_spec
from orchestrator.model_utils import is_image_focused_model, supports_image_mode
from orchestrator.server_manager import parse_launch_mode


class ImageModeTests(TestCase):
    def test_parse_launch_mode_accepts_image(self) -> None:
        self.assertEqual(parse_launch_mode("IMAGE"), "IMAGE")

    def test_infer_quantize_from_name_extracts_bits(self) -> None:
        self.assertEqual(infer_quantize_from_name("FLUX.1-schnell-4bit"), 4)
        self.assertEqual(infer_quantize_from_name("z-image-turbo-8bit"), 8)
        self.assertIsNone(infer_quantize_from_name("flux-dev"))

    def test_resolve_image_model_spec_flux_schnell(self) -> None:
        spec = resolve_image_model_spec(Path("FLUX.1-schnell-4bit"))
        self.assertEqual(spec.family, "flux1")
        self.assertEqual(spec.config_attr, "schnell")
        self.assertEqual(spec.quantize, 4)
        self.assertEqual(spec.default_steps, 4)

    def test_resolve_image_model_spec_z_image(self) -> None:
        spec = resolve_image_model_spec(Path("Z-Image-Turbo-4bit"))
        self.assertEqual(spec.family, "z_image")
        self.assertEqual(spec.config_attr, "z_image_turbo")

    def test_resolve_image_model_spec_flux2_klein(self) -> None:
        spec = resolve_image_model_spec(Path("flux2-klein-9b-4bit"))
        self.assertEqual(spec.family, "flux2")
        self.assertEqual(spec.config_attr, "flux2_klein_9b")

    def test_supports_image_mode_requires_weights(self) -> None:
        model_dir = Path(self._temp_dir()) / "FLUX.1-schnell-4bit"
        model_dir.mkdir()
        self.assertFalse(supports_image_mode(model_dir))

        weight_file = model_dir / "model.safetensors"
        weight_file.write_bytes(b"\x00" * 128)
        self.assertTrue(supports_image_mode(model_dir))
        self.assertTrue(is_image_focused_model(model_dir))

    def test_resolve_image_model_spec_rejects_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            resolve_image_model_spec(Path("Qwen3-Embedding-0.6B-4bit"))

    def _temp_dir(self) -> str:
        import tempfile

        return tempfile.mkdtemp()
