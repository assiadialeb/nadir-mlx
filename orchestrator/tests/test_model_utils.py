"""Tests for model completeness and capability detection."""

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from django.test import override_settings

from orchestrator.model_utils import (
    get_model_capabilities,
    is_model_complete,
    is_stt_focused_model,
    requires_relaxed_weight_loading,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class ModelUtilsCapabilityTests(TestCase):
    @patch("orchestrator.model_utils.supports_stt_mode", return_value=False)
    @patch("orchestrator.model_utils.supports_tts_mode", return_value=False)
    @patch("orchestrator.model_utils.supports_image_mode", return_value=False)
    @patch("orchestrator.model_utils.supports_rerank_mode", return_value=False)
    @patch("orchestrator.model_utils.supports_embedding_mode", return_value=True)
    @patch("orchestrator.model_utils.supports_multimodal_mode", return_value=False)
    @patch("orchestrator.model_utils.is_model_complete", return_value=True)
    def test_get_model_capabilities_flags_embedding(
        self,
        _mock_complete: object,
        _mock_vlm: object,
        _mock_embed: object,
        _mock_rerank: object,
        _mock_image: object,
        _mock_tts: object,
        _mock_stt: object,
    ) -> None:
        caps = get_model_capabilities("nomic-embed-text-v2")
        self.assertTrue(caps["supports_embedding"])
        self.assertFalse(caps["supports_text"])

    def test_requires_relaxed_weight_loading_when_backup_config_exists(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            with override_settings(MODELS_DIR=tmp_dir):
                model_dir = Path(tmp_dir) / "demo-model"
                model_dir.mkdir()
                (model_dir / "config.json.orig").write_text("{}", encoding="utf-8")
                self.assertTrue(requires_relaxed_weight_loading(model_dir))


class ModelUtilsCompletenessTests(TestCase):
    def test_whisper_npz_checkpoint_is_complete(self) -> None:
        model_path = REPO_ROOT / "models" / "whisper-small-mlx"
        if not model_path.is_dir():
            self.skipTest("whisper-small-mlx not downloaded locally")
        self.assertTrue(is_model_complete(model_path))
        self.assertTrue(is_stt_focused_model(model_path))

    def test_whisper_npz_4bit_checkpoint_is_complete(self) -> None:
        model_path = REPO_ROOT / "models" / "whisper-small-mlx-4bit"
        if not model_path.is_dir():
            self.skipTest("whisper-small-mlx-4bit not downloaded locally")
        self.assertTrue(is_model_complete(model_path))
        self.assertTrue(is_stt_focused_model(model_path))
