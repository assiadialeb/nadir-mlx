"""Tests for model completeness and capability detection."""

from pathlib import Path
from unittest import TestCase

from orchestrator.model_utils import is_model_complete, is_stt_focused_model

REPO_ROOT = Path(__file__).resolve().parents[2]


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
