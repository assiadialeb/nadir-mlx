"""Tests for legacy Whisper asset bootstrapping."""

from pathlib import Path
from unittest import TestCase

from orchestrator.whisper_assets import (
    ensure_whisper_hf_assets,
    is_legacy_mlx_whisper_checkpoint,
    is_stt_servable,
    resolve_legacy_whisper_processor_repo,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class WhisperAssetsTests(TestCase):
    def test_resolve_legacy_repo_for_small_mlx(self) -> None:
        self.assertEqual(
            resolve_legacy_whisper_processor_repo("whisper-small-mlx-4bit"),
            "openai/whisper-small",
        )

    def test_legacy_checkpoint_detection(self) -> None:
        model_path = REPO_ROOT / "models" / "whisper-small-mlx"
        if not model_path.is_dir():
            self.skipTest("whisper-small-mlx not present")
        self.assertTrue(is_legacy_mlx_whisper_checkpoint(model_path))

    def test_legacy_whisper_is_stt_servable(self) -> None:
        model_path = REPO_ROOT / "models" / "whisper-small-mlx"
        if not model_path.is_dir():
            self.skipTest("whisper-small-mlx not present")
        self.assertTrue(is_stt_servable(model_path))

    def test_bootstrap_adds_preprocessor_config(self) -> None:
        model_path = REPO_ROOT / "models" / "whisper-small-mlx-4bit"
        if not model_path.is_dir():
            self.skipTest("whisper-small-mlx-4bit not present")
        ensure_whisper_hf_assets(model_path)
        self.assertTrue((model_path / "preprocessor_config.json").is_file())
