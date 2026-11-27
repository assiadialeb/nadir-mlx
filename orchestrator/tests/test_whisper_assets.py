"""Tests for legacy Whisper asset bootstrapping."""

import json
import tempfile
import typing
from pathlib import Path
from unittest import TestCase

from orchestrator.whisper_assets import (
    ensure_whisper_hf_assets,
    is_legacy_mlx_whisper_checkpoint,
    is_stt_servable,
    resolve_legacy_whisper_processor_repo,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_legacy_whisper_checkpoint(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "weights.npz").write_bytes(b"")
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "whisper",
                "architectures": ["WhisperForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )


class WhisperAssetsTests(TestCase):
    def test_resolve_legacy_repo_for_small_mlx(self) -> None:
        self.assertEqual(
            resolve_legacy_whisper_processor_repo("whisper-small-mlx-4bit"),
            "openai/whisper-small",
        )

    def test_ensure_whisper_hf_assets_type_hints_resolve(self) -> None:
        hints = typing.get_type_hints(ensure_whisper_hf_assets)
        self.assertIn("model_path", hints)

    def test_is_stt_servable_type_hints_resolve(self) -> None:
        hints = typing.get_type_hints(is_stt_servable)
        self.assertIn("model_path", hints)

    def test_ensure_whisper_hf_assets_accepts_str_path_when_preprocessor_present(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "whisper-small-mlx"
            model_path.mkdir()
            (model_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            ensure_whisper_hf_assets(str(model_path))

    def test_is_stt_servable_accepts_str_path_for_legacy_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "whisper-small-mlx-4bit"
            _write_legacy_whisper_checkpoint(model_path)
            self.assertTrue(is_stt_servable(str(model_path)))

    def test_is_stt_servable_rejects_legacy_folder_without_known_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "whisper-custom-unknown"
            _write_legacy_whisper_checkpoint(model_path)
            self.assertFalse(is_stt_servable(model_path))

    def test_is_stt_servable_rejects_missing_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "whisper-small-mlx"
            model_path.mkdir()
            (model_path / "config.json").write_text(
                '{"model_type":"whisper"}', encoding="utf-8"
            )
            self.assertFalse(is_stt_servable(model_path))

    def test_is_stt_servable_rejects_invalid_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "whisper-small-mlx"
            _write_legacy_whisper_checkpoint(model_path)
            (model_path / "config.json").write_text("not-json", encoding="utf-8")
            self.assertFalse(is_stt_servable(model_path))

    def test_is_stt_servable_accepts_folder_with_preprocessor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "any-checkpoint"
            model_path.mkdir()
            (model_path / "config.json").write_text(
                '{"model_type":"whisper"}', encoding="utf-8"
            )
            (model_path / "weights.npz").write_bytes(b"")
            (model_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            self.assertTrue(is_stt_servable(model_path))

    def test_is_stt_servable_rejects_non_whisper_model_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "llama-model"
            model_path.mkdir()
            (model_path / "config.json").write_text(
                '{"model_type":"llama"}', encoding="utf-8"
            )
            (model_path / "weights.npz").write_bytes(b"")
            self.assertFalse(is_stt_servable(model_path))

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
