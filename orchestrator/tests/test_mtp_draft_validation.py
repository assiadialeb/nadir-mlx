"""Tests for Gemma 4 MTP drafter validation."""

import json
from pathlib import Path

from django.test import SimpleTestCase, override_settings

from orchestrator.mtp_draft_validation import (
    is_incompatible_mtp_ordered_assistant,
    validate_mtp_draft_advanced,
)
from orchestrator.server_config_schema import validate_and_normalize_server_config


def _write_assistant_config(model_dir: Path, *, quantized: bool) -> None:
    config = {
        "model_type": "gemma4_assistant",
        "use_ordered_embeddings": True,
    }
    if quantized:
        config["quantization"] = {"bits": 4, "group_size": 64, "mode": "affine"}
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle)


@override_settings(MODELS_DIR="/tmp/nadir-mtp-models-test")
class MtpDraftValidationTests(SimpleTestCase):
    def test_rejects_quantized_ordered_assistant_by_config(self) -> None:
        draft_dir = Path("/tmp/nadir-mtp-models-test/gemma-4-E4B-it-qat-assistant-4bit")
        _write_assistant_config(draft_dir, quantized=True)
        advanced = {
            "draft_kind": "mtp",
            "draft_model": str(draft_dir),
        }
        with self.assertRaisesRegex(ValueError, "unquantized drafter"):
            validate_mtp_draft_advanced("MULTIMODAL", advanced)

    def test_allows_bf16_ordered_assistant(self) -> None:
        draft_dir = Path("/tmp/nadir-mtp-models-test/gemma-4-E4B-it-assistant-bf16")
        _write_assistant_config(draft_dir, quantized=False)
        advanced = {
            "draft_kind": "mtp",
            "draft_model": "gemma-4-E4B-it-assistant-bf16",
        }
        validate_mtp_draft_advanced("MULTIMODAL", advanced)

    def test_rejects_quantized_hf_repo_id_heuristic(self) -> None:
        advanced = {
            "draft_kind": "mtp",
            "draft_model": "mlx-community/gemma-4-E4B-it-qat-assistant-4bit",
        }
        with self.assertRaisesRegex(ValueError, "unquantized drafter"):
            validate_mtp_draft_advanced("MULTIMODAL", advanced)

    def test_is_incompatible_helper(self) -> None:
        self.assertTrue(
            is_incompatible_mtp_ordered_assistant(
                {
                    "model_type": "gemma4_assistant",
                    "use_ordered_embeddings": True,
                    "quantization": {"bits": 4},
                }
            )
        )
        self.assertFalse(
            is_incompatible_mtp_ordered_assistant(
                {
                    "model_type": "gemma4_assistant",
                    "use_ordered_embeddings": False,
                    "quantization": {"bits": 4},
                }
            )
        )

    def test_validate_server_config_wires_mtp_check(self) -> None:
        draft_dir = Path("/tmp/nadir-mtp-models-test/gemma-4-E4B-it-qat-assistant-4bit")
        _write_assistant_config(draft_dir, quantized=True)
        with self.assertRaisesRegex(ValueError, "unquantized drafter"):
            validate_and_normalize_server_config(
                "MULTIMODAL",
                {
                    "advanced": {
                        "draft_kind": "mtp",
                        "draft_model": str(draft_dir),
                    }
                },
                "gemma-4-E4B-it-qat-4bit",
            )
