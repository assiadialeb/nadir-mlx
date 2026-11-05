"""Tests for mlx-lm server CLI capability detection."""

from __future__ import annotations

from unittest.mock import patch

from django.test import SimpleTestCase

from orchestrator.mlx_lm_capabilities import (
    clear_mlx_lm_capability_cache,
    mlx_lm_server_cli_flags,
    text_kv_quantization_enabled,
)


class MlxLmCapabilitiesTests(SimpleTestCase):
    def tearDown(self) -> None:
        clear_mlx_lm_capability_cache()

    def test_parses_help_flags(self) -> None:
        help_text = """
  --prefill-step-size PREFILL_STEP_SIZE
                        Step size for prefill
  --kv-bits KV_BITS     Quantize KV cache
"""
        with patch(
            "orchestrator.mlx_lm_capabilities.subprocess.run",
            return_value=type("R", (), {"stdout": help_text, "stderr": ""})(),
        ):
            clear_mlx_lm_capability_cache()
            flags = mlx_lm_server_cli_flags()
        self.assertIn("prefill-step-size", flags)
        self.assertIn("kv-bits", flags)

    def test_text_kv_disabled_when_flag_missing(self) -> None:
        with patch(
            "orchestrator.mlx_lm_capabilities.mlx_lm_server_cli_flags",
            return_value=frozenset({"prefill-step-size"}),
        ):
            self.assertFalse(text_kv_quantization_enabled())

    def test_text_kv_preview_env_override(self) -> None:
        with patch(
            "orchestrator.mlx_lm_capabilities.mlx_lm_server_cli_flags",
            return_value=frozenset({"prefill-step-size"}),
        ):
            with patch.dict("os.environ", {"NADIR_TEXT_KV_BITS_PREVIEW": "1"}):
                self.assertTrue(text_kv_quantization_enabled())

    def test_text_kv_requires_all_related_flags(self) -> None:
        with patch(
            "orchestrator.mlx_lm_capabilities.mlx_lm_server_cli_flags",
            return_value=frozenset({"kv-bits", "kv-quant-scheme"}),
        ):
            self.assertFalse(text_kv_quantization_enabled())
        with patch(
            "orchestrator.mlx_lm_capabilities.mlx_lm_server_cli_flags",
            return_value=frozenset({"kv-bits", "kv-quant-scheme", "kv-group-size"}),
        ):
            self.assertTrue(text_kv_quantization_enabled())

    def test_probe_failure_is_cached_briefly(self) -> None:
        import subprocess

        with patch(
            "orchestrator.mlx_lm_capabilities.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="mlx", timeout=15),
        ) as mock_run:
            clear_mlx_lm_capability_cache()
            self.assertEqual(mlx_lm_server_cli_flags(), frozenset())
            self.assertEqual(mock_run.call_count, 1)
            # Second call within TTL must not re-probe.
            self.assertEqual(mlx_lm_server_cli_flags(), frozenset())
            self.assertEqual(mock_run.call_count, 1)

        clear_mlx_lm_capability_cache()
        with patch(
            "orchestrator.mlx_lm_capabilities.subprocess.run",
            return_value=type(
                "R",
                (),
                {"stdout": "  --prefill-step-size N\n", "stderr": "", "returncode": 0},
            )(),
        ):
            self.assertIn("prefill-step-size", mlx_lm_server_cli_flags())
