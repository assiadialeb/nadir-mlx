"""Tests for RAM-aware mlx-lm prompt-cache defaults (MLX-125)."""

from __future__ import annotations

from unittest.mock import patch

from django.test import SimpleTestCase

from types import SimpleNamespace

from orchestrator.gateway_telemetry_services import prompt_cache_hit_rate
from orchestrator.prompt_cache import (
    default_prompt_cache_settings,
    resolve_prompt_cache_cli_flags,
)
from orchestrator.server_config_schema import (
    build_default_server_config,
    validate_and_normalize_server_config,
)


class PromptCacheDefaultsTests(SimpleTestCase):
    def test_default_settings_scale_with_ram(self) -> None:
        small = default_prompt_cache_settings(total_bytes=16 * 1024**3)
        self.assertEqual(small["prompt_cache_size"], 10)
        mid = default_prompt_cache_settings(total_bytes=48 * 1024**3)
        self.assertEqual(mid["prompt_cache_size"], 24)
        large = default_prompt_cache_settings(total_bytes=96 * 1024**3)
        self.assertEqual(large["prompt_cache_size"], 32)
        self.assertNotIn("prompt_cache_bytes", large)

    def test_resolve_cli_flags_prefer_explicit_advanced(self) -> None:
        flags = resolve_prompt_cache_cli_flags(
            {"prompt_cache_size": 7, "prompt_cache_bytes": 2_000_000}
        )
        self.assertEqual(flags["prompt-cache-size"], 7)
        self.assertEqual(flags["prompt-cache-bytes"], 2_000_000)

    def test_resolve_cli_flags_omits_bytes_by_default(self) -> None:
        flags = resolve_prompt_cache_cli_flags({"prompt_cache_size": 12})
        self.assertEqual(flags["prompt-cache-size"], 12)
        self.assertNotIn("prompt-cache-bytes", flags)
        flags_default = resolve_prompt_cache_cli_flags({})
        self.assertIn("prompt-cache-size", flags_default)
        self.assertNotIn("prompt-cache-bytes", flags_default)

    @patch(
        "orchestrator.prompt_cache.host_total_bytes",
        return_value=48 * 1024**3,
    )
    def test_build_default_text_includes_prompt_cache(
        self,
        _mock_total: object,
    ) -> None:
        config = build_default_server_config("TEXT")
        self.assertEqual(config["advanced"]["prompt_cache_size"], 24)
        self.assertNotIn("prompt_cache_bytes", config["advanced"])

    def test_validate_accepts_prompt_cache_knobs(self) -> None:
        config = validate_and_normalize_server_config(
            "TEXT",
            {
                "advanced": {
                    "prompt_cache_size": 12,
                    "prompt_cache_bytes": 1_500_000,
                }
            },
            "model-a",
        )
        self.assertEqual(config["advanced"]["prompt_cache_size"], 12)
        self.assertEqual(config["advanced"]["prompt_cache_bytes"], 1_500_000)

    def test_validate_rejects_prompt_cache_size_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            validate_and_normalize_server_config(
                "TEXT",
                {"advanced": {"prompt_cache_size": 0}},
                "model-a",
            )

    def test_prompt_cache_hit_rate_from_rows(self) -> None:
        rows = [
            SimpleNamespace(prompt_tokens=100, cached_prompt_tokens=0),
            SimpleNamespace(prompt_tokens=100, cached_prompt_tokens=80),
        ]
        rate = prompt_cache_hit_rate(rows)
        self.assertIsNotNone(rate)
        assert rate is not None
        self.assertAlmostEqual(rate, 80 / 200)
        self.assertIsNone(prompt_cache_hit_rate([]))
