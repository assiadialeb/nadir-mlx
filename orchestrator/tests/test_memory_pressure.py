"""Tests for KV + prompt-cache memory footprint estimates (MLX-127)."""

from __future__ import annotations

from unittest.mock import patch

from django.test import SimpleTestCase

from orchestrator.memory_pressure import (
    TransformerShape,
    estimate_instance_footprint,
    estimate_kv_cache_bytes,
    estimate_prompt_cache_budget_bytes,
    transformer_shape_from_config,
)


_QWEN_TEXT = {
    "model_type": "qwen3_5",
    "text_config": {
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "max_position_embeddings": 262144,
    },
}


class TransformerShapeTests(SimpleTestCase):
    def test_transformer_shape_from_nested_text_config(self) -> None:
        shape = transformer_shape_from_config(_QWEN_TEXT)
        assert shape is not None
        self.assertEqual(shape.num_layers, 64)
        self.assertEqual(shape.num_kv_heads, 4)
        self.assertEqual(shape.head_dim, 256)
        self.assertEqual(shape.max_position_embeddings, 262144)

    def test_transformer_shape_derives_head_dim_from_hidden_size(self) -> None:
        shape = transformer_shape_from_config(
            {
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "hidden_size": 1024,
            }
        )
        assert shape is not None
        self.assertEqual(shape.head_dim, 64)

    def test_transformer_shape_returns_none_when_incomplete(self) -> None:
        self.assertIsNone(transformer_shape_from_config({"num_hidden_layers": 12}))


class KvEstimateTests(SimpleTestCase):
    def test_estimate_kv_cache_bytes_fp16(self) -> None:
        shape = TransformerShape(num_layers=2, num_kv_heads=2, head_dim=64)
        # 2 (K+V) * 2 layers * 2 heads * 64 dim * 1024 tokens * 2 bytes
        self.assertEqual(
            estimate_kv_cache_bytes(shape, context_tokens=1024, kv_bits=16),
            2 * 2 * 2 * 64 * 1024 * 2,
        )

    def test_estimate_kv_cache_bytes_quantized(self) -> None:
        shape = TransformerShape(num_layers=1, num_kv_heads=1, head_dim=64)
        self.assertEqual(
            estimate_kv_cache_bytes(shape, context_tokens=1024, kv_bits=4),
            int(2 * 1 * 1 * 64 * 1024 * 0.5),
        )


class PromptCacheBudgetTests(SimpleTestCase):
    def test_explicit_prompt_cache_bytes_wins(self) -> None:
        shape = TransformerShape(num_layers=2, num_kv_heads=2, head_dim=64)
        budget = estimate_prompt_cache_budget_bytes(
            shape,
            server_config={"advanced": {"prompt_cache_bytes": 1_000_000}},
            kv_bits=16,
            prefix_tokens=4096,
            launch_mode="TEXT",
        )
        self.assertEqual(budget, 1_000_000)

    def test_budget_scales_with_entry_count(self) -> None:
        shape = TransformerShape(num_layers=1, num_kv_heads=1, head_dim=64)
        one = estimate_kv_cache_bytes(shape, context_tokens=1024, kv_bits=16)
        budget = estimate_prompt_cache_budget_bytes(
            shape,
            server_config={"advanced": {"prompt_cache_size": 3}},
            kv_bits=16,
            prefix_tokens=1024,
            launch_mode="TEXT",
        )
        self.assertEqual(budget, 3 * one)

    def test_multimodal_has_no_prompt_cache_budget(self) -> None:
        shape = TransformerShape(num_layers=2, num_kv_heads=2, head_dim=64)
        budget = estimate_prompt_cache_budget_bytes(
            shape,
            server_config={"advanced": {"prompt_cache_size": 24}},
            kv_bits=16,
            prefix_tokens=4096,
            launch_mode="MULTIMODAL",
        )
        self.assertEqual(budget, 0)


class FootprintIntegrationTests(SimpleTestCase):
    @patch("orchestrator.memory_pressure.planning_prompt_prefix_tokens", return_value=1024)
    @patch("orchestrator.memory_pressure.planning_context_tokens", return_value=2048)
    @patch("orchestrator.memory_pressure.load_transformer_shape")
    @patch(
        "orchestrator.memory_pressure.get_model_folder_size_bytes",
        return_value=10 * 1024**3,
    )
    def test_estimate_includes_kv_and_prompt_cache(
        self,
        _mock_disk: object,
        mock_shape: object,
        _mock_context: object,
        _mock_prefix: object,
    ) -> None:
        mock_shape.return_value = TransformerShape(
            num_layers=2, num_kv_heads=2, head_dim=64, max_position_embeddings=32768
        )
        estimate = estimate_instance_footprint(
            "chat-model",
            {"advanced": {"prompt_cache_size": 2}},
            launch_mode="TEXT",
        )
        expected_kv = estimate_kv_cache_bytes(
            mock_shape.return_value, context_tokens=2048, kv_bits=16
        )
        expected_cache = 2 * estimate_kv_cache_bytes(
            mock_shape.return_value, context_tokens=1024, kv_bits=16
        )
        expected_weights = int(10 * 1024**3 * 1.15)
        self.assertEqual(estimate.weights_bytes, expected_weights)
        self.assertEqual(estimate.kv_bytes, expected_kv)
        self.assertEqual(estimate.prompt_cache_bytes, expected_cache)
        self.assertEqual(
            estimate.total_bytes,
            expected_weights + expected_kv + expected_cache,
        )

    @patch("orchestrator.memory_pressure.load_transformer_shape")
    @patch(
        "orchestrator.memory_pressure.get_model_folder_size_bytes",
        return_value=4 * 1024**3,
    )
    def test_embedding_mode_skips_kv_terms(
        self,
        _mock_disk: object,
        mock_shape: object,
    ) -> None:
        mock_shape.return_value = TransformerShape(num_layers=8, num_kv_heads=4, head_dim=64)
        estimate = estimate_instance_footprint("embed", {}, launch_mode="EMBEDDING")
        self.assertEqual(estimate.kv_bytes, 0)
        self.assertEqual(estimate.prompt_cache_bytes, 0)
        mock_shape.assert_not_called()

    @patch("orchestrator.memory_pressure.planning_context_tokens", return_value=16384)
    @patch("orchestrator.memory_pressure.load_transformer_shape")
    @patch(
        "orchestrator.memory_pressure.get_model_folder_size_bytes",
        return_value=2 * 1024**3,
    )
    def test_max_kv_size_caps_planning_context(
        self,
        _mock_disk: object,
        mock_shape: object,
        _mock_context: object,
    ) -> None:
        mock_shape.return_value = TransformerShape(
            num_layers=1, num_kv_heads=1, head_dim=64, max_position_embeddings=100_000
        )
        estimate = estimate_instance_footprint(
            "vlm",
            {"max_kv_size": 2048},
            launch_mode="MULTIMODAL",
        )
        self.assertEqual(estimate.context_tokens, 2048)
        self.assertEqual(estimate.prompt_cache_bytes, 0)
