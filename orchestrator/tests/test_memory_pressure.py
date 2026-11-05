"""Tests for KV + prompt-cache memory footprint estimates (MLX-127)."""

from __future__ import annotations

from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from orchestrator.memory_pressure import (
    LinearAttentionShape,
    TransformerShape,
    estimate_instance_footprint,
    estimate_kv_cache_bytes,
    estimate_prompt_cache_budget_bytes,
    resolve_planning_parallel_sequences,
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
        "layer_types": (["linear_attention"] * 48) + (["full_attention"] * 16),
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    },
}

_GEMMA4_12B = {
    "model_type": "gemma4",
    "text_config": {
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "max_position_embeddings": 262144,
        "sliding_window": 1024,
        "num_kv_shared_layers": 0,
        "attention_k_eq_v": True,
        "layer_types": (["sliding_attention"] * 40) + (["full_attention"] * 8),
    },
}

_GEMMA4_E4B = {
    "model_type": "gemma4",
    "text_config": {
        "num_hidden_layers": 42,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "max_position_embeddings": 131072,
        "sliding_window": 512,
        "num_kv_shared_layers": 18,
        "attention_k_eq_v": False,
        "layer_types": (["sliding_attention"] * 35) + (["full_attention"] * 7),
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
        self.assertEqual(shape.layer_types.count("linear_attention"), 48)
        assert shape.linear is not None
        self.assertEqual(shape.linear.num_value_heads, 48)

    def test_transformer_shape_reads_gemma_sliding_and_shared_kv(self) -> None:
        shape = transformer_shape_from_config(_GEMMA4_E4B)
        assert shape is not None
        self.assertEqual(shape.sliding_window, 512)
        self.assertEqual(shape.num_kv_shared_layers, 18)
        self.assertEqual(shape.layer_types.count("sliding_attention"), 35)

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
    def test_estimate_kv_cache_bytes_fp16_dense(self) -> None:
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

    def test_sliding_layers_use_window_not_full_context(self) -> None:
        shape = TransformerShape(
            num_layers=2,
            num_kv_heads=1,
            head_dim=64,
            sliding_window=128,
            layer_types=("sliding_attention", "full_attention"),
        )
        sliding = 2 * 1 * 64 * 128 * 2
        full = 2 * 1 * 64 * 2048 * 2
        self.assertEqual(
            estimate_kv_cache_bytes(shape, context_tokens=2048, kv_bits=16),
            sliding + full,
        )

    def test_shared_kv_layers_are_skipped(self) -> None:
        shape = TransformerShape(
            num_layers=4,
            num_kv_heads=1,
            head_dim=64,
            num_kv_shared_layers=2,
        )
        # Only first 2 layers own KV.
        self.assertEqual(
            estimate_kv_cache_bytes(shape, context_tokens=100, kv_bits=16),
            2 * 2 * 1 * 64 * 100 * 2,
        )

    def test_attention_k_eq_v_halves_full_attention_storage(self) -> None:
        shape = TransformerShape(
            num_layers=1,
            num_kv_heads=2,
            head_dim=64,
            attention_k_eq_v=True,
            layer_types=("full_attention",),
        )
        self.assertEqual(
            estimate_kv_cache_bytes(shape, context_tokens=128, kv_bits=16),
            1 * 2 * 64 * 128 * 2,
        )

    def test_linear_attention_state_is_fixed_size(self) -> None:
        linear = LinearAttentionShape(
            num_key_heads=2,
            num_value_heads=4,
            key_head_dim=8,
            value_head_dim=8,
            conv_kernel_dim=4,
        )
        shape = TransformerShape(
            num_layers=1,
            num_kv_heads=1,
            head_dim=64,
            layer_types=("linear_attention",),
            linear=linear,
        )
        recurrent = 4 * 8 * 8 * 4
        conv = 3 * (2 * 2 * 8 + 4 * 8) * 2
        self.assertEqual(
            estimate_kv_cache_bytes(shape, context_tokens=8192, kv_bits=16),
            recurrent + conv,
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

    @override_settings(NADIR_MEMORY_PLANNING_PROMPT_CACHE_ENTRIES=1)
    def test_budget_caps_at_planning_entries(self) -> None:
        shape = TransformerShape(num_layers=1, num_kv_heads=1, head_dim=64)
        one = estimate_kv_cache_bytes(shape, context_tokens=1024, kv_bits=16)
        budget = estimate_prompt_cache_budget_bytes(
            shape,
            server_config={"advanced": {"prompt_cache_size": 24}},
            kv_bits=16,
            prefix_tokens=1024,
            launch_mode="TEXT",
        )
        self.assertEqual(budget, one)

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


class ParallelSequencesTests(SimpleTestCase):
    @override_settings(
        NADIR_MEMORY_PLANNING_PARALLEL_SEQUENCES=2,
        NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM=16,
        NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL=16,
    )
    def test_decode_concurrency_and_planning_cap_are_honored(self) -> None:
        self.assertEqual(
            resolve_planning_parallel_sequences(
                {"advanced": {"decode_concurrency": 8}}
            ),
            2,
        )
        self.assertEqual(
            resolve_planning_parallel_sequences(
                {"advanced": {"decode_concurrency": 1}}
            ),
            1,
        )


class FootprintIntegrationTests(SimpleTestCase):
    @override_settings(
        NADIR_MEMORY_PLANNING_PROMPT_CACHE_ENTRIES=1,
        NADIR_MEMORY_PLANNING_PARALLEL_SEQUENCES=1,
        NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM=16,
        NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL=16,
    )
    @patch("orchestrator.memory_pressure.planning_prompt_prefix_tokens", return_value=1024)
    @patch("orchestrator.memory_pressure.planning_context_tokens", return_value=2048)
    @patch("orchestrator.memory_pressure.load_transformer_shape")
    @patch(
        "orchestrator.memory_pressure.get_model_folder_size_bytes",
        return_value=10 * 1024**3,
    )
    def test_estimate_includes_kv_and_capped_prompt_cache(
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
        expected_cache = estimate_kv_cache_bytes(
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

    @override_settings(
        NADIR_MEMORY_PLANNING_PROMPT_CACHE_ENTRIES=1,
        NADIR_MEMORY_PLANNING_PARALLEL_SEQUENCES=1,
        NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM=16,
        NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL=16,
    )
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

    @override_settings(
        NADIR_MEMORY_PLANNING_CONTEXT_TOKENS=8192,
        NADIR_MEMORY_PLANNING_PROMPT_PREFIX_TOKENS=4096,
        NADIR_MEMORY_PLANNING_PROMPT_CACHE_ENTRIES=1,
        NADIR_MEMORY_PLANNING_PARALLEL_SEQUENCES=2,
        NADIR_GATEWAY_MAX_CONCURRENT_UPSTREAM=16,
        NADIR_GATEWAY_MAX_CONCURRENT_GLOBAL=16,
    )
    def test_gemma4_and_qwen_estimates_stay_near_disk(self) -> None:
        cases = (
            ("gemma-4-12B-it-4bit", _GEMMA4_12B, 6.3 * 1024**3),
            ("Qwen3.8-27B-4bit", _QWEN_TEXT, 15.0 * 1024**3),
            ("gemma-4-E4B-it-qat-4bit", _GEMMA4_E4B, 4.0 * 1024**3),
        )
        for model_name, config, disk in cases:
            with self.subTest(model=model_name):
                with (
                    patch(
                        "orchestrator.memory_pressure.load_transformer_shape",
                        return_value=transformer_shape_from_config(config),
                    ),
                    patch(
                        "orchestrator.memory_pressure.get_model_folder_size_bytes",
                        return_value=int(disk),
                    ),
                ):
                    estimate = estimate_instance_footprint(
                        model_name, {}, launch_mode="TEXT"
                    )
                ceiling = int(1.5 * disk + 2 * 1024**3)
                self.assertLessEqual(
                    estimate.total_bytes,
                    ceiling,
                    msg=(
                        f"{model_name}: {estimate.total_bytes / 1024**3:.1f} GiB "
                        f"> {ceiling / 1024**3:.1f} GiB "
                        f"(w={estimate.weights_bytes / 1024**3:.1f} "
                        f"kv={estimate.kv_bytes / 1024**3:.1f} "
                        f"pc={estimate.prompt_cache_bytes / 1024**3:.1f})"
                    ),
                )
