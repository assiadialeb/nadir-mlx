"""Tests for inference server launch command construction."""

from __future__ import annotations

from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from orchestrator.server_manager import _append_cli_args, _build_launch_command


@override_settings(MODELS_DIR="/tmp")
class ServerManagerLaunchCommandTests(SimpleTestCase):
    @patch("orchestrator.server_manager._get_python_bin", return_value="python")
    def test_build_launch_command_text_includes_sampling_flags(
        self,
        _mock_python_bin: object,
    ) -> None:
        command = _build_launch_command(
            "/tmp/llama",
            11400,
            "TEXT",
            {
                "host": "127.0.0.1",
                "max_tokens": 512,
                "advanced": {
                    "temp": 0.7,
                    "chat_template_args": {"enable_thinking": False},
                },
            },
            "llama-model",
        )
        self.assertIn("orchestrator.mlx_launcher", command)
        self.assertIn("--temp", command)
        self.assertIn("--chat-template-args", command)
        self.assertIn("--prompt-cache-size", command)
        self.assertNotIn("--prompt-cache-bytes", command)

    @patch("orchestrator.server_manager._get_python_bin", return_value="python")
    def test_build_launch_command_text_honors_explicit_prompt_cache(
        self,
        _mock_python_bin: object,
    ) -> None:
        command = _build_launch_command(
            "/tmp/llama",
            11400,
            "TEXT",
            {
                "host": "127.0.0.1",
                "advanced": {
                    "prompt_cache_size": 5,
                    "prompt_cache_bytes": 3_000_000,
                },
            },
            "llama-model",
        )
        size_index = command.index("--prompt-cache-size")
        bytes_index = command.index("--prompt-cache-bytes")
        self.assertEqual(command[size_index + 1], "5")
        self.assertEqual(command[bytes_index + 1], "3000000")

    @patch("orchestrator.server_manager._get_python_bin", return_value="python")
    def test_build_launch_command_text_passes_concurrency_knobs(
        self,
        _mock_python_bin: object,
    ) -> None:
        command = _build_launch_command(
            "/tmp/llama",
            11400,
            "TEXT",
            {
                "host": "127.0.0.1",
                "advanced": {"decode_concurrency": 8, "prompt_concurrency": 2},
            },
            "llama-model",
        )
        self.assertEqual(command[command.index("--decode-concurrency") + 1], "8")
        self.assertEqual(command[command.index("--prompt-concurrency") + 1], "2")

    @patch("orchestrator.server_manager._get_python_bin", return_value="python")
    def test_build_launch_command_text_passes_prefill_step_size(
        self,
        _mock_python_bin: object,
    ) -> None:
        command = _build_launch_command(
            "/tmp/llama",
            11400,
            "TEXT",
            {
                "host": "127.0.0.1",
                "advanced": {"prefill_step_size": 1024},
            },
            "llama-model",
        )
        self.assertEqual(command[command.index("--prefill-step-size") + 1], "1024")

    @patch("orchestrator.mlx_lm_capabilities.text_kv_quantization_enabled", return_value=True)
    @patch("orchestrator.server_manager._get_python_bin", return_value="python")
    def test_build_launch_command_text_passes_kv_when_supported(
        self,
        _mock_python_bin: object,
        _mock_kv: object,
    ) -> None:
        command = _build_launch_command(
            "/tmp/llama",
            11400,
            "TEXT",
            {
                "host": "127.0.0.1",
                "advanced": {"kv_bits": 4, "kv_group_size": 32},
            },
            "llama-model",
        )
        self.assertEqual(command[command.index("--kv-bits") + 1], "4")
        self.assertEqual(command[command.index("--kv-group-size") + 1], "32")

    @patch("orchestrator.server_manager._get_python_bin", return_value="python")
    def test_build_launch_command_multimodal_includes_mtp_flags(
        self,
        _mock_python_bin: object,
    ) -> None:
        command = _build_launch_command(
            "/tmp/gemma",
            11401,
            "MULTIMODAL",
            {
                "host": "127.0.0.1",
                "advanced": {
                    "draft_kind": "mtp",
                    "enable_thinking": True,
                },
            },
            "gemma-vlm",
        )
        self.assertIn("orchestrator.mlx_vlm_launcher", command)
        self.assertIn("--draft-kind", command)
        self.assertIn("--enable-thinking", command)

    @patch("orchestrator.server_manager._get_python_bin", return_value="python")
    def test_build_launch_command_embedding(self, _mock_python_bin: object) -> None:
        command = _build_launch_command(
            "/tmp/embed",
            11402,
            "EMBEDDING",
            {"host": "127.0.0.1", "model_id": "embed-alias"},
            "nomic-embed",
        )
        self.assertEqual(command[2], "orchestrator.mlx_embedding_launcher")
        self.assertIn("--model-id", command)
        self.assertEqual(command[command.index("--model-id") + 1], "embed-alias")

    @patch("orchestrator.server_manager._get_python_bin", return_value="python")
    def test_build_launch_command_reranker_disable_batching(
        self,
        _mock_python_bin: object,
    ) -> None:
        command = _build_launch_command(
            "/tmp/rerank",
            11403,
            "RERANKER",
            {"host": "127.0.0.1", "disable_batching": True},
            "bge-reranker",
        )
        self.assertIn("orchestrator.mlx_reranker_launcher", command)
        self.assertIn("--disable-batching", command)

    @patch("orchestrator.server_manager._get_python_bin", return_value="python")
    def test_build_launch_command_stt_passes_defaults(
        self,
        _mock_python_bin: object,
    ) -> None:
        command = _build_launch_command(
            "/tmp/whisper",
            11404,
            "STT",
            {
                "host": "127.0.0.1",
                "language": "fr",
                "chunk_duration": 15,
            },
            "whisper-model",
        )
        self.assertIn("orchestrator.mlx_stt_launcher", command)
        self.assertIn("--default-language", command)
        self.assertEqual(command[command.index("--default-language") + 1], "fr")

    def test_append_cli_args_skips_false_and_none(self) -> None:
        command: list[str] = ["python"]
        _append_cli_args(command, {
            "flag-a": None,
            "flag-b": False,
            "flag-c": True,
            "flag-d": {"k": "v"},
        })
        self.assertEqual(command, ["python", "--flag-c", "--flag-d", '{"k": "v"}'])

    @patch.dict("os.environ", {"NADIR_TEXT_MTP_PREVIEW": "1"})
    def test_text_launch_command_includes_draft_kind_when_preview_enabled(self) -> None:
        command = _build_launch_command(
            "/tmp/llama",
            11405,
            "TEXT",
            {
                "host": "127.0.0.1",
                "advanced": {
                    "draft_kind": "mtp",
                    "draft_block_size": 4,
                    "draft_model": "assistant-model",
                },
            },
            "llama-model",
        )
        self.assertIn("--draft-kind", command)
        self.assertIn("mtp", command)
        self.assertIn("--draft-block-size", command)
