"""Tests for inference server launch command construction."""

from __future__ import annotations

from unittest.mock import patch

from django.test import SimpleTestCase

from orchestrator.server_manager import _append_cli_args, _build_launch_command


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
