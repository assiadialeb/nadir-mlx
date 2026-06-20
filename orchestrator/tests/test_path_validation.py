"""Tests for model path and repo id validation."""

from pathlib import Path
from django.test import SimpleTestCase, override_settings

from orchestrator.model_utils import (
    get_folder_name,
    resolve_log_file_path,
    resolve_model_dir,
    validate_hf_repo_id,
    validate_model_folder_name,
)


@override_settings(MODELS_DIR="/tmp/mlx-models", LOGS_DIR="/tmp/mlx-logs")
class PathValidationTests(SimpleTestCase):
    def test_validate_model_folder_name_accepts_hf_style_names(self) -> None:
        self.assertEqual(validate_model_folder_name("whisper-small-mlx"), "whisper-small-mlx")
        self.assertEqual(validate_model_folder_name("Kokoro-82M-6bit"), "Kokoro-82M-6bit")

    def test_validate_model_folder_name_rejects_traversal(self) -> None:
        for invalid in ("..", ".", "../etc", "foo/bar", "foo\\bar", "foo..bar/baz"):
            with self.subTest(name=invalid):
                with self.assertRaises(ValueError):
                    validate_model_folder_name(invalid)

    def test_validate_hf_repo_id_accepts_org_model(self) -> None:
        self.assertEqual(
            validate_hf_repo_id("mlx-community/whisper-small-mlx"),
            "mlx-community/whisper-small-mlx",
        )

    def test_validate_hf_repo_id_rejects_traversal(self) -> None:
        for invalid in ("..", "org/../secret", "/abs/path", "no-slash", "a/b/c"):
            with self.subTest(repo_id=invalid):
                with self.assertRaises(ValueError):
                    validate_hf_repo_id(invalid)

    def test_get_folder_name_rejects_parent_segment(self) -> None:
        with self.assertRaises(ValueError):
            get_folder_name("mlx-community/..")

    def test_resolve_model_dir_stays_under_models_root(self) -> None:
        resolved = resolve_model_dir("Kokoro-82M-6bit")
        self.assertEqual(resolved, Path("/tmp/mlx-models/Kokoro-82M-6bit").resolve())

    def test_resolve_log_file_path_stays_under_logs_root(self) -> None:
        resolved = resolve_log_file_path("Kokoro-82M-6bit", 11444)
        self.assertEqual(
            resolved,
            Path("/tmp/mlx-logs/Kokoro-82M-6bit_11444.log").resolve(),
        )

    def test_resolve_log_file_path_rejects_invalid_port(self) -> None:
        with self.assertRaises(ValueError):
            resolve_log_file_path("Kokoro-82M-6bit", 0)
