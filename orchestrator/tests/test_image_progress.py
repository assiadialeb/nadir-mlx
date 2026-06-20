"""Unit tests for image progress logging."""

import time
from unittest import TestCase
from unittest.mock import MagicMock, patch

from orchestrator.image_progress import ImageProgressLogger, register_progress_logger


class ImageProgressLoggerTests(TestCase):
    def test_call_in_loop_prints_step_line(self) -> None:
        logger = ImageProgressLogger()
        logger._started_at = time.monotonic() - 12.5
        config = MagicMock(num_inference_steps=20)

        with patch("builtins.print") as mock_print:
            logger.call_in_loop(
                t=4,
                seed=1,
                prompt="test",
                latents=MagicMock(),
                config=config,
                time_steps=None,
            )

        mock_print.assert_called_once()
        message = mock_print.call_args[0][0]
        self.assertIn("step 5/20", message)
        self.assertIn("25%", message)
        self.assertRegex(message, r"elapsed=\d+\.\d+s")

    def test_call_before_loop_resets_timer(self) -> None:
        logger = ImageProgressLogger()
        with patch("orchestrator.image_progress.time.monotonic", return_value=100.0):
            logger.call_before_loop(1, "test", MagicMock(), MagicMock())
        self.assertEqual(logger._started_at, 100.0)

    def test_register_progress_logger_attaches_callback(self) -> None:
        model = MagicMock()
        register_progress_logger(model)
        model.callbacks.register.assert_called_once()
