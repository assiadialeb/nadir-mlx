"""Unit tests for image progress logging."""

from unittest import TestCase
from unittest.mock import MagicMock, patch

from orchestrator.image_progress import ImageProgressLogger, register_progress_logger


class ImageProgressLoggerTests(TestCase):
    def test_call_in_loop_prints_step_line(self) -> None:
        logger = ImageProgressLogger()
        config = MagicMock(num_inference_steps=20)
        time_steps = MagicMock(format_dict={"elapsed": 12.5})

        with patch("builtins.print") as mock_print:
            logger.call_in_loop(
                t=4,
                seed=1,
                prompt="test",
                latents=MagicMock(),
                config=config,
                time_steps=time_steps,
            )

        mock_print.assert_called_once()
        message = mock_print.call_args[0][0]
        self.assertIn("step 5/20", message)
        self.assertIn("25%", message)
        self.assertIn("elapsed=12.5s", message)

    def test_register_progress_logger_attaches_callback(self) -> None:
        model = MagicMock()
        register_progress_logger(model)
        model.callbacks.register.assert_called_once()
