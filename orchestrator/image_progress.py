"""Log image generation progress in a log-file-friendly format."""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
import PIL.Image

from mflux.models.common.config.config import Config


class ImageProgressLogger:
    """Emit one progress line per denoising step (works in redirected log files)."""

    def __init__(self) -> None:
        self._started_at: float | None = None

    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
    ) -> None:
        self._started_at = time.monotonic()

    def call_in_loop(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        time_steps: Any = None,
    ) -> None:
        if self._started_at is None:
            self._started_at = time.monotonic()

        step = t + 1
        total = config.num_inference_steps
        elapsed_seconds = time.monotonic() - self._started_at
        print(
            f"[image] step {step}/{total} ({int(step * 100 / total)}%) "
            f"elapsed={elapsed_seconds:.1f}s",
            flush=True,
        )


def register_progress_logger(model: Any) -> None:
    """Attach progress logging to an mflux model instance."""
    callbacks = getattr(model, "callbacks", None)
    if callbacks is None or not hasattr(callbacks, "register"):
        return
    callbacks.register(ImageProgressLogger())
