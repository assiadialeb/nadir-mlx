"""Log image generation progress in a log-file-friendly format."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import PIL.Image

from mflux.models.common.config.config import Config


class ImageProgressLogger:
    """Emit one progress line per denoising step (works in redirected log files)."""

    def call_in_loop(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        time_steps: Any = None,
    ) -> None:
        step = t + 1
        total = config.num_inference_steps
        elapsed = ""
        if time_steps is not None and hasattr(time_steps, "format_dict"):
            elapsed_seconds = time_steps.format_dict.get("elapsed")
            if elapsed_seconds is not None:
                elapsed = f" elapsed={elapsed_seconds:.1f}s"
        print(
            f"[image] step {step}/{total} ({int(step * 100 / total)}%){elapsed}",
            flush=True,
        )


def register_progress_logger(model: Any) -> None:
    """Attach progress logging to an mflux model instance."""
    callbacks = getattr(model, "callbacks", None)
    if callbacks is None or not hasattr(callbacks, "register"):
        return
    callbacks.register(ImageProgressLogger())
