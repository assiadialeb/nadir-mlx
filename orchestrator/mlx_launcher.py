"""Launch mlx_lm.server with compatibility fixes for newer model architectures."""

from __future__ import annotations

import sys
from pathlib import Path


def _patch_tokenizer_config() -> None:
    """Ensure Mistral/Qwen-style tokenizers load with the correct regex fix."""
    from mlx_lm.server import ModelProvider

    original_init = ModelProvider.__init__

    def patched_init(self, cli_args):
        original_init(self, cli_args)
        self._tokenizer_config["fix_mistral_regex"] = True

    ModelProvider.__init__ = patched_init


def _patch_load_model(model_path: Path) -> None:
    """Monkey-patch mlx_lm load to tolerate extra multimodal weight tensors."""
    from orchestrator.model_utils import (
        prepare_model_for_text_inference,
        requires_relaxed_weight_loading,
    )

    prepare_model_for_text_inference(model_path)
    if not requires_relaxed_weight_loading(model_path):
        return

    import mlx_lm.utils as mlx_utils

    original_load_model = mlx_utils.load_model

    def patched_load_model(path, *args, **kwargs):
        kwargs.setdefault("strict", False)
        return original_load_model(path, *args, **kwargs)

    mlx_utils.load_model = patched_load_model


def main() -> None:
    model_path: Path | None = None
    argv = sys.argv[1:]
    for index, arg in enumerate(argv):
        if arg == "--model" and index + 1 < len(argv):
            model_path = Path(argv[index + 1])
            break

    if model_path is not None:
        _patch_load_model(model_path)

    _patch_tokenizer_config()

    from mlx_lm.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
