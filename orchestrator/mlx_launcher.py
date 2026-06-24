"""Launch mlx_lm.server with compatibility fixes for newer model architectures."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from orchestrator.tokenizer_compat import install_auto_fix_mistral_regex


def _parse_cli_arg(argv: list[str], flag: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            value = argv[index + 1].strip()
            return value or None
    return None


def _strip_cli_flag(flag: str) -> None:
    """Remove a custom orchestrator flag before delegating to mlx_lm CLI parsing."""
    index = 1
    while index < len(sys.argv):
        if sys.argv[index] != flag:
            index += 1
            continue
        del sys.argv[index]
        if index < len(sys.argv) and not sys.argv[index].startswith("-"):
            del sys.argv[index]


def _install_gateway_alias_patch(
    api_model_id: str,
    local_model_path: Path,
) -> None:
    """Map gateway alias names to the preloaded local model directory."""
    resolved_path = str(local_model_path.resolve())
    folder_name = local_model_path.name
    alias_names = {api_model_id, folder_name, "default_model"}

    from mlx_lm.server import APIHandler, ModelProvider

    original_init = ModelProvider.__init__

    def patched_init(self, cli_args):
        original_init(self, cli_args)
        for alias in alias_names:
            if alias:
                self._model_map[alias] = resolved_path

    ModelProvider.__init__ = patched_init

    if not api_model_id:
        return

    def patched_models_request(self):
        self._set_completion_headers(200)
        self.end_headers()
        models = [
            {
                "id": api_model_id,
                "object": "model",
                "created": self.created,
            }
        ]
        response = {"object": "list", "data": models}
        self.wfile.write(json.dumps(response).encode())
        self.wfile.flush()

    APIHandler.handle_models_request = patched_models_request


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
    install_auto_fix_mistral_regex()

    argv = sys.argv[1:]
    model_path: Path | None = None
    for index, arg in enumerate(argv):
        if arg == "--model" and index + 1 < len(argv):
            model_path = Path(argv[index + 1])
            break

    api_model_id = (
        _parse_cli_arg(argv, "--model-id")
        or os.environ.get("NADIR_GATEWAY_ALIAS", "").strip()
        or None
    )
    _strip_cli_flag("--model-id")

    if model_path is not None:
        model_path = model_path.resolve()
        _patch_load_model(model_path)
        if api_model_id:
            _install_gateway_alias_patch(api_model_id, model_path)

    from mlx_lm.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
