"""Launch mlx_vlm.server with local model aliases for OpenAI-compatible clients."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def _parse_model_path(argv: list[str]) -> Path | None:
    for index, arg in enumerate(argv):
        if arg == "--model" and index + 1 < len(argv):
            return Path(argv[index + 1]).resolve()
    return None


def _install_model_alias_patch(local_model_path: Path) -> None:
    """Map client-facing model names to the preloaded local directory."""
    resolved_path = str(local_model_path)
    folder_name = local_model_path.name
    aliases = {
        "default_model": resolved_path,
        folder_name: resolved_path,
    }

    app_module = importlib.import_module("mlx_vlm.server.app")
    original_get_cached_model = app_module.get_cached_model
    inherit_adapter = app_module._INHERIT_ADAPTER

    def get_cached_model_with_aliases(
        model_path: str,
        adapter_path=inherit_adapter,
        *,
        model_kind: str = "auto",
    ):
        resolved = aliases.get(model_path, model_path)
        if not Path(resolved).is_dir() and Path(model_path).is_dir():
            resolved = str(Path(model_path).resolve())
        return original_get_cached_model(
            resolved,
            adapter_path,
            model_kind=model_kind,
        )

    app_module.get_cached_model = get_cached_model_with_aliases

    server_module = importlib.import_module("mlx_vlm.server")
    server_module.get_cached_model = get_cached_model_with_aliases

    openai_module = importlib.import_module("mlx_vlm.server.openai")
    if openai_module.get_cached_model is not None:
        openai_module.get_cached_model = get_cached_model_with_aliases

    anthropic_module = importlib.import_module("mlx_vlm.server.anthropic")
    if anthropic_module.get_cached_model is not None:
        anthropic_module.get_cached_model = get_cached_model_with_aliases


def main() -> None:
    model_path = _parse_model_path(sys.argv[1:])
    if model_path is None:
        raise SystemExit("Missing required argument: --model <local_model_path>")

    model_path = model_path.resolve()
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--model" and index + 1 < len(sys.argv):
            sys.argv[index + 1] = str(model_path)
            break

    os.environ["MLX_VLM_PRELOAD_MODEL"] = str(model_path)
    _install_model_alias_patch(model_path)

    from mlx_vlm.server.cli import main as server_main

    server_main()


if __name__ == "__main__":
    main()
