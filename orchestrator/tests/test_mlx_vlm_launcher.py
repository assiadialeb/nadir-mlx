import json
from pathlib import Path

import pytest

from orchestrator.mlx_vlm_launcher import model_requires_relaxed_weight_loading


def _write_gemma4_config(model_dir: Path, *, num_kv_shared_layers: int) -> None:
    config = {
        "model_type": "gemma4",
        "text_config": {"num_kv_shared_layers": num_kv_shared_layers},
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_model_requires_relaxed_weight_loading_for_mlx_gemma4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "gemma-4-e2b-it-4bit"
    model_dir.mkdir()
    _write_gemma4_config(model_dir, num_kv_shared_layers=20)
    (model_dir / "model.safetensors").write_bytes(b"placeholder")

    class FakeHandle:
        def metadata(self) -> dict[str, str]:
            return {"format": "mlx"}

        def __enter__(self) -> "FakeHandle":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_safe_open(path: str, framework: str) -> FakeHandle:
        return FakeHandle()

    monkeypatch.setattr("safetensors.safe_open", fake_safe_open)

    assert model_requires_relaxed_weight_loading(model_dir) is True


def test_model_requires_relaxed_weight_loading_false_without_kv_shared(tmp_path: Path) -> None:
    model_dir = tmp_path / "other-model"
    model_dir.mkdir()
    config = {"model_type": "gemma4", "text_config": {"num_kv_shared_layers": 0}}
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    assert model_requires_relaxed_weight_loading(model_dir) is False


def test_model_requires_relaxed_weight_loading_false_for_non_gemma4(tmp_path: Path) -> None:
    model_dir = tmp_path / "qwen-model"
    model_dir.mkdir()
    config = {"model_type": "qwen2_vl", "text_config": {"num_kv_shared_layers": 20}}
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    assert model_requires_relaxed_weight_loading(model_dir) is False
