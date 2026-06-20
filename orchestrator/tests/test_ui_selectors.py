"""Tests for UI selector helpers."""

from orchestrator.ui_selectors import build_models_by_server_type


def test_build_models_by_server_type_filters_by_capability() -> None:
    installed = [
        {
            "name": "chat-model",
            "supports_text": True,
            "supports_multimodal": False,
            "supports_embedding": False,
            "supports_rerank": False,
            "supports_image": False,
        },
        {
            "name": "flux-lite",
            "supports_text": False,
            "supports_multimodal": False,
            "supports_embedding": False,
            "supports_rerank": False,
            "supports_image": True,
        },
    ]
    mapping = build_models_by_server_type(installed)
    assert mapping["TEXT"] == ["chat-model"]
    assert mapping["IMAGE"] == ["flux-lite"]
    assert mapping["EMBEDDING"] == []
