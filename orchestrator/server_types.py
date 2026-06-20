"""Server type metadata for the orchestrator UI."""

from __future__ import annotations

from typing import TypedDict


class ServerTypeSpec(TypedDict):
    id: str
    label: str
    short_label: str
    capability: str
    backend: str
    api_hint: str


SERVER_TYPES: tuple[ServerTypeSpec, ...] = (
    {
        "id": "TEXT",
        "label": "Texte — mlx_lm",
        "short_label": "Texte",
        "capability": "supports_text",
        "backend": "mlx_lm",
        "api_hint": "/v1/chat/completions",
    },
    {
        "id": "MULTIMODAL",
        "label": "Multimodal — mlx_vlm",
        "short_label": "Multimodal",
        "capability": "supports_multimodal",
        "backend": "mlx_vlm",
        "api_hint": "/v1/chat/completions (vision)",
    },
    {
        "id": "EMBEDDING",
        "label": "Embeddings — mlx-embeddings",
        "short_label": "Embeddings",
        "capability": "supports_embedding",
        "backend": "mlx-embeddings",
        "api_hint": "/v1/embeddings",
    },
    {
        "id": "RERANKER",
        "label": "Rerank — local-reranker",
        "short_label": "Rerank",
        "capability": "supports_rerank",
        "backend": "local-reranker",
        "api_hint": "/v1/rerank",
    },
    {
        "id": "IMAGE",
        "label": "Image — mflux",
        "short_label": "Image",
        "capability": "supports_image",
        "backend": "mflux",
        "api_hint": "/v1/images/generations",
    },
    {
        "id": "TTS",
        "label": "TTS — mlx-audio (Kokoro)",
        "short_label": "TTS",
        "capability": "supports_tts",
        "backend": "mlx-audio",
        "api_hint": "/v1/audio/speech",
    },
    {
        "id": "STT",
        "label": "STT — mlx-audio (Whisper)",
        "short_label": "STT",
        "capability": "supports_stt",
        "backend": "mlx-audio",
        "api_hint": "/v1/audio/transcriptions",
    },
)


def get_server_type(server_type_id: str) -> ServerTypeSpec | None:
    for spec in SERVER_TYPES:
        if spec["id"] == server_type_id:
            return spec
    return None
