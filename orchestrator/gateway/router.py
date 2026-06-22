"""Gateway routing errors and resolved upstream targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LaunchMode = Literal[
    "TEXT",
    "MULTIMODAL",
    "EMBEDDING",
    "RERANKER",
    "IMAGE",
    "TTS",
    "STT",
]

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
COMPLETIONS_PATH = "/v1/completions"
EMBEDDINGS_PATH = "/v1/embeddings"
RERANK_PATH = "/v1/rerank"
IMAGES_PATH = "/v1/images/generations"
AUDIO_SPEECH_PATH = "/v1/audio/speech"
AUDIO_TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"
AUDIO_TRANSLATIONS_PATH = "/v1/audio/translations"

LAUNCH_MODE_API_PATH: dict[str, str] = {
    "TEXT": CHAT_COMPLETIONS_PATH,
    "MULTIMODAL": CHAT_COMPLETIONS_PATH,
    "EMBEDDING": EMBEDDINGS_PATH,
    "RERANKER": RERANK_PATH,
    "IMAGE": IMAGES_PATH,
    "TTS": AUDIO_SPEECH_PATH,
    "STT": AUDIO_TRANSCRIPTIONS_PATH,
}


@dataclass(frozen=True)
class GatewayRouteError(Exception):
    """Structured routing failure returned to API clients."""

    status_code: int
    code: str
    message: str

    def to_openai_error(self) -> dict[str, object]:
        return {
            "error": {
                "message": self.message,
                "type": self.code,
                "code": self.code,
            }
        }


@dataclass(frozen=True)
class GatewayTarget:
    """Resolved upstream MLX instance for a gateway alias."""

    alias: str
    instance_id: int
    launch_mode: str
    host: str
    port: int
    upstream_model: str
    api_path: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def upstream_url(self) -> str:
        return f"{self.base_url}{self.api_path}"
