"""OpenAI-compatible embedding response formatting helpers."""

from __future__ import annotations

import base64
import struct
from typing import Any, Final

SUPPORTED_ENCODING_FORMATS: Final[frozenset[str]] = frozenset({"float", "base64"})


class EmbeddingFormatError(ValueError):
    """Raised when the client requests an unsupported encoding_format."""

    def __init__(self, encoding_format: str) -> None:
        self.encoding_format = encoding_format
        supported = ", ".join(sorted(SUPPORTED_ENCODING_FORMATS))
        super().__init__(
            f"Unsupported encoding_format '{encoding_format}'. "
            f"Supported formats: {supported}."
        )


class EmbeddingDimensionError(ValueError):
    """Raised when dimensions is invalid for the embedding vector."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def normalize_encoding_format(raw: str | None) -> str:
    """Normalize and validate an OpenAI-style encoding_format value."""
    normalized = (raw or "float").strip().lower()
    if normalized not in SUPPORTED_ENCODING_FORMATS:
        raise EmbeddingFormatError(normalized)
    return normalized


def truncate_embedding_dimensions(
    vector: list[float],
    dimensions: int | None,
) -> list[float]:
    """Truncate an embedding to the first N dimensions (Matryoshka-style)."""
    if dimensions is None:
        return vector
    if dimensions <= 0:
        raise EmbeddingDimensionError("dimensions must be a positive integer.")
    if dimensions > len(vector):
        raise EmbeddingDimensionError(
            f"dimensions ({dimensions}) exceeds embedding length ({len(vector)})."
        )
    return vector[:dimensions]


def floats_to_base64(vector: list[float]) -> str:
    """Encode float32 values as OpenAI-style base64 (little-endian)."""
    packed = struct.pack(f"<{len(vector)}f", *vector)
    return base64.b64encode(packed).decode("ascii")


def base64_to_floats(encoded: str) -> list[float]:
    """Decode OpenAI-style base64 embedding bytes back to floats."""
    packed = base64.b64decode(encoded)
    if len(packed) % 4:
        raise ValueError("Invalid base64 embedding payload length.")
    count = len(packed) // 4
    return list(struct.unpack(f"<{count}f", packed))


def format_embedding_value(
    vector: list[float],
    encoding_format: str,
) -> list[float] | str:
    """Return either a float list or a base64 string for one embedding row."""
    if encoding_format == "base64":
        return floats_to_base64(vector)
    return vector


def prepare_embedding_row(
    vector: list[float],
    *,
    encoding_format: str,
    dimensions: int | None,
) -> list[float] | str:
    """Apply dimension truncation then encoding for a single embedding vector."""
    truncated = truncate_embedding_dimensions(vector, dimensions)
    return format_embedding_value(truncated, encoding_format)


def build_embedding_data_entries(
    vectors: list[list[float]],
    *,
    encoding_format: str,
    dimensions: int | None,
) -> list[dict[str, Any]]:
    """Build OpenAI-style data entries for one or more embedding rows."""
    return [
        {
            "object": "embedding",
            "index": index,
            "embedding": prepare_embedding_row(
                row,
                encoding_format=encoding_format,
                dimensions=dimensions,
            ),
        }
        for index, row in enumerate(vectors)
    ]
