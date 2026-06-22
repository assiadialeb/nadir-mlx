"""Tests for OpenAI-compatible embedding response formatting."""

from django.test import SimpleTestCase

from orchestrator.embedding_response import (
    EmbeddingDimensionError,
    EmbeddingFormatError,
    base64_to_floats,
    build_embedding_data_entries,
    floats_to_base64,
    normalize_encoding_format,
    prepare_embedding_row,
    truncate_embedding_dimensions,
)


class EmbeddingResponseTests(SimpleTestCase):
    def test_normalize_encoding_format_accepts_float_and_base64(self) -> None:
        self.assertEqual(normalize_encoding_format("FLOAT"), "float")
        self.assertEqual(normalize_encoding_format("base64"), "base64")

    def test_normalize_encoding_format_rejects_unknown(self) -> None:
        with self.assertRaises(EmbeddingFormatError):
            normalize_encoding_format("binary")

    def test_truncate_embedding_dimensions_returns_prefix(self) -> None:
        vector = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(truncate_embedding_dimensions(vector, 2), [1.0, 2.0])

    def test_truncate_embedding_dimensions_rejects_invalid_values(self) -> None:
        vector = [1.0, 2.0]
        with self.assertRaises(EmbeddingDimensionError):
            truncate_embedding_dimensions(vector, 0)
        with self.assertRaises(EmbeddingDimensionError):
            truncate_embedding_dimensions(vector, 5)

    def test_floats_to_base64_round_trip(self) -> None:
        vector = [0.25, -1.5, 3.0]
        encoded = floats_to_base64(vector)
        self.assertIsInstance(encoded, str)
        self.assertEqual(base64_to_floats(encoded), vector)

    def test_prepare_embedding_row_base64_with_dimensions(self) -> None:
        row = prepare_embedding_row(
            [1.0, 2.0, 3.0],
            encoding_format="base64",
            dimensions=2,
        )
        self.assertEqual(base64_to_floats(str(row)), [1.0, 2.0])

    def test_build_embedding_data_entries_batch_indices(self) -> None:
        entries = build_embedding_data_entries(
            [[1.0, 2.0], [3.0, 4.0]],
            encoding_format="float",
            dimensions=None,
        )
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["index"], 0)
        self.assertEqual(entries[1]["embedding"], [3.0, 4.0])
