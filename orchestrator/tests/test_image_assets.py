"""Tests for local image asset storage."""

from __future__ import annotations

import os
from pathlib import Path

from django.test import SimpleTestCase, override_settings

from orchestrator.image_assets import (
    build_generation_response_entry,
    build_public_image_url,
    is_valid_image_file_id,
    resolve_image_png_path,
    store_image_png,
)


class ImageAssetTests(SimpleTestCase):
    def test_store_and_resolve_png_round_trip(self) -> None:
        output_dir = Path("ImageAssetTests") / "tmp-images"
        os.environ["IMAGE_OUTPUT_DIR"] = str(output_dir.resolve())
        try:
            file_id = store_image_png(b"\x89PNG\r\n\x1a\n")
            self.assertTrue(is_valid_image_file_id(file_id))
            path = resolve_image_png_path(file_id)
            self.assertIsNotNone(path)
            assert path is not None
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        finally:
            os.environ.pop("IMAGE_OUTPUT_DIR", None)
            if output_dir.exists():
                for child in output_dir.glob("*.png"):
                    child.unlink()
                output_dir.rmdir()

    @override_settings(
        NADIR_GATEWAY_PUBLIC_BASE_URL="http://127.0.0.1:11380",
    )
    def test_build_public_image_url_uses_gateway_base(self) -> None:
        url = build_public_image_url("a" * 32)
        self.assertEqual(url, f"http://127.0.0.1:11380/v1/images/files/{'a' * 32}")

    def test_build_generation_response_entry_url_format(self) -> None:
        output_dir = Path("ImageAssetTests") / "tmp-url-entry"
        os.environ["IMAGE_OUTPUT_DIR"] = str(output_dir.resolve())
        os.environ["NADIR_GATEWAY_PUBLIC_BASE_URL"] = "http://127.0.0.1:11380"
        try:
            entry = build_generation_response_entry(b"\x89PNG", "url")
            self.assertIn("url", entry)
            self.assertTrue(entry["url"].startswith("http://127.0.0.1:11380/v1/images/files/"))
        finally:
            os.environ.pop("IMAGE_OUTPUT_DIR", None)
            os.environ.pop("NADIR_GATEWAY_PUBLIC_BASE_URL", None)
            if output_dir.exists():
                for child in output_dir.glob("*.png"):
                    child.unlink()
                output_dir.rmdir()

    def test_build_generation_response_entry_b64_format(self) -> None:
        entry = build_generation_response_entry(b"png", "b64_json")
        self.assertEqual(entry["b64_json"], "cG5n")

    def test_is_valid_image_file_id_rejects_path_traversal(self) -> None:
        self.assertFalse(is_valid_image_file_id("../etc/passwd"))
        self.assertFalse(is_valid_image_file_id("not-hex"))
