"""Unit tests for gateway multipart upload detection."""

from io import BytesIO
from unittest import TestCase

from starlette.datastructures import UploadFile as StarletteUploadFile

from orchestrator.gateway.services.mode_proxy import _is_upload_file


class GatewayUploadFileDetectionTests(TestCase):
    def test_is_upload_file_accepts_starlette_upload_file(self) -> None:
        upload = StarletteUploadFile(filename="sample.wav", file=BytesIO(b"RIFF"))
        self.assertTrue(_is_upload_file(upload))

    def test_is_upload_file_rejects_plain_string(self) -> None:
        self.assertFalse(_is_upload_file("not-a-file"))
