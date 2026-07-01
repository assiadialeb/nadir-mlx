"""Tests for environment variable helpers."""

from django.test import SimpleTestCase
from unittest.mock import patch

from orchestrator.env_utils import env_int, env_str


class EnvUtilsTests(SimpleTestCase):
    @patch.dict("os.environ", {"NADIR_TEST_STR": " 0.0.0.0 "}, clear=False)
    def test_env_str_returns_stripped_value(self) -> None:
        self.assertEqual(env_str("NADIR_TEST_STR", "127.0.0.1"), "0.0.0.0")

    @patch.dict("os.environ", {"NADIR_TEST_STR": ""}, clear=False)
    def test_env_str_treats_blank_as_unset(self) -> None:
        self.assertEqual(env_str("NADIR_TEST_STR", "127.0.0.1"), "127.0.0.1")

    @patch.dict("os.environ", {}, clear=False)
    def test_env_str_uses_fallback_when_missing(self) -> None:
        self.assertEqual(env_str("NADIR_TEST_MISSING", "11380"), "11380")

    @patch.dict("os.environ", {"NADIR_TEST_PORT": "11450"}, clear=False)
    def test_env_int_parses_value(self) -> None:
        self.assertEqual(env_int("NADIR_TEST_PORT", 11380), 11450)

    @patch.dict("os.environ", {"NADIR_TEST_PORT": "  "}, clear=False)
    def test_env_int_treats_blank_as_unset(self) -> None:
        self.assertEqual(env_int("NADIR_TEST_PORT", 11380), 11380)
