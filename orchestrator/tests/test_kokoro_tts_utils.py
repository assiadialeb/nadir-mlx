"""Tests for Kokoro voice and language resolution."""

from unittest import TestCase

from orchestrator.kokoro_tts_utils import (
    map_openai_voice_to_kokoro,
    normalize_lang_code,
    resolve_kokoro_voice,
    resolve_lang_code,
)


class KokoroTtsUtilsTests(TestCase):
    def test_normalize_lang_code_french_aliases(self) -> None:
        self.assertEqual(normalize_lang_code("fr"), "f")
        self.assertEqual(normalize_lang_code("fr-FR"), "f")
        self.assertEqual(normalize_lang_code("french"), "f")

    def test_map_openai_alloy_to_french_voice(self) -> None:
        self.assertEqual(map_openai_voice_to_kokoro("alloy", "f"), "ff_siwis")

    def test_resolve_openai_voice_for_french_server(self) -> None:
        voice, note = resolve_kokoro_voice("alloy", "f", "ff_siwis")
        self.assertEqual(voice, "ff_siwis")
        self.assertIn("alloy", note or "")

    def test_resolve_explicit_kokoro_voice(self) -> None:
        voice, note = resolve_kokoro_voice("af_heart", "a", "ff_siwis")
        self.assertEqual(voice, "af_heart")
        self.assertIsNone(note)

    def test_resolve_lang_code_from_language_field(self) -> None:
        self.assertEqual(resolve_lang_code(None, "fr", "a"), "f")

    def test_server_defaults_used_when_client_omits_voice(self) -> None:
        voice, _ = resolve_kokoro_voice(None, "f", "ff_siwis")
        self.assertEqual(voice, "ff_siwis")
