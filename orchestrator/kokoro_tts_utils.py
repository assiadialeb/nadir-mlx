"""Kokoro TTS voice and language resolution for OpenAI-compatible clients."""

from __future__ import annotations

import re
from typing import Optional

from orchestrator.kokoro_voices import (
    DEFAULT_KOKORO_LANG_CODE,
    DEFAULT_KOKORO_VOICE,
    KOKORO_VOICES,
)

KOKORO_VOICE_PATTERN = re.compile(r"^[a-z]{2}_[a-z0-9_]+$", re.IGNORECASE)

OPENAI_TTS_VOICES: frozenset[str] = frozenset({
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
})

OPENAI_FEMALE_VOICES: frozenset[str] = frozenset({
    "alloy",
    "nova",
    "shimmer",
    "coral",
    "sage",
    "ballad",
})

OPENAI_MALE_VOICES: frozenset[str] = frozenset({
    "echo",
    "onyx",
    "fable",
    "ash",
    "verse",
})

LANG_ALIASES: dict[str, str] = {
    "a": "a",
    "b": "b",
    "e": "e",
    "f": "f",
    "h": "h",
    "i": "i",
    "p": "p",
    "j": "j",
    "z": "z",
    "en": "a",
    "en-us": "a",
    "en_us": "a",
    "american": "a",
    "american english": "a",
    "en-gb": "b",
    "en_gb": "b",
    "british": "b",
    "british english": "b",
    "es": "e",
    "es-es": "e",
    "spanish": "e",
    "fr": "f",
    "fr-fr": "f",
    "fr_fr": "f",
    "french": "f",
    "français": "f",
    "francais": "f",
    "hi": "h",
    "hindi": "h",
    "it": "i",
    "italian": "i",
    "pt": "p",
    "pt-br": "p",
    "portuguese": "p",
    "ja": "j",
    "jp": "j",
    "japanese": "j",
    "zh": "z",
    "cmn": "z",
    "mandarin": "z",
    "chinese": "z",
}

DEFAULT_VOICE_BY_LANG: dict[str, str] = {
    "a": "af_heart",
    "b": "bf_emma",
    "e": "af_heart",
    "f": "ff_siwis",
    "h": "af_heart",
    "i": "af_heart",
    "p": "af_heart",
    "j": "jf_alpha",
    "z": "zf_xiaobei",
}

FEMALE_VOICE_BY_LANG: dict[str, str] = {
    "a": "af_heart",
    "b": "bf_emma",
    "e": "af_heart",
    "f": "ff_siwis",
    "h": "af_heart",
    "i": "af_heart",
    "p": "af_heart",
    "j": "jf_alpha",
    "z": "zf_xiaobei",
}

MALE_VOICE_BY_LANG: dict[str, str] = {
    "a": "am_adam",
    "b": "bm_george",
    "e": "am_adam",
    "f": "ff_siwis",
    "h": "am_adam",
    "i": "am_adam",
    "p": "am_adam",
    "j": "jf_gongitsune",
    "z": "zm_yunjian",
}

KNOWN_KOKORO_VOICE_IDS: frozenset[str] = frozenset(voice for voice, _ in KOKORO_VOICES)


def is_kokoro_voice_id(voice: str) -> bool:
    """Return True when the id looks like a Kokoro voice pack name."""
    normalized = voice.strip().lower()
    if normalized in KNOWN_KOKORO_VOICE_IDS:
        return True
    return bool(KOKORO_VOICE_PATTERN.match(normalized))


def normalize_lang_code(raw: Optional[str], fallback: str = DEFAULT_KOKORO_LANG_CODE) -> str:
    """Map API / UI language hints to Kokoro single-letter lang codes."""
    if raw is None or not str(raw).strip():
        return fallback
    key = str(raw).strip().lower().replace("_", "-")
    if key in LANG_ALIASES:
        return LANG_ALIASES[key]
    if len(key) == 1 and key in LANG_ALIASES:
        return key
    return fallback


def map_openai_voice_to_kokoro(openai_voice: str, lang_code: str) -> str:
    """Pick a Kokoro voice when the client sends an OpenAI TTS voice name."""
    normalized = openai_voice.strip().lower()
    if normalized in OPENAI_MALE_VOICES:
        return MALE_VOICE_BY_LANG.get(lang_code, DEFAULT_VOICE_BY_LANG.get(lang_code, DEFAULT_KOKORO_VOICE))
    if normalized in OPENAI_FEMALE_VOICES:
        return FEMALE_VOICE_BY_LANG.get(lang_code, DEFAULT_VOICE_BY_LANG.get(lang_code, DEFAULT_KOKORO_VOICE))
    return DEFAULT_VOICE_BY_LANG.get(lang_code, DEFAULT_KOKORO_VOICE)


def resolve_kokoro_voice(
    requested_voice: Optional[str],
    lang_code: str,
    server_default_voice: Optional[str],
) -> tuple[str, Optional[str]]:
    """Resolve the Kokoro voice id and optional remap note for logging."""
    normalized_lang = normalize_lang_code(lang_code)

    if requested_voice and is_kokoro_voice_id(requested_voice):
        return requested_voice.strip(), None

    if requested_voice and requested_voice.strip().lower() in OPENAI_TTS_VOICES:
        mapped = map_openai_voice_to_kokoro(requested_voice, normalized_lang)
        return mapped, f"OpenAI voice '{requested_voice}' mapped to Kokoro '{mapped}'"

    if server_default_voice and is_kokoro_voice_id(server_default_voice):
        return server_default_voice, None

    default_voice = DEFAULT_VOICE_BY_LANG.get(normalized_lang, DEFAULT_KOKORO_VOICE)
    return default_voice, None


def resolve_lang_code(
    lang_code: Optional[str],
    language: Optional[str],
    server_default: Optional[str],
) -> str:
    """Resolve Kokoro lang code from request fields and server defaults."""
    for candidate in (lang_code, language, server_default):
        if candidate and str(candidate).strip():
            return normalize_lang_code(
                str(candidate),
                fallback=normalize_lang_code(server_default, DEFAULT_KOKORO_LANG_CODE),
            )
    return DEFAULT_KOKORO_LANG_CODE
