"""Kokoro voice presets exposed in the orchestrator UI."""

from __future__ import annotations

KOKORO_VOICES: tuple[tuple[str, str], ...] = (
    ("af_heart", "af_heart — American English (F)"),
    ("af_bella", "af_bella — American English (F)"),
    ("af_sarah", "af_sarah — American English (F)"),
    ("af_nicole", "af_nicole — American English (F)"),
    ("af_sky", "af_sky — American English (F)"),
    ("am_adam", "am_adam — American English (M)"),
    ("am_michael", "am_michael — American English (M)"),
    ("bf_emma", "bf_emma — British English (F)"),
    ("bf_isabella", "bf_isabella — British English (F)"),
    ("bm_george", "bm_george — British English (M)"),
    ("bm_lewis", "bm_lewis — British English (M)"),
    ("ff_siwis", "ff_siwis — French (F)"),
    ("jf_alpha", "jf_alpha — Japanese (F)"),
    ("jf_gongitsune", "jf_gongitsune — Japanese (F)"),
    ("zf_xiaobei", "zf_xiaobei — Mandarin (F)"),
    ("zm_yunjian", "zm_yunjian — Mandarin (M)"),
)

DEFAULT_KOKORO_VOICE = "af_heart"

KOKORO_LANG_CODES: tuple[tuple[str, str], ...] = (
    ("a", "American English"),
    ("b", "British English"),
    ("e", "Spanish"),
    ("f", "French"),
    ("h", "Hindi"),
    ("i", "Italian"),
    ("p", "Portuguese"),
    ("j", "Japanese"),
    ("z", "Mandarin"),
)

DEFAULT_KOKORO_LANG_CODE = "a"


def is_valid_kokoro_voice(voice_id: str) -> bool:
    return any(voice_id == voice for voice, _ in KOKORO_VOICES)
