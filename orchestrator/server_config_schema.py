"""Server configuration schema, validation, and UI field metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from orchestrator.kokoro_voices import (
    DEFAULT_KOKORO_LANG_CODE,
    DEFAULT_KOKORO_VOICE,
    KOKORO_LANG_CODES,
    KOKORO_VOICES,
)
from orchestrator.model_registry import apply_registry_server_defaults

LaunchModeId = Literal["TEXT", "MULTIMODAL", "EMBEDDING", "RERANKER", "IMAGE", "TTS", "STT"]

FieldWidget = Literal["text", "number", "select", "checkbox", "resource"]


@dataclass(frozen=True)
class ConfigFieldSpec:
    name: str
    label: str
    widget: FieldWidget
    modes: tuple[str, ...]
    default: Any = None
    required: bool = False
    placeholder: str = ""
    help_text: str = ""
    choices: tuple[tuple[str, str], ...] = ()
    min_value: int | float | None = None
    max_value: int | float | None = None
    advanced_only: bool = False
    resource_type: str = ""


COMMON_FIELDS: tuple[ConfigFieldSpec, ...] = (
    ConfigFieldSpec(
        name="host",
        label="Interface réseau",
        widget="select",
        modes=("TEXT", "MULTIMODAL", "EMBEDDING", "RERANKER", "IMAGE", "TTS", "STT"),
        default="0.0.0.0",
        choices=(
            ("0.0.0.0", "Toutes les interfaces (0.0.0.0)"),
            ("127.0.0.1", "Local uniquement (127.0.0.1)"),
        ),
        help_text="Utilisez 127.0.0.1 pour limiter l'accès à la machine locale.",
    ),
    ConfigFieldSpec(
        name="model_id",
        label="ID modèle (API)",
        widget="text",
        modes=("TEXT", "MULTIMODAL", "EMBEDDING", "RERANKER", "IMAGE", "TTS", "STT"),
        placeholder="Identique au dossier si vide",
        help_text="Nom exposé via /v1/models (ex. pour LiteLLM ou clients OpenAI).",
    ),
)

MODE_FIELDS: tuple[ConfigFieldSpec, ...] = (
    ConfigFieldSpec(
        name="max_tokens",
        label="Max tokens (défaut serveur)",
        widget="number",
        modes=("TEXT", "MULTIMODAL"),
        placeholder="512",
        min_value=1,
        max_value=131_072,
        help_text="Plafond par défaut si le client n'envoie pas max_tokens.",
    ),
    ConfigFieldSpec(
        name="max_kv_size",
        label="Taille max du cache KV",
        widget="number",
        modes=("MULTIMODAL",),
        placeholder="Auto",
        min_value=512,
        max_value=1_000_000,
        help_text="Limite la mémoire contextuelle (tokens). Utile sur gros modèles VLM.",
    ),
    ConfigFieldSpec(
        name="trust_remote_code",
        label="Trust remote code",
        widget="checkbox",
        modes=("TEXT", "MULTIMODAL"),
        default=False,
        help_text="Requis pour certains tokenizers ou architectures custom.",
    ),
    ConfigFieldSpec(
        name="disable_batching",
        label="Désactiver le batching",
        widget="checkbox",
        modes=("RERANKER",),
        default=False,
        help_text="Débug ou compatibilité local-reranker.",
    ),
    ConfigFieldSpec(
        name="default_quality",
        label="Qualité par défaut",
        widget="select",
        modes=("IMAGE",),
        default="balanced",
        choices=(
            ("fast", "Rapide"),
            ("balanced", "Équilibré"),
            ("quality", "Qualité"),
        ),
        help_text="Preset utilisé si quality n'est pas précisé dans la requête API.",
    ),
    ConfigFieldSpec(
        name="voice_id",
        label="Voix",
        widget="select",
        modes=("TTS",),
        default=DEFAULT_KOKORO_VOICE,
        choices=KOKORO_VOICES,
        help_text="Voix Kokoro utilisée par défaut si le client n'envoie pas voice.",
    ),
    ConfigFieldSpec(
        name="speaking_rate",
        label="Vitesse de parole",
        widget="number",
        modes=("TTS",),
        default=1.0,
        placeholder="1.0",
        min_value=0.25,
        max_value=4.0,
        help_text="Multiplicateur de vitesse (selon le backend TTS).",
    ),
    ConfigFieldSpec(
        name="lang_code",
        label="Langue (Kokoro)",
        widget="select",
        modes=("TTS",),
        default=DEFAULT_KOKORO_LANG_CODE,
        choices=KOKORO_LANG_CODES,
        help_text="Code langue Kokoro (phonétique) utilisé par défaut.",
    ),
    ConfigFieldSpec(
        name="language",
        label="Langue (transcription)",
        widget="text",
        modes=("STT",),
        placeholder="Auto (détection)",
        help_text="Code ISO (ex. en, fr). Laisser vide pour la détection automatique.",
    ),
    ConfigFieldSpec(
        name="chunk_duration",
        label="Durée des segments (s)",
        widget="number",
        modes=("STT",),
        default=30.0,
        placeholder="30",
        min_value=1.0,
        max_value=120.0,
        help_text="Taille des fenêtres audio pour Whisper (secondes).",
    ),
)

ADVANCED_WHITELIST: dict[str, frozenset[str]] = {
    "TEXT": frozenset({
        "adapter_path",
        "draft_model",
        "num_draft_tokens",
        "chat_template_args",
        "temp",
        "top_p",
        "top_k",
        "min_p",
    }),
    "MULTIMODAL": frozenset({
        "adapter_path",
        "draft_model",
        "draft_kind",
        "draft_block_size",
        "kv_bits",
        "kv_quant_scheme",
        "kv_group_size",
        "enable_thinking",
        "thinking_budget",
    }),
    "EMBEDDING": frozenset(),
    "RERANKER": frozenset(),
    "IMAGE": frozenset({"quantize_override"}),
    "TTS": frozenset({"response_format"}),
    "STT": frozenset(),
}

ALL_FIELD_SPECS: tuple[ConfigFieldSpec, ...] = COMMON_FIELDS + MODE_FIELDS


def get_fields_for_mode(launch_mode: str) -> list[ConfigFieldSpec]:
    return [
        field
        for field in ALL_FIELD_SPECS
        if launch_mode in field.modes and not field.advanced_only
    ]


def build_default_server_config(
    launch_mode: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {"advanced": {}}
    for field in get_fields_for_mode(launch_mode):
        if field.default is not None:
            config[field.name] = field.default
    if model_name:
        config = apply_registry_server_defaults(launch_mode, model_name, config)
    return config


def _coerce_checkbox(raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).lower() in ("1", "true", "on", "yes")


def _coerce_number(field: ConfigFieldSpec, raw_value: Any) -> int | float:
    if isinstance(raw_value, (int, float)):
        number = raw_value
    else:
        text = str(raw_value).strip()
        number = float(text) if "." in text else int(text)
    if field.min_value is not None and number < field.min_value:
        raise ValueError(f"{field.label}: minimum {field.min_value}.")
    if field.max_value is not None and number > field.max_value:
        raise ValueError(f"{field.label}: maximum {field.max_value}.")
    if field.name not in ("speaking_rate", "chunk_duration") and isinstance(number, float):
        return int(number)
    return number


def _coerce_select(field: ConfigFieldSpec, raw_value: Any) -> str:
    value = str(raw_value).strip()
    allowed = {choice[0] for choice in field.choices}
    if value not in allowed:
        raise ValueError(f"{field.label}: valeur invalide '{value}'.")
    return value


def _coerce_required_text(field: ConfigFieldSpec, raw_value: Any) -> str:
    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"{field.label} ne peut pas être vide.")
    return value


def _coerce_field_value(field: ConfigFieldSpec, raw_value: Any) -> Any:
    if field.widget == "checkbox":
        return _coerce_checkbox(raw_value)

    if field.widget == "number":
        return _coerce_number(field, raw_value)

    if field.widget == "select":
        return _coerce_select(field, raw_value)

    if field.widget == "resource":
        value = str(raw_value).strip()
        if not value:
            raise ValueError(f"{field.label} est requis.")
        return value

    return _coerce_required_text(field, raw_value)


def _validate_advanced(launch_mode: str, advanced: Any) -> dict[str, Any]:
    if advanced is None:
        return {}
    if not isinstance(advanced, dict):
        raise ValueError("La section advanced doit être un objet JSON.")

    allowed = ADVANCED_WHITELIST.get(launch_mode, frozenset())
    unknown = set(advanced.keys()) - allowed
    if unknown:
        labels = ", ".join(sorted(unknown))
        raise ValueError(f"Clés advanced non autorisées pour {launch_mode}: {labels}.")

    normalized: dict[str, Any] = {}
    for key, value in advanced.items():
        if value is None:
            continue
        normalized[key] = value
    return normalized


def validate_and_normalize_server_config(
    launch_mode: str,
    raw_config: dict[str, Any] | None,
    model_name: str,
) -> dict[str, Any]:
    """Merge defaults, validate fields, and return a storable server_config dict."""
    raw = dict(raw_config or {})
    advanced_raw = raw.pop("advanced", {})
    normalized = build_default_server_config(launch_mode, model_name)

    for field in get_fields_for_mode(launch_mode):
        if field.name not in raw:
            continue
        value = raw[field.name]
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        normalized[field.name] = _coerce_field_value(field, value)

    if not normalized.get("model_id"):
        normalized["model_id"] = model_name

    normalized["advanced"] = _validate_advanced(launch_mode, advanced_raw)
    return normalized


def parse_server_config_from_post(
    post_data: dict[str, Any],
    launch_mode: str,
    model_name: str,
) -> dict[str, Any]:
    """Build server_config from HTML form fields (config_<name> + config_advanced)."""
    raw: dict[str, Any] = {}
    for field in get_fields_for_mode(launch_mode):
        form_key = f"config_{field.name}"
        if field.widget == "checkbox":
            raw[field.name] = form_key in post_data
            continue
        if form_key not in post_data:
            continue
        value = post_data.get(form_key)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        raw[field.name] = value

    advanced_text = (post_data.get("config_advanced") or "").strip()
    if advanced_text:
        try:
            raw["advanced"] = json.loads(advanced_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON avancé invalide: {exc}") from exc

    return validate_and_normalize_server_config(launch_mode, raw, model_name)


def resolve_server_config_from_request(
    post_data: dict[str, Any],
    launch_mode: str,
    model_name: str,
) -> dict[str, Any]:
    """Use explicit JSON payload (restart) or structured form fields (create)."""
    raw_json = (post_data.get("server_config_json") or "").strip()
    if raw_json:
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Configuration serveur invalide: {exc}") from exc
        return validate_and_normalize_server_config(launch_mode, payload, model_name)
    return parse_server_config_from_post(post_data, launch_mode, model_name)


def config_fields_for_ui_json() -> str:
    """Serialize field metadata for client-side form rendering."""
    payload: dict[str, list[dict[str, Any]]] = {}
    for mode in ("TEXT", "MULTIMODAL", "EMBEDDING", "RERANKER", "IMAGE", "TTS", "STT"):
        payload[mode] = [
            {
                "name": field.name,
                "label": field.label,
                "widget": field.widget,
                "default": field.default,
                "placeholder": field.placeholder,
                "help_text": field.help_text,
                "choices": [{"value": v, "label": lbl} for v, lbl in field.choices],
                "min": field.min_value,
                "max": field.max_value,
            }
            for field in get_fields_for_mode(mode)
        ]
    return json.dumps(payload)


def advanced_keys_for_ui_json() -> str:
    return json.dumps({mode: sorted(keys) for mode, keys in ADVANCED_WHITELIST.items()})
