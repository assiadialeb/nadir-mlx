"""OpenAI-compatible error envelopes for the Nadir Gateway."""

from __future__ import annotations

from typing import Any

OPENAI_TYPE_BY_CODE: dict[str, str] = {
    "model_not_found": "invalid_request_error",
    "invalid_model": "invalid_request_error",
    "unsupported_endpoint": "invalid_request_error",
    "invalid_request": "invalid_request_error",
    "invalid_api_key": "authentication_error",
    "model_unavailable": "server_error",
    "upstream_queue_timeout": "server_error",
    "gateway_timeout": "server_error",
    "bad_gateway": "server_error",
    "upstream_error": "server_error",
    "gateway_error": "server_error",
    "unsupported_launch_mode": "server_error",
}

PARAM_BY_CODE: dict[str, str] = {
    "model_not_found": "model",
    "invalid_model": "model",
}


def openai_error_type_for_code(code: str) -> str:
    """Map a gateway machine code to an OpenAI error ``type``."""
    return OPENAI_TYPE_BY_CODE.get(code, "server_error")


def openai_error_envelope(
    message: str,
    code: str,
    *,
    error_type: str | None = None,
    param: str | None = None,
) -> dict[str, Any]:
    """Build ``{"error": {...}}`` with OpenAI ``type`` and gateway ``code``."""
    resolved_type = error_type or openai_error_type_for_code(code)
    error: dict[str, Any] = {
        "message": message,
        "type": resolved_type,
        "code": code,
    }
    resolved_param = param if param is not None else PARAM_BY_CODE.get(code)
    if resolved_param is not None:
        error["param"] = resolved_param
    return {"error": error}
