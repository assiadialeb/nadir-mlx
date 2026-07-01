"""Shared OpenAPI response docs and structured errors for inference FastAPI apps."""

from __future__ import annotations

from fastapi import HTTPException

_OPENAPI_DESCRIPTIONS: dict[int, str] = {
    400: "Invalid request",
    404: "Resource not found",
    500: "Internal server error",
    501: "Not implemented",
    503: "Service unavailable",
}


def open_api_responses(*status_codes: int) -> dict[int, dict[str, str]]:
    """Build a FastAPI ``responses`` map for the given HTTP status codes."""
    return {
        code: {"description": _OPENAPI_DESCRIPTIONS[code]}
        for code in status_codes
        if code in _OPENAPI_DESCRIPTIONS
    }


OPENAPI_INFERENCE_ERRORS = open_api_responses(400, 500, 503)
OPENAPI_NOT_FOUND = open_api_responses(404)
OPENAPI_NOT_IMPLEMENTED = open_api_responses(501)
OPENAPI_IMAGE_GENERATION = open_api_responses(400, 500, 503)
OPENAPI_IMAGE_STUB = open_api_responses(501)


class InferenceApiError(Exception):
    """API error raised in services; converted to HTTPException on routes only."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def to_http_exception(error: InferenceApiError) -> HTTPException:
    """Map a structured inference error to FastAPI's HTTPException."""
    return HTTPException(status_code=error.status_code, detail=error.detail)
