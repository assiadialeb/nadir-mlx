"""Validated outbound requests to the Hugging Face API."""

from __future__ import annotations

from typing import Any

import requests

from orchestrator.security_utils import validate_huggingface_api_url


def huggingface_get(url: str, **kwargs: Any) -> requests.Response:
    """GET huggingface.co only — SSRF guard for hub/registry callers."""
    return requests.get(validate_huggingface_api_url(url), **kwargs)
