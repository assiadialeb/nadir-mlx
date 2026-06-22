"""Shared security helpers for outbound requests, paths, and error responses."""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from django.conf import settings

_BLOCKED_OUTBOUND_HOSTS = frozenset(
    {
        "metadata",
        "metadata.google.internal",
    }
)
_HOSTNAME_PATTERN = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?$")
_HF_SEARCH_MAX_LENGTH = 200
_HF_API_ORIGIN = "https://huggingface.co"


def extract_bearer_token(authorization: str | None) -> str:
    """Parse a Bearer token from an Authorization header value."""
    if not authorization:
        return ""
    value = authorization.strip()
    if value.lower().startswith("bearer "):
        return value[7:].strip()
    return value


def public_error_message(
    exc: Exception,
    *,
    fallback: str = "Request failed.",
) -> str:
    """Return a client-safe error string without stack traces or paths."""
    if isinstance(exc, ValueError):
        message = str(exc).strip()
        return message or fallback
    return fallback


def sanitize_hf_search_query(query: str) -> str:
    """Restrict Hugging Face hub search input to safe plain text."""
    cleaned = query.strip()
    if not cleaned:
        return ""
    if len(cleaned) > _HF_SEARCH_MAX_LENGTH:
        cleaned = cleaned[:_HF_SEARCH_MAX_LENGTH]
    if "://" in cleaned or "\n" in cleaned or "\r" in cleaned:
        raise ValueError("Invalid search query.")
    return cleaned


def validate_huggingface_api_url(url: str) -> str:
    """Ensure outbound requests stay on the official Hugging Face API origin."""
    parsed = urlparse(url)
    expected = urlparse(_HF_API_ORIGIN)
    if parsed.scheme != expected.scheme or parsed.netloc != expected.netloc:
        raise ValueError("Outbound URL must target huggingface.co.")
    return url


def validate_server_bind_host(host: str) -> str:
    """Validate a host value used for local inference bind or health probes."""
    cleaned = host.strip()
    if not cleaned:
        raise ValueError("Host is required.")
    if any(token in cleaned for token in ("://", "/", "@", "\\")):
        raise ValueError("Invalid host.")
    if cleaned.lower() in _BLOCKED_OUTBOUND_HOSTS:
        raise ValueError("Host is not allowed.")

    bracketed = cleaned.strip("[]")
    try:
        address = ipaddress.ip_address(bracketed)
    except ValueError:
        if not _HOSTNAME_PATTERN.fullmatch(cleaned):
            raise ValueError("Invalid host.") from None
        return cleaned

    if address.is_loopback or address.is_private:
        return cleaned
    if address.is_link_local or address.is_multicast or address.is_reserved:
        raise ValueError("Host is not allowed.")
    raise ValueError("Only loopback or private network hosts are allowed.")


def validate_outbound_http_host(host: str) -> str:
    """Validate a user-supplied host before server-side HTTP requests."""
    return validate_server_bind_host(host)


def safe_positive_int(value: int, *, field_name: str = "id") -> int:
    """Reject non-positive integers used in filesystem paths."""
    if value < 1:
        raise ValueError(f"Invalid {field_name}.")
    return value


def safe_path_under_root(root: Path, relative_name: str) -> Path:
    """Join a single relative segment and ensure the result stays under root."""
    if not relative_name or relative_name in {".", ".."}:
        raise ValueError("Invalid path segment.")
    if "/" in relative_name or "\\" in relative_name or ".." in relative_name:
        raise ValueError("Invalid path segment.")

    resolved_root = root.resolve()
    candidate = (resolved_root / relative_name).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError("Path escapes the allowed directory.")
    return candidate


def assert_path_under_directory(path: Path, root: Path) -> Path:
    """Ensure a resolved path remains inside a trusted root directory."""
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError("Path escapes the allowed directory.")
    return resolved_path


def models_root_path() -> Path:
    """Return the resolved local models directory."""
    return Path(settings.MODELS_DIR).resolve()
