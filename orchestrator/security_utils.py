"""Shared security helpers for outbound requests, paths, and error responses."""

from __future__ import annotations

import ipaddress
import os
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode, urlparse

from django.conf import settings
from django.urls import reverse

_BLOCKED_OUTBOUND_HOSTS = frozenset(
    {
        "metadata",
        "metadata.google.internal",
    }
)
_BENCHMARK_ENDPOINT_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]")
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


def benchmark_endpoint_enabled() -> bool:
    """Return whether custom ENDPOINT benchmark targets are permitted."""
    return bool(getattr(settings, "NADIR_BENCHMARK_ENDPOINT_ENABLED", settings.DEBUG))


def _benchmark_endpoint_allowed_hosts() -> list[str]:
    """Resolve allowed ENDPOINT hosts (empty list = private network OK in dev)."""
    env_raw = os.environ.get("NADIR_BENCHMARK_ENDPOINT_ALLOWED_HOSTS", "").strip()
    if env_raw:
        return [item.strip() for item in env_raw.split(",") if item.strip()]
    if settings.DEBUG:
        return []
    return list(_BENCHMARK_ENDPOINT_LOOPBACK_HOSTS)


def validate_benchmark_endpoint_host(host: str) -> str:
    """Restrict custom benchmark targets to approved hosts (anti-SSRF)."""
    if not benchmark_endpoint_enabled():
        raise ValueError(
            "Custom endpoint benchmarks are disabled. Use a running MLX instance."
        )

    cleaned = host.strip()
    if not cleaned:
        raise ValueError("Host is required for a custom endpoint.")

    safe_host = validate_server_bind_host(cleaned)
    allowed_hosts = _benchmark_endpoint_allowed_hosts()
    if not allowed_hosts:
        return safe_host

    normalized = safe_host.lower().strip("[]")
    allowed_normalized = {item.lower().strip("[]") for item in allowed_hosts}
    if normalized in allowed_normalized:
        return safe_host

    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        raise ValueError("Endpoint host is not allowed.") from None

    loopback_allowed = allowed_normalized.intersection({"127.0.0.1", "localhost", "::1"})
    if address.is_loopback and loopback_allowed:
        return safe_host

    raise ValueError("Endpoint host is not allowed.")


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


_MODELS_REDIRECT_KEYS = frozenset({"tab", "q", "cap", "sort"})


def build_models_redirect_url(params: Mapping[str, str]) -> str:
    """Build a same-origin /models redirect URL with whitelisted, sanitized query params."""
    safe_params: dict[str, str] = {}
    for key, raw_value in params.items():
        if key not in _MODELS_REDIRECT_KEYS:
            continue
        value = str(raw_value).strip()
        if not value:
            continue
        if key == "tab":
            if value not in ("installed", "hub"):
                continue
        elif key == "q":
            try:
                value = sanitize_hf_search_query(value)
            except ValueError:
                continue
            if not value:
                continue
        safe_params[key] = value

    base_path = reverse("models")
    if not safe_params:
        return base_path
    return f"{base_path}?{urlencode(safe_params)}"


def build_validated_http_url(host: str, port: int, path: str = "/") -> str:
    """Construct an http URL only after host/port validation (anti-SSRF)."""
    safe_host = validate_outbound_http_host(host)
    if port < 1 or port > 65535:
        raise ValueError("Port must be between 1 and 65535.")
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"http://{safe_host}:{port}{normalized_path}"


def benchmark_compare_export_filename(run_a_id: int, run_b_id: int) -> str:
    """Return a safe Content-Disposition filename for benchmark comparison exports."""
    safe_positive_int(run_a_id, field_name="benchmark run id")
    safe_positive_int(run_b_id, field_name="benchmark run id")
    return f"bench_compare_{run_a_id}_vs_{run_b_id}.json"


def validated_benchmark_artifact_path(run_id: int, artifact_basename: str) -> Path:
    """Return a benchmark artifact path confined under LOGS_DIR/benchmarks."""
    safe_positive_int(run_id, field_name="benchmark run id")
    benchmarks_dir = Path(settings.LOGS_DIR) / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    return safe_path_under_root(benchmarks_dir, artifact_basename)


def validated_sqlite_migration_path(raw_path: str) -> Path:
    """Validate a legacy SQLite source path for the migrate_sqlite_to_postgres command."""
    if ".." in raw_path.replace("\\", "/"):
        raise ValueError("Invalid sqlite path.")
    candidate = Path(raw_path).expanduser().resolve()
    if not candidate.is_file():
        raise ValueError(f"SQLite file not found: {candidate}")
    if candidate.suffix.lower() not in {".sqlite3", ".db"}:
        raise ValueError("Source must be a SQLite database file (.sqlite3 or .db).")
    return candidate
