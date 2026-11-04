"""Gateway configuration loaded from environment and Django settings."""

from __future__ import annotations

from dataclasses import dataclass

from django.conf import settings

from orchestrator.env_utils import env_int, env_str
from orchestrator.security_utils import is_loopback_bind_host


@dataclass(frozen=True)
class GatewayConfig:
    host: str
    port: int


def assert_gateway_bind_safe(host: str) -> None:
    """Refuse non-loopback binds when the gateway API key is unset."""
    if is_loopback_bind_host(host):
        return
    if env_str("NADIR_GATEWAY_API_KEY", ""):
        return
    raise SystemExit(
        "Refusing to bind Nadir Gateway to a non-loopback address "
        f"({host!r}) without NADIR_GATEWAY_API_KEY. "
        "Set the key or bind to 127.0.0.1 / localhost."
    )


def load_gateway_config() -> GatewayConfig:
    """Load gateway bind host/port and validate port allocation."""
    host = env_str("NADIR_GATEWAY_HOST", settings.NADIR_GATEWAY_HOST)
    port = env_int("NADIR_GATEWAY_PORT", settings.NADIR_GATEWAY_PORT)
    range_start = settings.INSTANCE_PORT_RANGE_START
    range_end = settings.INSTANCE_PORT_RANGE_END

    if range_start <= port <= range_end:
        raise ValueError(
            f"NADIR_GATEWAY_PORT ({port}) must stay outside the MLX instance range "
            f"{range_start}-{range_end}."
        )
    if port < 1024:
        raise ValueError(f"NADIR_GATEWAY_PORT ({port}) must be >= 1024.")

    assert_gateway_bind_safe(host)
    return GatewayConfig(host=host, port=port)
