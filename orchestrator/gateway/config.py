"""Gateway configuration loaded from environment and Django settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from django.conf import settings


@dataclass(frozen=True)
class GatewayConfig:
    host: str
    port: int


def load_gateway_config() -> GatewayConfig:
    """Load gateway bind host/port and validate port allocation."""
    host = os.environ.get("NADIR_GATEWAY_HOST", settings.NADIR_GATEWAY_HOST)
    port = int(os.environ.get("NADIR_GATEWAY_PORT", settings.NADIR_GATEWAY_PORT))
    range_start = settings.INSTANCE_PORT_RANGE_START
    range_end = settings.INSTANCE_PORT_RANGE_END

    if range_start <= port <= range_end:
        raise ValueError(
            f"NADIR_GATEWAY_PORT ({port}) must stay outside the MLX instance range "
            f"{range_start}-{range_end}."
        )
    if port < 1024:
        raise ValueError(f"NADIR_GATEWAY_PORT ({port}) must be >= 1024.")

    return GatewayConfig(host=host, port=port)
