"""Entrypoint: python -m orchestrator.gateway"""

from __future__ import annotations

import uvicorn

from orchestrator.gateway.config import load_gateway_config
from orchestrator.gateway.django_setup import setup_django


def main() -> None:
    setup_django()
    config = load_gateway_config()
    from orchestrator.gateway.app import create_app

    uvicorn.run(
        create_app(),
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
