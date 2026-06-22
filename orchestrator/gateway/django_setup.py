"""Initialize Django before gateway code touches the ORM."""

from __future__ import annotations

import os


def setup_django() -> None:
    """Configure Django settings for the standalone gateway worker."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mlx_orchestrator.settings")
    import django

    django.setup()
