"""Django management command to start the Nadir Gateway worker."""

from __future__ import annotations

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Start the Nadir Gateway OpenAI-compatible proxy (FastAPI + uvicorn)."

    def handle(self, *args, **options) -> None:
        from orchestrator.gateway.__main__ import main

        self.stdout.write(self.style.SUCCESS("Starting Nadir Gateway…"))
        main()
