from django.apps import AppConfig


class OrchestratorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'orchestrator'

    def ready(self) -> None:
        from django.conf import settings

        if not getattr(settings, "INSTANCE_WATCHDOG_ENABLED", True):
            return
        from orchestrator.instance_watchdog import start_watchdog_if_needed

        start_watchdog_if_needed()
