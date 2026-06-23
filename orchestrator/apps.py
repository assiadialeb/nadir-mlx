from django.apps import AppConfig


class OrchestratorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'orchestrator'

    def ready(self) -> None:
        from orchestrator.instance_health import should_skip_watchdog

        if should_skip_watchdog():
            return

        from django.conf import settings

        if getattr(settings, "INSTANCE_WATCHDOG_ENABLED", True):
            from orchestrator.instance_watchdog import start_watchdog_if_needed

            start_watchdog_if_needed()

        if getattr(settings, "NADIR_IDLE_OFFLOAD_ENABLED", True):
            from orchestrator.instance_idle_watcher import start_idle_watcher_if_needed

            start_idle_watcher_if_needed()
