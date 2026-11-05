from django.apps import AppConfig


class OrchestratorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'orchestrator'

    def ready(self) -> None:
        from orchestrator.serve_supervisor import start_parent_death_watch_if_needed

        # Orphaned serve children exit when the parent is SIGKILL'd / OOM'd.
        start_parent_death_watch_if_needed()

        import sys

        from orchestrator.instance_health import should_skip_watchdog

        # pytest argv has no "test" token; loops would race the test database.
        if "pytest" in sys.modules or should_skip_watchdog():
            return

        from django.conf import settings

        from orchestrator.startup_reconcile import start_startup_reconcile_if_needed

        start_startup_reconcile_if_needed()

        if getattr(settings, "INSTANCE_WATCHDOG_ENABLED", True):
            from orchestrator.instance_watchdog import start_watchdog_if_needed

            start_watchdog_if_needed()

        if getattr(settings, "NADIR_IDLE_OFFLOAD_ENABLED", True):
            from orchestrator.instance_idle_watcher import start_idle_watcher_if_needed

            start_idle_watcher_if_needed()

        if getattr(settings, "NADIR_GATEWAY_BATCH_WORKER_ENABLED", True):
            from orchestrator.gateway_batch_worker import start_gateway_batch_worker_if_needed

            start_gateway_batch_worker_if_needed()
