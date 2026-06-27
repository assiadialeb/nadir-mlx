"""Copy orchestrator data (and auth users) from a legacy SQLite file into PostgreSQL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.db import connections, transaction

from orchestrator.security_utils import validated_sqlite_migration_path


_LEGACY_ALIAS = "legacy_sqlite"
_MIGRATION_MODELS = (
    "orchestrator.ModelDownload",
    "orchestrator.InferenceInstance",
    "orchestrator.BenchmarkRun",
)


class Command(BaseCommand):
    help = (
        "Migrate data from db.sqlite3 into the configured PostgreSQL database. "
        "Run `python manage.py migrate` on PostgreSQL first."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--sqlite-path",
            default=str(Path(settings.BASE_DIR) / "db.sqlite3"),
            help="Path to the source SQLite database (default: project db.sqlite3).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Count rows only; do not write to PostgreSQL.",
        )
        parser.add_argument(
            "--clear-target",
            action="store_true",
            help="Delete existing orchestrator rows and non-superusers before import.",
        )

    def handle(self, *args, **options) -> None:
        default_engine = settings.DATABASES["default"]["ENGINE"]
        if "postgresql" not in default_engine:
            raise CommandError(
                "Default database is not PostgreSQL. Set NADIR_DB_HOST or NADIR_DATABASE_URL."
            )

        try:
            sqlite_path = validated_sqlite_migration_path(str(options["sqlite_path"]))
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

        self._register_legacy_connection(sqlite_path)
        counts = self._count_legacy_rows()
        self.stdout.write("Legacy SQLite row counts:")
        for label, count in counts.items():
            self.stdout.write(f"  {label}: {count}")

        if options["dry_run"]:
            self.stdout.write(self.style.WARNING("Dry run — no data copied."))
            return

        with transaction.atomic():
            if options["clear_target"]:
                self._clear_target_tables()
            self._copy_users()
            self._copy_orchestrator_models()
            self._reset_sequences()

        self.stdout.write(self.style.SUCCESS("SQLite → PostgreSQL migration completed."))

    def _register_legacy_connection(self, sqlite_path: Path) -> None:
        legacy_config = {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": sqlite_path,
            "ATOMIC_REQUESTS": False,
            "AUTOCOMMIT": True,
            "CONN_MAX_AGE": 0,
            "CONN_HEALTH_CHECKS": False,
            "OPTIONS": {},
            "TIME_ZONE": settings.TIME_ZONE,
            "USER": "",
            "PASSWORD": "",
            "HOST": "",
            "PORT": "",
            "TEST": {
                "CHARSET": None,
                "COLLATION": None,
                "MIGRATE": True,
                "MIRROR": None,
                "NAME": None,
            },
        }
        settings.DATABASES[_LEGACY_ALIAS] = legacy_config
        connections.databases[_LEGACY_ALIAS] = legacy_config
        connections[_LEGACY_ALIAS].close()

    def _count_legacy_rows(self) -> dict[str, int]:
        user_model = get_user_model()
        counts = {user_model._meta.label: user_model.objects.using(_LEGACY_ALIAS).count()}
        for label in _MIGRATION_MODELS:
            model = apps.get_model(label)
            counts[label] = model.objects.using(_LEGACY_ALIAS).count()
        return counts

    def _clear_target_tables(self) -> None:
        benchmark_model = apps.get_model("orchestrator", "BenchmarkRun")
        benchmark_model.objects.all().delete()
        apps.get_model("orchestrator", "InferenceInstance").objects.all().delete()
        apps.get_model("orchestrator", "ModelDownload").objects.all().delete()
        user_model = get_user_model()
        user_model.objects.filter(is_superuser=False).delete()
        self.stdout.write("Cleared target orchestrator tables and non-superuser accounts.")

    def _copy_users(self) -> None:
        user_model = get_user_model()
        copied = 0
        for legacy_user in user_model.objects.using(_LEGACY_ALIAS).all():
            payload = {
                field.name: getattr(legacy_user, field.name)
                for field in user_model._meta.fields
                if field.name != "id"
            }
            user_model.objects.update_or_create(
                username=legacy_user.username,
                defaults=payload,
            )
            copied += 1
        self.stdout.write(f"Users synced: {copied}")

    def _copy_orchestrator_models(self) -> None:
        for label in _MIGRATION_MODELS:
            model = apps.get_model(label)
            copied = 0
            for legacy_row in model.objects.using(_LEGACY_ALIAS).all().order_by("pk"):
                payload = self._model_payload(model, legacy_row)
                model.objects.update_or_create(pk=legacy_row.pk, defaults=payload)
                copied += 1
            self.stdout.write(f"{label}: {copied} row(s)")

    def _model_payload(self, model, legacy_row) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for field in model._meta.fields:
            if field.name == "id":
                continue
            if field.is_relation and field.many_to_one:
                payload[field.name + "_id"] = getattr(legacy_row, field.name + "_id")
                continue
            payload[field.name] = getattr(legacy_row, field.name)
        return payload

    def _reset_sequences(self) -> None:
        from django.core.management.color import no_style

        connection = connections["default"]
        if connection.vendor != "postgresql":
            return

        models = [apps.get_model(label) for label in _MIGRATION_MODELS]
        models.append(get_user_model())
        statements = connection.ops.sequence_reset_sql(no_style(), models)
        with connection.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)
        self.stdout.write("PostgreSQL sequences reset.")
