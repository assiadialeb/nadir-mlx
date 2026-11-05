"""Run pending OpenAI-style batch jobs (MLX-132)."""

from __future__ import annotations

from django.core.management.base import BaseCommand

from orchestrator.batch_services import process_pending_batch_jobs, reconcile_stale_in_progress_jobs


class Command(BaseCommand):
    help = "Process validating gateway batch jobs (JSONL chat completions)."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--max-jobs",
            type=int,
            default=1,
            help="Maximum number of jobs to process in this run.",
        )
        parser.add_argument(
            "--reconcile-stale",
            action="store_true",
            help="Mark in_progress jobs as failed before processing (fail-closed v1).",
        )

    def handle(self, *args, **options) -> None:
        if options["reconcile_stale"]:
            count = reconcile_stale_in_progress_jobs()
            if count:
                self.stdout.write(f"Marked {count} stale in_progress batch(es) failed.")
        processed = process_pending_batch_jobs(max_jobs=int(options["max_jobs"]))
        self.stdout.write(f"Processed {processed} batch job(s).")
