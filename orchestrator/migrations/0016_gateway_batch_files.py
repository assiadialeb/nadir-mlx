# Generated manually for MLX-132

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("orchestrator", "0015_inferenceinstance_desired_state"),
    ]

    operations = [
        migrations.CreateModel(
            name="GatewayBatchFile",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("file_id", models.CharField(db_index=True, max_length=128, unique=True)),
                ("purpose", models.CharField(max_length=32)),
                ("filename", models.CharField(max_length=512)),
                ("bytes", models.BigIntegerField()),
                ("storage_name", models.CharField(max_length=64)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["-created_at", "-id"],
            },
        ),
        migrations.CreateModel(
            name="GatewayBatchJob",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("batch_id", models.CharField(db_index=True, max_length=128, unique=True)),
                ("endpoint", models.CharField(max_length=128)),
                ("completion_window", models.CharField(default="24h", max_length=16)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("validating", "Validating"),
                            ("in_progress", "In progress"),
                            ("completed", "Completed"),
                            ("failed", "Failed"),
                            ("cancelling", "Cancelling"),
                            ("cancelled", "Cancelled"),
                        ],
                        default="validating",
                        max_length=32,
                    ),
                ),
                ("errors", models.JSONField(blank=True, null=True)),
                ("request_counts", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("in_progress_at", models.DateTimeField(blank=True, null=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                ("cancelled_at", models.DateTimeField(blank=True, null=True)),
                (
                    "error_file",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="batch_jobs_as_errors",
                        to="orchestrator.gatewaybatchfile",
                    ),
                ),
                (
                    "input_file",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="batch_jobs_as_input",
                        to="orchestrator.gatewaybatchfile",
                    ),
                ),
                (
                    "output_file",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="batch_jobs_as_output",
                        to="orchestrator.gatewaybatchfile",
                    ),
                ),
            ],
            options={
                "ordering": ["-created_at", "-id"],
            },
        ),
    ]
