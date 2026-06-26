# Generated manually for MLX-49

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("orchestrator", "0012_inferenceinstance_last_used_at"),
    ]

    operations = [
        migrations.AddField(
            model_name="benchmarkrun",
            name="benchmark_kind",
            field=models.CharField(
                choices=[
                    ("PERF", "Performance"),
                    ("QUALITY", "Quality"),
                    ("COMPLETE", "Complete"),
                ],
                default="PERF",
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name="benchmarkrun",
            name="parent_run",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="child_runs",
                to="orchestrator.benchmarkrun",
            ),
        ),
    ]
