"""Use empty string instead of NULL for error_message fields."""

from django.db import migrations, models


def clear_null_error_messages(apps, schema_editor) -> None:
    model_download = apps.get_model("orchestrator", "ModelDownload")
    benchmark_run = apps.get_model("orchestrator", "BenchmarkRun")
    model_download.objects.filter(error_message__isnull=True).update(error_message="")
    benchmark_run.objects.filter(error_message__isnull=True).update(error_message="")


class Migration(migrations.Migration):

    dependencies = [
        ("orchestrator", "0009_alter_inferenceinstance_launch_mode"),
    ]

    operations = [
        migrations.RunPython(clear_null_error_messages, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="modeldownload",
            name="error_message",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AlterField(
            model_name="benchmarkrun",
            name="error_message",
            field=models.TextField(blank=True, default=""),
        ),
    ]
