from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("orchestrator", "0010_alter_error_message_no_null"),
    ]

    operations = [
        migrations.AddField(
            model_name="inferenceinstance",
            name="health_status",
            field=models.CharField(
                blank=True,
                choices=[
                    ("HEALTHY", "Healthy"),
                    ("DEGRADED", "Degraded"),
                    ("DOWN", "Down"),
                    ("UNKNOWN", "Unknown"),
                ],
                default="UNKNOWN",
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name="inferenceinstance",
            name="last_health_check_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
