from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("orchestrator", "0011_inferenceinstance_health_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="inferenceinstance",
            name="last_used_at",
            field=models.DateTimeField(
                blank=True,
                help_text="Last gateway-proxied request time (UTC); used for idle offload.",
                null=True,
            ),
        ),
    ]
