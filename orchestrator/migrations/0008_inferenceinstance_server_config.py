from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("orchestrator", "0007_alter_inferenceinstance_launch_mode"),
    ]

    operations = [
        migrations.AddField(
            model_name="inferenceinstance",
            name="server_config",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
