from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("orchestrator", "0008_inferenceinstance_server_config"),
    ]

    operations = [
        migrations.AlterField(
            model_name="inferenceinstance",
            name="launch_mode",
            field=models.CharField(
                choices=[
                    ("TEXT", "Text"),
                    ("MULTIMODAL", "Multimodal"),
                    ("EMBEDDING", "Embedding"),
                    ("RERANKER", "Reranker"),
                    ("IMAGE", "Image"),
                    ("TTS", "TTS"),
                    ("STT", "STT"),
                ],
                default="TEXT",
                max_length=20,
            ),
        ),
    ]
