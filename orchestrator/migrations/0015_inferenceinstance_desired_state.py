from django.db import migrations, models

_LOOPBACK_HOST = "127.0.0.1"
_LEGACY_STOP_FLAG = "manual_stop_in_progress"


def _seed_desired_state_and_clean_config(apps, schema_editor) -> None:
    """Seed intent from current status, drop stuck stop flags, rebind legacy 0.0.0.0 hosts."""
    instance_model = apps.get_model("orchestrator", "InferenceInstance")
    for instance in instance_model.objects.all().iterator():
        instance.desired_state = (
            "running" if instance.status in ("RUNNING", "LOADING") else "stopped"
        )
        config = dict(instance.server_config or {})
        ops = dict(config.get("ops") or {})
        ops.pop(_LEGACY_STOP_FLAG, None)
        config["ops"] = ops
        # mlx-lm / mlx-vlm have no auth: legacy rows bound to every interface.
        if config.get("host") == "0.0.0.0":
            config["host"] = _LOOPBACK_HOST
        instance.server_config = config
        instance.save(update_fields=["desired_state", "server_config"])


class Migration(migrations.Migration):

    dependencies = [
        ("orchestrator", "0014_gateway_request_metric"),
    ]

    operations = [
        migrations.AddField(
            model_name="inferenceinstance",
            name="desired_state",
            field=models.CharField(
                choices=[("running", "Running"), ("stopped", "Stopped")],
                default="running",
                help_text="Operator/system intent; watchdog and reconcile never override 'stopped'.",
                max_length=10,
            ),
        ),
        migrations.AddField(
            model_name="inferenceinstance",
            name="stopping_since",
            field=models.DateTimeField(
                blank=True,
                help_text="Set while a stop is in progress; considered abandoned after a timeout.",
                null=True,
            ),
        ),
        migrations.RunPython(
            _seed_desired_state_and_clean_config,
            migrations.RunPython.noop,
        ),
    ]
