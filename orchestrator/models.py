from django.db import models

class ModelDownload(models.Model):
    STATUS_CHOICES = [
        ('DOWNLOADING', 'Downloading'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]

    repo_id = models.CharField(max_length=255, unique=True)
    local_path = models.CharField(max_length=512)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='DOWNLOADING')
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.repo_id} ({self.status})"


class InferenceInstance(models.Model):
    STATUS_CHOICES = [
        ('LOADING', 'Loading'),
        ('RUNNING', 'Running'),
        ('STOPPED', 'Stopped'),
        ('FAILED', 'Failed'),
    ]
    LAUNCH_MODE_CHOICES = [
        ('TEXT', 'Text'),
        ('MULTIMODAL', 'Multimodal'),
        ('EMBEDDING', 'Embedding'),
        ('RERANKER', 'Reranker'),
        ('IMAGE', 'Image'),
        ('TTS', 'TTS'),
        ('STT', 'STT'),
    ]

    model_name = models.CharField(max_length=255)
    port = models.IntegerField()
    launch_mode = models.CharField(
        max_length=20,
        choices=LAUNCH_MODE_CHOICES,
        default='TEXT',
    )
    server_config = models.JSONField(default=dict, blank=True)
    pid = models.IntegerField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='LOADING')
    health_status = models.CharField(
        max_length=20,
        choices=[
            ('HEALTHY', 'Healthy'),
            ('DEGRADED', 'Degraded'),
            ('DOWN', 'Down'),
            ('UNKNOWN', 'Unknown'),
        ],
        default='UNKNOWN',
        blank=True,
    )
    last_health_check_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    stopped_at = models.DateTimeField(null=True, blank=True)
    last_used_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Last gateway-proxied request time (UTC); used for idle offload.",
    )
    DESIRED_STATE_RUNNING = 'running'
    DESIRED_STATE_STOPPED = 'stopped'
    desired_state = models.CharField(
        max_length=10,
        choices=[
            (DESIRED_STATE_RUNNING, 'Running'),
            (DESIRED_STATE_STOPPED, 'Stopped'),
        ],
        default=DESIRED_STATE_RUNNING,
        help_text="Operator/system intent; watchdog and reconcile never override 'stopped'.",
    )
    stopping_since = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Set while a stop is in progress; considered abandoned after a timeout.",
    )

    def __str__(self):
        return f"{self.model_name} on port {self.port} ({self.launch_mode}, PID: {self.pid})"


class GatewayRequestMetric(models.Model):
    """Per-request gateway timing (TTFT, prefill, cache) for local diagnostics."""

    request_id = models.CharField(max_length=64, db_index=True)
    endpoint = models.CharField(max_length=128)
    alias = models.CharField(max_length=255, blank=True, default="")
    instance = models.ForeignKey(
        InferenceInstance,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="gateway_request_metrics",
    )
    stream = models.BooleanField(default=False)
    started_at = models.DateTimeField(db_index=True)
    queue_wait_ms = models.FloatField(null=True, blank=True)
    wake_ms = models.FloatField(null=True, blank=True)
    cold_start = models.BooleanField(default=False)
    prefill_ms = models.FloatField(null=True, blank=True)
    ttft_ms = models.FloatField(null=True, blank=True)
    visible_ttft_ms = models.FloatField(null=True, blank=True)
    generation_ms = models.FloatField(null=True, blank=True)
    total_ms = models.FloatField(null=True, blank=True)
    prompt_tokens = models.IntegerField(null=True, blank=True)
    completion_tokens = models.IntegerField(null=True, blank=True)
    cached_prompt_tokens = models.IntegerField(null=True, blank=True)
    prefill_tokens_per_s = models.FloatField(null=True, blank=True)
    generation_tokens_per_s = models.FloatField(null=True, blank=True)
    error_code = models.CharField(max_length=64, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-started_at", "-id"]

    def __str__(self) -> str:
        return f"GatewayTiming {self.request_id} alias={self.alias}"


class BenchmarkRun(models.Model):
    TARGET_CHOICES = [
        ("INSTANCE", "MLX instance"),
        ("ENDPOINT", "Custom endpoint"),
    ]
    BENCHMARK_KIND_CHOICES = [
        ("PERF", "Performance"),
        ("QUALITY", "Quality"),
        ("COMPLETE", "Complete"),
    ]
    STATUS_CHOICES = [
        ("PENDING", "Pending"),
        ("RUNNING", "Running"),
        ("COMPLETED", "Completed"),
        ("FAILED", "Failed"),
    ]

    benchmark_kind = models.CharField(
        max_length=20,
        choices=BENCHMARK_KIND_CHOICES,
        default="PERF",
    )
    target_type = models.CharField(max_length=20, choices=TARGET_CHOICES)
    instance = models.ForeignKey(
        InferenceInstance,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="benchmark_runs",
    )
    parent_run = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="child_runs",
    )
    endpoint_url = models.CharField(max_length=512)
    model_id = models.CharField(max_length=255, blank=True)
    params = models.JSONField(default=dict)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="PENDING")
    results = models.JSONField(null=True, blank=True)
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Benchmark #{self.id} ({self.status}) — {self.endpoint_url}"

    @property
    def summaries(self) -> list[dict]:
        if not self.results:
            return []
        if self.benchmark_kind in ("QUALITY", "COMPLETE"):
            return []
        summaries = []
        for entry in self.results.get("results", []):
            summary = entry.get("summary")
            if summary:
                summaries.append(summary)
        return summaries

    @property
    def quality_warnings(self) -> list[str]:
        """Non-fatal phase warnings stored on completed quality runs."""
        if not self.results or self.status != "COMPLETED":
            return []
        warnings = self.results.get("warnings")
        if not isinstance(warnings, list):
            return []
        return [str(item) for item in warnings if item]

    @property
    def quality_metrics(self) -> dict:
        """Normalized industry + platform metrics for quality or complete runs."""
        if not self.results:
            return {}
        if self.benchmark_kind == "QUALITY":
            return self.results.get("metrics", {})
        if self.benchmark_kind == "COMPLETE":
            return self.results.get("quality_summary", {})
        return {}
