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
    error_message = models.TextField(null=True, blank=True)
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
    created_at = models.DateTimeField(auto_now_add=True)
    stopped_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.model_name} on port {self.port} ({self.launch_mode}, PID: {self.pid})"


class BenchmarkRun(models.Model):
    TARGET_CHOICES = [
        ("INSTANCE", "MLX instance"),
        ("ENDPOINT", "Custom endpoint"),
    ]
    STATUS_CHOICES = [
        ("PENDING", "Pending"),
        ("RUNNING", "Running"),
        ("COMPLETED", "Completed"),
        ("FAILED", "Failed"),
    ]

    target_type = models.CharField(max_length=20, choices=TARGET_CHOICES)
    instance = models.ForeignKey(
        InferenceInstance,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="benchmark_runs",
    )
    endpoint_url = models.CharField(max_length=512)
    model_id = models.CharField(max_length=255, blank=True)
    params = models.JSONField(default=dict)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="PENDING")
    results = models.JSONField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
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
        summaries = []
        for entry in self.results.get("results", []):
            summary = entry.get("summary")
            if summary:
                summaries.append(summary)
        return summaries
