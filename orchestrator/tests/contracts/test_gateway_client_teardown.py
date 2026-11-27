"""The contract gateway client drains telemetry writes before teardown (MLX-137)."""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest
from django.db import transaction

from orchestrator import gateway_telemetry_services as telemetry
from orchestrator.models import InferenceInstance


@pytest.mark.contract
@pytest.mark.ticket("MLX-137")
@pytest.mark.ac("MLX-137:AC1")
@pytest.mark.django_db(transaction=True)
def test_gateway_client_drains_telemetry_writer_before_flush_ac1(gateway_client: Any) -> None:
    """AC1: a telemetry write still running when the test ends does not lock the flush.

    The writer thread holds a write transaction past the end of the test body.
    Without the drain in ``gateway_client`` the teardown flush of the shared
    in-memory SQLite DB fails with ``database table is locked`` (CI run
    44479483504).
    """
    writing = threading.Event()

    def slow_write(_snapshot: Any) -> None:
        with transaction.atomic():
            InferenceInstance.objects.create(
                model_name="teardown-probe",
                port=11470,
                launch_mode="TEXT",
                server_config={},
                status="RUNNING",
            )
            writing.set()
            time.sleep(1)

    with patch.object(telemetry, "record_gateway_request_timing", slow_write):
        telemetry.enqueue_gateway_request_timing(object())
        assert writing.wait(timeout=5), "the telemetry writer did not start the write"
