"""Tests for OpenAI Files + Batches services (MLX-132)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from orchestrator.batch_file_store import batch_files_dir
from orchestrator.batch_services import (
    create_batch_file,
    create_batch_job,
    delete_batch_file,
    file_to_openai_dict,
    get_batch_file,
    process_pending_batch_jobs,
)
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH
from orchestrator.models import GatewayBatchJob


@pytest.fixture
def batch_files_tmp(tmp_path, settings, monkeypatch):
    monkeypatch.setenv("NADIR_BATCH_FILES_DIR", str(tmp_path / "batch_files"))
    batch_files_dir().mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.mark.django_db
def test_create_and_delete_batch_file(batch_files_tmp) -> None:
    record = create_batch_file(
        purpose="batch",
        filename="input.jsonl",
        data=b'{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"x","messages":[]}}\n',
    )
    from orchestrator.batch_services import BatchServiceError

    assert not isinstance(record, BatchServiceError)
    assert record.file_id.startswith("file-")
    payload = file_to_openai_dict(record)
    assert payload["object"] == "file"
    assert payload["purpose"] == "batch"

    fetched = get_batch_file(record.file_id)
    assert fetched is not None
    assert delete_batch_file(record.file_id) is None
    assert get_batch_file(record.file_id) is None


@pytest.mark.django_db
def test_rejects_non_batch_purpose(batch_files_tmp) -> None:
    result = create_batch_file(purpose="fine-tune", filename="x", data=b"data")
    assert result.status_code == 400


@pytest.mark.django_db
def test_process_batch_job_calls_gateway(batch_files_tmp) -> None:
    line = {
        "custom_id": "req-1",
        "method": "POST",
        "url": CHAT_COMPLETIONS_PATH,
        "body": {"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
    }
    record = create_batch_file(
        purpose="batch",
        filename="in.jsonl",
        data=(json.dumps(line) + "\n").encode("utf-8"),
    )
    job = create_batch_job(
        input_file_id=record.file_id,
        endpoint=CHAT_COMPLETIONS_PATH,
    )
    assert isinstance(job, GatewayBatchJob)

    mock_response = type(
        "R",
        (),
        {"status_code": 200, "json": lambda self: {"id": "chatcmpl-1", "choices": []}},
    )()

    with patch("orchestrator.batch_services.httpx.post", return_value=mock_response):
        processed = process_pending_batch_jobs(max_jobs=1)

    assert processed == 1
    job.refresh_from_db()
    assert job.status == GatewayBatchJob.STATUS_COMPLETED
    assert job.output_file is not None
    assert job.request_counts.get("completed") == 1
