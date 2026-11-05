"""Tests for OpenAI Files + Batches services (MLX-132)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from django.conf import settings

from orchestrator.batch_file_store import batch_files_dir
from io import BytesIO

from orchestrator.batch_services import (
    BatchServiceError,
    _batch_errors_payload,
    _claim_batch_job,
    _execute_batch_lines,
    _normalize_batch_errors,
    _post_chat_completion,
    _run_claimed_batch_job,
    batch_to_openai_dict,
    cancel_batch_job,
    create_batch_file,
    create_batch_file_from_stream,
    create_batch_job,
    delete_batch_file,
    file_to_openai_dict,
    get_batch_file,
    process_pending_batch_jobs,
    purge_expired_batch_jobs,
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

    with patch("orchestrator.batch_services.httpx.post", return_value=mock_response) as post_mock:
        processed = process_pending_batch_jobs(max_jobs=1)

    assert processed == 1
    job.refresh_from_db()
    assert job.status == GatewayBatchJob.STATUS_COMPLETED
    assert job.output_file is not None
    assert job.request_counts.get("completed") == 1
    called_url = post_mock.call_args[0][0]
    assert called_url == (
        f"http://127.0.0.1:{settings.NADIR_GATEWAY_PORT}{CHAT_COMPLETIONS_PATH}"
    )


@pytest.mark.django_db
def test_cancel_validating_batch_is_immediately_cancelled(batch_files_tmp) -> None:
    record = create_batch_file(
        purpose="batch",
        filename="in.jsonl",
        data=b'{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"x","messages":[]}}\n',
    )
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)

    result = cancel_batch_job(job.batch_id)
    assert not isinstance(result, BatchServiceError)
    job.refresh_from_db()
    assert job.status == GatewayBatchJob.STATUS_CANCELLED
    assert job.cancelled_at is not None

    assert process_pending_batch_jobs(max_jobs=1) == 0


@pytest.mark.django_db
def test_claim_batch_job_allows_only_one_worker(batch_files_tmp) -> None:
    record = create_batch_file(
        purpose="batch",
        filename="in.jsonl",
        data=b'{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"x","messages":[]}}\n',
    )
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)

    assert _claim_batch_job(job.pk) is True
    job.refresh_from_db()
    assert job.status == GatewayBatchJob.STATUS_IN_PROGRESS
    assert _claim_batch_job(job.pk) is False


@pytest.mark.django_db
def test_line_exception_completes_batch_with_failed_line(batch_files_tmp) -> None:
    line = {
        "custom_id": "req-1",
        "url": CHAT_COMPLETIONS_PATH,
        "body": {"model": "demo", "messages": []},
    }
    record = create_batch_file(
        purpose="batch",
        filename="in.jsonl",
        data=(json.dumps(line) + "\n").encode("utf-8"),
    )
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)

    with patch(
        "orchestrator.batch_services._post_chat_completion",
        side_effect=TypeError("probe"),
    ):
        process_pending_batch_jobs(max_jobs=1)

    job.refresh_from_db()
    assert job.status == GatewayBatchJob.STATUS_COMPLETED
    assert job.request_counts.get("failed") == 1


@pytest.mark.django_db
def test_unexpected_worker_failure_marks_batch_failed(batch_files_tmp) -> None:
    record = create_batch_file(
        purpose="batch",
        filename="in.jsonl",
        data=b'{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"x","messages":[]}}\n',
    )
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)
    assert _claim_batch_job(job.pk)

    with patch(
        "orchestrator.batch_services._execute_batch_lines",
        side_effect=RuntimeError("worker boom"),
    ):
        _run_claimed_batch_job(job.pk)

    job.refresh_from_db()
    assert job.status == GatewayBatchJob.STATUS_FAILED


@pytest.mark.django_db
def test_cancel_while_in_progress_uses_cancelling(batch_files_tmp) -> None:
    record = create_batch_file(
        purpose="batch",
        filename="in.jsonl",
        data=b'{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"x","messages":[]}}\n',
    )
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)
    GatewayBatchJob.objects.filter(pk=job.pk).update(
        status=GatewayBatchJob.STATUS_IN_PROGRESS,
    )

    result = cancel_batch_job(job.batch_id)
    assert not isinstance(result, BatchServiceError)
    job.refresh_from_db()
    assert job.status == GatewayBatchJob.STATUS_CANCELLING


@pytest.mark.django_db
def test_cancel_after_claim_uses_cancelling_not_cancelled(batch_files_tmp) -> None:
    record = create_batch_file(
        purpose="batch",
        filename="in.jsonl",
        data=b'{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"x","messages":[]}}\n',
    )
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)
    assert _claim_batch_job(job.pk)

    result = cancel_batch_job(job.batch_id)
    assert not isinstance(result, BatchServiceError)
    job.refresh_from_db()
    assert job.status == GatewayBatchJob.STATUS_CANCELLING


@pytest.mark.django_db
def test_cancel_cancelling_batch_is_idempotent(batch_files_tmp) -> None:
    record = create_batch_file(
        purpose="batch",
        filename="in.jsonl",
        data=b'{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"x","messages":[]}}\n',
    )
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)
    GatewayBatchJob.objects.filter(pk=job.pk).update(status=GatewayBatchJob.STATUS_CANCELLING)

    result = cancel_batch_job(job.batch_id)
    assert not isinstance(result, BatchServiceError)
    job.refresh_from_db()
    assert job.status == GatewayBatchJob.STATUS_CANCELLING


@pytest.mark.django_db
def test_cancel_finished_batch_returns_error(batch_files_tmp) -> None:
    record = create_batch_file(
        purpose="batch",
        filename="in.jsonl",
        data=b'{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"x","messages":[]}}\n',
    )
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)
    GatewayBatchJob.objects.filter(pk=job.pk).update(status=GatewayBatchJob.STATUS_COMPLETED)

    result = cancel_batch_job(job.batch_id)
    assert isinstance(result, BatchServiceError)
    assert result.status_code == 400


@pytest.mark.django_db
def test_invalid_jsonl_marks_batch_failed(batch_files_tmp) -> None:
    record = create_batch_file(purpose="batch", filename="bad.jsonl", data=b"not-json\n")
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)

    process_pending_batch_jobs(max_jobs=1)
    job.refresh_from_db()
    assert job.status == GatewayBatchJob.STATUS_FAILED


@pytest.mark.django_db
def test_missing_input_blob_marks_batch_failed(batch_files_tmp) -> None:
    record = create_batch_file(
        purpose="batch",
        filename="in.jsonl",
        data=b'{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"x","messages":[]}}\n',
    )
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)

    with patch("orchestrator.batch_services.read_batch_file_content", return_value=None):
        process_pending_batch_jobs(max_jobs=1)

    job.refresh_from_db()
    assert job.status == GatewayBatchJob.STATUS_FAILED


@pytest.mark.django_db
def test_post_chat_completion_maps_http_error(batch_files_tmp) -> None:
    mock_response = type(
        "R",
        (),
        {
            "status_code": 503,
            "text": "busy",
            "json": lambda self: (_ for _ in ()).throw(ValueError("no json")),
        },
    )()
    with patch("orchestrator.batch_services.httpx.post", return_value=mock_response):
        result = _post_chat_completion("cid-1", {"model": "m", "messages": []})
    assert result["custom_id"] == "cid-1"
    assert "error" in result


@pytest.mark.django_db
def test_process_line_missing_body_and_bad_url(batch_files_tmp) -> None:
    payload = (
        json.dumps({"custom_id": "a", "url": "/v1/other", "body": {}}) + "\n"
        + json.dumps({"custom_id": "b", "url": CHAT_COMPLETIONS_PATH}) + "\n"
    )
    record = create_batch_file(purpose="batch", filename="in.jsonl", data=payload.encode())
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)

    mock_response = type(
        "R",
        (),
        {"status_code": 200, "json": lambda self: {"choices": []}},
    )()
    with patch("orchestrator.batch_services.httpx.post", return_value=mock_response):
        process_pending_batch_jobs(max_jobs=1)

    job.refresh_from_db()
    assert job.status == GatewayBatchJob.STATUS_COMPLETED
    assert job.request_counts.get("failed") == 2


@pytest.mark.django_db
def test_multi_line_output_preserves_order(batch_files_tmp) -> None:
    lines = [
        {"custom_id": "first", "url": CHAT_COMPLETIONS_PATH, "body": {"model": "m", "messages": []}},
        {"custom_id": "second", "url": CHAT_COMPLETIONS_PATH, "body": {"model": "m", "messages": []}},
    ]
    payload = "\n".join(json.dumps(line) for line in lines) + "\n"
    record = create_batch_file(purpose="batch", filename="in.jsonl", data=payload.encode())
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)

    mock_response = type(
        "R",
        (),
        {"status_code": 200, "json": lambda self: {"choices": []}},
    )()
    with patch("orchestrator.batch_services.httpx.post", return_value=mock_response):
        process_pending_batch_jobs(max_jobs=1)

    job.refresh_from_db()
    from orchestrator.batch_services import read_batch_file_content

    assert job.output_file is not None
    raw = read_batch_file_content(job.output_file)
    assert raw is not None
    out_lines = raw.decode().strip().split("\n")
    assert json.loads(out_lines[0])["custom_id"] == "first"
    assert json.loads(out_lines[1])["custom_id"] == "second"


@pytest.mark.django_db
def test_cancel_during_execution_keeps_first_line(batch_files_tmp) -> None:
    lines = [
        {"custom_id": "keep", "url": CHAT_COMPLETIONS_PATH, "body": {"model": "m", "messages": []}},
        {"custom_id": "skip", "url": CHAT_COMPLETIONS_PATH, "body": {"model": "m", "messages": []}},
    ]
    payload = "\n".join(json.dumps(line) for line in lines) + "\n"
    record = create_batch_file(purpose="batch", filename="in.jsonl", data=payload.encode())
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)
    assert _claim_batch_job(job.pk)

    mock_response = type(
        "R",
        (),
        {"status_code": 200, "json": lambda self: {"choices": []}},
    )()
    cancel_checks = {"n": 0}

    def cancel_after_first_record(_job_pk: int) -> bool:
        cancel_checks["n"] += 1
        return cancel_checks["n"] == 1

    with patch("orchestrator.batch_services.batch_line_parallelism", return_value=1):
        with patch(
            "orchestrator.batch_services._batch_cancel_requested",
            side_effect=cancel_after_first_record,
        ):
            with patch("orchestrator.batch_services.httpx.post", return_value=mock_response):
                _run_claimed_batch_job(job.pk)

    job.refresh_from_db()
    assert job.status == GatewayBatchJob.STATUS_CANCELLED
    assert job.request_counts.get("completed", 0) >= 1
    from orchestrator.batch_services import read_batch_file_content

    assert job.output_file is not None
    raw = read_batch_file_content(job.output_file)
    assert raw is not None
    assert "keep" in raw.decode()


@pytest.mark.django_db
def test_execute_batch_lines_empty_returns_empty(batch_files_tmp) -> None:
    record = create_batch_file(
        purpose="batch",
        filename="in.jsonl",
        data=b'{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"x","messages":[]}}\n',
    )
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)

    output, counts, cancelled = _execute_batch_lines(
        job.pk,
        [],
        {"total": 0, "completed": 0, "failed": 0},
    )
    assert output == []
    assert counts == {"total": 0, "completed": 0, "failed": 0}
    assert cancelled is False


@pytest.mark.django_db
def test_create_batch_file_from_stream(batch_files_tmp) -> None:
    record = create_batch_file_from_stream(
        purpose="batch",
        filename="stream.jsonl",
        readable=BytesIO(b'{"custom_id":"s"}\n'),
    )
    assert not isinstance(record, BatchServiceError)
    assert record.bytes == len(b'{"custom_id":"s"}\n')


@pytest.mark.django_db
def test_output_lines_include_openai_batch_req_id(batch_files_tmp) -> None:
    line = {
        "custom_id": "req-1",
        "url": CHAT_COMPLETIONS_PATH,
        "body": {"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
    }
    record = create_batch_file(
        purpose="batch",
        filename="in.jsonl",
        data=(json.dumps(line) + "\n").encode("utf-8"),
    )
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)
    mock_response = type(
        "R",
        (),
        {"status_code": 200, "json": lambda self: {"id": "chatcmpl-1", "choices": []}},
    )()
    with patch("orchestrator.batch_services.httpx.post", return_value=mock_response):
        process_pending_batch_jobs(max_jobs=1)
    job.refresh_from_db()
    from orchestrator.batch_services import read_batch_file_content

    raw = read_batch_file_content(job.output_file)
    assert raw is not None
    parsed = json.loads(raw.decode().strip().split("\n")[0])
    assert parsed["id"].startswith("batch_req_")
    assert parsed["custom_id"] == "req-1"


@pytest.mark.django_db
def test_failed_job_uses_openai_errors_list(batch_files_tmp) -> None:
    record = create_batch_file(purpose="batch", filename="bad.jsonl", data=b"not-json\n")
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)
    process_pending_batch_jobs(max_jobs=1)
    job.refresh_from_db()
    assert job.status == GatewayBatchJob.STATUS_FAILED
    payload = batch_to_openai_dict(job)
    assert payload["errors"]["object"] == "list"
    assert payload["errors"]["data"][0]["message"]


@pytest.mark.django_db
def test_purge_expired_batch_jobs(batch_files_tmp) -> None:
    record = create_batch_file(
        purpose="batch",
        filename="in.jsonl",
        data=b'{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"x","messages":[]}}\n',
    )
    job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(job, GatewayBatchJob)
    GatewayBatchJob.objects.filter(pk=job.pk).update(
        status=GatewayBatchJob.STATUS_COMPLETED,
        completed_at=datetime.now(timezone.utc) - timedelta(days=60),
    )
    with patch.dict("os.environ", {"NADIR_BATCH_RETENTION_DAYS": "30"}):
        removed = purge_expired_batch_jobs()
    assert removed == 1
    assert GatewayBatchJob.objects.filter(pk=job.pk).exists() is False


@pytest.mark.django_db
def test_purge_keeps_shared_input_for_newer_batch(batch_files_tmp) -> None:
    from orchestrator.batch_services import read_batch_file_content

    payload = (
        b'{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"x","messages":[]}}\n'
    )
    record = create_batch_file(purpose="batch", filename="shared.jsonl", data=payload)
    assert not isinstance(record, BatchServiceError)
    old_job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(old_job, GatewayBatchJob)
    GatewayBatchJob.objects.filter(pk=old_job.pk).update(
        status=GatewayBatchJob.STATUS_COMPLETED,
        completed_at=datetime.now(timezone.utc) - timedelta(days=60),
    )
    new_job = create_batch_job(input_file_id=record.file_id, endpoint=CHAT_COMPLETIONS_PATH)
    assert isinstance(new_job, GatewayBatchJob)

    with patch.dict("os.environ", {"NADIR_BATCH_RETENTION_DAYS": "30"}):
        removed = purge_expired_batch_jobs()
    assert removed == 1
    assert get_batch_file(record.file_id) is not None
    assert read_batch_file_content(record) is not None

    mock_response = type(
        "R",
        (),
        {"status_code": 200, "json": lambda self: {"id": "chatcmpl-1", "choices": []}},
    )()
    with patch("orchestrator.batch_services.httpx.post", return_value=mock_response):
        process_pending_batch_jobs(max_jobs=1)

    new_job.refresh_from_db()
    assert new_job.status == GatewayBatchJob.STATUS_COMPLETED


def test_normalize_batch_errors_legacy_message() -> None:
    normalized = _normalize_batch_errors({"message": "legacy"})
    assert normalized == _batch_errors_payload("legacy")
