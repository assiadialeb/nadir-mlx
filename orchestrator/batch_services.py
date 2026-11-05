"""OpenAI Files + Batches business logic (MLX-132)."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from orchestrator.batch_file_store import (
    batch_file_max_bytes,
    delete_storage_blob,
    is_valid_batch_file_id,
    new_file_id,
    new_storage_name,
    read_file_bytes,
    write_upload_bytes,
)
from orchestrator.env_utils import env_float
from orchestrator.gateway.gpu_scheduler import PRIORITY_HEADER, PRIORITY_BATCH
from orchestrator.gateway.router import CHAT_COMPLETIONS_PATH
from orchestrator.gateway_client_auth import gateway_api_key
from orchestrator.image_assets import gateway_public_base_url
from orchestrator.models import GatewayBatchFile, GatewayBatchJob

logger = logging.getLogger(__name__)

ALLOWED_FILE_PURPOSES = frozenset({"batch"})
ALLOWED_BATCH_ENDPOINTS = frozenset({CHAT_COMPLETIONS_PATH})


@dataclass(frozen=True)
class BatchServiceError:
    message: str
    code: str = "invalid_request"
    status_code: int = 400


def create_batch_file(*, purpose: str, filename: str, data: bytes) -> GatewayBatchFile | BatchServiceError:
    if purpose not in ALLOWED_FILE_PURPOSES:
        return BatchServiceError(
            f"Unsupported purpose '{purpose}'. Only 'batch' is supported.",
            status_code=400,
        )
    if not data:
        return BatchServiceError("Empty file.", status_code=400)
    if len(data) > batch_file_max_bytes():
        return BatchServiceError("File too large.", code="payload_too_large", status_code=413)

    storage_name = new_storage_name()
    write_upload_bytes(storage_name, data)
    record = GatewayBatchFile.objects.create(
        file_id=new_file_id(),
        purpose=purpose,
        filename=filename or "upload.jsonl",
        bytes=len(data),
        storage_name=storage_name,
    )
    return record


def get_batch_file(file_id: str) -> GatewayBatchFile | None:
    if not is_valid_batch_file_id(file_id):
        return None
    return GatewayBatchFile.objects.filter(file_id=file_id).first()


def list_batch_files(*, limit: int = 100) -> list[GatewayBatchFile]:
    cap = max(1, min(limit, 500))
    return list(GatewayBatchFile.objects.all()[:cap])


def delete_batch_file(file_id: str) -> BatchServiceError | None:
    record = get_batch_file(file_id)
    if record is None:
        return BatchServiceError("No such file object.", status_code=404)
    if GatewayBatchJob.objects.filter(input_file=record).exists():
        return BatchServiceError("File is attached to a batch job.", status_code=409)
    delete_storage_blob(record.storage_name)
    record.delete()
    return None


def read_batch_file_content(record: GatewayBatchFile) -> bytes | None:
    return read_file_bytes(record.storage_name)


def create_batch_job(
    *,
    input_file_id: str,
    endpoint: str,
    completion_window: str = "24h",
) -> GatewayBatchJob | BatchServiceError:
    if endpoint not in ALLOWED_BATCH_ENDPOINTS:
        return BatchServiceError(
            f"Unsupported batch endpoint '{endpoint}'.",
            status_code=400,
        )
    input_file = get_batch_file(input_file_id)
    if input_file is None:
        return BatchServiceError("Invalid input_file_id.", status_code=400)
    if input_file.purpose != "batch":
        return BatchServiceError("Input file purpose must be 'batch'.", status_code=400)

    return GatewayBatchJob.objects.create(
        batch_id=_new_batch_id(),
        input_file=input_file,
        endpoint=endpoint,
        completion_window=completion_window or "24h",
        status=GatewayBatchJob.STATUS_VALIDATING,
    )


def get_batch_job(batch_id: str) -> GatewayBatchJob | None:
    return GatewayBatchJob.objects.filter(batch_id=batch_id).select_related(
        "input_file",
        "output_file",
        "error_file",
    ).first()


def list_batch_jobs(*, limit: int = 100) -> list[GatewayBatchJob]:
    cap = max(1, min(limit, 500))
    return list(
        GatewayBatchJob.objects.select_related("input_file", "output_file", "error_file").all()[:cap]
    )


def cancel_batch_job(batch_id: str) -> GatewayBatchJob | BatchServiceError:
    job = get_batch_job(batch_id)
    if job is None:
        return BatchServiceError("No such batch.", status_code=404)
    if job.status in {
        GatewayBatchJob.STATUS_COMPLETED,
        GatewayBatchJob.STATUS_FAILED,
        GatewayBatchJob.STATUS_CANCELLED,
    }:
        return BatchServiceError("Batch already finished.", status_code=400)
    job.status = GatewayBatchJob.STATUS_CANCELLING
    job.save(update_fields=["status"])
    return job


def file_to_openai_dict(record: GatewayBatchFile) -> dict[str, Any]:
    created = record.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return {
        "id": record.file_id,
        "object": "file",
        "bytes": record.bytes,
        "created_at": int(created.timestamp()),
        "filename": record.filename,
        "purpose": record.purpose,
    }


def batch_to_openai_dict(job: GatewayBatchJob) -> dict[str, Any]:
    created = job.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    payload: dict[str, Any] = {
        "id": job.batch_id,
        "object": "batch",
        "endpoint": job.endpoint,
        "errors": job.errors,
        "input_file_id": job.input_file.file_id,
        "completion_window": job.completion_window,
        "status": job.status,
        "output_file_id": job.output_file.file_id if job.output_file else None,
        "error_file_id": job.error_file.file_id if job.error_file else None,
        "created_at": int(created.timestamp()),
        "request_counts": job.request_counts or {},
    }
    if job.completed_at:
        payload["completed_at"] = int(job.completed_at.timestamp())
    if job.in_progress_at:
        payload["in_progress_at"] = int(job.in_progress_at.timestamp())
    return payload


def process_pending_batch_jobs(*, max_jobs: int = 1) -> int:
    """Run up to ``max_jobs`` validating jobs (management command / cron)."""
    processed = 0
    for _ in range(max(1, max_jobs)):
        job = (
            GatewayBatchJob.objects.filter(status=GatewayBatchJob.STATUS_VALIDATING)
            .select_related("input_file")
            .order_by("created_at")
            .first()
        )
        if job is None:
            break
        _execute_batch_job(job.pk)
        processed += 1
    return processed


def _execute_batch_job(job_pk: int) -> None:
    job = GatewayBatchJob.objects.select_related("input_file").filter(pk=job_pk).first()
    if job is None or job.status != GatewayBatchJob.STATUS_VALIDATING:
        return

    now = datetime.now(timezone.utc)
    job.status = GatewayBatchJob.STATUS_IN_PROGRESS
    job.in_progress_at = now
    job.save(update_fields=["status", "in_progress_at"])

    raw = read_batch_file_content(job.input_file)
    if raw is None:
        _fail_job(job, "Input file missing on disk.")
        return

    try:
        lines = _parse_jsonl_lines(raw)
    except ValueError as exc:
        _fail_job(job, str(exc))
        return

    output_lines: list[str] = []
    counts = {"total": len(lines), "completed": 0, "failed": 0}
    for line_no, line in enumerate(lines, start=1):
        job.refresh_from_db(fields=["status"])
        if job.status == GatewayBatchJob.STATUS_CANCELLING:
            _cancel_job(job, output_lines, counts)
            return
        result = _process_batch_line(line, line_no=line_no)
        output_lines.append(json.dumps(result, separators=(",", ":")))
        if result.get("error"):
            counts["failed"] += 1
        else:
            counts["completed"] += 1

    _complete_job(job, output_lines, counts)


def _parse_jsonl_lines(raw: bytes) -> list[dict[str, Any]]:
    text = raw.decode("utf-8")
    lines: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"JSONL line {line_no} must be an object.")
        lines.append(parsed)
    if not lines:
        raise ValueError("Input JSONL is empty.")
    return lines


def _process_batch_line(line: dict[str, Any], *, line_no: int) -> dict[str, Any]:
    custom_id = str(line.get("custom_id") or f"line-{line_no}")
    body = line.get("body")
    if not isinstance(body, dict):
        return {"custom_id": custom_id, "error": {"message": "Missing body object."}}
    url = str(line.get("url") or CHAT_COMPLETIONS_PATH)
    if url != CHAT_COMPLETIONS_PATH:
        return {
            "custom_id": custom_id,
            "error": {"message": f"Unsupported url '{url}'."},
        }
    return _post_chat_completion(custom_id, body)


def _post_chat_completion(custom_id: str, body: dict[str, Any]) -> dict[str, Any]:
    timeout = env_float("NADIR_BATCH_LINE_TIMEOUT_SECONDS", 600.0)
    url = f"{gateway_public_base_url().rstrip('/')}{CHAT_COMPLETIONS_PATH}"
    headers = {"Content-Type": "application/json", PRIORITY_HEADER: PRIORITY_BATCH}
    api_key = gateway_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
    try:
        response = httpx.post(url, json=body, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        return {"custom_id": custom_id, "error": {"message": str(exc)}}
    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = {"error": {"message": response.text}}
        return {"custom_id": custom_id, "error": detail.get("error", detail)}
    try:
        payload = response.json()
    except ValueError:
        return {"custom_id": custom_id, "error": {"message": "Invalid JSON response."}}
    return {"custom_id": custom_id, "response": {"status_code": response.status_code, "body": payload}}


def _complete_job(
    job: GatewayBatchJob,
    output_lines: list[str],
    counts: dict[str, int],
) -> None:
    output_bytes = ("\n".join(output_lines) + "\n").encode("utf-8")
    output_file = _persist_output_file(job.batch_id, "output", output_bytes)
    now = datetime.now(timezone.utc)
    job.output_file = output_file
    job.request_counts = counts
    job.status = GatewayBatchJob.STATUS_COMPLETED
    job.completed_at = now
    job.save(update_fields=["output_file", "request_counts", "status", "completed_at"])


def _cancel_job(
    job: GatewayBatchJob,
    output_lines: list[str],
    counts: dict[str, int],
) -> None:
    if output_lines:
        output_bytes = ("\n".join(output_lines) + "\n").encode("utf-8")
        job.output_file = _persist_output_file(job.batch_id, "output", output_bytes)
    now = datetime.now(timezone.utc)
    job.status = GatewayBatchJob.STATUS_CANCELLED
    job.cancelled_at = now
    job.request_counts = counts
    job.save(
        update_fields=["output_file", "status", "cancelled_at", "request_counts"]
    )


def _fail_job(job: GatewayBatchJob, message: str) -> None:
    job.status = GatewayBatchJob.STATUS_FAILED
    job.errors = {"message": message}
    job.completed_at = datetime.now(timezone.utc)
    job.save(update_fields=["status", "errors", "completed_at"])


def _persist_output_file(batch_id: str, suffix: str, data: bytes) -> GatewayBatchFile:
    storage_name = new_storage_name()
    write_upload_bytes(storage_name, data)
    filename = f"{batch_id}_{suffix}.jsonl"
    return GatewayBatchFile.objects.create(
        file_id=new_file_id(),
        purpose="batch",
        filename=filename,
        bytes=len(data),
        storage_name=storage_name,
    )


def _new_batch_id() -> str:
    return f"batch_{uuid.uuid4().hex}"


def reconcile_stale_in_progress_jobs() -> int:
    """Mark orphaned in_progress jobs failed after gateway restart (fail-closed v1)."""
    stale = GatewayBatchJob.objects.filter(status=GatewayBatchJob.STATUS_IN_PROGRESS)
    count = stale.count()
    if count:
        stale.update(
            status=GatewayBatchJob.STATUS_FAILED,
            errors={"message": "Batch interrupted; re-submit input file."},
        )
    return count
