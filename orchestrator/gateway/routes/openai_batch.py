"""OpenAI Files + Batches routes (local batch MVP)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from asgiref.sync import sync_to_async

from orchestrator.gateway.auth import verify_gateway_api_key
from orchestrator.gateway.routes.common import parse_json_body

from orchestrator.batch_services import (
    BatchServiceError,
    batch_to_openai_dict,
    cancel_batch_job,
    create_batch_file_from_stream,
    create_batch_job,
    delete_batch_file,
    file_to_openai_dict,
    get_batch_file,
    get_batch_job,
    list_batch_files,
    list_batch_jobs,
    read_batch_file_content,
)
from orchestrator.fastapi_openapi import OPENAPI_NOT_FOUND

router = APIRouter(dependencies=[Depends(verify_gateway_api_key)])


def _service_error(exc: BatchServiceError) -> JSONResponse:
    from orchestrator.gateway.errors import openai_error_envelope

    return JSONResponse(
        status_code=exc.status_code,
        content=openai_error_envelope(exc.message, exc.code),
    )


@router.post("/v1/files")
def upload_file(
    purpose: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
):
    result = create_batch_file_from_stream(
        purpose=purpose.strip(),
        filename=file.filename or "upload.jsonl",
        readable=file.file,
    )
    if isinstance(result, BatchServiceError):
        return _service_error(result)
    return file_to_openai_dict(result)


@router.get("/v1/files")
def list_files(limit: int = 100) -> dict[str, object]:
    records = list_batch_files(limit=limit)
    return {
        "object": "list",
        "data": [file_to_openai_dict(record) for record in records],
    }


@router.get("/v1/files/{file_id}", responses=OPENAPI_NOT_FOUND)
def retrieve_file(file_id: str):
    record = get_batch_file(file_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No such file object.")
    return file_to_openai_dict(record)


@router.get("/v1/files/{file_id}/content", responses=OPENAPI_NOT_FOUND)
def download_file_content(file_id: str):
    record = get_batch_file(file_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No such file object.")
    content = read_batch_file_content(record)
    if content is None:
        raise HTTPException(status_code=404, detail="File content missing.")
    return PlainTextResponse(content.decode("utf-8"), media_type="application/jsonl")


@router.delete("/v1/files/{file_id}", responses=OPENAPI_NOT_FOUND)
def remove_file(file_id: str):
    err = delete_batch_file(file_id)
    if isinstance(err, BatchServiceError):
        return _service_error(err)
    return {"id": file_id, "object": "file", "deleted": True}


@router.post("/v1/batches")
async def create_batch(request: Request):
    body = await parse_json_body(request)
    if isinstance(body, JSONResponse):
        return body
    input_file_id = str(body.get("input_file_id") or "").strip()
    endpoint = str(body.get("endpoint") or "").strip()
    completion_window = str(body.get("completion_window") or "24h").strip()
    if not input_file_id or not endpoint:
        return _service_error(
            BatchServiceError("input_file_id and endpoint are required.")
        )
    result = await sync_to_async(create_batch_job)(
        input_file_id=input_file_id,
        endpoint=endpoint,
        completion_window=completion_window,
    )
    if isinstance(result, BatchServiceError):
        return _service_error(result)
    return batch_to_openai_dict(result)


@router.get("/v1/batches")
def list_batches(limit: int = 100) -> dict[str, object]:
    jobs = list_batch_jobs(limit=limit)
    return {
        "object": "list",
        "data": [batch_to_openai_dict(job) for job in jobs],
    }


@router.get("/v1/batches/{batch_id}", responses=OPENAPI_NOT_FOUND)
def retrieve_batch(batch_id: str):
    job = get_batch_job(batch_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such batch.")
    return batch_to_openai_dict(job)


@router.post("/v1/batches/{batch_id}/cancel", responses=OPENAPI_NOT_FOUND)
def cancel_batch(batch_id: str):
    result = cancel_batch_job(batch_id)
    if isinstance(result, BatchServiceError):
        return _service_error(result)
    return batch_to_openai_dict(result)
