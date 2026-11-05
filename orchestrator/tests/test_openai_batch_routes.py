"""Gateway route tests for /v1/files and /v1/batches."""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from orchestrator.batch_file_store import batch_files_dir
from orchestrator.gateway.app import create_app


@pytest.fixture
def batch_files_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("NADIR_BATCH_FILES_DIR", str(tmp_path / "batch_files"))
    batch_files_dir().mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def gateway_client(batch_files_tmp, monkeypatch):
    monkeypatch.setenv("NADIR_BATCH_FILES_DIR", str(batch_files_tmp / "batch_files"))
    return TestClient(create_app())


@pytest.mark.django_db
def test_upload_and_list_files(gateway_client) -> None:
    files = {
        "file": ("batch.jsonl", io.BytesIO(b"{}\n"), "application/jsonl"),
    }
    data = {"purpose": "batch"}
    response = gateway_client.post("/v1/files", data=data, files=files)
    assert response.status_code == 200
    file_id = response.json()["id"]

    listed = gateway_client.get("/v1/files")
    assert listed.status_code == 200
    ids = {item["id"] for item in listed.json()["data"]}
    assert file_id in ids


@pytest.mark.django_db
def test_create_batch_endpoint(gateway_client) -> None:
    upload = gateway_client.post(
        "/v1/files",
        data={"purpose": "batch"},
        files={"file": ("in.jsonl", io.BytesIO(b"{}\n"), "application/jsonl")},
    )
    file_id = upload.json()["id"]
    create = gateway_client.post(
        "/v1/batches",
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    assert create.status_code == 200
    body = create.json()
    assert body["object"] == "batch"
    assert body["status"] == "validating"
    assert body["input_file_id"] == file_id
