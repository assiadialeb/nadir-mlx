import os
import threading

from huggingface_hub import snapshot_download
from django.conf import settings

from .models import ModelDownload
from .model_utils import (
    get_folder_name,
    get_model_path,
    is_model_complete,
    validate_hf_repo_id,
)


def _download_thread(repo_id: str, download_id: int) -> None:
    record = ModelDownload.objects.get(id=download_id)
    local_dir = record.local_path

    try:
        os.makedirs(settings.MODELS_DIR, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

        if not is_model_complete(local_dir):
            raise RuntimeError("Download finished but model files are incomplete.")

        record.status = "COMPLETED"
        record.error_message = ""
        record.save(update_fields=["status", "error_message"])
    except Exception as exc:
        record.refresh_from_db()
        if is_model_complete(local_dir):
            record.status = "COMPLETED"
            record.error_message = ""
        else:
            record.status = "FAILED"
            record.error_message = str(exc)
        record.save(update_fields=["status", "error_message"])


def start_model_download(repo_id: str) -> ModelDownload:
    validate_hf_repo_id(repo_id)
    folder_name = get_folder_name(repo_id)
    local_dir = str(get_model_path(folder_name))

    if is_model_complete(local_dir):
        record, _ = ModelDownload.objects.update_or_create(
            repo_id=repo_id,
            defaults={
                "local_path": local_dir,
                "status": "COMPLETED",
                "error_message": "",
            },
        )
        return record

    record = ModelDownload.objects.filter(repo_id=repo_id).first()
    if record:
        if record.status == "DOWNLOADING":
            return record
        record.status = "DOWNLOADING"
        record.error_message = ""
        record.local_path = local_dir
        record.save(update_fields=["status", "error_message", "local_path"])
    else:
        record = ModelDownload.objects.create(
            repo_id=repo_id,
            local_path=local_dir,
            status="DOWNLOADING",
        )

    thread = threading.Thread(
        target=_download_thread,
        args=(repo_id, record.id),
        daemon=True,
    )
    thread.start()
    return record
