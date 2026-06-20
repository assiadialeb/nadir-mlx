import os
import random
import re
import socket
import subprocess
import signal
import time
from typing import Literal, Optional

from django.conf import settings
from django.utils import timezone

from .models import InferenceInstance
from .model_utils import (
    is_model_complete,
    prepare_model_for_multimodal_inference,
    prepare_model_for_text_inference,
    supports_embedding_mode,
    supports_image_mode,
    supports_multimodal_mode,
    supports_rerank_mode,
)

LaunchMode = Literal["TEXT", "MULTIMODAL", "EMBEDDING", "RERANKER", "IMAGE"]
SERVER_HOST = "0.0.0.0"
STARTUP_WAIT_SECONDS = {
    "TEXT": 2,
    "MULTIMODAL": 8,
    "EMBEDDING": 5,
    "RERANKER": 8,
    "IMAGE": 25,
}

_LOG_FAILURE_PATTERNS = (
    re.compile(r"ValueError: Model type .+ not supported"),
    re.compile(r"ModuleNotFoundError:"),
    re.compile(r"Failed to launch mlx_lm server:"),
    re.compile(r"FileNotFoundError:"),
    re.compile(r"Model output does not contain embeddings"),
    re.compile(r"Model path not found:"),
    re.compile(r"local-reranker is not installed"),
    re.compile(r"No module named 'local_reranker'"),
    re.compile(r"No module named 'mflux'"),
    re.compile(r"Unsupported image model folder"),
)


def get_downloaded_models() -> list[str]:
    """Scan the models directory for complete, ready-to-serve model folders."""
    from .model_utils import sync_model_download_status

    sync_model_download_status()

    models_dir = settings.MODELS_DIR
    if not os.path.exists(models_dir):
        return []

    models = []
    for name in os.listdir(models_dir):
        path = os.path.join(models_dir, name)
        if os.path.isdir(path) and is_model_complete(path):
            models.append(name)
    return sorted(models)


def parse_launch_mode(raw_mode: Optional[str]) -> LaunchMode:
    """Normalize and validate the requested launch mode."""
    mode = (raw_mode or "TEXT").upper()
    if mode not in ("TEXT", "MULTIMODAL", "EMBEDDING", "RERANKER", "IMAGE"):
        raise ValueError(
            "Launch mode must be TEXT, MULTIMODAL, EMBEDDING, RERANKER, or IMAGE."
        )
    return mode


def is_port_free(port: int) -> bool:
    """Check whether a TCP port is free on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _find_listener_pids(port: int) -> list[int]:
    """Return PIDs listening on a TCP port (macOS/Linux via lsof)."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    pids: list[int] = []
    for line in result.stdout.splitlines():
        token = line.strip()
        if token.isdigit():
            pids.append(int(token))
    return pids


def _find_orchestrator_launcher_pids(port: int) -> list[int]:
    """Return PIDs for orchestrator launcher modules bound to a port."""
    pattern = f"orchestrator\\.mlx_.*launcher.*--port {port}"
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    if result.returncode not in (0, 1):
        return []

    pids: list[int] = []
    for line in result.stdout.splitlines():
        token = line.strip()
        if token.isdigit():
            pids.append(int(token))
    return pids


def _terminate_launchers_on_port(port: int) -> None:
    """Stop orchestrator launcher processes and free the TCP port."""
    target_pids = set(_find_orchestrator_launcher_pids(port))
    target_pids.update(_find_listener_pids(port))

    for pid in target_pids:
        _terminate_pid(pid, signal.SIGTERM)

    if not _wait_for_port_release(port, 3.0):
        target_pids.update(_find_orchestrator_launcher_pids(port))
        target_pids.update(_find_listener_pids(port))
        for pid in target_pids:
            _terminate_pid(pid, signal.SIGKILL)
        _wait_for_port_release(port, 2.0)


def _terminate_pid(pid: int, sig: signal.Signals) -> None:
    """Send a signal to a process, preferring its entire process group."""
    try:
        os.killpg(os.getpgid(pid), sig)
    except OSError:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def _wait_for_port_release(port: int, timeout_seconds: float) -> bool:
    """Poll until the port is free or the timeout expires."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if is_port_free(port):
            return True
        time.sleep(0.2)
    return is_port_free(port)


def get_free_port() -> int:
    """Select a random free port in the range 11400-11500."""
    ports = list(range(11400, 11501))
    random.shuffle(ports)
    for port in ports:
        if is_port_free(port):
            return port
    raise RuntimeError("No free ports available in range 11400-11500")


def _detect_log_failure(log_file_path: str) -> Optional[str]:
    if not os.path.exists(log_file_path):
        return None
    try:
        with open(log_file_path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except OSError:
        return None

    for pattern in _LOG_FAILURE_PATTERNS:
        match = pattern.search(content)
        if match:
            return match.group(0)
    return None


def _get_python_bin() -> str:
    python_bin = os.path.join(settings.BASE_DIR, "venv", "bin", "python")
    if os.path.exists(python_bin):
        return python_bin
    return "python"


def _build_launch_command(
    model_path: str,
    port: int,
    launch_mode: LaunchMode,
) -> list[str]:
    python_bin = _get_python_bin()
    if launch_mode == "MULTIMODAL":
        return [
            python_bin,
            "-m",
            "orchestrator.mlx_vlm_launcher",
            "--model",
            model_path,
            "--host",
            SERVER_HOST,
            "--port",
            str(port),
        ]
    if launch_mode == "EMBEDDING":
        return [
            python_bin,
            "-m",
            "orchestrator.mlx_embedding_launcher",
            "--model",
            model_path,
            "--host",
            SERVER_HOST,
            "--port",
            str(port),
        ]
    if launch_mode == "RERANKER":
        return [
            python_bin,
            "-m",
            "orchestrator.mlx_reranker_launcher",
            "--model",
            model_path,
            "--host",
            SERVER_HOST,
            "--port",
            str(port),
        ]
    if launch_mode == "IMAGE":
        return [
            python_bin,
            "-m",
            "orchestrator.mlx_image_launcher",
            "--model",
            model_path,
            "--host",
            SERVER_HOST,
            "--port",
            str(port),
        ]
    return [
        python_bin,
        "-m",
        "orchestrator.mlx_launcher",
        "--model",
        model_path,
        "--host",
        SERVER_HOST,
        "--port",
        str(port),
    ]


def _prepare_model_for_launch(model_path: str, launch_mode: LaunchMode) -> None:
    if launch_mode == "MULTIMODAL":
        if not supports_multimodal_mode(model_path):
            raise ValueError("This model does not support multimodal inference.")
        prepare_model_for_multimodal_inference(model_path)
        return
    if launch_mode == "EMBEDDING":
        if not supports_embedding_mode(model_path):
            raise ValueError("This model does not support embedding inference.")
        return
    if launch_mode == "RERANKER":
        if not supports_rerank_mode(model_path):
            raise ValueError("This model does not support rerank inference.")
        return
    if launch_mode == "IMAGE":
        if not supports_image_mode(model_path):
            raise ValueError("This model does not support image generation.")
        return
    prepare_model_for_text_inference(model_path)


def _get_launch_env(launch_mode: LaunchMode) -> dict[str, str]:
    env = os.environ.copy()
    if launch_mode == "IMAGE":
        env["TQDM_DISABLE"] = "1"
    return env


def _get_or_create_instance(
    model_name: str,
    port: int,
    launch_mode: LaunchMode,
) -> InferenceInstance:
    existing = InferenceInstance.objects.filter(
        model_name=model_name,
        port=port,
        status="STOPPED",
    ).order_by("-created_at").first()
    if existing:
        existing.status = "LOADING"
        existing.launch_mode = launch_mode
        existing.pid = None
        existing.stopped_at = None
        existing.save(update_fields=["status", "launch_mode", "pid", "stopped_at"])
        return existing

    return InferenceInstance.objects.create(
        model_name=model_name,
        port=port,
        launch_mode=launch_mode,
        status="LOADING",
    )


def start_instance(
    model_name: str,
    port: Optional[int] = None,
    launch_mode: LaunchMode = "TEXT",
) -> InferenceInstance:
    """Launch a text or multimodal inference server in the background."""
    launch_mode = parse_launch_mode(launch_mode)
    model_path = os.path.join(settings.MODELS_DIR, model_name)
    if not os.path.isdir(model_path):
        raise ValueError(f"Model folder '{model_name}' was not found in ./models.")
    if not is_model_complete(model_path):
        raise ValueError(
            f"Model '{model_name}' is incomplete. Wait for the download to finish."
        )

    _prepare_model_for_launch(model_path, launch_mode)

    if not port:
        port = get_free_port()
    else:
        port = int(port)

    _terminate_launchers_on_port(port)
    if not is_port_free(port):
        raise ValueError(f"Port {port} is already in use.")

    os.makedirs(settings.LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(settings.LOGS_DIR, f"{model_name}_{port}.log")
    log_file = open(log_file_path, "w", encoding="utf-8")

    instance = _get_or_create_instance(model_name, port, launch_mode)
    cmd = _build_launch_command(model_path, port, launch_mode)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=_get_launch_env(launch_mode),
        )
        instance.pid = process.pid
        instance.status = "LOADING"
        instance.save(update_fields=["pid", "status"])

        time.sleep(STARTUP_WAIT_SECONDS[launch_mode])
        failure = _detect_log_failure(log_file_path)
        if failure:
            stop_instance(instance)
            instance.status = "FAILED"
            instance.save(update_fields=["status"])
            raise RuntimeError(f"Server failed to load model: {failure}")

        if process.poll() is not None:
            failure = _detect_log_failure(log_file_path) or "Process exited unexpectedly."
            instance.status = "FAILED"
            instance.save(update_fields=["status"])
            raise RuntimeError(f"Server failed to start: {failure}")

        instance.status = "RUNNING"
        instance.save(update_fields=["status"])
        return instance
    except Exception:
        if instance.status != "FAILED":
            instance.status = "FAILED"
            instance.save(update_fields=["status"])
        raise
    finally:
        log_file.close()


def check_instance_status(instance: InferenceInstance) -> str:
    """Check whether the subprocess PID is still active and update status."""
    if instance.status not in ("RUNNING", "LOADING"):
        return instance.status

    if not instance.pid:
        instance.status = "STOPPED"
        instance.save(update_fields=["status"])
        return "STOPPED"

    try:
        os.kill(instance.pid, 0)
    except OSError:
        log_path = os.path.join(
            settings.LOGS_DIR,
            f"{instance.model_name}_{instance.port}.log",
        )
        failure = _detect_log_failure(log_path)
        instance.status = "FAILED" if failure else "STOPPED"
        instance.pid = None
        instance.stopped_at = timezone.now()
        instance.save(update_fields=["status", "pid", "stopped_at"])
        return instance.status

    if instance.status == "LOADING":
        log_path = os.path.join(
            settings.LOGS_DIR,
            f"{instance.model_name}_{instance.port}.log",
        )
        failure = _detect_log_failure(log_path)
        if failure:
            stop_instance(instance)
            instance.status = "FAILED"
            instance.save(update_fields=["status"])
            return "FAILED"

        instance.status = "RUNNING"
        instance.save(update_fields=["status"])

    return instance.status


def stop_instance(instance: InferenceInstance) -> None:
    """Gracefully terminate or force-kill the server and free its port."""
    _terminate_launchers_on_port(instance.port)

    if not _wait_for_port_release(instance.port, 2.0):
        remaining_pids = _find_listener_pids(instance.port)
        pid_list = ", ".join(str(pid) for pid in remaining_pids) or "unknown"
        raise RuntimeError(
            f"Port {instance.port} is still in use (PID: {pid_list}). "
            "Another process may be holding it."
        )

    instance.pid = None
    instance.status = "STOPPED"
    instance.stopped_at = timezone.now()
    instance.save(update_fields=["status", "pid", "stopped_at"])


def delete_instance(instance: InferenceInstance) -> None:
    """Stop a running instance if needed, then remove its database record."""
    if instance.status in ("RUNNING", "LOADING") and instance.pid:
        stop_instance(instance)
    instance.delete()


def get_instance_logs(model_name: str, port: int) -> str:
    """Read the last 500 lines of logs for the given instance."""
    log_file_path = os.path.join(settings.LOGS_DIR, f"{model_name}_{port}.log")
    if not os.path.exists(log_file_path):
        return "Log file not found."

    try:
        with open(log_file_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
            return "".join(lines[-500:])
    except OSError as exc:
        return f"Error reading logs: {exc}"
