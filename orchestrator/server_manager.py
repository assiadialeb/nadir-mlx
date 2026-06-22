import json
import os
import random
import re
import socket
import subprocess
import signal
import time
from typing import Any, Literal, Optional

from django.conf import settings
from django.utils import timezone

from .gateway_aliases import validate_gateway_alias_unique
from .models import InferenceInstance
from .server_config_schema import build_default_server_config, validate_and_normalize_server_config
from .model_utils import (
    is_model_complete,
    prepare_model_for_multimodal_inference,
    prepare_model_for_text_inference,
    resolve_log_file_path,
    resolve_model_dir,
    supports_embedding_mode,
    supports_image_mode,
    supports_multimodal_mode,
    supports_rerank_mode,
    supports_stt_mode,
    supports_tts_mode,
)

LaunchMode = Literal["TEXT", "MULTIMODAL", "EMBEDDING", "RERANKER", "IMAGE", "TTS", "STT"]
_REUSABLE_INSTANCE_STATUSES = ("STOPPED", "FAILED")


def default_server_host() -> str:
    """Return the default bind address for inference servers."""
    return str(getattr(settings, "MLX_DEFAULT_SERVER_HOST", "127.0.0.1"))


DEFAULT_SERVER_HOST = "127.0.0.1"
STARTUP_WAIT_SECONDS = {
    "TEXT": 2,
    "MULTIMODAL": 8,
    "EMBEDDING": 5,
    "RERANKER": 8,
    "IMAGE": 25,
    "TTS": 15,
    "STT": 20,
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
    re.compile(r"No module named 'misaki'"),
    re.compile(r"No module named 'num2words'"),
    re.compile(r"requires misaki with English extras"),
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
    if mode not in ("TEXT", "MULTIMODAL", "EMBEDDING", "RERANKER", "IMAGE", "TTS", "STT"):
        raise ValueError(
            "Launch mode must be TEXT, MULTIMODAL, EMBEDDING, RERANKER, IMAGE, TTS, or STT."
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


def _is_process_alive(pid: int) -> bool:
    """Return True when a PID still exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _find_child_pids(parent_pid: int) -> list[int]:
    """Return direct child PIDs for a parent process."""
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(parent_pid)],
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


def _collect_descendant_pids(root_pid: int) -> set[int]:
    """Collect a process and all of its descendants."""
    collected: set[int] = set()
    queue = [root_pid]
    while queue:
        pid = queue.pop()
        if pid in collected or pid <= 0:
            continue
        collected.add(pid)
        queue.extend(_find_child_pids(pid))
    return collected


def _collect_stop_targets(instance: InferenceInstance) -> set[int]:
    """Gather every PID that should be stopped for an inference instance."""
    targets = set(_find_orchestrator_launcher_pids(instance.port))
    targets.update(_find_listener_pids(instance.port))
    if instance.pid:
        targets.update(_collect_descendant_pids(instance.pid))
    return {pid for pid in targets if pid > 0}


def _force_stop_pids(pids: set[int], grace_seconds: float = 2.0) -> set[int]:
    """Terminate processes gracefully, then SIGKILL any survivors."""
    if not pids:
        return set()

    for pid in pids:
        _terminate_pid(pid, signal.SIGTERM)

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if not any(_is_process_alive(pid) for pid in pids):
            return set()
        time.sleep(0.2)

    survivors = {pid for pid in pids if _is_process_alive(pid)}
    for pid in survivors:
        _terminate_pid(pid, signal.SIGKILL)

    time.sleep(0.3)
    return {pid for pid in survivors if _is_process_alive(pid)}


def _terminate_launchers_on_port(port: int) -> None:
    """Stop orchestrator launcher processes and free the TCP port."""
    targets = set(_find_orchestrator_launcher_pids(port))
    targets.update(_find_listener_pids(port))
    survivors = _force_stop_pids(targets, grace_seconds=2.0)

    if survivors:
        for pid in survivors:
            _terminate_pid(pid, signal.SIGKILL)
        time.sleep(0.3)

    _wait_for_port_release(port, 5.0)


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


def _resolve_server_config(
    launch_mode: LaunchMode,
    server_config: dict[str, Any] | None,
    model_name: str,
    *,
    exclude_instance_id: int | None = None,
) -> dict[str, Any]:
    normalized = validate_and_normalize_server_config(
        launch_mode,
        server_config or build_default_server_config(launch_mode),
        model_name,
    )
    validate_gateway_alias_unique(
        _config_model_id(normalized, model_name),
        exclude_instance_id=exclude_instance_id,
    )
    return normalized


def _config_host(server_config: dict[str, Any]) -> str:
    return str(server_config.get("host") or default_server_host())


def _config_model_id(server_config: dict[str, Any], model_name: str) -> str:
    return str(server_config.get("model_id") or model_name)


def _append_cli_args(command: list[str], flags: dict[str, Any]) -> None:
    for flag, value in flags.items():
        if value is None or value is False:
            continue
        if value is True:
            command.append(f"--{flag}")
            continue
        if isinstance(value, dict):
            command.extend([f"--{flag}", json.dumps(value)])
            continue
        command.extend([f"--{flag}", str(value)])


def _build_launch_command(
    model_path: str,
    port: int,
    launch_mode: LaunchMode,
    server_config: dict[str, Any],
    model_name: str,
) -> list[str]:
    host = _config_host(server_config)
    model_id = _config_model_id(server_config, model_name)
    advanced = server_config.get("advanced") or {}
    python_bin = _get_python_bin()

    if launch_mode == "MULTIMODAL":
        command = [
            python_bin,
            "-m",
            "orchestrator.mlx_vlm_launcher",
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
        ]
        _append_cli_args(command, {
            "model-id": model_id,
            "max-tokens": server_config.get("max_tokens"),
            "max-kv-size": server_config.get("max_kv_size"),
            "trust-remote-code": server_config.get("trust_remote_code"),
            "adapter-path": advanced.get("adapter_path"),
            "draft-model": advanced.get("draft_model"),
            "draft-kind": advanced.get("draft_kind"),
            "draft-block-size": advanced.get("draft_block_size"),
            "kv-bits": advanced.get("kv_bits"),
            "kv-quant-scheme": advanced.get("kv_quant_scheme"),
            "kv-group-size": advanced.get("kv_group_size"),
            "enable-thinking": advanced.get("enable_thinking"),
            "thinking-budget": advanced.get("thinking_budget"),
        })
        return command

    if launch_mode == "EMBEDDING":
        return [
            python_bin,
            "-m",
            "orchestrator.mlx_embedding_launcher",
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--model-id",
            model_id,
        ]

    if launch_mode == "RERANKER":
        command = [
            python_bin,
            "-m",
            "orchestrator.mlx_reranker_launcher",
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
        ]
        if server_config.get("disable_batching"):
            command.append("--disable-batching")
        command.extend(["--model-id", model_id])
        return command

    if launch_mode == "IMAGE":
        return [
            python_bin,
            "-m",
            "orchestrator.mlx_image_launcher",
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--model-id",
            model_id,
        ]

    if launch_mode == "TTS":
        command = [
            python_bin,
            "-m",
            "orchestrator.mlx_tts_launcher",
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--model-id",
            model_id,
        ]
        _append_cli_args(command, {
            "default-voice": server_config.get("voice_id"),
            "default-speed": server_config.get("speaking_rate"),
            "default-lang-code": server_config.get("lang_code"),
        })
        return command

    if launch_mode == "STT":
        command = [
            python_bin,
            "-m",
            "orchestrator.mlx_stt_launcher",
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--model-id",
            model_id,
        ]
        _append_cli_args(command, {
            "default-language": server_config.get("language"),
            "default-chunk-duration": server_config.get("chunk_duration"),
        })
        return command

    command = [
        python_bin,
        "-m",
        "orchestrator.mlx_launcher",
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--model-id",
        model_id,
    ]
    _append_cli_args(command, {
        "max-tokens": server_config.get("max_tokens"),
        "trust-remote-code": server_config.get("trust_remote_code"),
        "adapter-path": advanced.get("adapter_path"),
        "draft-model": advanced.get("draft_model"),
        "num-draft-tokens": advanced.get("num_draft_tokens"),
        "chat-template-args": advanced.get("chat_template_args"),
        "temp": advanced.get("temp"),
        "top-p": advanced.get("top_p"),
        "top-k": advanced.get("top_k"),
        "min-p": advanced.get("min_p"),
    })
    return command


_MODE_SUPPORT_CHECKS: dict[
    LaunchMode,
    tuple[Any, str],
] = {
    "MULTIMODAL": (supports_multimodal_mode, "This model does not support multimodal inference."),
    "EMBEDDING": (supports_embedding_mode, "This model does not support embedding inference."),
    "RERANKER": (supports_rerank_mode, "This model does not support rerank inference."),
    "IMAGE": (supports_image_mode, "This model does not support image generation."),
    "TTS": (supports_tts_mode, "This model does not support TTS inference."),
    "STT": (supports_stt_mode, "This model does not support STT inference."),
}


def _prepare_model_for_launch(model_path: str, launch_mode: LaunchMode) -> None:
    check = _MODE_SUPPORT_CHECKS.get(launch_mode)
    if check is None:
        prepare_model_for_text_inference(model_path)
        return

    supports_fn, error_message = check
    if not supports_fn(model_path):
        raise ValueError(error_message)

    if launch_mode == "MULTIMODAL":
        prepare_model_for_multimodal_inference(model_path)


def _get_launch_env(
    launch_mode: LaunchMode,
    server_config: dict[str, Any],
    *,
    model_name: str = "",
) -> dict[str, str]:
    env = os.environ.copy()
    if launch_mode in ("TEXT", "MULTIMODAL"):
        env["NADIR_GATEWAY_ALIAS"] = _config_model_id(server_config, model_name)
    if launch_mode in ("IMAGE", "TTS", "STT"):
        env["TQDM_DISABLE"] = "1"
        if launch_mode == "IMAGE":
            quality = server_config.get("default_quality")
            if quality:
                env["IMAGE_DEFAULT_QUALITY"] = str(quality)
            env.setdefault(
                "IMAGE_OUTPUT_DIR",
                str(getattr(settings, "IMAGE_OUTPUT_DIR", "")),
            )
            env.setdefault(
                "NADIR_GATEWAY_PUBLIC_BASE_URL",
                str(getattr(settings, "NADIR_GATEWAY_PUBLIC_BASE_URL", "")),
            )
    return env


def _find_reusable_instance(model_name: str, port: int) -> InferenceInstance | None:
    """Return a stopped or failed instance row that can be relaunched on the same slot."""
    return (
        InferenceInstance.objects.filter(
            model_name=model_name,
            port=port,
            status__in=_REUSABLE_INSTANCE_STATUSES,
        )
        .order_by("-created_at")
        .first()
    )


def _get_or_create_instance(
    model_name: str,
    port: int,
    launch_mode: LaunchMode,
    server_config: dict[str, Any],
) -> InferenceInstance:
    existing = _find_reusable_instance(model_name, port)
    if existing:
        existing.status = "LOADING"
        existing.launch_mode = launch_mode
        existing.server_config = server_config
        existing.pid = None
        existing.stopped_at = None
        existing.save(update_fields=["status", "launch_mode", "server_config", "pid", "stopped_at"])
        return existing

    return InferenceInstance.objects.create(
        model_name=model_name,
        port=port,
        launch_mode=launch_mode,
        server_config=server_config,
        status="LOADING",
    )


def start_instance(
    model_name: str,
    port: Optional[int] = None,
    launch_mode: LaunchMode = "TEXT",
    server_config: dict[str, Any] | None = None,
) -> InferenceInstance:
    """Launch an inference server in the background."""
    launch_mode = parse_launch_mode(launch_mode)
    model_path = str(resolve_model_dir(model_name))
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

    reusable = _find_reusable_instance(model_name, port)
    normalized_config = _resolve_server_config(
        launch_mode,
        server_config,
        model_name,
        exclude_instance_id=reusable.pk if reusable else None,
    )

    _terminate_launchers_on_port(port)
    if not is_port_free(port):
        raise ValueError(f"Port {port} is already in use.")

    os.makedirs(settings.LOGS_DIR, exist_ok=True)
    log_file_path = str(resolve_log_file_path(model_name, port))
    log_file = open(log_file_path, "w", encoding="utf-8")

    instance = _get_or_create_instance(model_name, port, launch_mode, normalized_config)
    cmd = _build_launch_command(model_path, port, launch_mode, normalized_config, model_name)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=_get_launch_env(launch_mode, normalized_config, model_name=model_name),
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


def _stop_port_release_timeout_seconds() -> float:
    """How long to wait for a TCP port to become free after stopping an instance."""
    raw = os.environ.get("MLX_STOP_PORT_RELEASE_TIMEOUT_SECONDS")
    if raw:
        try:
            return max(3.0, float(raw))
        except ValueError:
            pass
    return float(getattr(settings, "MLX_STOP_PORT_RELEASE_TIMEOUT_SECONDS", 12.0))


def is_manual_stop_in_progress(instance: InferenceInstance) -> bool:
    """Return True while the UI is stopping this instance (watchdog should stay idle)."""
    ops = (instance.server_config or {}).get("ops") or {}
    return bool(ops.get("manual_stop_in_progress"))


def _set_manual_stop_in_progress(
    instance: InferenceInstance,
    *,
    active: bool,
    save: bool = True,
) -> None:
    config = dict(instance.server_config or {})
    ops = dict(config.get("ops") or {})
    if active:
        ops["manual_stop_in_progress"] = True
    else:
        ops.pop("manual_stop_in_progress", None)
    config["ops"] = ops
    instance.server_config = config
    if save:
        instance.save(update_fields=["server_config"])


def _port_blocker_pids(port: int) -> list[int]:
    """Return launcher and listener PIDs still bound to a port."""
    seen: set[int] = set()
    ordered: list[int] = []
    for pid in _find_orchestrator_launcher_pids(port) + _find_listener_pids(port):
        if pid in seen or pid <= 0:
            continue
        seen.add(pid)
        ordered.append(pid)
    return ordered


def _ensure_port_released(port: int, timeout_seconds: float) -> list[int]:
    """Kill blockers and poll until the port is free or the timeout expires."""
    deadline = time.time() + timeout_seconds
    last_known: list[int] = []
    while time.time() < deadline:
        if is_port_free(port):
            return []
        blockers = _port_blocker_pids(port)
        if blockers:
            last_known = blockers
            for pid in blockers:
                _terminate_pid(pid, signal.SIGKILL)
            time.sleep(0.4)
            continue
        time.sleep(0.25)
    if is_port_free(port):
        return []
    return _port_blocker_pids(port) or last_known


def check_instance_status(instance: InferenceInstance) -> str:
    """Check whether the subprocess PID is still active and update status."""
    if is_manual_stop_in_progress(instance):
        return instance.status

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
    """Terminate the inference server and ensure the port is released."""
    instance.refresh_from_db()
    _set_manual_stop_in_progress(instance, active=True)
    try:
        targets = _collect_stop_targets(instance)
        _force_stop_pids(targets, grace_seconds=3.0)
        remaining = _ensure_port_released(
            instance.port,
            _stop_port_release_timeout_seconds(),
        )
        if remaining:
            pid_list = ", ".join(str(pid) for pid in remaining)
            raise RuntimeError(
                f"Port {instance.port} is still in use (PID: {pid_list}). "
                "The server could not be stopped completely."
            )

        config = dict(instance.server_config or {})
        ops = dict(config.get("ops") or {})
        ops.pop("manual_stop_in_progress", None)
        config["ops"] = ops
        instance.server_config = config
        instance.pid = None
        instance.status = "STOPPED"
        instance.stopped_at = timezone.now()
        instance.save(update_fields=["status", "pid", "stopped_at", "server_config"])
    except Exception:
        _set_manual_stop_in_progress(instance, active=False)
        raise


def delete_instance(instance: InferenceInstance) -> None:
    """Stop a running instance if needed, then remove its database record."""
    if instance.status in ("RUNNING", "LOADING"):
        stop_instance(instance)
    instance.delete()


def update_stopped_instance(
    instance: InferenceInstance,
    *,
    port: int | None = None,
    launch_mode: LaunchMode | None = None,
    server_config: dict[str, Any] | None = None,
) -> InferenceInstance:
    """Update port, mode, or config for a stopped or failed instance."""
    if instance.status in ("RUNNING", "LOADING"):
        raise ValueError("Stop the server before editing its configuration.")

    new_port = int(port) if port is not None else instance.port
    new_mode = parse_launch_mode(launch_mode) if launch_mode else instance.launch_mode
    if server_config is not None:
        new_config = _resolve_server_config(
            new_mode,
            server_config,
            instance.model_name,
            exclude_instance_id=instance.pk,
        )
    else:
        new_config = dict(instance.server_config or {})

    if new_port != instance.port:
        conflict = (
            InferenceInstance.objects.filter(port=new_port)
            .exclude(pk=instance.pk)
            .exists()
        )
        if conflict:
            raise ValueError(f"Port {new_port} is already used by another instance.")
        if not is_port_free(new_port):
            raise ValueError(f"Port {new_port} is already in use.")

    instance.port = new_port
    instance.launch_mode = new_mode
    instance.server_config = new_config
    instance.save(update_fields=["port", "launch_mode", "server_config"])
    return instance


def restart_instance(instance: InferenceInstance) -> InferenceInstance:
    """Gracefully stop a running instance and relaunch with the same configuration."""
    model_name = instance.model_name
    port = instance.port
    launch_mode = parse_launch_mode(instance.launch_mode)
    server_config = dict(instance.server_config or {})

    if instance.status in ("RUNNING", "LOADING"):
        stop_instance(instance)
        instance.refresh_from_db()

    return start_instance(model_name, port, launch_mode, server_config)


def get_instance_logs(model_name: str, port: int) -> str:
    """Read the last 500 lines of logs for the given instance."""
    try:
        log_file_path = resolve_log_file_path(model_name, port)
    except ValueError:
        return "Log file not found."

    if not log_file_path.is_file():
        return "Log file not found."

    try:
        with open(log_file_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
            return "".join(lines[-500:])
    except OSError:
        return "Error reading logs."
