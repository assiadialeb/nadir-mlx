import json
import logging
import os
import random
import re
import socket
import subprocess
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from django.conf import settings
from django.utils import timezone

from orchestrator.gateway.route_cache import clear_gateway_route_cache

from .gateway_aliases import validate_instance_gateway_aliases
from .instance_locks import (
    DEFAULT_SLOT_LOCK_TIMEOUT_SECONDS,
    USER_ACTION_SLOT_WAIT_SECONDS,
    instance_slot_lock,
    is_slot_lock_held_elsewhere,
)
from .models import InferenceInstance
from .server_config_schema import (
    build_default_server_config,
    text_mtp_preview_enabled,
    validate_and_normalize_server_config,
)
from .model_utils import (
    is_model_complete,
    prepare_model_for_multimodal_inference,
    prepare_model_for_text_inference,
    resolve_log_file_path,
    resolve_model_dir,
    validate_model_folder_name,
    validated_optional_model_filesystem_ref,
    supports_embedding_mode,
    supports_image_mode,
    supports_multimodal_mode,
    supports_rerank_mode,
    supports_stt_mode,
    supports_tts_mode,
)
from .security_utils import (
    assert_path_under_directory,
    models_root_path,
    validate_server_bind_host,
    validated_launch_port,
)

LaunchMode = Literal["TEXT", "MULTIMODAL", "EMBEDDING", "RERANKER", "IMAGE", "TTS", "STT"]
_REUSABLE_INSTANCE_STATUSES = ("STOPPED", "FAILED")
_ACTIVE_INSTANCE_STATUSES = ("LOADING", "RUNNING")
_INSTANCE_COMMAND_MARKER = "orchestrator."
_READY_POLL_SECONDS = 0.5
# A stop whose actor died (Ctrl+C, SIGKILL) must not freeze the instance forever.
_STOP_IN_PROGRESS_STALE_SECONDS = 60.0

logger = logging.getLogger(__name__)

START_CANCELLED_STOP_REQUESTED_MSG = "Start cancelled: a stop was requested."


class InstanceStartCancelled(RuntimeError):
    """A stop was requested while the instance was loading."""


@dataclass(frozen=True)
class InstanceLaunchPlan:
    """Validated launch parameters; the DB row is already LOADING / desired running."""

    instance_id: int
    port: int
    model_path: str
    launch_mode: LaunchMode
    requested_model_name: str
    already_active: bool = False


def default_server_host() -> str:
    """Return the default bind address for inference servers."""
    return str(getattr(settings, "MLX_DEFAULT_SERVER_HOST", "127.0.0.1"))


DEFAULT_SERVER_HOST = "127.0.0.1"


def instance_ready_timeout_seconds() -> float:
    """Max wait for /health after spawn before a start is declared FAILED."""
    return float(getattr(settings, "MLX_INSTANCE_READY_TIMEOUT_SECONDS", 300.0))

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
    re.compile(
        r"ValueError: \[reshape\] Cannot reshape array of size \d+ "
        r"into shape \(1,1,4096,256\)"
    ),
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
        try:
            path = resolve_model_dir(name)
        except ValueError:
            continue
        if path.is_dir() and is_model_complete(path):
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
    safe_port = validated_launch_port(port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", safe_port))
            return True
        except OSError:
            return False


def _find_listener_pids(port: int) -> list[int]:
    """Return PIDs listening on a TCP port (macOS/Linux via lsof)."""
    safe_port = validated_launch_port(port)
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{safe_port}", "-sTCP:LISTEN", "-t"],
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
    safe_port = validated_launch_port(port)
    # Anchored port token: "--port 1145" must not match "--port 11454".
    pattern = f"orchestrator\\.[a-z_]+ .*--port {safe_port}( |$)"
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


def _process_command(pid: int) -> str:
    """Return the full command line of a PID, or an empty string."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _command_matches_instance_port(command: str, port: int) -> bool:
    if _INSTANCE_COMMAND_MARKER not in command:
        return False
    tokens = command.split()
    return any(
        token == "--port" and tokens[index + 1] == str(port)
        for index, token in enumerate(tokens[:-1])
    )


def is_instance_process(pid: int | None, port: int) -> bool:
    """Return True only for a live orchestrator launcher bound to ``port``.

    Stored PIDs survive reboots and crashes and may be recycled by unrelated
    programs; never signal or trust a PID without checking its command line.
    """
    if not pid or not _is_process_alive(pid):
        return False
    return _command_matches_instance_port(_process_command(pid), port)


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


def _owned_port_pids(port: int, *, extra_root: int | None = None) -> set[int]:
    """Return launcher trees and listeners on ``port`` that belong to Nadir.

    Foreign listeners (not an orchestrator launcher, not a descendant of one)
    are never returned, so stop/start cannot kill unrelated programs.
    """
    roots = set(_find_orchestrator_launcher_pids(port))
    if extra_root and is_instance_process(extra_root, port):
        roots.add(extra_root)
    owned: set[int] = set()
    for root in roots:
        owned.update(_collect_descendant_pids(root))
    for pid in _find_listener_pids(port):
        if pid in owned or is_instance_process(pid, port):
            owned.add(pid)
    return {pid for pid in owned if pid > 0}


def _collect_stop_targets(instance: InferenceInstance) -> set[int]:
    """Gather every PID that should be stopped for an inference instance."""
    return _owned_port_pids(instance.port, extra_root=instance.pid)


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
    survivors = _force_stop_pids(_owned_port_pids(port), grace_seconds=2.0)

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
    validate_instance_gateway_aliases(
        normalized,
        model_name,
        exclude_instance_id=exclude_instance_id,
    )
    return normalized


def _config_host(server_config: dict[str, Any]) -> str:
    return validate_server_bind_host(str(server_config.get("host") or default_server_host()))


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
    safe_model_path = str(
        assert_path_under_directory(Path(model_path).resolve(), models_root_path()),
    )
    host = _config_host(server_config)
    safe_port = validated_launch_port(port)
    model_id = _config_model_id(server_config, model_name)
    advanced = server_config.get("advanced") or {}
    adapter_path = validated_optional_model_filesystem_ref(advanced.get("adapter_path"))
    draft_model = validated_optional_model_filesystem_ref(advanced.get("draft_model"))
    python_bin = _get_python_bin()

    if launch_mode == "MULTIMODAL":
        command = [
            python_bin,
            "-m",
            "orchestrator.mlx_vlm_launcher",
            "--model",
            safe_model_path,
            "--host",
            host,
            "--port",
            str(safe_port),
        ]
        _append_cli_args(command, {
            "model-id": model_id,
            "max-tokens": server_config.get("max_tokens"),
            "max-kv-size": server_config.get("max_kv_size"),
            "trust-remote-code": server_config.get("trust_remote_code"),
            "adapter-path": adapter_path,
            "draft-model": draft_model,
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
            safe_model_path,
            "--host",
            host,
            "--port",
            str(safe_port),
            "--model-id",
            model_id,
        ]

    if launch_mode == "RERANKER":
        command = [
            python_bin,
            "-m",
            "orchestrator.mlx_reranker_launcher",
            "--model",
            safe_model_path,
            "--host",
            host,
            "--port",
            str(safe_port),
        ]
        if server_config.get("disable_batching"):
            command.append("--disable-batching")
        command.extend(["--model-id", model_id])
        return command

    if launch_mode == "IMAGE":
        command = [
            python_bin,
            "-m",
            "orchestrator.mlx_image_launcher",
            "--model",
            safe_model_path,
            "--host",
            host,
            "--port",
            str(safe_port),
            "--model-id",
            model_id,
        ]
        _append_cli_args(command, {
            "quantize-override": advanced.get("quantize_override"),
        })
        return command

    if launch_mode == "TTS":
        command = [
            python_bin,
            "-m",
            "orchestrator.mlx_tts_launcher",
            "--model",
            safe_model_path,
            "--host",
            host,
            "--port",
            str(safe_port),
            "--model-id",
            model_id,
        ]
        _append_cli_args(command, {
            "default-voice": server_config.get("voice_id"),
            "default-speed": server_config.get("speaking_rate"),
            "default-lang-code": server_config.get("lang_code"),
            "default-response-format": advanced.get("response_format"),
        })
        return command

    if launch_mode == "STT":
        command = [
            python_bin,
            "-m",
            "orchestrator.mlx_stt_launcher",
            "--model",
            safe_model_path,
            "--host",
            host,
            "--port",
            str(safe_port),
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
        safe_model_path,
        "--host",
        host,
        "--port",
        str(safe_port),
        "--model-id",
        model_id,
    ]
    _append_cli_args(command, {
        "max-tokens": server_config.get("max_tokens"),
        "trust-remote-code": server_config.get("trust_remote_code"),
        "adapter-path": adapter_path,
        "draft-model": draft_model,
        "num-draft-tokens": advanced.get("num_draft_tokens"),
        "chat-template-args": advanced.get("chat_template_args"),
        "temp": advanced.get("temp"),
        "top-p": advanced.get("top_p"),
        "top-k": advanced.get("top_k"),
        "min-p": advanced.get("min_p"),
        "decode-concurrency": advanced.get("decode_concurrency"),
        "prompt-concurrency": advanced.get("prompt_concurrency"),
        "prefill-step-size": advanced.get("prefill_step_size"),
    })
    from orchestrator.mlx_lm_capabilities import text_kv_quantization_enabled
    from orchestrator.prompt_cache import resolve_prompt_cache_cli_flags

    if text_kv_quantization_enabled():
        _append_cli_args(command, {
            "kv-bits": advanced.get("kv_bits"),
            "kv-quant-scheme": advanced.get("kv_quant_scheme"),
            "kv-group-size": advanced.get("kv_group_size"),
        })

    _append_cli_args(command, resolve_prompt_cache_cli_flags(advanced))
    if text_mtp_preview_enabled():
        _append_cli_args(command, {
            "draft-kind": advanced.get("draft_kind"),
            "draft-block-size": advanced.get("draft_block_size"),
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
    # Serve supervisor vars must not reach MLX workers (wrong ppid / ownership).
    from orchestrator.serve_supervisor import BACKGROUND_OWNER_ENV, SERVE_PARENT_PID_ENV

    env.pop(BACKGROUND_OWNER_ENV, None)
    env.pop(SERVE_PARENT_PID_ENV, None)
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


def _find_active_instance(model_name: str, port: int) -> InferenceInstance | None:
    """Return a LOADING/RUNNING row for this slot whose process is really alive."""
    rows = InferenceInstance.objects.filter(
        model_name=model_name,
        port=port,
        status__in=_ACTIVE_INSTANCE_STATUSES,
    )
    for row in rows:
        if is_instance_process(row.pid, port):
            return row
    return None


def _find_reusable_instance(model_name: str, port: int) -> InferenceInstance | None:
    """Return a stopped, failed, or stale LOADING row that can be relaunched on this slot."""
    rows = InferenceInstance.objects.filter(model_name=model_name, port=port).order_by(
        "-created_at"
    )
    for row in rows:
        if row.status in _REUSABLE_INSTANCE_STATUSES:
            return row
        if row.status == "LOADING" and not is_instance_process(row.pid, port):
            return row
    return None


def _ensure_port_not_owned_by_other_instance(port: int, instance_id: int | None) -> None:
    """Refuse to launch over another live instance (start would kill its process)."""
    others = InferenceInstance.objects.filter(
        port=port,
        status__in=_ACTIVE_INSTANCE_STATUSES,
    ).exclude(pk=instance_id)
    for other in others:
        if is_instance_process(other.pid, port):
            raise ValueError(
                f"Port {port} is already used by running instance '{other.model_name}'."
            )


def _get_or_create_instance(
    reusable: InferenceInstance | None,
    *,
    model_name: str,
    port: int,
    launch_mode: LaunchMode,
    server_config: dict[str, Any],
    mark_desired_running: bool,
) -> InferenceInstance:
    if reusable is None:
        return InferenceInstance.objects.create(
            model_name=model_name,
            port=port,
            launch_mode=launch_mode,
            server_config=server_config,
            status="LOADING",
        )

    reusable.status = "LOADING"
    reusable.launch_mode = launch_mode
    reusable.server_config = server_config
    reusable.pid = None
    reusable.stopped_at = None
    reusable.stopping_since = None
    update_fields = [
        "status",
        "launch_mode",
        "server_config",
        "pid",
        "stopped_at",
        "stopping_since",
    ]
    if mark_desired_running:
        reusable.desired_state = InferenceInstance.DESIRED_STATE_RUNNING
        update_fields.append("desired_state")
    reusable.save(update_fields=update_fields)
    return reusable


def _validated_model_path(safe_model_name: str, launch_mode: LaunchMode) -> str:
    model_path = str(resolve_model_dir(safe_model_name))
    if not os.path.isdir(model_path):
        raise ValueError(f"Model folder '{safe_model_name}' was not found in ./models.")
    if not is_model_complete(model_path):
        raise ValueError(
            f"Model '{safe_model_name}' is incomplete. Wait for the download to finish."
        )
    _prepare_model_for_launch(model_path, launch_mode)
    return model_path


def prepare_instance_launch(
    model_name: str,
    port: Optional[int] = None,
    launch_mode: LaunchMode = "TEXT",
    server_config: dict[str, Any] | None = None,
    *,
    wait_for_slot: bool = True,
    mark_desired_running: bool = True,
    slot_timeout_seconds: float = DEFAULT_SLOT_LOCK_TIMEOUT_SECONDS,
) -> InstanceLaunchPlan:
    """Validate a launch and reserve its DB row (LOADING) under the slot lock.

    Raises:
        ValueError: invalid model, config, or port owned by another instance.
        InstanceSlotBusy: ``wait_for_slot=False`` and another actor holds the slot.
    """
    launch_mode = parse_launch_mode(launch_mode)
    safe_model_name = validate_model_folder_name(model_name)
    model_path = _validated_model_path(safe_model_name, launch_mode)
    resolved_port = validated_launch_port(int(port)) if port else get_free_port()

    with instance_slot_lock(
        resolved_port,
        blocking=wait_for_slot,
        timeout_seconds=slot_timeout_seconds,
    ):
        active = _find_active_instance(safe_model_name, resolved_port)
        if active is not None:
            return InstanceLaunchPlan(
                instance_id=active.pk,
                port=resolved_port,
                model_path=model_path,
                launch_mode=launch_mode,
                requested_model_name=model_name,
                already_active=True,
            )
        reusable = _find_reusable_instance(safe_model_name, resolved_port)
        _ensure_port_not_owned_by_other_instance(
            resolved_port, reusable.pk if reusable else None
        )
        normalized_config = _resolve_server_config(
            launch_mode,
            server_config,
            safe_model_name,
            exclude_instance_id=reusable.pk if reusable else None,
        )
        instance = _get_or_create_instance(
            reusable,
            model_name=safe_model_name,
            port=resolved_port,
            launch_mode=launch_mode,
            server_config=normalized_config,
            mark_desired_running=mark_desired_running,
        )
    return InstanceLaunchPlan(
        instance_id=instance.pk,
        port=resolved_port,
        model_path=model_path,
        launch_mode=launch_mode,
        requested_model_name=model_name,
    )


def launch_prepared_instance(
    plan: InstanceLaunchPlan,
    *,
    wait_for_slot: bool = True,
    slot_timeout_seconds: float = DEFAULT_SLOT_LOCK_TIMEOUT_SECONDS,
) -> InferenceInstance:
    """Spawn the process, wait for /health and warmup, then mark RUNNING.

    Idempotent: returns the row untouched when its process is already alive.
    """
    with instance_slot_lock(
        plan.port,
        blocking=wait_for_slot,
        timeout_seconds=slot_timeout_seconds,
    ):
        instance = InferenceInstance.objects.get(pk=plan.instance_id)
        if instance.status in _ACTIVE_INSTANCE_STATUSES and is_instance_process(
            instance.pid, plan.port
        ):
            return instance
        if instance.desired_state == InferenceInstance.DESIRED_STATE_STOPPED:
            _persist_stop_outcome(instance, "STOPPED")
            raise InstanceStartCancelled(START_CANCELLED_STOP_REQUESTED_MSG)
        return _launch_locked(instance, plan)


def start_instance(
    model_name: str,
    port: Optional[int] = None,
    launch_mode: LaunchMode = "TEXT",
    server_config: dict[str, Any] | None = None,
    *,
    wait_for_slot: bool = True,
    slot_timeout_seconds: float = DEFAULT_SLOT_LOCK_TIMEOUT_SECONDS,
) -> InferenceInstance:
    """Launch an inference server and return once it is RUNNING (healthy and warm)."""
    plan = prepare_instance_launch(
        model_name,
        port,
        launch_mode,
        server_config,
        wait_for_slot=wait_for_slot,
        slot_timeout_seconds=slot_timeout_seconds,
    )
    return launch_prepared_instance(
        plan,
        wait_for_slot=wait_for_slot,
        slot_timeout_seconds=slot_timeout_seconds,
    )


def _launch_locked(instance: InferenceInstance, plan: InstanceLaunchPlan) -> InferenceInstance:
    _ensure_port_not_owned_by_other_instance(plan.port, instance.pk)
    _terminate_launchers_on_port(plan.port)
    if not is_port_free(plan.port):
        _persist_stop_outcome(instance, "FAILED")
        raise ValueError(f"Port {plan.port} is already in use by another program.")

    os.makedirs(settings.LOGS_DIR, exist_ok=True)
    safe_model_name = validate_model_folder_name(instance.model_name)
    log_file_path = resolve_log_file_path(safe_model_name, plan.port)
    try:
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            process = _spawn_instance_process(instance, plan, log_file)
        _wait_until_healthy(instance, process, str(log_file_path))
        from orchestrator.instance_warmup import maybe_run_load_warmup

        maybe_run_load_warmup(instance)
        if _stop_requested(instance.pk):
            raise InstanceStartCancelled(START_CANCELLED_STOP_REQUESTED_MSG)
        _mark_instance_running(instance)
        return instance
    except InstanceStartCancelled:
        _stop_instance_locked(instance, desired_state=None)
        raise
    except Exception:
        _abort_failed_launch(instance)
        raise


def _spawn_instance_process(
    instance: InferenceInstance,
    plan: InstanceLaunchPlan,
    log_file: Any,
) -> subprocess.Popen:
    # Free on_demand LRU victims when unified memory is tight (MLX-112).
    from orchestrator.instance_eviction import ensure_headroom_for_load

    ensure_headroom_for_load(instance)
    server_config = dict(instance.server_config or {})
    cmd = _build_launch_command(
        plan.model_path,
        plan.port,
        plan.launch_mode,
        server_config,
        instance.model_name,
    )
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=_get_launch_env(
            plan.launch_mode,
            server_config,
            model_name=plan.requested_model_name,
        ),
    )
    instance.pid = process.pid
    instance.status = "LOADING"
    instance.save(update_fields=["pid", "status"])
    return process


def _stop_requested(instance_id: int) -> bool:
    return InferenceInstance.objects.filter(
        pk=instance_id,
        desired_state=InferenceInstance.DESIRED_STATE_STOPPED,
    ).exists()


def _wait_until_healthy(
    instance: InferenceInstance,
    process: subprocess.Popen,
    log_file_path: str,
) -> None:
    """Stay LOADING until /health answers; mlx-lm only serves it after weights load."""
    from orchestrator.instance_health import probe_http_health

    timeout_seconds = instance_ready_timeout_seconds()
    deadline = time.monotonic() + timeout_seconds
    while True:
        failure = _detect_log_failure(log_file_path)
        if failure:
            raise RuntimeError(f"Server failed to load model: {failure}")
        if process.poll() is not None:
            raise RuntimeError("Server failed to start: Process exited unexpectedly.")
        if _stop_requested(instance.pk):
            raise InstanceStartCancelled(START_CANCELLED_STOP_REQUESTED_MSG)
        if probe_http_health(instance):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Server did not answer /health within {int(timeout_seconds)} seconds."
            )
        time.sleep(_READY_POLL_SECONDS)


def _mark_instance_running(instance: InferenceInstance) -> None:
    # Seed activity so idle offload does not treat created_at / old stopped_at
    # as "idle for days" and stop a freshly started on_demand instance.
    instance.status = "RUNNING"
    instance.stopped_at = None
    instance.last_used_at = timezone.now()
    instance.save(update_fields=["status", "stopped_at", "last_used_at"])
    clear_gateway_route_cache()


def _abort_failed_launch(instance: InferenceInstance) -> None:
    """Kill whatever was spawned and leave the row FAILED (never stuck LOADING)."""
    if instance.pid:
        try:
            _stop_instance_locked(instance, desired_state=None)
        except Exception:
            logger.exception(
                "Cleanup after failed launch of %s on port %s failed",
                instance.model_name,
                instance.port,
            )
    _persist_stop_outcome(instance, "FAILED")


def _stop_port_release_timeout_seconds() -> float:
    """How long to wait for a TCP port to become free after stopping an instance."""
    raw = os.environ.get("MLX_STOP_PORT_RELEASE_TIMEOUT_SECONDS")
    if raw:
        try:
            return max(3.0, float(raw))
        except ValueError:
            pass
    return float(getattr(settings, "MLX_STOP_PORT_RELEASE_TIMEOUT_SECONDS", 12.0))


def is_stop_in_progress(instance: InferenceInstance) -> bool:
    """Return True while a stop is running (stale after a timeout if its actor died)."""
    since = instance.stopping_since
    if since is None:
        return False
    return (timezone.now() - since).total_seconds() < _STOP_IN_PROGRESS_STALE_SECONDS


def _ensure_port_released(port: int, timeout_seconds: float) -> list[int]:
    """Kill owned blockers and poll until the port is free or the timeout expires."""
    deadline = time.time() + timeout_seconds
    last_known: list[int] = []
    while time.time() < deadline:
        if is_port_free(port):
            return []
        blockers = sorted(_owned_port_pids(port))
        if blockers:
            last_known = blockers
            for pid in blockers:
                _terminate_pid(pid, signal.SIGKILL)
            time.sleep(0.4)
            continue
        time.sleep(0.25)
    if is_port_free(port):
        return []
    return sorted(_owned_port_pids(port)) or last_known


def _instance_log_path(instance: InferenceInstance) -> str:
    return os.path.join(settings.LOGS_DIR, f"{instance.model_name}_{instance.port}.log")


def check_instance_status(instance: InferenceInstance) -> str:
    """Reconcile DB status with the real process; never touch a slot mid-transition."""
    if instance.status not in _ACTIVE_INSTANCE_STATUSES:
        return instance.status
    if is_stop_in_progress(instance) or is_slot_lock_held_elsewhere(instance.port):
        return instance.status
    if not is_instance_process(instance.pid, instance.port):
        return _mark_process_gone(instance)
    if instance.status == "LOADING":
        return _settle_orphan_loading(instance)
    return instance.status


def _mark_process_gone(instance: InferenceInstance) -> str:
    """Record an exit nobody asked for: always_on → FAILED, sleepers → STOPPED."""
    from orchestrator.lifecycle_selectors import LIFECYCLE_MODE_ALWAYS_ON, get_lifecycle_mode

    failure = _detect_log_failure(_instance_log_path(instance))
    expected_running = (
        instance.desired_state == InferenceInstance.DESIRED_STATE_RUNNING
        and get_lifecycle_mode(instance.server_config) == LIFECYCLE_MODE_ALWAYS_ON
    )
    instance.status = "FAILED" if failure or expected_running else "STOPPED"
    if instance.status == "STOPPED":
        instance.desired_state = InferenceInstance.DESIRED_STATE_STOPPED
    instance.pid = None
    instance.stopped_at = timezone.now()
    instance.save(update_fields=["status", "desired_state", "pid", "stopped_at"])
    clear_gateway_route_cache()
    return instance.status


def _settle_orphan_loading(instance: InferenceInstance) -> str:
    """Finish a LOADING row whose starter died (e.g. runserver reload mid-start)."""
    from orchestrator.instance_health import probe_http_health
    from orchestrator.instance_locks import InstanceSlotBusy

    if _detect_log_failure(_instance_log_path(instance)):
        try:
            stop_instance(instance, blocking=False)
        except InstanceSlotBusy:
            return instance.status
        _persist_stop_outcome(instance, "FAILED")
        return "FAILED"
    if probe_http_health(instance):
        _mark_instance_running(instance)
    return instance.status


def _persist_stop_outcome(instance: InferenceInstance, status: str) -> None:
    instance.status = status
    instance.pid = None
    instance.stopped_at = timezone.now()
    instance.stopping_since = None
    instance.save(update_fields=["status", "pid", "stopped_at", "stopping_since"])
    clear_gateway_route_cache()


def stop_instance(
    instance: InferenceInstance,
    *,
    desired_state: str | None = None,
    blocking: bool = True,
    timeout_seconds: float = DEFAULT_SLOT_LOCK_TIMEOUT_SECONDS,
) -> None:
    """Terminate the inference server and ensure the port is released.

    Args:
        instance: Row to stop.
        desired_state: Intent to persist (``stopped`` for user / idle / eviction
            stops); ``None`` for internal stops (restart, failed launch).
        blocking: Wait for an in-flight start/stop on the slot instead of failing.
        timeout_seconds: Max wait for the slot when ``blocking``.

    Raises:
        InstanceSlotBusy: the slot could not be acquired in time.
        RuntimeError: the port could not be released (row is left FAILED).
    """
    with instance_slot_lock(instance.port, blocking=blocking, timeout_seconds=timeout_seconds):
        _stop_instance_locked(instance, desired_state=desired_state)


def _stop_instance_locked(instance: InferenceInstance, *, desired_state: str | None) -> None:
    instance.refresh_from_db()
    instance.stopping_since = timezone.now()
    update_fields = ["stopping_since"]
    if desired_state:
        instance.desired_state = desired_state
        update_fields.append("desired_state")
    instance.save(update_fields=update_fields)
    try:
        _force_stop_pids(_collect_stop_targets(instance), grace_seconds=3.0)
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
        _persist_stop_outcome(instance, "STOPPED")
    except Exception:
        # Process may already be dead while status still says RUNNING; fail closed.
        _persist_stop_outcome(instance, "FAILED")
        raise


def request_instance_stop(instance: InferenceInstance) -> bool:
    """Record a user stop; stop now, or leave it to the in-flight start / watchdog.

    The persisted ``desired_state=stopped`` is the queue: an in-flight start
    cancels itself, and the watchdog converges any active row still meant to
    be stopped once the slot is free.

    Returns:
        True when the instance was stopped synchronously, False when the stop
        was queued behind another transition on the same slot.
    """
    from orchestrator.instance_locks import InstanceSlotBusy

    InferenceInstance.objects.filter(pk=instance.pk).update(
        desired_state=InferenceInstance.DESIRED_STATE_STOPPED,
    )
    try:
        stop_instance(
            instance,
            desired_state=InferenceInstance.DESIRED_STATE_STOPPED,
            timeout_seconds=USER_ACTION_SLOT_WAIT_SECONDS,
        )
    except InstanceSlotBusy:
        return False
    return True


def delete_instance(
    instance: InferenceInstance,
    *,
    timeout_seconds: float = DEFAULT_SLOT_LOCK_TIMEOUT_SECONDS,
) -> None:
    """Stop a running instance if needed, then remove its database record."""
    with instance_slot_lock(instance.port, timeout_seconds=timeout_seconds):
        instance.refresh_from_db()
        if instance.status in _ACTIVE_INSTANCE_STATUSES:
            stop_instance(instance, desired_state=InferenceInstance.DESIRED_STATE_STOPPED)
        instance.delete()


def update_stopped_instance(
    instance: InferenceInstance,
    *,
    port: int | None = None,
    launch_mode: LaunchMode | None = None,
    server_config: dict[str, Any] | None = None,
) -> InferenceInstance:
    """Update port, mode, or config for a stopped or failed instance.

    Holds the slot lock of both the current and the target port so a
    concurrent start on the new port cannot race the move.
    """
    target_port = int(port) if port is not None else instance.port
    with instance_slot_lock(instance.port, timeout_seconds=USER_ACTION_SLOT_WAIT_SECONDS):
        with instance_slot_lock(target_port, timeout_seconds=USER_ACTION_SLOT_WAIT_SECONDS):
            instance.refresh_from_db()
            return _update_stopped_instance_locked(
                instance,
                port=port,
                launch_mode=launch_mode,
                server_config=server_config,
            )


def _update_stopped_instance_locked(
    instance: InferenceInstance,
    *,
    port: int | None,
    launch_mode: LaunchMode | None,
    server_config: dict[str, Any] | None,
) -> InferenceInstance:
    if instance.status in _ACTIVE_INSTANCE_STATUSES:
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
    clear_gateway_route_cache()
    return instance


def restart_instance(
    instance: InferenceInstance,
    *,
    automatic: bool = False,
    wait_for_slot: bool = True,
) -> InferenceInstance:
    """Stop (if active) and relaunch with the same configuration, atomically per slot.

    Args:
        instance: Row to restart.
        automatic: Watchdog / startup reconcile caller. Skips rows whose
            ``desired_state`` is ``stopped`` (re-read under the lock) and never
            flips that intent back to ``running``.
        wait_for_slot: Wait for an in-flight transition instead of failing.
    """
    model_name = validate_model_folder_name(instance.model_name)
    port = validated_launch_port(instance.port)
    launch_mode = parse_launch_mode(instance.launch_mode)

    with instance_slot_lock(port, blocking=wait_for_slot):
        instance.refresh_from_db()
        if automatic and instance.desired_state != InferenceInstance.DESIRED_STATE_RUNNING:
            return instance
        if instance.status in _ACTIVE_INSTANCE_STATUSES:
            _stop_instance_locked(instance, desired_state=None)
        plan = prepare_instance_launch(
            model_name,
            port,
            launch_mode,
            dict(instance.server_config or {}),
            mark_desired_running=not automatic,
        )
        return launch_prepared_instance(plan)


def get_instance_logs(model_name: str, port: int) -> str:
    """Read the last 500 lines of logs for the given instance."""
    try:
        safe_name = validate_model_folder_name(model_name)
        safe_port = validated_launch_port(port)
        log_file_path = resolve_log_file_path(safe_name, safe_port)
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
