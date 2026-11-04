"""Process supervisor helpers for ``manage.py serve`` (UI + gateway [+ MCP])."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Sequence

BACKGROUND_OWNER_ENV = "NADIR_BACKGROUND_OWNER"
SERVE_PARENT_PID_ENV = "NADIR_SERVE_PARENT_PID"


@dataclass(frozen=True)
class ServeSpec:
    """Resolved bind targets for a local Nadir stack."""

    ui_host: str
    ui_port: int
    with_mcp: bool
    mcp_host: str
    mcp_port: int
    project_root: Path


@dataclass(frozen=True)
class BoundService:
    """A named listen address checked before starting children."""

    label: str
    host: str
    port: int


def default_serve_spec(*, with_mcp: bool = False) -> ServeSpec:
    """Build a ServeSpec from environment variables."""
    ui_host = (os.environ.get("DJANGO_BIND_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    ui_port = int((os.environ.get("DJANGO_HTTP_PORT") or "8000").strip() or "8000")
    mcp_host = (os.environ.get("NADIR_MCP_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    mcp_port = int((os.environ.get("NADIR_MCP_PORT") or "11381").strip() or "11381")
    return ServeSpec(
        ui_host=ui_host,
        ui_port=ui_port,
        with_mcp=with_mcp,
        mcp_host=mcp_host,
        mcp_port=mcp_port,
        project_root=Path(__file__).resolve().parents[1],
    )


def gateway_bind_from_env() -> tuple[str, int]:
    """Return gateway host/port from environment (defaults match settings)."""
    host = (os.environ.get("NADIR_GATEWAY_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    port = int((os.environ.get("NADIR_GATEWAY_PORT") or "11380").strip() or "11380")
    return host, port


def required_bind_services(spec: ServeSpec) -> list[BoundService]:
    """Ports that must be free before ``serve`` starts children."""
    gateway_host, gateway_port = gateway_bind_from_env()
    services = [
        BoundService("UI", spec.ui_host, spec.ui_port),
        BoundService("Gateway", gateway_host, gateway_port),
    ]
    if spec.with_mcp:
        services.append(BoundService("MCP", spec.mcp_host, spec.mcp_port))
    return services


def _port_in_use(host: str, port: int) -> bool:
    """Return True when a real listener blocks ``host:port``.

    Uses ``SO_REUSEADDR`` like Django/uvicorn so a leftover TIME_WAIT socket
    after Ctrl+C is not reported as ``already in use``.
    """
    family = socket.AF_INET6 if ":" in host and not host.startswith(":") else socket.AF_INET
    bind_host = host
    if host in {"0.0.0.0", "::"}:
        bind_host = "127.0.0.1" if family == socket.AF_INET else "::1"
        family = socket.AF_INET if bind_host == "127.0.0.1" else socket.AF_INET6
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, port))
            return False
        except OSError:
            return True


def _listener_pids(port: int) -> list[int]:
    """Best-effort PIDs listening on a TCP port (macOS/Linux via lsof)."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except subprocess.TimeoutExpired:
        return []
    except OSError:
        return []
    pids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def assert_serve_ports_free(spec: ServeSpec) -> None:
    """Raise SystemExit with a clear message when a serve port is occupied."""
    for service in required_bind_services(spec):
        if not _port_in_use(service.host, service.port):
            continue
        pids = _listener_pids(service.port)
        if pids:
            details = ", ".join(
                f"{pid} ({_process_command(pid)[:80] or 'unknown'})" for pid in pids[:3]
            )
            pid_hint = f" — listener(s): {details}"
            kill_hint = f" Try: kill {' '.join(str(pid) for pid in pids)}"
        else:
            pid_hint = ""
            kill_hint = ""
        raise SystemExit(
            f"Port {service.port} ({service.label}) is already in use{pid_hint}. "
            "Nadir seems already running — stop the existing process or change the port."
            f"{kill_hint}"
        )


def build_child_commands(spec: ServeSpec, python_executable: str | None = None) -> list[list[str]]:
    """Return argv lists for gateway, UI, and optional MCP children."""
    python = python_executable or sys.executable
    manage = str(spec.project_root / "manage.py")
    commands = [
        [python, manage, "run_gateway"],
        [python, manage, "runserver", f"{spec.ui_host}:{spec.ui_port}", "--noreload"],
    ]
    if spec.with_mcp:
        commands.append(
            [
                python,
                manage,
                "run_mcp",
                "--host",
                spec.mcp_host,
                "--port",
                str(spec.mcp_port),
            ]
        )
    return commands


def supervised_child_env(
    extra: dict[str, str] | None = None,
    *,
    parent_pid: int | None = None,
) -> dict[str, str]:
    """Environment for serve children: one background owner + parent death watch."""
    env = os.environ.copy()
    if extra:
        env.update(extra)
    env[BACKGROUND_OWNER_ENV] = "gateway"
    env[SERVE_PARENT_PID_ENV] = str(parent_pid if parent_pid is not None else os.getpid())
    return env


_parent_watch_started = False
_parent_watch_lock = threading.Lock()


def start_parent_death_watch_if_needed() -> None:
    """Exit this process if the supervising ``serve`` parent disappears."""
    global _parent_watch_started
    raw = (os.environ.get(SERVE_PARENT_PID_ENV) or "").strip()
    if not raw:
        return
    try:
        parent_pid = int(raw)
    except ValueError:
        return
    if parent_pid <= 0 or parent_pid == os.getpid():
        return
    with _parent_watch_lock:
        if _parent_watch_started:
            return
        _parent_watch_started = True

        def _watch() -> None:
            while True:
                if os.getppid() != parent_pid:
                    os._exit(1)
                time.sleep(1.0)

        threading.Thread(target=_watch, name="nadir-parent-death-watch", daemon=True).start()


def _safe_signal_pid(pid: int, sig: signal.Signals) -> bool:
    """Send ``sig`` to ``pid``; return False when the process is gone or denied."""
    try:
        os.kill(pid, sig)
        return True
    except OSError:
        return False


def _signal_process_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    """Deliver ``sig`` to the child's process group (fallback: the child only)."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
        return
    except OSError:
        pass
    try:
        # Negative PID = process group; mirrors killpg when setsid() was used.
        os.kill(-process.pid, sig)
        return
    except OSError:
        pass
    try:
        process.send_signal(sig)
    except OSError:
        pass


def _terminate_process_group(process: subprocess.Popen[bytes], *, grace_seconds: float = 5.0) -> None:
    """SIGTERM the process group, then SIGKILL if needed."""
    if process.poll() is not None:
        return
    _signal_process_group(process, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)
    _signal_process_group(process, signal.SIGKILL)
    try:
        process.kill()
    except OSError:
        pass


def _process_command(pid: int) -> str:
    """Best-effort process command line (lowercased)."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except subprocess.TimeoutExpired:
        return ""
    except OSError:
        return ""
    return (result.stdout or "").strip().lower()


def _process_ppid(pid: int) -> int | None:
    """Return parent PID for ``pid``, or None when unknown."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "ppid="],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except subprocess.TimeoutExpired:
        return None
    except OSError:
        return None
    raw = (result.stdout or "").strip()
    if not raw.isdigit():
        return None
    return int(raw)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _is_reclaimable_nadir_orphan(pid: int, project_root: Path) -> bool:
    """True only for orphaned listeners of *this* project's manage.py.

    Refuses to signal other Django apps (or a healthy second ``serve``) that
    happen to share port 8000 / 11380.
    """
    command = _process_command(pid)
    if not command:
        return False
    manage_path = str((project_root / "manage.py").resolve()).lower()
    if manage_path not in command:
        return False
    if not any(token in command for token in ("runserver", "run_gateway", "run_mcp")):
        return False
    ppid = _process_ppid(pid)
    # After Ctrl+C of the supervisor, children are typically reparented to 1.
    if ppid in (None, 0, 1):
        return True
    return not _pid_is_alive(ppid)


def _wait_until_port_free(host: str, port: int, *, timeout_seconds: float = 2.0) -> bool:
    """Poll until the port is free or ``timeout_seconds`` elapses."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _port_in_use(host, port):
            return True
        time.sleep(0.1)
    return not _port_in_use(host, port)


def _signal_nadir_orphans(
    port: int,
    sig: signal.Signals,
    *,
    project_root: Path,
    reclaimed: list[int],
    skip_reclaimed: bool,
) -> None:
    """Signal reclaimable Nadir orphans on ``port`` and track reclaimed PIDs."""
    for pid in _listener_pids(port):
        if skip_reclaimed and pid in reclaimed:
            continue
        if not _is_reclaimable_nadir_orphan(pid, project_root):
            continue
        if not _safe_signal_pid(pid, sig):
            continue
        if pid not in reclaimed:
            reclaimed.append(pid)


def _reclaim_port_listeners(
    host: str,
    port: int,
    *,
    project_root: Path,
    reclaimed: list[int],
) -> None:
    """SIGTERM then SIGKILL orphaned Nadir listeners still holding ``port``."""
    _signal_nadir_orphans(
        port,
        signal.SIGTERM,
        project_root=project_root,
        reclaimed=reclaimed,
        skip_reclaimed=True,
    )
    if _wait_until_port_free(host, port):
        return
    _signal_nadir_orphans(
        port,
        signal.SIGKILL,
        project_root=project_root,
        reclaimed=reclaimed,
        skip_reclaimed=False,
    )


def reclaim_stale_serve_ports(spec: ServeSpec) -> list[int]:
    """SIGTERM/SIGKILL leftover listeners on serve ports (orphans after bad Ctrl+C)."""
    reclaimed: list[int] = []
    for service in required_bind_services(spec):
        if not _port_in_use(service.host, service.port):
            continue
        _reclaim_port_listeners(
            service.host,
            service.port,
            project_root=spec.project_root,
            reclaimed=reclaimed,
        )
    return reclaimed


def stop_children(
    processes: Sequence[subprocess.Popen[bytes]],
    *,
    grace_seconds: float = 5.0,
    spec: ServeSpec | None = None,
) -> None:
    """Stop all supervised children (gateway / UI / MCP) and free their ports."""
    for process in processes:
        _terminate_process_group(process, grace_seconds=grace_seconds)
    for process in processes:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    if spec is not None:
        reclaim_stale_serve_ports(spec)


def start_children(commands: Sequence[Sequence[str]], *, env: dict[str, str] | None = None) -> list[subprocess.Popen[bytes]]:
    """Start each command in its own process group and return the Popen handles."""
    child_env = supervised_child_env(env)
    processes: list[subprocess.Popen[bytes]] = []
    for argv in commands:
        process = subprocess.Popen(
            list(argv),
            env=child_env,
            start_new_session=True,
        )
        processes.append(process)
    return processes


def wait_for_exit(
    processes: Sequence[subprocess.Popen[bytes]],
    *,
    should_stop: Callable[[], bool] | None = None,
) -> int | None:
    """Block until one child exits or ``should_stop`` is true.

    Returns the child exit code, or ``None`` when stop was requested.
    """
    while True:
        if should_stop is not None and should_stop():
            return None
        for process in processes:
            code = process.poll()
            if code is not None:
                return int(code)
        time.sleep(0.25)
