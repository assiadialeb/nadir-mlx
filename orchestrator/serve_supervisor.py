"""Process supervisor helpers for ``manage.py serve`` (UI + gateway [+ MCP])."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ServeSpec:
    """Resolved bind targets for a local Nadir stack."""

    ui_host: str
    ui_port: int
    with_mcp: bool
    mcp_host: str
    mcp_port: int
    project_root: Path


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


def _terminate_process_group(process: subprocess.Popen[bytes], *, grace_seconds: float = 5.0) -> None:
    """SIGTERM the process group, then SIGKILL if needed."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


def stop_children(processes: Sequence[subprocess.Popen[bytes]], *, grace_seconds: float = 5.0) -> None:
    """Stop all supervised children (gateway / UI / MCP)."""
    for process in processes:
        _terminate_process_group(process, grace_seconds=grace_seconds)
    for process in processes:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def start_children(commands: Sequence[Sequence[str]], *, env: dict[str, str] | None = None) -> list[subprocess.Popen[bytes]]:
    """Start each command in its own process group and return the Popen handles."""
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    processes: list[subprocess.Popen[bytes]] = []
    for argv in commands:
        process = subprocess.Popen(
            list(argv),
            env=child_env,
            start_new_session=True,
        )
        processes.append(process)
    return processes


def wait_for_exit(processes: Sequence[subprocess.Popen[bytes]]) -> int:
    """Block until one child exits; return that exit code (or 1)."""
    while True:
        for process in processes:
            code = process.poll()
            if code is not None:
                return int(code)
        time.sleep(0.4)
