"""Unit tests for manage.py serve supervisor helpers."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.serve_supervisor import (
    BACKGROUND_OWNER_ENV,
    SERVE_PARENT_PID_ENV,
    ServeSpec,
    assert_serve_ports_free,
    build_child_commands,
    default_serve_spec,
    gateway_bind_from_env,
    required_bind_services,
    supervised_child_env,
)


def test_default_serve_spec_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("DJANGO_BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("DJANGO_HTTP_PORT", "8001")
    monkeypatch.setenv("NADIR_MCP_HOST", "127.0.0.1")
    monkeypatch.setenv("NADIR_MCP_PORT", "11382")
    spec = default_serve_spec(with_mcp=True)
    assert spec.ui_host == "127.0.0.1"
    assert spec.ui_port == 8001
    assert spec.with_mcp is True
    assert spec.mcp_port == 11382
    assert (spec.project_root / "manage.py").is_file()


def test_build_child_commands_without_mcp() -> None:
    spec = ServeSpec(
        ui_host="127.0.0.1",
        ui_port=8000,
        with_mcp=False,
        mcp_host="127.0.0.1",
        mcp_port=11381,
        project_root=Path("/tmp/nadir"),
    )
    commands = build_child_commands(spec, python_executable="/usr/bin/python3")
    assert len(commands) == 2
    assert commands[0][-1] == "run_gateway"
    assert "runserver" in commands[1]
    assert "127.0.0.1:8000" in commands[1]
    assert "--noreload" in commands[1]


def test_build_child_commands_with_mcp() -> None:
    spec = ServeSpec(
        ui_host="127.0.0.1",
        ui_port=8000,
        with_mcp=True,
        mcp_host="127.0.0.1",
        mcp_port=11381,
        project_root=Path("/tmp/nadir"),
    )
    commands = build_child_commands(spec, python_executable="/usr/bin/python3")
    assert len(commands) == 3
    assert commands[2][2] == "run_mcp"
    assert commands[2][-2:] == ["--port", "11381"]


def test_supervised_child_env_sets_owner_and_parent_pid(monkeypatch) -> None:
    monkeypatch.delenv(BACKGROUND_OWNER_ENV, raising=False)
    monkeypatch.delenv(SERVE_PARENT_PID_ENV, raising=False)
    env = supervised_child_env(parent_pid=4242)
    assert env[BACKGROUND_OWNER_ENV] == "gateway"
    assert env[SERVE_PARENT_PID_ENV] == "4242"
    assert "PATH" in env


def test_required_bind_services_includes_mcp_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("NADIR_GATEWAY_HOST", "127.0.0.1")
    monkeypatch.setenv("NADIR_GATEWAY_PORT", "11380")
    spec = ServeSpec(
        ui_host="127.0.0.1",
        ui_port=8000,
        with_mcp=True,
        mcp_host="127.0.0.1",
        mcp_port=11381,
        project_root=Path("/tmp/nadir"),
    )
    labels = [service.label for service in required_bind_services(spec)]
    assert labels == ["UI", "Gateway", "MCP"]
    host, port = gateway_bind_from_env()
    assert host == "127.0.0.1"
    assert port == 11380


def test_assert_serve_ports_free_raises_when_busy() -> None:
    spec = ServeSpec(
        ui_host="127.0.0.1",
        ui_port=8000,
        with_mcp=False,
        mcp_host="127.0.0.1",
        mcp_port=11381,
        project_root=Path("/tmp/nadir"),
    )
    with patch("orchestrator.serve_supervisor._port_in_use", return_value=True):
        with patch("orchestrator.serve_supervisor._listener_pids", return_value=[999]):
            with patch(
                "orchestrator.serve_supervisor._process_command",
                return_value="python manage.py runserver",
            ):
                with pytest.raises(SystemExit, match="already in use"):
                    assert_serve_ports_free(spec)


def test_port_in_use_allows_bind_with_reuseaddr_after_busy_connect() -> None:
    """TIME_WAIT must not look like a live listener (servers use SO_REUSEADDR)."""
    from orchestrator.serve_supervisor import _port_in_use

    import socket as socket_mod

    server = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
    server.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.listen(1)
    client = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    server.accept()
    client.close()
    server.close()
    # Port may be in TIME_WAIT; our check should still report free.
    assert _port_in_use("127.0.0.1", port) is False


def test_reclaim_stale_serve_ports_signals_manage_py_listeners() -> None:
    from orchestrator.serve_supervisor import reclaim_stale_serve_ports

    spec = ServeSpec(
        ui_host="127.0.0.1",
        ui_port=8000,
        with_mcp=False,
        mcp_host="127.0.0.1",
        mcp_port=11381,
        project_root=Path("/tmp/nadir"),
    )
    busy_ports = {8000}

    def port_in_use(host: str, port: int) -> bool:
        del host
        return port in busy_ports

    def kill_pid(pid: int, sig: int) -> None:
        del sig
        if pid == 4242:
            busy_ports.discard(8000)

    with patch("orchestrator.serve_supervisor._port_in_use", side_effect=port_in_use):
        with patch("orchestrator.serve_supervisor._listener_pids", return_value=[4242]):
            with patch(
                "orchestrator.serve_supervisor._is_reclaimable_nadir_orphan",
                return_value=True,
            ):
                with patch("orchestrator.serve_supervisor.os.kill", side_effect=kill_pid) as mock_kill:
                    reclaimed = reclaim_stale_serve_ports(spec)
    assert reclaimed == [4242]
    assert mock_kill.called
    assert 8000 not in busy_ports


def test_is_reclaimable_requires_this_manage_py_and_dead_parent(tmp_path: Path) -> None:
    from orchestrator.serve_supervisor import _is_reclaimable_nadir_orphan

    manage = tmp_path / "manage.py"
    manage.write_text("# stub\n", encoding="utf-8")
    command = f"python {manage.resolve()} runserver 127.0.0.1:8000 --noreload"
    with patch(
        "orchestrator.serve_supervisor._process_command",
        return_value=command.lower(),
    ):
        with patch("orchestrator.serve_supervisor._process_ppid", return_value=1):
            assert _is_reclaimable_nadir_orphan(99, tmp_path) is True
        with patch("orchestrator.serve_supervisor._process_ppid", return_value=4242):
            with patch("orchestrator.serve_supervisor._pid_is_alive", return_value=True):
                assert _is_reclaimable_nadir_orphan(99, tmp_path) is False
    other = f"python /other/project/manage.py runserver 127.0.0.1:8000"
    with patch("orchestrator.serve_supervisor._process_command", return_value=other):
        with patch("orchestrator.serve_supervisor._process_ppid", return_value=1):
            assert _is_reclaimable_nadir_orphan(99, tmp_path) is False


def test_wait_for_exit_returns_none_when_stop_requested() -> None:
    from orchestrator.serve_supervisor import wait_for_exit

    class _FakeProc:
        def poll(self):
            return None

    assert wait_for_exit([_FakeProc()], should_stop=lambda: True) is None
