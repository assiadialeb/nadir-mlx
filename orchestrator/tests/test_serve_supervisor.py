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
            with pytest.raises(SystemExit, match="already in use"):
                assert_serve_ports_free(spec)
