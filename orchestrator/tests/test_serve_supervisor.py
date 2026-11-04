"""Unit tests for manage.py serve supervisor helpers."""

from __future__ import annotations

from pathlib import Path

from orchestrator.serve_supervisor import ServeSpec, build_child_commands, default_serve_spec


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
