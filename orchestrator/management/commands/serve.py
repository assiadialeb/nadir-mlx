"""Start the full local Nadir stack (UI + gateway, optional MCP) in one command."""

from __future__ import annotations

import signal
from dataclasses import replace
from typing import Any

from django.core.management.base import BaseCommand

from orchestrator.serve_supervisor import (
    assert_serve_ports_free,
    build_child_commands,
    default_serve_spec,
    gateway_bind_from_env,
    start_children,
    stop_children,
    wait_for_exit,
)


class Command(BaseCommand):
    help = (
        "Start Django UI + Nadir Gateway together (optional MCP). "
        "Ctrl+C stops all supervised processes."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--with-mcp",
            action="store_true",
            help="Also start manage.py run_mcp (loopback Streamable HTTP MCP).",
        )
        parser.add_argument(
            "--ui-host",
            default=None,
            help="Django bind host (default DJANGO_BIND_HOST or 127.0.0.1).",
        )
        parser.add_argument(
            "--ui-port",
            type=int,
            default=None,
            help="Django bind port (default DJANGO_HTTP_PORT or 8000).",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        spec = default_serve_spec(with_mcp=bool(options["with_mcp"]))
        if options["ui_host"]:
            spec = replace(spec, ui_host=str(options["ui_host"]).strip() or spec.ui_host)
        if options["ui_port"] is not None:
            spec = replace(spec, ui_port=int(options["ui_port"]))

        assert_serve_ports_free(spec)
        commands = build_child_commands(spec)
        gateway_host, gateway_port = gateway_bind_from_env()
        self.stdout.write(self.style.SUCCESS("Starting Nadir stack…"))
        self.stdout.write(f"  UI:      http://{spec.ui_host}:{spec.ui_port}/")
        self.stdout.write(f"  Gateway: http://{gateway_host}:{gateway_port}/v1")
        if spec.with_mcp:
            self.stdout.write(
                f"  MCP:     http://{spec.mcp_host}:{spec.mcp_port}/mcp"
            )
        else:
            self.stdout.write("  MCP:     skipped (pass --with-mcp to enable)")

        processes = start_children(commands)
        stopping = False

        def _shutdown(signum: int, _frame: Any) -> None:
            nonlocal stopping
            if stopping:
                return
            stopping = True
            self.stdout.write(self.style.WARNING(f"\nSignal {signum}: stopping children…"))
            stop_children(processes)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        try:
            exit_code = wait_for_exit(processes)
            if not stopping:
                self.stdout.write(
                    self.style.ERROR(
                        f"A supervised process exited (code {exit_code}); shutting down."
                    )
                )
                stop_children(processes)
            if exit_code not in (0, None) and not stopping:
                raise SystemExit(exit_code)
        finally:
            stop_children(processes)
