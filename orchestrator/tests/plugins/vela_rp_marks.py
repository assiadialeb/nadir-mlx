"""Pytest plugin: register ReportPortal ticket / ac markers (SPCTR-159)."""

from __future__ import annotations

from typing import Any

from orchestrator.rp_marks import (  # noqa: F401
    attributes_from_item_marks,
    build_ci_launch_attributes,
    is_valid_ac_mark_value,
)


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        "ticket(issue_key): ReportPortal attribute ticket=<issue_key>",
    )
    config.addinivalue_line(
        "markers",
        'ac("KEY:ACn"): ReportPortal attribute ac=<KEY:ACn> (acceptance criterion)',
    )
