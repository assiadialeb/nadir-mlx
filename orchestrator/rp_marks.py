"""ReportPortal pytest mark helpers shared across the Vela suite (SPCTR-159).

pytest-reportportal maps ``@pytest.mark.<name>("value")`` to launch/item attributes
``name:value``. This module documents the contract and validates ``ac`` values.
"""

from __future__ import annotations

import re
from typing import Any

AC_MARK_PATTERN = re.compile(
    r"^[A-Z][A-Z0-9]*-\d+:AC\d+$",
    flags=re.IGNORECASE,
)


def is_valid_ac_mark_value(value: str) -> bool:
    """Return True when ``value`` matches ``KEY:ACn`` (e.g. ``SPCTR-159:AC1``)."""
    return bool(AC_MARK_PATTERN.fullmatch(str(value or "").strip()))


def attributes_from_item_marks(item: Any) -> list[tuple[str, str]]:
    """Collect ``(key, value)`` pairs from ``ticket`` / ``ac`` marks on a pytest item."""
    pairs: list[tuple[str, str]] = []
    for mark_name in ("ticket", "ac"):
        for mark in item.iter_markers(name=mark_name):
            if not mark.args:
                continue
            value = str(mark.args[0]).strip()
            if value:
                pairs.append((mark_name, value))
    return pairs


def build_ci_launch_attributes(
    *,
    event_name: str,
    git_ref: str,
    ref_name: str,
    head_ref: str,
    commit_sha: str,
    release_tag: str = "",
    ci_name: str = "github",
) -> str:
    """Build the ``RP_LAUNCH_ATTRIBUTES`` semicolon list for CI (SPCTR-159).

    On tag or GitHub ``release`` events, appends ``tag`` and ``release``.
    """
    branch_ref = (head_ref or ref_name or "").strip()
    attrs = [
        f"ci:{ci_name}",
        f"ref:{branch_ref}",
        f"branch:{branch_ref}",
        f"event:{event_name}",
        f"commit:{commit_sha}",
    ]
    tag_value = ""
    if event_name == "release" and release_tag.strip():
        tag_value = release_tag.strip()
    elif git_ref.startswith("refs/tags/"):
        tag_value = ref_name.strip()
    if tag_value:
        attrs.append(f"tag:{tag_value}")
        attrs.append(f"release:{tag_value}")
    return ";".join(attrs)
