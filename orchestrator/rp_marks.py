"""ReportPortal pytest mark helpers shared across the Vela suite (SPCTR-159).

pytest-reportportal maps ``@pytest.mark.<name>("value")`` to launch/item attributes
``name:value``. This module documents the contract and validates ``ac`` values.

It also resolves the CI launch name and attributes (SPCTR-311). Run it as a
script from a workflow step (no third-party imports, works before
``pip install``)::

    python3 <path>/rp_marks.py   # reads CI env vars, appends to $GITHUB_OUTPUT
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Mapping

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


# Fixed ReportPortal launch names, one per CI event (SPCTR-311). The RP project
# is already per product, so no product prefix. Branch, tag, PR and SHA live in
# the launch attributes, never in the name, so RP filters can group runs.
LAUNCH_PULL_REQUEST = "pull request"
LAUNCH_MERGE = "merge"
LAUNCH_COMMIT = "commit"
LAUNCH_TAG = "tag"
LAUNCH_RELEASE = "release"
LAUNCH_MANUAL = "manual"
CI_LAUNCH_NAMES = (
    LAUNCH_PULL_REQUEST,
    LAUNCH_MERGE,
    LAUNCH_COMMIT,
    LAUNCH_TAG,
    LAUNCH_RELEASE,
    LAUNCH_MANUAL,
)

_PULL_REQUEST_EVENTS = frozenset({"pull_request", "pull_request_target"})


def is_ci_launch_name(name: str | None) -> bool:
    """True when ``name`` is one of the fixed CI launch names (case-insensitive)."""
    return (name or "").strip().lower() in CI_LAUNCH_NAMES


def resolve_ci_launch_name(
    *,
    event_name: str,
    git_ref: str,
    ref_name: str,
    default_branch: str = "main",
    override: str = "",
) -> str:
    """Return the fixed ReportPortal launch name for a CI event (SPCTR-311).

    A non-empty manual ``override`` (workflow_dispatch input) wins. Otherwise:
    pull request -> ``pull request``; release -> ``release``; manual dispatch ->
    ``manual``; tag push -> ``tag``; push on the default branch -> ``merge``;
    any other push (or unknown event) -> ``commit``.
    """
    manual_name = " ".join((override or "").split())
    if manual_name:
        return manual_name
    event = (event_name or "").strip()
    if event in _PULL_REQUEST_EVENTS:
        return LAUNCH_PULL_REQUEST
    if event == "release":
        return LAUNCH_RELEASE
    if event == "workflow_dispatch":
        return LAUNCH_MANUAL
    if (git_ref or "").startswith("refs/tags/"):
        return LAUNCH_TAG
    if event == "push" and (ref_name or "").strip() == (default_branch or "main").strip():
        return LAUNCH_MERGE
    return LAUNCH_COMMIT


def resolve_ci_launch(env: Mapping[str, str]) -> tuple[str, str]:
    """Return ``(name, attributes)`` from CI environment variables.

    Reads ``EVENT_NAME``, ``GIT_REF``, ``REF_NAME``, ``HEAD_REF``, ``COMMIT_SHA``,
    ``RELEASE_TAG``, ``DEFAULT_BRANCH``, ``LAUNCH_OVERRIDE`` and ``CI_NAME``.
    """
    event_name = env.get("EVENT_NAME", "")
    git_ref = env.get("GIT_REF", "")
    ref_name = env.get("REF_NAME", "")
    name = resolve_ci_launch_name(
        event_name=event_name,
        git_ref=git_ref,
        ref_name=ref_name,
        default_branch=env.get("DEFAULT_BRANCH", "") or "main",
        override=env.get("LAUNCH_OVERRIDE", ""),
    )
    attributes = build_ci_launch_attributes(
        event_name=event_name,
        git_ref=git_ref,
        ref_name=ref_name,
        head_ref=env.get("HEAD_REF", ""),
        commit_sha=env.get("COMMIT_SHA", ""),
        release_tag=env.get("RELEASE_TAG", ""),
        ci_name=env.get("CI_NAME", "") or "github",
    )
    return name, attributes


def main(env: Mapping[str, str] | None = None) -> int:
    """CI entry point: print the launch name/attributes and export them as step outputs."""
    env = os.environ if env is None else env
    name, attributes = resolve_ci_launch(env)
    print(f"Resolved ReportPortal launch name: {name}")
    print(f"Resolved ReportPortal launch attributes: {attributes}")
    output_path = env.get("GITHUB_OUTPUT", "")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(f"name={name}\nattributes={attributes}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
