"""Unit tests for ReportPortal mark helpers (SPCTR-159)."""

from __future__ import annotations

import pytest

from orchestrator.rp_marks import (
    attributes_from_item_marks,
    build_ci_launch_attributes,
    is_valid_ac_mark_value,
)


def test_is_valid_ac_mark_value_accepts_canonical_form() -> None:
    assert is_valid_ac_mark_value("SPCTR-159:AC1")
    assert is_valid_ac_mark_value("grvt-118:AC12")


def test_is_valid_ac_mark_value_rejects_malformed() -> None:
    assert not is_valid_ac_mark_value("")
    assert not is_valid_ac_mark_value("SPCTR-159")
    assert not is_valid_ac_mark_value("SPCTR-159:AC")


def test_build_ci_launch_attributes_adds_tag_and_release_on_tag_ref() -> None:
    attrs = build_ci_launch_attributes(
        event_name="push",
        git_ref="refs/tags/v0.5.0",
        ref_name="v0.5.0",
        head_ref="",
        commit_sha="abc123",
    )
    assert "tag:v0.5.0" in attrs
    assert "release:v0.5.0" in attrs


def test_build_ci_launch_attributes_omits_tag_on_pull_request() -> None:
    attrs = build_ci_launch_attributes(
        event_name="pull_request",
        git_ref="refs/pull/1/merge",
        ref_name="1/merge",
        head_ref="feat/demo",
        commit_sha="prsha",
    )
    assert "branch:feat/demo" in attrs
    assert "tag:" not in attrs


@pytest.mark.ticket("SPCTR-159")
@pytest.mark.ac("SPCTR-159:AC2")
@pytest.mark.ac("SPCTR-159:AC3")
def test_ac_and_ticket_marks_are_visible_on_the_test_node(request: pytest.FixtureRequest) -> None:
    pairs = attributes_from_item_marks(request.node)
    assert ("ticket", "SPCTR-159") in pairs
    assert ("ac", "SPCTR-159:AC2") in pairs
    assert ("ac", "SPCTR-159:AC3") in pairs
