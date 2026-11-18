"""Fixed ReportPortal launch names per CI event (MLX-135)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.rp_marks import (
    CI_LAUNCH_NAMES,
    build_ci_launch_attributes,
    is_ci_launch_name,
    main,
    resolve_ci_launch,
    resolve_ci_launch_name,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "orchestrator/rp_marks.py"
WORKFLOWS = [".github/workflows/spectra.yml"]


@pytest.mark.ticket("MLX-135")
@pytest.mark.ac("MLX-135:AC1")
@pytest.mark.parametrize(
    ("event_name", "git_ref", "ref_name", "expected"),
    [
        ("pull_request", "refs/pull/12/merge", "12/merge", "pull request"),
        ("pull_request_target", "refs/heads/main", "main", "pull request"),
        ("push", "refs/heads/main", "main", "merge"),
        ("push", "refs/heads/feat/x", "feat/x", "commit"),
        ("push", "refs/tags/v1.2.0", "v1.2.0", "tag"),
        ("release", "refs/tags/v1.2.0", "v1.2.0", "release"),
        ("workflow_dispatch", "refs/heads/main", "main", "manual"),
        ("workflow_dispatch", "refs/tags/v1.2.0", "v1.2.0", "manual"),
        ("schedule", "refs/heads/main", "main", "commit"),
    ],
)
def test_resolve_ci_launch_name_is_fixed_per_event_ac1(event_name, git_ref, ref_name, expected) -> None:
    """AC1: the launch name depends on the event type only, never on branch/tag/SHA."""
    name = resolve_ci_launch_name(event_name=event_name, git_ref=git_ref, ref_name=ref_name)
    assert name == expected
    assert name in CI_LAUNCH_NAMES


@pytest.mark.ticket("MLX-135")
@pytest.mark.ac("MLX-135:AC1")
def test_resolve_ci_launch_name_uses_repository_default_branch_ac1() -> None:
    """AC1: ``merge`` follows the repo default branch, not a hard-coded ``main``."""
    kwargs = {"event_name": "push", "git_ref": "refs/heads/develop", "ref_name": "develop"}
    assert resolve_ci_launch_name(**kwargs, default_branch="develop") == "merge"
    assert resolve_ci_launch_name(**kwargs, default_branch="main") == "commit"
    assert resolve_ci_launch_name(**kwargs, default_branch="") == "commit"


@pytest.mark.ticket("MLX-135")
@pytest.mark.ac("MLX-135:AC1")
def test_manual_override_wins_and_stays_on_one_line_ac1() -> None:
    """AC1: the workflow_dispatch override is kept, newlines cannot inject step outputs."""
    name = resolve_ci_launch_name(
        event_name="workflow_dispatch",
        git_ref="refs/heads/main",
        ref_name="main",
        override="  Release v0.5.0\nattributes=x  ",
    )
    assert name == "Release v0.5.0 attributes=x"
    assert resolve_ci_launch_name(event_name="push", git_ref="", ref_name="", override="   ") == "commit"


@pytest.mark.ticket("MLX-135")
@pytest.mark.ac("MLX-135:AC1")
def test_is_ci_launch_name_matches_fixed_names_only_ac1() -> None:
    assert is_ci_launch_name("Pull Request")
    assert is_ci_launch_name(" release ")
    assert not is_ci_launch_name("Release v0.5.0")
    assert not is_ci_launch_name(None)


@pytest.mark.ticket("MLX-135")
@pytest.mark.ac("MLX-135:AC2")
def test_main_exports_name_and_unchanged_attributes_ac2(tmp_path) -> None:
    """AC2: attributes come from build_ci_launch_attributes, exported as step outputs."""
    output = tmp_path / "github_output"
    env = {
        "EVENT_NAME": "release",
        "GIT_REF": "refs/tags/v1.2.0",
        "REF_NAME": "v1.2.0",
        "HEAD_REF": "",
        "COMMIT_SHA": "0123abcd",
        "RELEASE_TAG": "v1.2.0",
        "DEFAULT_BRANCH": "main",
        "GITHUB_OUTPUT": str(output),
    }
    assert main(env) == 0
    expected_attrs = build_ci_launch_attributes(
        event_name="release",
        git_ref="refs/tags/v1.2.0",
        ref_name="v1.2.0",
        head_ref="",
        commit_sha="0123abcd",
        release_tag="v1.2.0",
    )
    assert expected_attrs == (
        "ci:github;ref:v1.2.0;branch:v1.2.0;event:release;commit:0123abcd;tag:v1.2.0;release:v1.2.0"
    )
    assert output.read_text(encoding="utf-8").splitlines() == [
        "name=release",
        f"attributes={expected_attrs}",
    ]


@pytest.mark.ticket("MLX-135")
@pytest.mark.ac("MLX-135:AC2")
def test_pull_request_attributes_keep_head_branch_and_sha_ac2() -> None:
    name, attrs = resolve_ci_launch(
        {
            "EVENT_NAME": "pull_request",
            "GIT_REF": "refs/pull/7/merge",
            "REF_NAME": "7/merge",
            "HEAD_REF": "feat/demo",
            "COMMIT_SHA": "headsha",
            "CI_NAME": "forgejo",
        }
    )
    assert name == "pull request"
    assert attrs == "ci:forgejo;ref:feat/demo;branch:feat/demo;event:pull_request;commit:headsha"


@pytest.mark.ticket("MLX-135")
@pytest.mark.ac("MLX-135:AC1")
def test_script_runs_with_stdlib_only_before_pip_install_ac1(tmp_path) -> None:
    """AC1: the workflow runs the module as a plain script (isolated mode, no deps)."""
    output = tmp_path / "out"
    env = {
        "EVENT_NAME": "push",
        "GIT_REF": "refs/heads/main",
        "REF_NAME": "main",
        "COMMIT_SHA": "abc",
        "DEFAULT_BRANCH": "main",
        "GITHUB_OUTPUT": str(output),
    }
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "name=merge" in output.read_text(encoding="utf-8")


@pytest.mark.ticket("MLX-135")
@pytest.mark.ac("MLX-135:AC1")
@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_workflow_delegates_launch_name_to_module_ac1(workflow) -> None:
    """AC1/AC2: no inline bash copy of the naming/attribute logic left in CI."""
    content = (REPO_ROOT / workflow).read_text(encoding="utf-8")
    assert "run: python3 orchestrator/rp_marks.py" in content
    assert "DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}" in content
    assert 'name="Release' not in content
    assert 'name="Tag' not in content
    assert "attrs=" not in content
