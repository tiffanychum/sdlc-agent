"""Tests for the router's tool-coverage filter in `_derive_required_steps`.

Motivating bug: on `golden_020` the default team derived pipeline
[coder, devops]. Coder has no `github` tools and narrated instead of calling
them; that narration then poisoned the handoff context so `devops` also
narrated. The tool-coverage gate drops agents that can't contribute tool
groups the task requires.
"""

from __future__ import annotations

import pytest

from src.orchestrator import (
    _infer_required_tool_groups,
    _enforce_tool_coverage,
)


DEFAULT_AGENTS = [
    {"role": "planner", "tool_groups": ["memory"]},
    {"role": "researcher", "tool_groups": ["web", "rag", "memory"]},
    {"role": "coder", "tool_groups": ["filesystem", "shell", "rag", "memory"]},
    {"role": "qa", "tool_groups": ["filesystem", "shell", "memory"]},
    {"role": "project_manager", "tool_groups": ["jira", "memory"]},
    {"role": "devops", "tool_groups": ["git", "github", "memory"]},
]


# ── _infer_required_tool_groups ────────────────────────────────────────────


def test_github_only_task_requires_github_not_filesystem():
    """The golden_020 scenario: a GitHub-REST-only task has no local writes."""
    text = (
        "in the github repository tiffanychum/sdlc-agent-integration-test, "
        "using only the github rest api, create a branch called test-integration, "
        "add a file test.py with a pytest function, and open a pull request."
    )
    groups = _infer_required_tool_groups(
        text,
        is_build_task=True,
        needs_git=True,
        needs_jira=False,
        needs_research=False,
        needs_testing=True,
        needs_qa=False,
    )
    assert "github" in groups
    assert "filesystem" not in groups, "github-only task should not require local FS"
    assert "shell" not in groups, "github-only task should not require local shell"


def test_github_plus_local_path_requires_both_filesystem_and_github():
    """
    golden_021 scenario: prompt says "using ONLY the GitHub REST tools" BUT also
    references a local path like /tmp/discount-util/. The "only GitHub" phrasing
    constrains git operations, not file writes — filesystem is still required.
    """
    text = (
        "fetch jira task sdlc-11 and implement it. work only in /tmp/discount-util/. "
        "write the implementation and tests under /tmp/discount-util/tests/ and run "
        "pytest there. once tests pass locally, push using only the github rest tools "
        "(github_create_branch, github_create_pr)."
    )
    groups = _infer_required_tool_groups(
        text,
        is_build_task=True,
        needs_git=True,
        needs_jira=True,
        needs_research=False,
        needs_testing=True,
        needs_qa=False,
    )
    assert "filesystem" in groups, "local /tmp path means filesystem IS needed"
    assert "shell" in groups, "local pytest run means shell IS needed"
    assert "github" in groups
    assert "jira" in groups


def test_local_build_with_git_init_requires_fs_shell_and_git():
    text = (
        "build a validate_email utility in /tmp/validate-util/, write pytest tests, "
        "initialize a local git repo and commit the changes."
    )
    groups = _infer_required_tool_groups(
        text,
        is_build_task=True,
        needs_git=True,
        needs_jira=False,
        needs_research=False,
        needs_testing=True,
        needs_qa=False,
    )
    assert "filesystem" in groups
    assert "shell" in groups
    assert groups & {"git", "github"}, "must require at least one git-family tool"


def test_review_task_requires_filesystem_only():
    """golden_001 / golden_007 style — path is given, just read & summarise."""
    text = "review src/orchestrator.py and point out any issues."
    groups = _infer_required_tool_groups(
        text,
        is_build_task=False,
        needs_git=False,
        needs_jira=False,
        needs_research=False,
        needs_testing=False,
        needs_qa=False,
    )
    assert groups == set(), "pure review task needs no required tool groups"


def test_research_task_requires_web_or_rag():
    text = "research best practices for async fastapi handlers."
    groups = _infer_required_tool_groups(
        text,
        is_build_task=False,
        needs_git=False,
        needs_jira=False,
        needs_research=True,
        needs_testing=False,
        needs_qa=False,
    )
    assert "web_or_rag" in groups


# ── _enforce_tool_coverage ─────────────────────────────────────────────────


def test_drops_coder_for_github_only_task():
    steps = ["coder", "devops"]
    required = {"github"}
    out = _enforce_tool_coverage(steps, DEFAULT_AGENTS, required)
    assert "coder" not in out, "coder has no github tools — must be dropped"
    assert out == ["devops"]


def test_keeps_coder_and_devops_for_local_build_plus_git():
    steps = ["coder", "devops"]
    required = {"filesystem", "shell", "git"}
    out = _enforce_tool_coverage(steps, DEFAULT_AGENTS, required)
    assert out == ["coder", "devops"], "both agents contribute coverage"


def test_planner_and_researcher_are_coverage_neutral():
    steps = ["planner", "researcher", "coder"]
    required = {"filesystem"}
    out = _enforce_tool_coverage(steps, DEFAULT_AGENTS, required)
    assert "planner" in out and "researcher" in out
    assert "coder" in out


def test_appends_devops_when_task_needs_github_but_not_in_pipeline():
    steps = ["coder"]
    required = {"github"}
    out = _enforce_tool_coverage(steps, DEFAULT_AGENTS, required)
    assert "devops" in out, "must add devops to cover uncovered github requirement"


def test_empty_required_returns_pipeline_untouched():
    steps = ["coder", "devops"]
    out = _enforce_tool_coverage(steps, DEFAULT_AGENTS, set())
    assert out == steps


def test_web_or_rag_satisfied_by_researcher():
    steps = ["researcher", "coder"]
    required = {"web_or_rag", "filesystem"}
    out = _enforce_tool_coverage(steps, DEFAULT_AGENTS, required)
    assert out == ["researcher", "coder"]


def test_does_not_add_agent_already_in_pipeline():
    steps = ["devops"]
    required = {"github"}
    out = _enforce_tool_coverage(steps, DEFAULT_AGENTS, required)
    assert out == ["devops"]


def test_agent_with_no_tool_groups_is_dropped_when_required_nonempty():
    """Defensive: an agent whose tool_groups aren't declared contributes nothing."""
    agents = DEFAULT_AGENTS + [{"role": "ghost", "tool_groups": []}]
    steps = ["ghost", "devops"]
    required = {"github"}
    out = _enforce_tool_coverage(steps, agents, required)
    assert "ghost" not in out
    assert "devops" in out


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
