"""Unit tests for Fix #1 (plan-in-handoff) and Fix #3 (per-turn read dedup).

Both are pure in-process logic — no network / DB / LLM — so these tests run
in <1s and catch regressions the moment someone tweaks the wiring.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from src.orchestrator import _build_handoff_context, _extract_last_ai_text  # noqa: PLC2701
from src.tools.read_dedup import activate_read_dedup, wrap_read_dedup  # noqa: PLC2701


# ─── Fix #1: _extract_last_ai_text ──────────────────────────────────────────


def test_extract_last_ai_text_finds_most_recent_string_content() -> None:
    msgs = [
        HumanMessage(content="task"),
        AIMessage(content="first draft"),
        HumanMessage(content="refine"),
        AIMessage(content="final plan text"),
    ]
    assert _extract_last_ai_text(msgs) == "final plan text"


def test_extract_last_ai_text_skips_empty_tool_call_messages() -> None:
    # A tool-only AIMessage has empty .content — we must skip past it and
    # return the previous textual AIMessage.
    msgs = [
        HumanMessage(content="task"),
        AIMessage(content="plan body text"),
        AIMessage(content=""),  # tool-call-only prefill
    ]
    assert _extract_last_ai_text(msgs) == "plan body text"


def test_extract_last_ai_text_handles_anthropic_list_content() -> None:
    # Claude sometimes returns content as a list of {type, text} dicts.
    msgs = [
        AIMessage(content=[{"type": "text", "text": "step 1"}, {"type": "text", "text": "step 2"}]),
    ]
    out = _extract_last_ai_text(msgs)
    assert "step 1" in out and "step 2" in out


def test_extract_last_ai_text_returns_empty_when_none() -> None:
    assert _extract_last_ai_text([]) == ""
    assert _extract_last_ai_text([HumanMessage(content="no AI here")]) == ""


# ─── Fix #1: _build_handoff_context injects plan from planner_v2 ────────────


def test_handoff_injects_plan_block_when_prior_agent_was_planner_v2() -> None:
    state = {
        "messages": [
            HumanMessage(content="Build feature X"),
            AIMessage(content="## Plan\n1. Builder — edit app/x.py\n2. Builder — add tests"),
        ],
        "agent_trace": [
            {"step": "execution", "agent": "planner_v2", "tool_calls": []},
        ],
    }
    out = _build_handoff_context(state, current_role="builder")
    assert out is not None
    assert "PLAN FROM PRIOR PLANNER_V2" in out
    assert "do NOT re-plan" in out
    assert "1. Builder — edit app/x.py" in out
    assert "ORIGINAL REQUEST: Build feature X" in out


def test_handoff_injects_plan_block_when_prior_agent_was_planner() -> None:
    # Dev-team role is "planner" (not "planner_v2") — same behaviour expected.
    state = {
        "messages": [
            HumanMessage(content="Fix bug Y"),
            AIMessage(content="## Plan\n1. Do thing"),
        ],
        "agent_trace": [
            {"step": "execution", "agent": "planner", "tool_calls": []},
        ],
    }
    out = _build_handoff_context(state, current_role="coder")
    assert out is not None
    assert "PLAN FROM PRIOR PLANNER" in out


def test_handoff_no_plan_block_when_prior_agent_is_not_planner() -> None:
    state = {
        "messages": [
            HumanMessage(content="task"),
            AIMessage(content="coder finished"),
        ],
        "agent_trace": [
            {"step": "execution", "agent": "coder", "tool_calls": [{"tool": "write_file"}]},
        ],
    }
    out = _build_handoff_context(state, current_role="qa")
    assert out is not None
    assert "PLAN FROM PRIOR" not in out


def test_handoff_returns_none_when_no_prior_agents() -> None:
    state = {
        "messages": [HumanMessage(content="task")],
        "agent_trace": [],
    }
    assert _build_handoff_context(state, current_role="planner_v2") is None


# ─── Fix #3: read_dedup wrapper ─────────────────────────────────────────────


class _ReadArgs(BaseModel):
    path: str
    start_line: int = 1


def _make_fake_tool(calls_log: list[dict]) -> StructuredTool:
    """Build a StructuredTool whose coroutine just records its args."""
    async def _coro(**kwargs):
        calls_log.append(dict(kwargs))
        return f"ok:{kwargs.get('path', '')}"

    return StructuredTool.from_function(
        coroutine=_coro,
        name="read_file",
        description="fake read_file",
        args_schema=_ReadArgs,
        func=lambda **kw: None,
    )


def test_dedup_no_op_outside_active_scope() -> None:
    """With no active scope, the wrapper must pass through every call."""
    calls: list[dict] = []
    wrapped = wrap_read_dedup(_make_fake_tool(calls))

    async def _run():
        r1 = await wrapped.coroutine(path="a.py")
        r2 = await wrapped.coroutine(path="a.py")
        return r1, r2

    r1, r2 = asyncio.run(_run())
    assert r1 == "ok:a.py"
    assert r2 == "ok:a.py"
    assert len(calls) == 2  # no dedup → both executed


def test_dedup_short_circuits_duplicate_call_in_scope() -> None:
    calls: list[dict] = []
    wrapped = wrap_read_dedup(_make_fake_tool(calls))

    async def _run():
        with activate_read_dedup():
            r1 = await wrapped.coroutine(path="a.py")
            r2 = await wrapped.coroutine(path="a.py")  # duplicate → dedup hint
        return r1, r2

    r1, r2 = asyncio.run(_run())
    assert r1 == "ok:a.py"
    assert "[dedup]" in r2
    assert "Already called" in r2
    assert len(calls) == 1  # only the first executed


def test_dedup_distinguishes_different_args() -> None:
    calls: list[dict] = []
    wrapped = wrap_read_dedup(_make_fake_tool(calls))

    async def _run():
        with activate_read_dedup():
            a = await wrapped.coroutine(path="a.py")
            b = await wrapped.coroutine(path="b.py")
            a2 = await wrapped.coroutine(path="a.py")  # dup
        return a, b, a2

    a, b, a2 = asyncio.run(_run())
    assert a == "ok:a.py"
    assert b == "ok:b.py"
    assert "[dedup]" in a2
    assert len(calls) == 2  # a.py and b.py each ran once


def test_dedup_distinguishes_same_path_different_line_range() -> None:
    """Different kwargs (e.g. start_line) are a distinct read and must not dedup."""
    calls: list[dict] = []
    wrapped = wrap_read_dedup(_make_fake_tool(calls))

    async def _run():
        with activate_read_dedup():
            await wrapped.coroutine(path="a.py", start_line=1)
            await wrapped.coroutine(path="a.py", start_line=200)

    asyncio.run(_run())
    assert len(calls) == 2


def test_dedup_scope_resets_between_invocations() -> None:
    """Each `with` block is a fresh scope — dedup does not leak across turns."""
    calls: list[dict] = []
    wrapped = wrap_read_dedup(_make_fake_tool(calls))

    async def _run():
        with activate_read_dedup():
            await wrapped.coroutine(path="a.py")
        # Scope exited — new scope must re-execute the call.
        with activate_read_dedup():
            await wrapped.coroutine(path="a.py")

    asyncio.run(_run())
    assert len(calls) == 2


def test_dedup_tool_metadata_preserved() -> None:
    tool = _make_fake_tool([])
    wrapped = wrap_read_dedup(tool)
    assert wrapped.name == tool.name
    assert wrapped.description == tool.description
    # args_schema identity or structural equivalence both acceptable — check by name.
    assert wrapped.args_schema.__name__ == tool.args_schema.__name__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
