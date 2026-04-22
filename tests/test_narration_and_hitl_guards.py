"""Unit tests for Fix #1 (narration guard) and Fix #2 (HITL auto-confirm).

Both are pure heuristics over message strings — no network / DB involved —
so these tests run in <1s and catch regressions the moment someone tweaks
the regex sets.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.orchestrator import _looks_like_narration  # noqa: PLC2701 — internal helper under test
from src.evaluation.regression import _looks_like_hitl_wait  # noqa: PLC2701


# ─── Fix #1: narration heuristic ────────────────────────────────────────


def test_narration_detects_classic_preamble() -> None:
    assert _looks_like_narration("Now writing the report.")
    assert _looks_like_narration("I will now compile findings.")
    assert _looks_like_narration("About to produce the final review.")
    assert _looks_like_narration("Let me start writing the report now.")
    assert _looks_like_narration("Confidence: 4/5")


def test_narration_ignores_real_deliverable() -> None:
    # Realistic reviewer deliverable with all four sections — should NOT fire.
    deliverable = (
        "SUMMARY: Reviewed src/llm/client.py (89 lines). Verdict: needs-work.\n\n"
        "FINDINGS:\n"
        "- WARNING: src/llm/client.py:45 — No timeout on httpx client; requests can hang indefinitely.\n"
        "- CRITICAL: src/llm/client.py:67 — API key logged in plain text via logger.debug().\n\n"
        "VERIFIED: get_llm() correctly injects extra_body for thinking models. Error handling present on lines 71-78.\n\n"
        "RECOMMENDATIONS:\n"
        "1. Add httpx.Timeout(read=60, connect=10) to all client instantiations.\n"
        "2. Redact API keys in all log statements before shipping.\n"
    )
    assert not _looks_like_narration(deliverable)


def test_narration_ignores_empty_and_short_non_preamble() -> None:
    assert not _looks_like_narration("")
    assert not _looks_like_narration("Done.")
    assert not _looks_like_narration("All checks passed.")


def test_narration_ignores_long_message_even_if_preamble_word() -> None:
    # If an agent writes ≥ 250 chars that *also* contains "now writing",
    # we assume it's a legitimate deliverable that just happens to mention
    # the phrase.  Keeps false-positive rate low.
    long_msg = (
        "The module under review handles authentication and now writing "
        "tests for it is out of scope.  "
    ) * 5
    assert len(long_msg) >= 250
    assert not _looks_like_narration(long_msg)


# ─── Fix #2: HITL-wait detector ─────────────────────────────────────────


def _ai(text: str, tool_calls=None) -> SimpleNamespace:
    """Build a minimal duck-typed AIMessage for the heuristic."""
    # isinstance(..., AIMessage) is checked inside the function, so we need
    # a real AIMessage.  Import lazily so missing deps don't blow up the
    # module import path.
    from langchain_core.messages import AIMessage
    m = AIMessage(content=text)
    m.tool_calls = tool_calls or []  # type: ignore[assignment]
    return m  # type: ignore[return-value]


def test_hitl_wait_detects_shall_i_proceed() -> None:
    msg = _ai("Here is the plan.\n\nShall I proceed with step 2?")
    assert _looks_like_hitl_wait([msg])


def test_hitl_wait_detects_await_confirmation() -> None:
    msg = _ai("Plan ready. Awaiting your confirmation before executing.")
    assert _looks_like_hitl_wait([msg])


def test_hitl_wait_detects_reply_yes() -> None:
    msg = _ai("Reply 'yes' to confirm and I will execute the push.")
    assert _looks_like_hitl_wait([msg])


def test_hitl_wait_ignores_when_tool_calls_present() -> None:
    # If the agent already scheduled a tool call, it isn't really waiting.
    msg = _ai(
        "Shall I proceed?",
        tool_calls=[{"name": "git_push", "args": {}, "id": "tc1"}],
    )
    assert not _looks_like_hitl_wait([msg])


def test_hitl_wait_ignores_empty_and_non_ai() -> None:
    from langchain_core.messages import HumanMessage
    assert not _looks_like_hitl_wait([])
    assert not _looks_like_hitl_wait([HumanMessage(content="Shall I proceed?")])


def test_hitl_wait_ignores_genuine_completion() -> None:
    msg = _ai(
        "I've pushed the branch and opened PR #42. All tests pass. "
        "Task complete."
    )
    assert not _looks_like_hitl_wait([msg])


def test_hitl_wait_handles_curly_quotes() -> None:
    # Real output from gpt-5 on golden_020 in the first fix-123 run — curly
    # quotes foiled the first regex set.
    msg = _ai(
        "Pre-flight check\n\nI will proceed using only GitHub REST tools. "
        "Please confirm or adjust the following:\n\n"
        "- Repository: org/repo\n"
        "Reply \u201cYes\u201d to proceed, or provide any changes."
    )
    assert _looks_like_hitl_wait([msg])


def test_hitl_wait_detects_reply_approved_past_tense() -> None:
    # Real output from gpt-5 on default/golden_020 — "Approved" past tense with curly quotes.
    msg = _ai(
        "If approved, I will proceed to:\n- Create the branch\n- Commit test.py\n\n"
        "Reply \u201cApproved\u201d to proceed, or provide edits to the file path/name."
    )
    assert _looks_like_hitl_wait([msg])


def test_hitl_wait_detects_approve_plan_cursor_style() -> None:
    msg = _ai(
        "## Plan\n1. Read the file\n2. Patch it\n3. Run tests\n\n"
        "Approve this plan, or suggest changes?\n"
        "A) Approve — proceed\nB) Modify — describe changes\nC) Reject — stop"
    )
    assert _looks_like_hitl_wait([msg])
