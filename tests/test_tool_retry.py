"""
Unit tests for src/tools/tool_retry.py — error classification, retry behaviour,
and structured-error formatting for tool invocations.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from src.tools.tool_retry import (
    Classification,
    ErrorClass,
    classify_error,
    format_error,
    invoke_with_recovery,
)


# ── Classification ──────────────────────────────────────────────────

class _MockHttpError(Exception):
    pass


def test_classify_500_is_transient():
    cls = classify_error(_MockHttpError("Server error: 500 Internal Server Error"))
    assert cls.error_class == ErrorClass.TRANSIENT
    assert cls.http_status == 500


def test_classify_502_is_transient():
    cls = classify_error(_MockHttpError("HTTP 502 bad gateway"))
    assert cls.error_class == ErrorClass.TRANSIENT


def test_classify_429_rate_limit_is_transient():
    cls = classify_error(_MockHttpError("429 Too Many Requests — rate limit"))
    assert cls.error_class == ErrorClass.TRANSIENT


def test_classify_404_is_correctable():
    cls = classify_error(_MockHttpError("HTTP 404 not found"))
    assert cls.error_class == ErrorClass.CORRECTABLE
    assert cls.http_status == 404


def test_classify_422_conflict_is_correctable():
    cls = classify_error(_MockHttpError("Reference already exists: 422"))
    assert cls.error_class == ErrorClass.CORRECTABLE


def test_classify_401_is_fatal():
    cls = classify_error(_MockHttpError("HTTP 401 unauthorized"))
    assert cls.error_class == ErrorClass.FATAL


def test_classify_403_is_fatal():
    cls = classify_error(_MockHttpError("HTTP 403 forbidden — bad token"))
    assert cls.error_class == ErrorClass.FATAL


def test_classify_remote_protocol_error_by_name():
    class RemoteProtocolError(Exception):
        pass
    cls = classify_error(RemoteProtocolError("Server disconnected"))
    assert cls.error_class == ErrorClass.TRANSIENT


def test_classify_server_disconnected_message_is_transient():
    cls = classify_error(Exception("Server disconnected without sending a response."))
    assert cls.error_class == ErrorClass.TRANSIENT


def test_classify_file_not_found_is_correctable():
    cls = classify_error(FileNotFoundError("File not found: src/missing.py"))
    assert cls.error_class == ErrorClass.CORRECTABLE


def test_classify_connection_error_is_transient():
    class ConnectError(Exception):
        pass
    cls = classify_error(ConnectError("connection refused"))
    assert cls.error_class == ErrorClass.TRANSIENT


def test_classify_invalid_api_key_is_fatal():
    cls = classify_error(Exception("Invalid API key provided"))
    assert cls.error_class == ErrorClass.FATAL


def test_classify_unknown_error_defaults_correctable():
    """Unknown errors default to CORRECTABLE so the agent gets a chance to adapt."""
    cls = classify_error(Exception("some unexpected error string with no signal"))
    assert cls.error_class == ErrorClass.CORRECTABLE


# ── format_error ────────────────────────────────────────────────────

def test_format_error_includes_tool_name_class_hint_for_github_422():
    msg = format_error(
        "github_create_file",
        Classification(ErrorClass.CORRECTABLE, 422, "HTTP 422"),
        Exception("Reference already exists"),
        {"path": "test.py", "branch": "feature/x"},
        retry_count=0,
    )
    assert "TOOL_ERROR github_create_file" in msg
    assert "[CORRECTABLE]" in msg
    assert "HTTP 422" in msg
    assert "Attempted: path=" in msg
    assert "HINT:" in msg
    assert "github_update_file" in msg


def test_format_error_filesystem_not_found_hints_list_directory():
    msg = format_error(
        "read_file",
        Classification(ErrorClass.CORRECTABLE, None, "File not found"),
        FileNotFoundError("File not found: mcp_servers/shell_server.py"),
        {"path": "mcp_servers/shell_server.py"},
        retry_count=0,
    )
    assert "HINT:" in msg
    assert "list_directory" in msg


def test_format_error_includes_retry_count_when_exhausted():
    msg = format_error(
        "github_create_pr",
        Classification(ErrorClass.TRANSIENT, 503, "HTTP 503"),
        Exception("Service Unavailable"),
        {"title": "x"},
        retry_count=2,
    )
    assert "Retries: 2 (exhausted)" in msg


# ── invoke_with_recovery async behaviour ────────────────────────────

def _run(coro):
    return asyncio.run(coro)


def test_invoke_success_first_attempt():
    calls = {"n": 0}

    async def _call():
        calls["n"] += 1
        return "ok"

    result = _run(invoke_with_recovery("my_tool", {}, _call))
    assert result == "ok"
    assert calls["n"] == 1


def test_invoke_transient_retries_then_succeeds():
    """A TRANSIENT error followed by success should return success after retry."""
    calls = {"n": 0}

    async def _call():
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("Server disconnected without sending a response.")
        return "ok-after-retry"

    with patch("src.tools.tool_retry._BASE_DELAY_S", 0.0):
        result = _run(invoke_with_recovery("my_tool", {}, _call))

    assert result == "ok-after-retry"
    assert calls["n"] == 2


def test_invoke_correctable_returns_structured_error_no_retry():
    """CORRECTABLE errors must NOT retry — they need agent to change inputs."""
    calls = {"n": 0}

    async def _call():
        calls["n"] += 1
        raise FileNotFoundError("File not found: bogus/path.py")

    result = _run(invoke_with_recovery("read_file", {"path": "bogus/path.py"}, _call))
    assert calls["n"] == 1
    assert "TOOL_ERROR read_file" in result
    assert "[CORRECTABLE]" in result
    assert "HINT:" in result
    assert "list_directory" in result


def test_invoke_fatal_returns_error_no_retry():
    calls = {"n": 0}

    async def _call():
        calls["n"] += 1
        raise Exception("HTTP 401 unauthorized")

    result = _run(invoke_with_recovery("github_create_pr", {"title": "x"}, _call))
    assert calls["n"] == 1
    assert "[FATAL]" in result


def test_invoke_transient_exhausts_returns_error_after_max_retries():
    calls = {"n": 0}

    async def _call():
        calls["n"] += 1
        raise Exception("Server disconnected")

    with patch("src.tools.tool_retry._BASE_DELAY_S", 0.0), \
         patch("src.tools.tool_retry._MAX_RETRIES", 2):
        result = _run(invoke_with_recovery("github_create_file", {"path": "x"}, _call))

    # 1 original + 2 retries = 3 attempts
    assert calls["n"] == 3
    assert "[TRANSIENT]" in result
    assert "Retries: 2 (exhausted)" in result


def test_invoke_disabled_via_flag_raises_through():
    """When TOOL_RECOVERY_ENABLED is false, original exception must propagate."""
    async def _call():
        raise RuntimeError("boom")

    with patch("src.tools.tool_retry._ENABLED", False):
        with pytest.raises(RuntimeError, match="boom"):
            _run(invoke_with_recovery("any_tool", {}, _call))
