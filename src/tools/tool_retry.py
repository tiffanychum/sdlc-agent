"""
Tool-invocation error recovery layer.

Every MCP tool call in `src/tools/registry.py` is routed through `invoke_with_recovery`.
Exceptions from `mcp_server.call_tool(...)` are classified into three buckets:

- TRANSIENT   — network disconnects / timeouts / 5xx. Retried with exponential
                backoff (default 2 retries, cumulative cap 10s).
- CORRECTABLE — file-not-found / 404 / 422 conflict / bad argument. Returned as
                a structured string the LLM can read and self-correct from.
                NOT retried (retrying won't help — the agent must change inputs).
- FATAL       — auth errors / schema violations on required fields. Returned as
                an error string so the agent can abandon or escalate.

Everything goes through the same code path so every tool (filesystem, github,
jira, git, web, planner, memory, rag, sql) gets the same treatment.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# ── Tunables (env-overridable) ──────────────────────────────────────

_ENABLED = os.getenv("TOOL_RECOVERY_ENABLED", "true").lower() in ("1", "true", "yes")
_MAX_RETRIES = int(os.getenv("TOOL_RECOVERY_MAX_RETRIES", "2"))
_BASE_DELAY_S = float(os.getenv("TOOL_RECOVERY_BASE_DELAY_S", "0.5"))
_CUMULATIVE_CAP_S = float(os.getenv("TOOL_RECOVERY_CUMULATIVE_CAP_S", "10.0"))


class ErrorClass(str, Enum):
    TRANSIENT = "TRANSIENT"
    CORRECTABLE = "CORRECTABLE"
    FATAL = "FATAL"


@dataclass
class Classification:
    error_class: ErrorClass
    http_status: Optional[int]
    short_reason: str


# ── Classifier ──────────────────────────────────────────────────────

_TRANSIENT_EXC_NAMES = {
    # httpx / network
    "RemoteProtocolError", "ConnectError", "ConnectTimeout", "ReadTimeout",
    "WriteTimeout", "PoolTimeout", "NetworkError", "ProtocolError",
    # stdlib
    "TimeoutError", "ConnectionError", "ConnectionResetError",
    "ConnectionAbortedError", "BrokenPipeError",
    # asyncio
    "CancelledError",  # treat as transient — often a provider-side shutdown
}

_TRANSIENT_HINT_PATTERNS = (
    re.compile(r"server disconnected", re.I),
    re.compile(r"connection reset", re.I),
    re.compile(r"timed? ?out", re.I),
    re.compile(r"temporarily unavailable", re.I),
    re.compile(r"bad gateway", re.I),
    re.compile(r"gateway timeout", re.I),
    re.compile(r"service unavailable", re.I),
    re.compile(r"rate limit", re.I),  # retry with backoff
)

_CORRECTABLE_HINT_PATTERNS = (
    re.compile(r"file not found", re.I),
    re.compile(r"no such file", re.I),
    re.compile(r"path (does not exist|not found)", re.I),
    re.compile(r"not a directory", re.I),
    re.compile(r"already exists", re.I),
    re.compile(r"conflict", re.I),
    re.compile(r"invalid argument", re.I),
    re.compile(r"bad request", re.I),
    re.compile(r"validation (error|failed)", re.I),
    re.compile(r"\b404\b"),
    re.compile(r"\b422\b"),
    re.compile(r"\b409\b"),
)

_FATAL_HINT_PATTERNS = (
    re.compile(r"unauthor", re.I),
    re.compile(r"forbidden", re.I),
    re.compile(r"permission denied", re.I),
    re.compile(r"invalid (api )?(key|token|credentials)", re.I),
    re.compile(r"\b401\b"),
    re.compile(r"\b403\b"),
)


def _extract_http_status(message: str) -> Optional[int]:
    m = re.search(r"\b(4\d\d|5\d\d)\b", message)
    if m:
        return int(m.group(1))
    return None


def classify_error(exc: BaseException) -> Classification:
    """Classify a raised exception from `mcp_server.call_tool`."""
    exc_name = type(exc).__name__
    msg = str(exc) or exc_name
    status = _extract_http_status(msg)

    if status is not None:
        if 500 <= status <= 599:
            return Classification(ErrorClass.TRANSIENT, status, f"HTTP {status}")
        if status in (408, 429):
            return Classification(ErrorClass.TRANSIENT, status, f"HTTP {status}")
        if status in (401, 403):
            return Classification(ErrorClass.FATAL, status, f"HTTP {status} auth")
        if 400 <= status <= 499:
            return Classification(ErrorClass.CORRECTABLE, status, f"HTTP {status}")

    if exc_name in _TRANSIENT_EXC_NAMES:
        return Classification(ErrorClass.TRANSIENT, None, exc_name)

    for pat in _FATAL_HINT_PATTERNS:
        if pat.search(msg):
            return Classification(ErrorClass.FATAL, status, pat.pattern)

    for pat in _TRANSIENT_HINT_PATTERNS:
        if pat.search(msg):
            return Classification(ErrorClass.TRANSIENT, status, pat.pattern)

    for pat in _CORRECTABLE_HINT_PATTERNS:
        if pat.search(msg):
            return Classification(ErrorClass.CORRECTABLE, status, pat.pattern)

    return Classification(ErrorClass.CORRECTABLE, status, f"{exc_name}")


# ── Per-tool hint map ───────────────────────────────────────────────
#
# Keyed by a substring of the tool name. The first match wins. Each entry
# returns a short, actionable hint the LLM can read to self-correct.

def _hint_for(tool_name: str, cls: Classification, exc_msg: str) -> str:
    lname = tool_name.lower()
    status = cls.http_status

    # GitHub tools
    if lname.startswith("github_"):
        if status == 422 or "already exists" in exc_msg.lower():
            return ("The resource already exists on GitHub. If creating a branch, check "
                    "with github_list_branches first. If creating a file, use "
                    "github_update_file instead, or github_get_file to inspect current contents. "
                    "If creating a PR, check github_list_prs for an open one on the same branch.")
        if status == 404:
            return ("GitHub returned 404. Verify the repo/branch/path spelling and that "
                    "the branch exists (github_list_branches). Repo names are case-sensitive.")
        if status in (401, 403):
            return ("GitHub auth failed. The token may be missing or lack repo scope — this "
                    "is a FATAL error; report it back to the user rather than retrying.")
        return ""

    # Filesystem tools (read_file, write_file, list_directory, search_files)
    if lname in ("read_file", "write_file", "list_directory", "search_files"):
        if cls.error_class == ErrorClass.CORRECTABLE:
            return ("File/path not found. Call list_directory on the parent folder first "
                    "to discover correct paths. Common missing prefixes: 'src/', 'tests/'. "
                    "Never guess — list, then read.")
        return ""

    # Shell / run_command
    if lname in ("run_command", "shell", "exec"):
        if cls.error_class == ErrorClass.CORRECTABLE:
            return ("Shell command failed. Check the working directory, ensure required "
                    "binaries are installed, and verify arg quoting. Run a simpler probe "
                    "first (e.g. 'which <bin>' or 'ls').")
        return ""

    # Git tools
    if lname.startswith("git_"):
        if "not a git repository" in exc_msg.lower() or "not a working tree" in exc_msg.lower():
            return ("Directory is not a git repo. Run git_init first, or cd into the correct repo. "
                    "If the repo should already exist, list_directory to verify.")
        if "nothing to commit" in exc_msg.lower():
            return ("Nothing to commit. Confirm files were actually written (list_directory) and "
                    "staged before attempting git_commit.")
        return ""

    # Jira tools
    if lname.startswith("jira_"):
        if status == 404:
            return ("Jira returned 404. Verify the issue key format (e.g. SDLC-123) and project. "
                    "Use jira_list_projects or jira_search to find valid keys.")
        if status == 400:
            return ("Jira rejected the request. Check required fields (summary, issuetype) and "
                    "that the assignee username exists (jira_list_users).")
        return ""

    # RAG
    if "rag" in lname or "perf_search" in lname:
        if cls.error_class == ErrorClass.CORRECTABLE:
            return ("RAG lookup produced no match. Rephrase the query or broaden terms; "
                    "consider switching between perf_search and rag_search.")
        return ""

    # SQL / data_analyst
    if lname == "query_db" or lname == "sql":
        if "syntax error" in exc_msg.lower():
            return ("SQL syntax error. Only SELECT / WITH ... SELECT are allowed. Re-check "
                    "table/column names against the schema.")
        return ""

    # Memory / planner — usually argument validation
    if lname.startswith("memory_") or lname.startswith("plan_") or lname in ("create_plan", "update_plan_step"):
        if cls.error_class == ErrorClass.CORRECTABLE:
            return ("Argument validation failed. Re-check the tool's required fields; for "
                    "planner tools, step_id must reference an existing plan step.")
        return ""

    return ""


def format_error(
    tool_name: str,
    cls: Classification,
    exc: BaseException,
    args_preview: dict,
    retry_count: int,
) -> str:
    """Build the structured error string returned to the LLM after exhaustion."""
    msg = str(exc) or type(exc).__name__
    hint = _hint_for(tool_name, cls, msg)
    parts = [
        f"TOOL_ERROR {tool_name} [{cls.error_class.value}]"
        + (f" (HTTP {cls.http_status})" if cls.http_status else "")
    ]
    if args_preview:
        preview = ", ".join(f"{k}={_short(v)}" for k, v in list(args_preview.items())[:5])
        parts.append(f"Attempted: {preview}")
    parts.append(f"Cause: {msg[:500]}")
    if retry_count > 0:
        parts.append(f"Retries: {retry_count} (exhausted)")
    if hint:
        parts.append(f"HINT: {hint}")
    return "\n".join(parts)


def _short(v: Any) -> str:
    s = repr(v)
    return s if len(s) <= 80 else s[:77] + "..."


# ── Invoker ─────────────────────────────────────────────────────────

async def invoke_with_recovery(
    tool_name: str,
    args: dict,
    call: Callable[[], Awaitable[Any]],
) -> Any:
    """
    Execute `call()` with classification + retry for TRANSIENT errors.

    - TRANSIENT: retry with exponential backoff, max `_MAX_RETRIES`, capped at
      `_CUMULATIVE_CAP_S` seconds total wait.
    - CORRECTABLE / FATAL: return a structured error string; NO retry. The
      agent receives the hint and can self-correct or abandon.

    If recovery is disabled via env, raises immediately (legacy behaviour).
    """
    if not _ENABLED:
        return await call()

    attempt = 0
    total_wait = 0.0

    while True:
        try:
            return await call()
        except BaseException as exc:  # noqa: BLE001 — we re-classify below
            cls = classify_error(exc)

            if cls.error_class != ErrorClass.TRANSIENT:
                logger.warning(
                    "tool=%s class=%s status=%s reason=%s — returning error to LLM",
                    tool_name, cls.error_class.value, cls.http_status, cls.short_reason,
                )
                return format_error(tool_name, cls, exc, args, retry_count=attempt)

            if attempt >= _MAX_RETRIES:
                logger.error(
                    "tool=%s TRANSIENT exhausted after %d retries (%.1fs total): %s",
                    tool_name, attempt, total_wait, exc,
                )
                return format_error(tool_name, cls, exc, args, retry_count=attempt)

            delay = _BASE_DELAY_S * (2 ** attempt)
            if total_wait + delay > _CUMULATIVE_CAP_S:
                delay = max(0.0, _CUMULATIVE_CAP_S - total_wait)
                if delay <= 0:
                    logger.error(
                        "tool=%s TRANSIENT cap %.1fs hit after %d retries: %s",
                        tool_name, _CUMULATIVE_CAP_S, attempt, exc,
                    )
                    return format_error(tool_name, cls, exc, args, retry_count=attempt)

            logger.warning(
                "tool=%s TRANSIENT attempt=%d/%d reason=%s — retrying in %.2fs",
                tool_name, attempt + 1, _MAX_RETRIES, cls.short_reason, delay,
            )
            attempt += 1
            t0 = time.monotonic()
            await asyncio.sleep(delay)
            total_wait += time.monotonic() - t0
