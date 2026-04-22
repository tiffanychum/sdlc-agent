"""Per-turn read deduplication for agents.

Some agents (notably sdlc_2_0's Builder) occasionally issue the SAME
`read_file` / `list_directory` call more than once inside a single ReAct turn.
The prior ToolMessage is still in the context window, so the LLM already has
the answer — it just forgets and re-reads.  Each duplicate read costs an extra
network round-trip and ~1-3 s of wall-clock.

This module provides a small, explicit dedup layer:

- `activate_read_dedup()` — context manager; sets a per-invocation cache in a
  contextvar, then resets on exit.  Nested entries reuse the outer cache so
  you get one consistent dedup scope per agent invocation.
- `wrap_read_dedup(tool)` — wraps a LangChain `StructuredTool`'s coroutine so
  when the SAME `(tool.name, normalized_args)` tuple is seen twice inside the
  active scope, the second call returns a short notice instead of re-executing.

The wrapper is a NO-OP when the contextvar is unset (i.e. outside any active
scope) — tools behave exactly as before.  That keeps the blast radius tiny:
only Builder turns activate the cache.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
from typing import Iterator, Optional

from langchain_core.tools import StructuredTool


# Contextvar holds the live dedup cache for one agent invocation.
# None = dedup disabled (default).
_read_cache_var: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "read_dedup_cache", default=None
)


@contextlib.contextmanager
def activate_read_dedup() -> Iterator[dict]:
    """Activate a fresh dedup scope for the current async task.

    Usage:
        with activate_read_dedup() as cache:
            await agent.ainvoke(...)   # all wrapped tools share `cache`

    Nested entries re-use the outer cache (idempotent).
    """
    existing = _read_cache_var.get()
    if existing is not None:
        yield existing
        return

    cache: dict = {}
    token = _read_cache_var.set(cache)
    try:
        yield cache
    finally:
        _read_cache_var.reset(token)


def _normalize_args(kwargs: dict) -> str:
    """Build a stable cache key from tool kwargs.

    JSON with sorted keys is sufficient — our filesystem tool args are flat
    primitives (path, start_line, end_line, query, pattern, recursive).
    """
    try:
        return json.dumps(kwargs, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(sorted(kwargs.items()))


def wrap_read_dedup(tool: StructuredTool) -> StructuredTool:
    """Wrap a read-style tool so repeat calls in the same scope short-circuit.

    The wrapped tool:
      - Defers to the original coroutine when no dedup scope is active.
      - Returns a short "already read" string when the same args are seen twice
        inside the active scope.

    Only the coroutine path is wrapped — our filesystem tools are async, and
    the sync `func` is a no-op placeholder (see `registry.py`).
    """
    orig_coro = tool.coroutine
    if orig_coro is None:
        return tool  # defensive — nothing to wrap

    tool_name = tool.name

    async def _deduped_coro(**kwargs):
        cache = _read_cache_var.get()
        if cache is None:
            return await orig_coro(**kwargs)

        key = f"{tool_name}::{_normalize_args(kwargs)}"
        if key in cache:
            # Build a human-readable hint pointing back to the prior result.
            # Keep it SHORT — the actual content is already in the message
            # history, so we only need to nudge the LLM to look there.
            args_preview = ", ".join(
                f"{k}={v!r}" for k, v in kwargs.items() if v not in (None, "")
            )[:200]
            return (
                f"[dedup] Already called `{tool_name}({args_preview})` earlier "
                "this turn. The prior result is in your message history above — "
                "use it instead of re-reading."
            )

        result = await orig_coro(**kwargs)
        cache[key] = True
        return result

    return StructuredTool.from_function(
        coroutine=_deduped_coro,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=tool.func,
    )


# Names of the read-style filesystem tools we want to dedup by default.  Other
# tools (shell, git, write_file, etc.) are NEVER deduped — they're side-effectful.
DEDUPABLE_READ_TOOLS = frozenset({
    "read_file", "list_directory", "search_files", "find_files",
})
