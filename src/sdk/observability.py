"""
Observability primitives for the SDLC Hub SDK.

Wraps the existing ``TraceCollector`` (``src/tracing/collector.py``) so that
traces produced via the SDK look identical to traces produced by the
orchestrator — same DB tables, same span types, same cost estimation,
same OTel pipeline.

The decorator adds two niceties on top of TraceCollector:

1. Automatic team_id tagging (no way to forget).
2. Capturing the wrapped function's return value as ``Trace.agent_response``
   so the result shows up in the UI's trace list and is eligible for
   later DeepEval scoring.
"""

from __future__ import annotations

import contextvars
import inspect
import logging
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Iterator

from src.db.database import get_session
from src.db.models import Trace as TraceModel
from src.tracing.collector import TraceCollector

logger = logging.getLogger(__name__)


# Context-var so nested ``hub.trace.span(...)`` calls find the active
# TraceCollector created by ``trace_agent`` without manual plumbing.
_active_collector: contextvars.ContextVar[TraceCollector | None] = (
    contextvars.ContextVar("hub_active_collector", default=None)
)


@dataclass
class TraceResult:
    """Return value of a function decorated by ``@trace_agent``.

    The wrapped function's actual return value is exposed via ``output``;
    ``trace_id`` is the persistent SQLite trace key — useful for jumping
    to the UI or running ``hub.eval.run(trace_id)`` afterwards.
    """

    trace_id: str
    output: Any
    error: str | None = None
    latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


class _SpanHandle:
    """Lightweight handle returned by ``hub.trace.span(...)``.

    Lets user code stamp tokens, model name, and arbitrary output_data
    onto the active span — exactly the contract used by the orchestrator's
    own span calls.
    """

    def __init__(self, span_id: str, collector: TraceCollector):
        self._span_id = span_id
        self._collector = collector
        self._tokens_in = 0
        self._tokens_out = 0
        self._model = ""
        self._output_data: dict[str, Any] = {}
        self._error: str | None = None

    def set_tokens(self, in_: int = 0, out_: int = 0, model: str = "") -> None:
        self._tokens_in = int(in_ or 0)
        self._tokens_out = int(out_ or 0)
        if model:
            self._model = str(model)

    def set_output(self, **fields: Any) -> None:
        for k, v in fields.items():
            self._output_data[str(k)] = v

    def set_error(self, message: str) -> None:
        self._error = message[:300]

    # The collector is closed by the surrounding context manager, not here.
    @property
    def span_id(self) -> str:
        return self._span_id


class ObservabilityClient:
    """Owns the team_id, exposes trace_agent + span."""

    def __init__(self, team_id: str) -> None:
        self.team_id = team_id

    # ── Public API ────────────────────────────────────────────────────

    def trace_agent(
        self,
        name: str,
        agent_used: str = "",
        version: str = "v1",
    ) -> Callable:
        """Decorator factory.

        Args:
            name:        Span name for the wrapping execution span. Convention
                         is ``"<agent>:<task>"`` to keep consistent with the
                         orchestrator's span naming.
            agent_used:  Logical agent label persisted on ``Trace.agent_used``.
                         Defaults to ``name`` when empty.
            version:     Free-form version string stored on the wrapping span.
        """

        def decorator(func: Callable) -> Callable:
            if inspect.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    return await self._run(func, args, kwargs, name, agent_used, version, is_async=True)
                async_wrapper.__wrapped__ = func  # type: ignore[attr-defined]
                async_wrapper.__name__ = func.__name__
                return async_wrapper

            def sync_wrapper(*args, **kwargs):
                return self._run_sync(func, args, kwargs, name, agent_used, version)
            sync_wrapper.__wrapped__ = func  # type: ignore[attr-defined]
            sync_wrapper.__name__ = func.__name__
            return sync_wrapper

        return decorator

    @contextmanager
    def span(
        self,
        name: str,
        span_type: str = "tool_call",
        input_data: dict | None = None,
    ) -> Iterator[_SpanHandle]:
        """Context manager creating a child span on the active trace.

        Must be used inside a function wrapped by ``@trace_agent``. If no
        trace is active a no-op handle is returned and a single warning is
        logged so user code never crashes.
        """
        collector = _active_collector.get()
        if collector is None:
            logger.warning(
                "hub.trace.span(%r) called outside an active trace_agent. "
                "Span data will not be persisted.", name,
            )
            yield _SpanHandle(span_id="<noop>", collector=_NullCollector())
            return

        span_id = collector.start_span(name=name, span_type=span_type, input_data=input_data or {})
        handle = _SpanHandle(span_id=span_id, collector=collector)
        try:
            yield handle
        except Exception as exc:
            handle.set_error(str(exc))
            collector.end_span(
                span_id,
                output_data=handle._output_data or None,
                tokens_in=handle._tokens_in,
                tokens_out=handle._tokens_out,
                model=handle._model,
                error=handle._error,
            )
            raise
        else:
            collector.end_span(
                span_id,
                output_data=handle._output_data or None,
                tokens_in=handle._tokens_in,
                tokens_out=handle._tokens_out,
                model=handle._model,
                error=handle._error,
            )

    @asynccontextmanager
    async def aspan(
        self,
        name: str,
        span_type: str = "tool_call",
        input_data: dict | None = None,
    ) -> AsyncIterator[_SpanHandle]:
        """Async-friendly variant of ``span``. Behaves identically — provided
        for consistency with code that wants ``async with hub.trace.aspan(...)``.
        """
        with self.span(name=name, span_type=span_type, input_data=input_data) as handle:
            yield handle

    def current_trace_id(self) -> str | None:
        """Return the trace_id of the trace currently active on this task."""
        c = _active_collector.get()
        return c.trace_id if c else None

    # ── Internals ─────────────────────────────────────────────────────

    async def _run(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        name: str,
        agent_used: str,
        version: str,
        *,
        is_async: bool,
    ) -> TraceResult:
        user_prompt = self._extract_user_prompt(args, kwargs)
        collector = TraceCollector(team_id=self.team_id, user_prompt=user_prompt)
        token = _active_collector.set(collector)
        outer_span = collector.start_span(
            name=name,
            span_type="agent_execution",
            input_data={"prompt": user_prompt[:300], "version": version},
        )
        error: str | None = None
        output: Any = None
        try:
            output = await func(*args, **kwargs) if is_async else func(*args, **kwargs)
            return self._persist(
                collector,
                outer_span,
                output=output,
                agent_used=agent_used or name,
                error=None,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._persist(
                collector,
                outer_span,
                output=None,
                agent_used=agent_used or name,
                error=error,
            )
            raise
        finally:
            _active_collector.reset(token)

    def _run_sync(self, func, args, kwargs, name, agent_used, version):
        user_prompt = self._extract_user_prompt(args, kwargs)
        collector = TraceCollector(team_id=self.team_id, user_prompt=user_prompt)
        token = _active_collector.set(collector)
        outer_span = collector.start_span(
            name=name, span_type="agent_execution",
            input_data={"prompt": user_prompt[:300], "version": version},
        )
        try:
            output = func(*args, **kwargs)
            return self._persist(collector, outer_span, output=output,
                                 agent_used=agent_used or name, error=None)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._persist(collector, outer_span, output=None,
                          agent_used=agent_used or name, error=error)
            raise
        finally:
            _active_collector.reset(token)

    def _persist(
        self,
        collector: TraceCollector,
        outer_span_id: str,
        *,
        output: Any,
        agent_used: str,
        error: str | None,
    ) -> TraceResult:
        response_text = self._stringify_output(output)
        collector.end_span(
            outer_span_id,
            output_data={"response_preview": response_text[:300]} if response_text else None,
            error=error,
        )
        collector.save()

        # Set agent_used / agent_response on the persisted Trace row so the
        # UI's trace list and the /api/traces/evaluate endpoint can find it.
        # Mirror the same fields the orchestrator sets on its traces.
        session = get_session()
        try:
            tr = session.query(TraceModel).filter(TraceModel.id == collector.trace_id).one_or_none()
            if tr is not None:
                tr.agent_used = agent_used or "sdk"
                if response_text:
                    tr.agent_response = response_text[:2000]
                tr.tool_calls_json = [
                    {"tool": s.get("name", ""), "args": s.get("input_data", {})}
                    for s in collector.spans
                    if s.get("span_type") == "tool_call"
                ]
                if error:
                    tr.status = "error"
                session.commit()
        finally:
            session.close()

        latency_ms = sum(
            ((s.get("end_time") or s.get("start_time")) - s["start_time"]).total_seconds() * 1000
            for s in collector.spans
            if s.get("start_time") and s.get("end_time")
        )
        total_tokens = sum(s.get("tokens_in", 0) + s.get("tokens_out", 0) for s in collector.spans)
        total_cost = sum(s.get("cost", 0.0) for s in collector.spans)

        return TraceResult(
            trace_id=collector.trace_id,
            output=output,
            error=error,
            latency_ms=round(latency_ms, 1),
            total_tokens=int(total_tokens),
            total_cost=round(total_cost, 6),
        )

    @staticmethod
    def _extract_user_prompt(args: tuple, kwargs: dict) -> str:
        """Best-effort capture of the wrapped function's main input.

        Convention: the first positional string argument or a kwarg called
        ``prompt`` / ``question`` / ``query`` is taken as the user prompt.
        Falls back to ``""`` when no string-like argument is found.
        """
        for k in ("prompt", "question", "query", "user_prompt", "input"):
            if k in kwargs and isinstance(kwargs[k], str):
                return kwargs[k]
        for a in args:
            if isinstance(a, str):
                return a
        return ""

    @staticmethod
    def _stringify_output(output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        try:
            return str(output)
        except Exception:
            return "<unstringifiable response>"


class _NullCollector:
    """Sentinel collector used when ``span`` is called outside a trace.

    It implements just enough of the TraceCollector surface that
    ``_SpanHandle`` doesn't crash. All operations are silent no-ops.
    """

    spans: list = []
    trace_id: str = ""

    def start_span(self, *_a, **_kw) -> str:  # pragma: no cover - sentinel
        return uuid.uuid4().hex[:12]

    def end_span(self, *_a, **_kw) -> None:  # pragma: no cover - sentinel
        return None
