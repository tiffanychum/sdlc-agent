"""
ReportsClient — programmatic, team-scoped retrieval of observability data.

Calls the same in-process helpers used by the FastAPI endpoints so the
numbers shown in the UI and the numbers returned to SDK callers are
always identical.

Methods:
    list_traces        — paginated trace list for the team
    trace              — single trace + spans
    performance_report — full ``/api/traces/performance-report`` payload
    anomalies          — z-score latency anomalies (subset of the report)
    stats              — quick aggregate (counts, evaluated, errors)
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import desc

from src.db.database import get_session
from src.db.models import Span, Trace as TraceModel


class ReportsClient:
    def __init__(self, team_id: str) -> None:
        self.team_id = team_id

    # ── Traces ────────────────────────────────────────────────────────

    def list_traces(self, *, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
        """Most recent traces for this team, newest first.

        Args:
            limit:   page size, capped at 200.
            status:  optional filter — ``"completed"``, ``"error"`` or ``"running"``.
        """
        limit = max(1, min(int(limit), 200))
        session = get_session()
        try:
            q = session.query(TraceModel).filter(TraceModel.team_id == self.team_id)
            if status:
                q = q.filter(TraceModel.status == status)
            rows = q.order_by(desc(TraceModel.created_at)).limit(limit).all()
            return [self._trace_to_dict(t) for t in rows]
        finally:
            session.close()

    def trace(self, trace_id: str) -> dict[str, Any] | None:
        """Single trace + ordered span list (None if not found / wrong team)."""
        session = get_session()
        try:
            tr = session.query(TraceModel).filter(TraceModel.id == trace_id).one_or_none()
            if tr is None or (tr.team_id and tr.team_id != self.team_id):
                return None
            spans = (
                session.query(Span)
                .filter(Span.trace_id == trace_id)
                .order_by(Span.start_time.asc())
                .all()
            )
            payload = self._trace_to_dict(tr)
            payload["spans"] = [self._span_to_dict(s) for s in spans]
            return payload
        finally:
            session.close()

    # ── Aggregate analytics ──────────────────────────────────────────

    def performance_report(self, *, days: int = 7) -> dict[str, Any]:
        """Full performance report scoped to this team.

        Same payload shape as ``GET /api/traces/performance-report``.
        """
        from src.tracing.analyzer import TraceAnalyzer
        analyzer = TraceAnalyzer(team_id=self.team_id)
        return analyzer.full_report(days=days)

    def anomalies(self, *, days: int = 7, z_threshold: float = 2.5) -> list[dict[str, Any]]:
        """Latency outliers (z-score >= threshold) for this team."""
        from src.tracing.analyzer import TraceAnalyzer
        analyzer = TraceAnalyzer(team_id=self.team_id)
        return analyzer.performance_anomalies(days=days, z_threshold=z_threshold)

    def otel_stats(self, *, days: int = 30) -> dict[str, Any]:
        """Team-scoped OTel span analytics — same payload as
        ``GET /api/otel/spans/stats?team_id=…``.

        Includes:
          * ``by_model`` — cost/tokens/throughput per LLM model with
            differently-spelled aliases consolidated.
          * ``top_errors`` — top-10 failure categories.
          * ``errors_by_model`` — top-5 failures per model.
        """
        # We import lazily to avoid a circular import at module load —
        # `server` imports `src.sdk` indirectly through TraceCollector.
        from server import otel_span_stats
        return otel_span_stats(days=days, team_id=self.team_id)

    def stats(self) -> dict[str, Any]:
        """Lightweight dashboard-style aggregate — counts, eval rate, errors."""
        session = get_session()
        try:
            base = session.query(TraceModel).filter(TraceModel.team_id == self.team_id)
            total = base.count()
            evaluated = base.filter(TraceModel.eval_status == "evaluated").count()
            errored = base.filter(TraceModel.status == "error").count()
            completed = base.filter(TraceModel.status == "completed").count()
            return {
                "team_id": self.team_id,
                "total_traces": total,
                "completed": completed,
                "errored": errored,
                "evaluated": evaluated,
                "pending_evaluation": total - evaluated,
            }
        finally:
            session.close()

    # ── Convenience ──────────────────────────────────────────────────

    def ui_url(self, base_url: str = "http://localhost:3000") -> str:
        """Return a deep link to the monitoring page filtered to this team."""
        return f"{base_url.rstrip('/')}/monitoring?team={self.team_id}"

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _trace_to_dict(t: TraceModel) -> dict[str, Any]:
        return {
            "id": t.id,
            "team_id": t.team_id,
            "user_prompt": t.user_prompt,
            "agent_used": t.agent_used,
            "agent_response": t.agent_response,
            "tool_calls": t.tool_calls_json or [],
            "total_latency_ms": t.total_latency_ms,
            "total_tokens": t.total_tokens,
            "total_cost": t.total_cost,
            "status": t.status,
            "eval_status": t.eval_status,
            "eval_scores": t.eval_scores or {},
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }

    @staticmethod
    def _span_to_dict(s: Span) -> dict[str, Any]:
        duration_ms: float | None = None
        if s.start_time and s.end_time:
            duration_ms = round((s.end_time - s.start_time).total_seconds() * 1000, 1)
        return {
            "id": s.id,
            "trace_id": s.trace_id,
            "parent_span_id": s.parent_span_id,
            "name": s.name,
            "span_type": s.span_type,
            "duration_ms": duration_ms,
            "tokens_in": s.tokens_in,
            "tokens_out": s.tokens_out,
            "cost": s.cost,
            "model": s.model,
            "status": s.status,
            "error": s.error,
        }
