"""
TraceAnalyzer — performance profiling for agent execution traces.

Mirrors the observability methodology used in production performance analysis
systems: compute latency percentiles, failure rates, utilisation trends, and
cost breakdowns across the agent execution history stored in SQLite.

Key metrics:
  - Agent call latency: p50 / p95 / p99 in milliseconds
  - Tool failure rates: per tool and overall
  - Context window utilisation: tokens_in / max_tokens as a trend
  - Cost breakdown: per agent role and per model

All methods operate on data already in the DB (Trace + Span tables), so no
additional instrumentation is required — this is a pure analytics layer.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta
from typing import Optional


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of *values* (0 ≤ p ≤ 100).

    Uses the nearest-rank method (same as NumPy's percentile with method='higher').
    Returns 0.0 for empty lists.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if p == 0:
        return sorted_vals[0]
    if p == 100:
        return sorted_vals[-1]
    idx = math.ceil((p / 100) * len(sorted_vals)) - 1
    return sorted_vals[max(0, idx)]


def _normalize_model_name(raw: str) -> str:
    """
    Deduplicate and normalise model name strings.

    Some model names get recorded as concatenated duplicates in the DB
    (e.g. "claude-haiku-3claude-haiku-3") due to a recording bug where
    the model is appended instead of replaced.  This function detects
    the pattern and returns the single-occurrence version.

    Also normalises casing differences like "Claude-Sonnet-4" → "claude-sonnet-4".
    """
    if not raw:
        return raw
    # Deduplicate: if the string is an exact N-fold repetition, return 1 copy
    for n in range(2, 7):
        chunk = len(raw) // n
        if chunk > 0 and raw == raw[:chunk] * n:
            return raw[:chunk]
    # Also catch partial repetition (repeated with different separator wouldn't
    # be caught above, but the DB pattern is always exact concatenation)
    return raw


# Canonical model name aliases — map raw API names to display names
_MODEL_ALIASES: dict[str, str] = {
    "claude-sonnet-4.6": "claude-sonnet-4.6",
    "claude-sonnet-4": "claude-sonnet-4",
    "Claude-Sonnet-4": "claude-sonnet-4",
    "claude-3-5-sonnet-20241022": "claude-3.5-sonnet",
    "claude-haiku-3": "claude-haiku-3",
    "claude-haiku-3.5": "claude-haiku-3.5",
    "Claude-Haiku-3.5": "claude-haiku-3.5",
    "claude-opus-4.5": "claude-opus-4.5",
    "claude-opus-4.6": "claude-opus-4.6",
    "Claude-Opus-4": "claude-opus-4",
    "gpt-5": "gpt-5",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-5.3-codex": "gpt-5.3-codex",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-3-flash": "gemini-3-flash",
    "gemini-3.1-pro": "gemini-3.1-pro",
}


def _canonical_model(raw: str) -> str:
    """Deduplicate + apply alias → returns a clean, display-ready model name."""
    deduped = _normalize_model_name(raw)
    return _MODEL_ALIASES.get(deduped, deduped) or deduped or "unknown"


class TraceAnalyzer:
    """
    Computes performance analytics over the agent execution history.

    Usage:
        analyzer = TraceAnalyzer()
        report = analyzer.full_report(days=7)
    """

    def __init__(self, team_id: Optional[str] = None):
        self.team_id = team_id

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_spans(
        self,
        since: Optional[datetime] = None,
        span_type: Optional[str] = None,
    ) -> list[dict]:
        """Load spans from DB, optionally filtered by time window and type."""
        from src.db.database import get_session
        from src.db.models import Span, Trace

        session = get_session()
        try:
            q = session.query(Span)
            if self.team_id or since:
                q = q.join(Trace, Span.trace_id == Trace.id)
                if self.team_id:
                    q = q.filter(Trace.team_id == self.team_id)
                if since:
                    q = q.filter(Trace.created_at >= since)
            if span_type:
                q = q.filter(Span.span_type == span_type)

            spans = q.all()
            return [
                {
                    "id": s.id,
                    "trace_id": s.trace_id,
                    "name": s.name,
                    "span_type": s.span_type,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "tokens_in": s.tokens_in or 0,
                    "tokens_out": s.tokens_out or 0,
                    "cost": s.cost or 0.0,
                    "model": _canonical_model(s.model or ""),
                    "status": s.status or "completed",
                    "error": s.error,
                }
                for s in spans
            ]
        finally:
            session.close()

    def _load_traces(self, since: Optional[datetime] = None) -> list[dict]:
        """Load traces from DB for cost/agent breakdown."""
        from src.db.database import get_session
        from src.db.models import Trace

        session = get_session()
        try:
            q = session.query(Trace)
            if self.team_id:
                q = q.filter(Trace.team_id == self.team_id)
            if since:
                q = q.filter(Trace.created_at >= since)
            traces = q.all()

            # For traces where agent_used was never written (recording bug),
            # fall back to reading routing/supervisor spans for this trace.
            trace_ids_missing_agent = [t.id for t in traces if not t.agent_used]
            agent_from_spans: dict[str, str] = {}
            if trace_ids_missing_agent:
                import json as _json
                from src.db.models import Span
                # 1st preference: routing spans with output_data["selected_agent"]
                routing_spans = (
                    session.query(Span)
                    .filter(
                        Span.trace_id.in_(trace_ids_missing_agent),
                        Span.span_type.in_(["routing", "supervisor"]),
                    )
                    .all()
                )
                for sp in routing_spans:
                    try:
                        od = (_json.loads(sp.output_data)
                              if isinstance(sp.output_data, str) else sp.output_data or {})
                    except Exception:
                        od = {}
                    sel = od.get("selected_agent") or od.get("decision", "")
                    if sel and sel not in ("unknown", "FINISH", "END"):
                        existing = agent_from_spans.get(sp.trace_id, "")
                        if sel not in existing:
                            agent_from_spans[sp.trace_id] = (
                                f"{existing} > {sel}" if existing else sel
                            )
                # 2nd preference: named agent_execution spans (e.g. "agent:coder")
                agent_exec_spans = (
                    session.query(Span)
                    .filter(
                        Span.trace_id.in_(trace_ids_missing_agent),
                        Span.span_type == "agent_execution",
                    )
                    .all()
                )
                for sp in agent_exec_spans:
                    if sp.trace_id in agent_from_spans:
                        continue  # already resolved via routing span
                    role = sp.name.split(":")[-1] if ":" in sp.name else sp.name
                    if role and role != "agent":  # skip generic "agent" label
                        existing = agent_from_spans.get(sp.trace_id, "")
                        if role not in existing:
                            agent_from_spans[sp.trace_id] = (
                                f"{existing} > {role}" if existing else role
                            )

            # Collect trace IDs that have any agent_execution spans (multi-agent runs)
            traces_with_exec: set[str] = set()
            if trace_ids_missing_agent:
                from src.db.models import Span as _Span2
                exec_trace_ids = (
                    session.query(_Span2.trace_id)
                    .filter(
                        _Span2.trace_id.in_(trace_ids_missing_agent),
                        _Span2.span_type == "agent_execution",
                    )
                    .distinct()
                    .all()
                )
                traces_with_exec = {r[0] for r in exec_trace_ids}

            return [
                {
                    "id": t.id,
                    "agent_used": (
                        t.agent_used
                        or agent_from_spans.get(t.id, "")
                        or ("orchestrated" if t.id in traces_with_exec else "chat")
                    ),
                    "total_tokens": t.total_tokens or 0,
                    "total_cost": t.total_cost or 0.0,
                    "total_latency_ms": t.total_latency_ms or 0.0,
                    "status": t.status or "completed",
                    "created_at": t.created_at,
                }
                for t in traces
            ]
        finally:
            session.close()

    # ── Latency percentiles ───────────────────────────────────────────────────

    def agent_latency_percentiles(
        self,
        days: int = 7,
    ) -> dict[str, dict]:
        """
        Compute p50 / p95 / p99 call latency (ms) per agent role.

        Uses named agent spans ("agent:role") from the orchestrator for per-role
        breakdown.  Generic "agent" spans (from LangGraph auto-instrumentation)
        are included in _overall only — they don't carry role information.

        Returns:
            {
              "planner":  {"p50": 1234, "p95": 3456, "p99": 5678, "count": 42, "avg": 2000},
              "coder":    {...},
              ...
              "_overall": {"p50": ..., ...}   # all agent_execution spans combined
            }
        """
        since = datetime.utcnow() - timedelta(days=days)
        spans = self._load_spans(since=since, span_type="agent_execution")

        latencies_by_role: dict[str, list[float]] = {}
        all_latencies: list[float] = []

        for s in spans:
            if not (s["start_time"] and s["end_time"]):
                continue
            dur_ms = (s["end_time"] - s["start_time"]).total_seconds() * 1000
            if dur_ms < 1:
                # Skip near-zero durations — these are synthetic post-hoc spans
                # without real timing (recorded before the latency_ms fix).
                continue

            raw_name = s["name"]
            if ":" in raw_name:
                role = raw_name.split(":")[-1]
            elif raw_name == "agent":
                # Generic LangGraph auto-instrumented span — contributes to
                # _overall only, not per-role breakdown.
                all_latencies.append(dur_ms)
                continue
            else:
                role = raw_name

            latencies_by_role.setdefault(role, []).append(dur_ms)
            all_latencies.append(dur_ms)

        # Supplement with trace-level latency for single-agent traces.
        # For traces where only ONE agent ran (agent_used has no " > "), the total
        # trace latency is a good proxy for that agent's execution time.
        traces = self._load_traces(since=since)
        for t in traces:
            agent = t["agent_used"]
            if not agent or " > " in agent or agent in ("orchestrated", "chat", "agent"):
                continue  # skip multi-agent or unattributable traces
            lat = t["total_latency_ms"]
            if lat and lat > 100:
                latencies_by_role.setdefault(agent, []).append(lat)
                all_latencies.append(lat)

        result: dict[str, dict] = {}

        for role, lats in sorted(latencies_by_role.items()):
            result[role] = {
                "p50": round(_percentile(lats, 50), 1),
                "p95": round(_percentile(lats, 95), 1),
                "p99": round(_percentile(lats, 99), 1),
                "avg": round(statistics.mean(lats), 1) if lats else 0.0,
                "count": len(lats),
                "max": round(max(lats), 1) if lats else 0.0,
            }

        if all_latencies:
            result["_overall"] = {
                "p50": round(_percentile(all_latencies, 50), 1),
                "p95": round(_percentile(all_latencies, 95), 1),
                "p99": round(_percentile(all_latencies, 99), 1),
                "avg": round(statistics.mean(all_latencies), 1),
                "count": len(all_latencies),
                "max": round(max(all_latencies), 1),
            }

        return result

    def tool_latency_percentiles(self, days: int = 7) -> dict[str, dict]:
        """
        Compute p50 / p95 / p99 latency (ms) per tool call.

        Returns per-tool percentiles, sorted by p99 descending (slowest first).
        Useful for identifying which tools are bottlenecks.
        """
        since = datetime.utcnow() - timedelta(days=days)
        spans = self._load_spans(since=since, span_type="tool_call")

        latencies_by_tool: dict[str, list[float]] = {}
        for s in spans:
            if s["start_time"] and s["end_time"]:
                dur_ms = (s["end_time"] - s["start_time"]).total_seconds() * 1000
                tool = s["name"].split(":")[-1] if ":" in s["name"] else s["name"]
                latencies_by_tool.setdefault(tool, []).append(dur_ms)

        result: dict[str, dict] = {}
        for tool, lats in latencies_by_tool.items():
            result[tool] = {
                "p50": round(_percentile(lats, 50), 1),
                "p95": round(_percentile(lats, 95), 1),
                "p99": round(_percentile(lats, 99), 1),
                "avg": round(statistics.mean(lats), 1) if lats else 0.0,
                "count": len(lats),
            }

        # Sort by p99 descending
        return dict(sorted(result.items(), key=lambda kv: -kv[1]["p99"]))

    # ── Tool failure rates ────────────────────────────────────────────────────

    def tool_failure_rates(self, days: int = 7) -> dict[str, dict]:
        """
        Compute per-tool call success/failure rates.

        Returns:
            {
              "write_file":   {"total": 45, "failed": 3, "failure_rate_pct": 6.7},
              "web_search":   {"total": 30, "failed": 1, "failure_rate_pct": 3.3},
              ...
              "_overall":     {"total": ..., "failed": ..., "failure_rate_pct": ...}
            }
        """
        since = datetime.utcnow() - timedelta(days=days)
        spans = self._load_spans(since=since, span_type="tool_call")

        stats: dict[str, dict] = {}
        for s in spans:
            tool = s["name"].split(":")[-1] if ":" in s["name"] else s["name"]
            if tool not in stats:
                stats[tool] = {"total": 0, "failed": 0}
            stats[tool]["total"] += 1
            if s["status"] == "error" or s["error"]:
                stats[tool]["failed"] += 1

        result: dict[str, dict] = {}
        total_calls = 0
        total_failed = 0

        for tool, s in sorted(stats.items()):
            rate = (s["failed"] / s["total"] * 100) if s["total"] else 0.0
            result[tool] = {
                "total": s["total"],
                "failed": s["failed"],
                "failure_rate_pct": round(rate, 1),
            }
            total_calls += s["total"]
            total_failed += s["failed"]

        if total_calls:
            result["_overall"] = {
                "total": total_calls,
                "failed": total_failed,
                "failure_rate_pct": round(total_failed / total_calls * 100, 1),
            }

        return result

    # ── Context window utilisation ────────────────────────────────────────────

    def context_window_utilization(
        self,
        days: int = 7,
        model_max_tokens: Optional[dict[str, int]] = None,
    ) -> dict[str, dict]:
        """
        Compute context window utilisation trend: (tokens_in / model_max_tokens) per model.

        Args:
            days: Look-back window.
            model_max_tokens: Map of model name → context window size.
                Defaults to known model sizes.

        Returns:
            {
              "Claude-Sonnet-4": {
                "max_context_tokens": 200000,
                "avg_tokens_in": 12345,
                "p99_tokens_in": 45678,
                "avg_utilization_pct": 6.2,
                "p99_utilization_pct": 22.8,
                "count": 15,
                "at_risk_count": 1,    # calls where utilization > 80%
              },
              ...
            }
        """
        _defaults: dict[str, int] = {
            "Claude-Sonnet-4": 200_000,
            "claude-sonnet-4": 200_000,
            "Claude-Haiku-3.5": 200_000,
            "Claude-Opus-4": 200_000,
            "gpt-4o": 128_000,
            "gpt-4o-mini": 128_000,
            "gpt-5-mini": 128_000,
            "gemini-2.5-flash-lite": 1_000_000,
            "gemini-3-flash": 1_000_000,
            "default": 128_000,
        }
        limits = {**_defaults, **(model_max_tokens or {})}

        since = datetime.utcnow() - timedelta(days=days)
        spans = self._load_spans(since=since, span_type="llm_call")

        # Group token_in values by model. Normalise so multiple spellings
        # of the same model don't produce separate context-utilisation
        # rows in the dashboard.
        from src.utils.model_names import normalize_model_name

        by_model: dict[str, list[int]] = {}
        for s in spans:
            raw = s["model"] or "default"
            model = normalize_model_name(raw) or "default"
            if s["tokens_in"] > 0:
                by_model.setdefault(model, []).append(s["tokens_in"])

        result: dict[str, dict] = {}
        for model, token_counts in by_model.items():
            max_ctx = limits.get(model, limits["default"])
            utilizations = [t / max_ctx * 100 for t in token_counts]
            at_risk = sum(1 for u in utilizations if u > 80)
            result[model] = {
                "max_context_tokens": max_ctx,
                "avg_tokens_in": round(statistics.mean(token_counts)),
                "p95_tokens_in": round(_percentile(token_counts, 95)),
                "p99_tokens_in": round(_percentile(token_counts, 99)),
                "avg_utilization_pct": round(statistics.mean(utilizations), 1),
                "p99_utilization_pct": round(_percentile(utilizations, 99), 1),
                "count": len(token_counts),
                "at_risk_count": at_risk,  # calls > 80% context filled
            }

        return result

    # ── Cost breakdown ────────────────────────────────────────────────────────

    def cost_breakdown(self, days: int = 7) -> dict[str, dict]:
        """
        Cost breakdown per agent role and per model.

        Returns:
            {
              "by_agent": {
                "coder":      {"total_cost_usd": 0.045, "avg_cost_usd": 0.003, "count": 15},
                "researcher": {...},
                ...
              },
              "by_model": {
                "Claude-Sonnet-4": {"total_cost_usd": 0.12, "calls": 40},
                ...
              },
              "summary": {
                "total_cost_usd": 0.18,
                "total_calls": 55,
                "avg_cost_per_call_usd": 0.0033,
                "most_expensive_agent": "coder",
                "most_expensive_model": "Claude-Sonnet-4",
              }
            }
        """
        since = datetime.utcnow() - timedelta(days=days)
        traces = self._load_traces(since=since)

        by_agent: dict[str, dict] = {}
        for t in traces:
            agent = t["agent_used"] or "unknown"
            if agent not in by_agent:
                by_agent[agent] = {"total_cost_usd": 0.0, "count": 0}
            by_agent[agent]["total_cost_usd"] += t["total_cost"]
            by_agent[agent]["count"] += 1

        for agent, stats in by_agent.items():
            stats["avg_cost_usd"] = round(
                stats["total_cost_usd"] / stats["count"], 6
            ) if stats["count"] else 0.0
            stats["total_cost_usd"] = round(stats["total_cost_usd"], 6)

        # Per-model breakdown from spans. We normalise the model name so
        # different spellings of the same underlying model (e.g.
        # "claude-sonnet-4-6" and "claude-sonnet-4.6") collapse into one
        # row, matching what the OTel Observability tab shows.
        from src.utils.model_names import normalize_model_name

        spans = self._load_spans(since=since, span_type="llm_call")
        by_model: dict[str, dict] = {}
        for s in spans:
            raw = s["model"] or "unknown"
            model = normalize_model_name(raw) or "unknown"
            if model not in by_model:
                by_model[model] = {"total_cost_usd": 0.0, "calls": 0}
            by_model[model]["total_cost_usd"] += s["cost"]
            by_model[model]["calls"] += 1

        for m in by_model.values():
            m["total_cost_usd"] = round(m["total_cost_usd"], 6)

        total_cost = sum(t["total_cost"] for t in traces)
        total_calls = len(traces)

        most_expensive_agent = max(
            by_agent.items(), key=lambda kv: kv[1]["total_cost_usd"], default=(None, {})
        )[0]
        most_expensive_model = max(
            by_model.items(), key=lambda kv: kv[1]["total_cost_usd"], default=(None, {})
        )[0]

        return {
            "by_agent": dict(sorted(
                by_agent.items(), key=lambda kv: -kv[1]["total_cost_usd"]
            )),
            "by_model": dict(sorted(
                by_model.items(), key=lambda kv: -kv[1]["total_cost_usd"]
            )),
            "summary": {
                "total_cost_usd": round(total_cost, 6),
                "total_calls": total_calls,
                "avg_cost_per_call_usd": round(total_cost / total_calls, 6) if total_calls else 0.0,
                "most_expensive_agent": most_expensive_agent,
                "most_expensive_model": most_expensive_model,
                "period_days": days,
            },
        }

    # ── Anomaly detection on agent traces ─────────────────────────────────────

    def performance_anomalies(self, days: int = 7, z_threshold: float = 2.5) -> list[dict]:
        """
        Detect latency anomalies in agent/tool spans using z-score thresholding.

        Any span whose duration is more than *z_threshold* standard deviations above
        the mean for its span type is flagged as an anomaly.

        Returns a list of anomaly dicts sorted by severity (z-score descending):
            [
              {
                "span_type": "tool_call",
                "name": "read_file",
                "duration_ms": 4500,
                "mean_ms": 120,
                "z_score": 3.8,
                "severity": "CRITICAL",
                "trace_id": "abc123",
              },
              ...
            ]
        """
        since = datetime.utcnow() - timedelta(days=days)

        # Load spans with their parent trace context for richer anomaly details
        from src.db.database import get_session
        from src.db.models import Span as SpanModel, Trace as TraceModel
        session = get_session()
        try:
            rows = (
                session.query(SpanModel, TraceModel)
                .join(TraceModel, SpanModel.trace_id == TraceModel.id)
                .filter(TraceModel.created_at >= since)
            )
            if self.team_id:
                rows = rows.filter(TraceModel.team_id == self.team_id)
            span_trace_pairs = rows.all()
        finally:
            session.close()

        durations_by_type: dict[str, list[tuple]] = {}
        for s, t in span_trace_pairs:
            if s.start_time and s.end_time:
                dur = (s.end_time - s.start_time).total_seconds() * 1000
                key = s.span_type or "unknown"
                durations_by_type.setdefault(key, []).append((dur, s, t))

        anomalies: list[dict] = []
        for span_type, entries in durations_by_type.items():
            durs = [e[0] for e in entries]
            if len(durs) < 5:
                continue
            mean = statistics.mean(durs)
            stdev = statistics.stdev(durs)
            if stdev == 0:
                continue
            for dur, s, t in entries:
                z = (dur - mean) / stdev
                if z >= z_threshold:
                    severity = "CRITICAL" if z >= 4.0 else "WARNING"
                    raw_name = s.name or ""
                    tool_name = raw_name.split(":")[-1] if ":" in raw_name else raw_name
                    # Derive agent role from span name (convention: "agent_name:tool_name")
                    agent_role = raw_name.split(":")[0] if ":" in raw_name else ""
                    # Task description from parent trace user_prompt
                    task_desc = (getattr(t, "user_prompt", "") or "")[:120]
                    anomalies.append({
                        "span_type": span_type,
                        "name": tool_name,
                        "agent_role": agent_role,
                        "task": task_desc,
                        "prompt_version": getattr(t, "prompt_version", "v1") or "v1",
                        "duration_ms": round(dur, 1),
                        "mean_ms": round(mean, 1),
                        "stdev_ms": round(stdev, 1),
                        "z_score": round(z, 2),
                        "severity": severity,
                        "trace_id": s.trace_id,
                        "model": _canonical_model(s.model or ""),
                    })

        return sorted(anomalies, key=lambda a: -a["z_score"])

    # ── Regression metric insights ────────────────────────────────────────────

    def regression_metric_insights(self, days: int = 30) -> dict:
        """
        Aggregate DeepEval and G-Eval scores across all RegressionResult rows to
        surface which metrics score consistently lowest, which golden tests are
        most expensive/slowest, and which agent roles appear most in failed runs.

        Returns:
            {
              "metric_averages": {            # per-metric avg across all runs
                "faithfulness": {"avg": 0.68, "min": 0.21, "count": 45, "below_threshold": 12},
                "answer_relevancy": {...},
                ...
              },
              "worst_metrics": [             # sorted lowest avg first
                {"metric": "faithfulness", "avg": 0.68, "below_threshold": 12},
                ...
              ],
              "costliest_tests": [           # golden tests by avg cost
                {"golden_case_id": "golden_030", "golden_case_name": "...",
                 "avg_cost": 0.045, "avg_latency_ms": 120000, "runs": 5},
                ...
              ],
              "slowest_tests": [...],        # golden tests by avg latency
              "failed_agent_patterns": {    # agent roles in failed-run delegation patterns
                "coder": 12, "devops": 8, ...
              },
              "pass_rate_by_test": {         # pass rate per golden test
                "golden_001": {"passes": 4, "total": 5, "pass_rate": 0.80},
                ...
              },
              "summary": {
                "total_runs": 150,
                "pass_rate": 0.73,
                "worst_metric": "faithfulness",
                "worst_test": "golden_030",
                "most_failed_agent": "coder",
              }
            }
        """
        from src.db.database import get_session
        from src.db.models import RegressionResult

        since = datetime.utcnow() - timedelta(days=days)
        session = get_session()
        try:
            rows = session.query(RegressionResult).filter(
                RegressionResult.created_at >= since
            ).all()
        finally:
            session.close()

        if not rows:
            return {
                "metric_averages": {},
                "worst_metrics": [],
                "costliest_tests": [],
                "slowest_tests": [],
                "failed_agent_patterns": {},
                "pass_rate_by_test": {},
                "summary": {"total_runs": 0, "pass_rate": 0.0},
            }

        # ── Aggregate DeepEval + G-Eval scores per metric ─────────────────────
        metric_values: dict[str, list[float]] = {}
        # DeepEval thresholds
        _de_thresholds: dict[str, float] = {
            "answer_relevancy": 0.70, "faithfulness": 0.70,
            "contextual_recall": 0.65, "hallucination": 0.30,
            "semantic_similarity": 0.60,
        }
        # G-Eval thresholds
        _geval_thresholds: dict[str, float] = {
            "correctness": 3.5, "relevance": 3.5, "coherence": 3.5,
            "tool_usage_quality": 3.5, "completeness": 3.5,
        }

        for row in rows:
            de = row.deepeval_scores or {}
            qe = row.quality_scores or {}
            for k, v in de.items():
                if isinstance(v, (int, float)):
                    metric_values.setdefault(k, []).append(float(v))
            for k, v in qe.items():
                if isinstance(v, (int, float)):
                    metric_values.setdefault(k, []).append(float(v))

        all_thresholds = {**_de_thresholds, **_geval_thresholds}
        metric_averages: dict[str, dict] = {}
        for metric, vals in metric_values.items():
            threshold = all_thresholds.get(metric, 3.5 if max(vals) > 1.5 else 0.7)
            # hallucination: lower is better — flag when too high
            if metric == "hallucination":
                below = sum(1 for v in vals if v > threshold)
            else:
                below = sum(1 for v in vals if v < threshold)
            metric_averages[metric] = {
                "avg": round(statistics.mean(vals), 3),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
                "count": len(vals),
                "below_threshold": below,
                "threshold": threshold,
            }

        # Sort: for hallucination sort by avg descending (higher=worse),
        # for all others sort ascending (lower=worse)
        def _sort_key(item):
            name, d = item
            if name == "hallucination":
                return -d["avg"]
            return d["avg"]

        worst_metrics = sorted(
            [{"metric": m, **v} for m, v in metric_averages.items()],
            key=lambda x: (x["avg"] if x["metric"] != "hallucination" else -x["avg"])
        )

        # ── Cost and latency by golden test ───────────────────────────────────
        test_stats: dict[str, dict] = {}
        for row in rows:
            tid = row.golden_case_id
            if tid not in test_stats:
                test_stats[tid] = {
                    "golden_case_id": tid,
                    "golden_case_name": row.golden_case_name or tid,
                    "costs": [], "latencies": [], "passes": 0, "total": 0,
                }
            test_stats[tid]["costs"].append(row.actual_cost or 0.0)
            test_stats[tid]["latencies"].append(row.actual_latency_ms or 0.0)
            test_stats[tid]["total"] += 1
            if row.overall_pass:
                test_stats[tid]["passes"] += 1

        costliest_tests = sorted(
            [
                {
                    "golden_case_id": v["golden_case_id"],
                    "golden_case_name": v["golden_case_name"],
                    "avg_cost": round(statistics.mean(v["costs"]), 6),
                    "avg_latency_ms": round(statistics.mean(v["latencies"]), 0),
                    "runs": v["total"],
                }
                for v in test_stats.values() if v["costs"]
            ],
            key=lambda x: -x["avg_cost"],
        )

        slowest_tests = sorted(costliest_tests, key=lambda x: -x["avg_latency_ms"])

        # ── Agent roles in failed delegation patterns ─────────────────────────
        failed_agent_counts: dict[str, int] = {}
        for row in rows:
            if not row.overall_pass:
                pattern = row.actual_delegation_pattern or []
                for agent in pattern:
                    if agent:
                        failed_agent_counts[agent] = failed_agent_counts.get(agent, 0) + 1

        # ── Pass rate by test ─────────────────────────────────────────────────
        pass_rate_by_test = {
            v["golden_case_id"]: {
                "passes": v["passes"],
                "total": v["total"],
                "pass_rate": round(v["passes"] / v["total"], 3) if v["total"] else 0.0,
            }
            for v in test_stats.values()
        }

        total_runs = len(rows)
        total_passes = sum(1 for r in rows if r.overall_pass)
        worst_metric = worst_metrics[0]["metric"] if worst_metrics else None
        worst_test = costliest_tests[0]["golden_case_id"] if costliest_tests else None
        most_failed_agent = max(
            failed_agent_counts.items(), key=lambda kv: kv[1], default=(None, 0)
        )[0]

        return {
            "metric_averages": metric_averages,
            "worst_metrics": worst_metrics[:10],
            "costliest_tests": costliest_tests[:10],
            "slowest_tests": slowest_tests[:10],
            "failed_agent_patterns": dict(sorted(
                failed_agent_counts.items(), key=lambda kv: -kv[1]
            )),
            "pass_rate_by_test": pass_rate_by_test,
            "summary": {
                "total_runs": total_runs,
                "pass_rate": round(total_passes / total_runs, 3) if total_runs else 0.0,
                "worst_metric": worst_metric,
                "worst_test": worst_test,
                "most_failed_agent": most_failed_agent,
                "period_days": days,
            },
        }

    def ab_compare(self, run_a_id: str, run_b_id: str) -> dict:
        """
        Side-by-side comparison of two eval runs (A/B analysis).

        Computes metric deltas, highlights statistically meaningful changes,
        and produces a radar-chart-ready data structure.

        Returns:
            {
              "run_a": {"id": "...", "model": "...", "created_at": "..."},
              "run_b": {"id": "...", "model": "...", "created_at": "..."},
              "metrics": {
                "faithfulness": {"run_a_avg": 0.72, "run_b_avg": 0.68,
                                 "delta": -0.04, "significant": false},
                ...
              },
              "performance": {
                "avg_cost_usd": {"run_a": 0.0052, "run_b": 0.0031, "delta_pct": -40.4},
                "avg_latency_ms": {"run_a": 28000, "run_b": 19000, "delta_pct": -32.1},
                "pass_rate": {"run_a": 0.75, "run_b": 0.85, "delta_pct": +13.3},
              },
              "radar_data": [   # for radar chart — normalised 0–1 scale
                {"metric": "faithfulness", "run_a": 0.72, "run_b": 0.68},
                ...
              ],
              "recommendation": "Run B is better — higher pass rate (+13%) at lower cost (-40%)",
              "test_comparison": [   # per-golden-test breakdown
                {"golden_case_id": "golden_001", "run_a_pass": true, "run_b_pass": true, ...},
              ]
            }
        """
        from src.db.database import get_session
        from src.db.models import RegressionResult, EvalRun

        session = get_session()
        try:
            run_a_meta = session.query(EvalRun).filter_by(id=run_a_id).first()
            run_b_meta = session.query(EvalRun).filter_by(id=run_b_id).first()
            rows_a = session.query(RegressionResult).filter_by(run_id=run_a_id).all()
            rows_b = session.query(RegressionResult).filter_by(run_id=run_b_id).all()
        finally:
            session.close()

        def _run_info(meta, run_id: str) -> dict:
            if not meta:
                return {"id": run_id, "model": "unknown", "created_at": None, "num_tasks": 0}
            return {
                "id": run_id,
                "model": meta.model or "unknown",
                "created_at": meta.created_at.isoformat() if meta.created_at else None,
                "num_tasks": meta.num_tasks or 0,
                "prompt_version": meta.prompt_version or "v1",
            }

        def _agg_metrics(rows) -> dict[str, list[float]]:
            agg: dict[str, list[float]] = {}
            for row in rows:
                for k, v in (row.deepeval_scores or {}).items():
                    if isinstance(v, (int, float)):
                        agg.setdefault(k, []).append(float(v))
                for k, v in (row.quality_scores or {}).items():
                    if isinstance(v, (int, float)):
                        agg.setdefault(k, []).append(float(v))
            return agg

        agg_a = _agg_metrics(rows_a)
        agg_b = _agg_metrics(rows_b)

        # Significance thresholds: 0.1 for DeepEval (0-1 scale), 0.3 for G-Eval (1-5 scale)
        all_metrics = set(agg_a.keys()) | set(agg_b.keys())
        metrics_compare: dict[str, dict] = {}
        for m in sorted(all_metrics):
            vals_a = agg_a.get(m, [])
            vals_b = agg_b.get(m, [])
            avg_a = round(statistics.mean(vals_a), 3) if vals_a else None
            avg_b = round(statistics.mean(vals_b), 3) if vals_b else None
            if avg_a is not None and avg_b is not None:
                delta = round(avg_b - avg_a, 3)
                sig_threshold = 0.3 if max(vals_a + vals_b) > 1.5 else 0.1
                metrics_compare[m] = {
                    "run_a_avg": avg_a, "run_b_avg": avg_b,
                    "delta": delta,
                    "significant": abs(delta) >= sig_threshold,
                    "better": "B" if delta > 0 else ("A" if delta < 0 else "equal"),
                }

        # Performance metrics
        def _perf(rows):
            if not rows:
                return {"avg_cost": 0.0, "avg_latency_ms": 0.0, "pass_rate": 0.0}
            costs = [r.actual_cost or 0 for r in rows]
            lats = [r.actual_latency_ms or 0 for r in rows]
            passes = sum(1 for r in rows if r.overall_pass)
            return {
                "avg_cost": round(statistics.mean(costs), 6),
                "avg_latency_ms": round(statistics.mean(lats), 0),
                "pass_rate": round(passes / len(rows), 3),
            }

        perf_a = _perf(rows_a)
        perf_b = _perf(rows_b)

        def _pct_delta(a, b):
            if a == 0:
                return 0.0
            return round((b - a) / a * 100, 1)

        performance = {
            "avg_cost_usd": {
                "run_a": perf_a["avg_cost"], "run_b": perf_b["avg_cost"],
                "delta_pct": _pct_delta(perf_a["avg_cost"], perf_b["avg_cost"]),
            },
            "avg_latency_ms": {
                "run_a": perf_a["avg_latency_ms"], "run_b": perf_b["avg_latency_ms"],
                "delta_pct": _pct_delta(perf_a["avg_latency_ms"], perf_b["avg_latency_ms"]),
            },
            "pass_rate": {
                "run_a": perf_a["pass_rate"], "run_b": perf_b["pass_rate"],
                "delta_pct": _pct_delta(perf_a["pass_rate"], perf_b["pass_rate"]),
            },
        }

        # Radar data: normalise all metrics to 0-1 for uniform radar display
        radar_data = []
        for m, v in metrics_compare.items():
            scale = 5.0 if (v["run_a_avg"] or 0) > 1.5 else 1.0
            radar_data.append({
                "metric": m,
                "run_a": round(v["run_a_avg"] / scale, 3),
                "run_b": round(v["run_b_avg"] / scale, 3),
            })

        # Generate a plain-English recommendation
        b_wins_quality = sum(1 for v in metrics_compare.values() if v.get("better") == "B" and v.get("significant"))
        a_wins_quality = sum(1 for v in metrics_compare.values() if v.get("better") == "A" and v.get("significant"))
        cost_delta = performance["avg_cost_usd"]["delta_pct"]
        latency_delta = performance["avg_latency_ms"]["delta_pct"]
        pass_delta = performance["pass_rate"]["delta_pct"]

        if b_wins_quality > a_wins_quality and cost_delta <= 0:
            rec = f"Run B is better — {b_wins_quality} quality metric(s) improved with {abs(cost_delta):.0f}% cost reduction."
        elif b_wins_quality > a_wins_quality and cost_delta > 20:
            rec = f"Run B has better quality ({b_wins_quality} metrics improved) but {cost_delta:.0f}% higher cost — evaluate tradeoff."
        elif a_wins_quality > b_wins_quality:
            rec = f"Run A is better — {a_wins_quality} quality metric(s) regressed in Run B."
        elif cost_delta < -15 and abs(pass_delta) < 5:
            rec = f"Runs are quality-equivalent; Run B is {abs(cost_delta):.0f}% cheaper — prefer Run B."
        else:
            rec = "Runs are equivalent across quality and performance metrics."

        # Per-test breakdown
        test_map_a = {r.golden_case_id: r for r in rows_a}
        test_map_b = {r.golden_case_id: r for r in rows_b}
        all_tests = sorted(set(test_map_a.keys()) | set(test_map_b.keys()))
        test_comparison = []
        for tid in all_tests:
            ra = test_map_a.get(tid)
            rb = test_map_b.get(tid)
            test_comparison.append({
                "golden_case_id": tid,
                "name": (ra or rb).golden_case_name if (ra or rb) else tid,
                "run_a_pass": ra.overall_pass if ra else None,
                "run_b_pass": rb.overall_pass if rb else None,
                "run_a_cost": ra.actual_cost if ra else None,
                "run_b_cost": rb.actual_cost if rb else None,
                "run_a_latency_ms": ra.actual_latency_ms if ra else None,
                "run_b_latency_ms": rb.actual_latency_ms if rb else None,
            })

        return {
            "run_a": _run_info(run_a_meta, run_a_id),
            "run_b": _run_info(run_b_meta, run_b_id),
            "metrics": metrics_compare,
            "performance": performance,
            "radar_data": radar_data,
            "recommendation": rec,
            "test_comparison": test_comparison,
        }

    # ── Full report ───────────────────────────────────────────────────────────

    def full_report(self, days: int = 7) -> dict:
        """
        Generate a comprehensive performance report covering all metrics.

        Returns a structured dict suitable for JSON serialization and
        display in the monitoring dashboard.
        """
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "period_days": days,
            "agent_latency_percentiles": self.agent_latency_percentiles(days),
            "tool_latency_percentiles": self.tool_latency_percentiles(days),
            "tool_failure_rates": self.tool_failure_rates(days),
            "context_window_utilization": self.context_window_utilization(days),
            "cost_breakdown": self.cost_breakdown(days),
            "performance_anomalies": self.performance_anomalies(days),
            "regression_insights": self.regression_metric_insights(days=30),
        }
