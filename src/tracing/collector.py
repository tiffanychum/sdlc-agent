"""
Trace collector — records every LLM call, tool invocation, and routing decision.

Follows OpenTelemetry trace/span semantics without the full OTEL dependency:
- Trace: one complete user request → response cycle
- Span: individual operation within a trace (LLM call, tool call, routing)
"""

import time
import uuid
from datetime import datetime
from contextlib import contextmanager

from src.db.database import get_session
from src.db.models import Trace, Span

COST_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-5.3-codex": {"input": 0.003, "output": 0.012},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "default": {"input": 0.002, "output": 0.008},
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    rates = COST_PER_1K_TOKENS.get(model, COST_PER_1K_TOKENS["default"])
    return (tokens_in / 1000 * rates["input"]) + (tokens_out / 1000 * rates["output"])


class TraceCollector:
    """Collects trace and span data for a single request lifecycle."""

    def __init__(self, team_id: str = None, user_prompt: str = ""):
        self.trace_id = uuid.uuid4().hex[:12]
        self.team_id = team_id
        self.user_prompt = user_prompt
        self.start_time = time.time()
        self.spans: list[dict] = []
        self._current_span_id = None

    def start_span(self, name: str, span_type: str, parent_id: str = None, input_data: dict = None) -> str:
        span_id = uuid.uuid4().hex[:12]
        self.spans.append({
            "id": span_id,
            "parent_span_id": parent_id or self._current_span_id,
            "name": name,
            "span_type": span_type,
            "start_time": datetime.utcnow(),
            "end_time": None,
            "input_data": input_data or {},
            "output_data": {},
            "tokens_in": 0,
            "tokens_out": 0,
            "cost": 0.0,
            "model": "",
            "status": "running",
            "error": None,
        })
        self._current_span_id = span_id
        return span_id

    def end_span(self, span_id: str, output_data: dict = None, tokens_in: int = 0,
                 tokens_out: int = 0, model: str = "", error: str = None):
        for span in self.spans:
            if span["id"] == span_id:
                span["end_time"] = datetime.utcnow()
                span["output_data"] = _truncate(output_data or {})
                span["tokens_in"] = tokens_in
                span["tokens_out"] = tokens_out
                span["model"] = model
                span["cost"] = estimate_cost(model, tokens_in, tokens_out)
                span["status"] = "error" if error else "completed"
                span["error"] = error
                break

    def save(self):
        """Persist the trace and all spans to the database."""
        elapsed_ms = (time.time() - self.start_time) * 1000
        total_tokens = sum(s["tokens_in"] + s["tokens_out"] for s in self.spans)
        total_cost = sum(s["cost"] for s in self.spans)

        session = get_session()
        try:
            trace = Trace(
                id=self.trace_id,
                team_id=self.team_id,
                user_prompt=self.user_prompt[:500],
                total_latency_ms=elapsed_ms,
                total_tokens=total_tokens,
                total_cost=total_cost,
                status="completed" if not any(s["error"] for s in self.spans) else "error",
            )
            session.add(trace)

            for s in self.spans:
                span = Span(
                    id=s["id"],
                    trace_id=self.trace_id,
                    parent_span_id=s["parent_span_id"],
                    name=s["name"],
                    span_type=s["span_type"],
                    start_time=s["start_time"],
                    end_time=s["end_time"],
                    input_data=s["input_data"],
                    output_data=s["output_data"],
                    tokens_in=s["tokens_in"],
                    tokens_out=s["tokens_out"],
                    cost=s["cost"],
                    model=s["model"],
                    status=s["status"],
                    error=s["error"],
                )
                session.add(span)

            session.commit()
        finally:
            session.close()

    def to_dict(self) -> dict:
        elapsed_ms = (time.time() - self.start_time) * 1000
        return {
            "trace_id": self.trace_id,
            "team_id": self.team_id,
            "user_prompt": self.user_prompt,
            "total_latency_ms": round(elapsed_ms, 1),
            "total_tokens": sum(s["tokens_in"] + s["tokens_out"] for s in self.spans),
            "total_cost": round(sum(s["cost"] for s in self.spans), 6),
            "spans": [{
                "id": s["id"],
                "name": s["name"],
                "span_type": s["span_type"],
                "tokens_in": s["tokens_in"],
                "tokens_out": s["tokens_out"],
                "cost": round(s["cost"], 6),
                "model": s["model"],
                "status": s["status"],
                "error": s["error"],
            } for s in self.spans],
        }


def _truncate(data: dict, max_len: int = 500) -> dict:
    """Truncate string values in dict for storage."""
    result = {}
    for k, v in data.items():
        if isinstance(v, str) and len(v) > max_len:
            result[k] = v[:max_len] + "..."
        else:
            result[k] = v
    return result
