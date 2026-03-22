"""
OpenTelemetry-based trace collector for the SDLC Agent platform.

Uses the OTel SDK with GenAI semantic conventions (gen_ai.*) for standardized
agent observability. Supports export to any OTel-compatible backend (Phoenix,
Langfuse, Jaeger, Datadog) via OTLP.

Also persists spans to local SQLite for the built-in dashboard.

Architecture:
- OpenInference auto-instruments LangChain (LLM calls, tool calls)
- Custom spans added for routing, agent execution, supervisor decisions
- CostEstimationProcessor enriches spans with cost data on export
- Dual export: OTLP (if configured) + local DB
"""

import os
import time
import uuid
from datetime import datetime

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import StatusCode
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

from src.db.database import get_session
from src.db.models import Trace as TraceModel, Span as SpanModel


# ── Cost Estimation ──────────────────────────────────────────────

COST_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-5.3-codex": {"input": 0.003, "output": 0.012},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "gemini-2.5-flash-lite": {"input": 0.001, "output": 0.004},
    "llama-3.1-8b-cs": {"input": 0.0005, "output": 0.002},
    "mistral-small-3": {"input": 0.001, "output": 0.003},
    "default": {"input": 0.002, "output": 0.008},
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    rates = COST_PER_1K_TOKENS.get(model, COST_PER_1K_TOKENS["default"])
    return (tokens_in / 1000 * rates["input"]) + (tokens_out / 1000 * rates["output"])


# ── OTel Setup ───────────────────────────────────────────────────

_provider_initialized = False


def init_otel():
    """Initialize the OpenTelemetry TracerProvider with GenAI resource attributes."""
    global _provider_initialized
    if _provider_initialized or not HAS_OTEL:
        return

    resource = Resource.create({
        "service.name": "sdlc-agent",
        "service.version": "0.3.0",
        "gen_ai.system": "langchain",
    })

    provider = TracerProvider(resource=resource)

    provider.add_span_processor(DBSpanProcessor())

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            otlp_exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        except ImportError:
            pass

    trace.set_tracer_provider(provider)
    _auto_instrument_langchain()
    _provider_initialized = True


def _auto_instrument_langchain():
    """Auto-instrument LangChain via OpenInference if available."""
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
    except ImportError:
        pass


def get_tracer(name: str = "sdlc-agent"):
    """Get an OTel tracer, or None if OTel is not installed."""
    if not HAS_OTEL:
        return None
    init_otel()
    return trace.get_tracer(name, "0.3.0")


# ── DB Span Processor ────────────────────────────────────────────

if HAS_OTEL:
    class DBSpanProcessor(SpanProcessor):
        """
        Custom OTel SpanProcessor that persists completed spans to SQLite.
        Enriches spans with cost estimation based on token usage and GenAI attributes.
        """

        def on_start(self, span, parent_context=None):
            pass

        def on_end(self, span):
            attrs = dict(span.attributes) if span.attributes else {}

            model = attrs.get("gen_ai.request.model", attrs.get("llm.model", ""))
            tokens_in = attrs.get("gen_ai.usage.input_tokens", attrs.get("llm.token_count.prompt", 0))
            tokens_out = attrs.get("gen_ai.usage.output_tokens", attrs.get("llm.token_count.completion", 0))
            cost = estimate_cost(str(model), int(tokens_in), int(tokens_out))

            span_type = attrs.get("span.type", "unknown")
            if "tool" in span.name.lower():
                span_type = "tool_call"
            elif "llm" in span.name.lower() or "chatmodel" in span.name.lower():
                span_type = "llm_call"
            elif "agent" in span.name.lower():
                span_type = "agent_execution"
            elif "rout" in span.name.lower():
                span_type = "routing"

            input_data = {}
            output_data = {}
            for k, v in attrs.items():
                if k.startswith("input.") or k.startswith("gen_ai.prompt"):
                    input_data[k] = str(v)[:500]
                elif k.startswith("output.") or k.startswith("gen_ai.completion"):
                    output_data[k] = str(v)[:500]

            span_record = {
                "id": format(span.context.span_id, '016x')[:12],
                "trace_id": format(span.context.trace_id, '032x')[:12],
                "parent_span_id": format(span.parent.span_id, '016x')[:12] if span.parent else None,
                "name": span.name,
                "span_type": span_type,
                "start_time": datetime.utcfromtimestamp(span.start_time / 1e9) if span.start_time else datetime.utcnow(),
                "end_time": datetime.utcfromtimestamp(span.end_time / 1e9) if span.end_time else datetime.utcnow(),
                "input_data": input_data,
                "output_data": output_data,
                "tokens_in": int(tokens_in),
                "tokens_out": int(tokens_out),
                "cost": cost,
                "model": str(model),
                "status": "completed" if span.status.status_code != StatusCode.ERROR else "error",
                "error": span.status.description if span.status.status_code == StatusCode.ERROR else None,
            }

            _pending_spans.setdefault(span_record["trace_id"], []).append(span_record)

        def shutdown(self):
            pass

        def force_flush(self, timeout_millis=None):
            pass

_pending_spans: dict[str, list[dict]] = {}


# ── TraceCollector (OTel-backed) ─────────────────────────────────

class TraceCollector:
    """
    Collects trace/span data using the OTel SDK.
    Provides the same interface as the previous custom collector for
    backward compatibility with server.py and orchestrator.py.
    """

    def __init__(self, team_id: str = None, user_prompt: str = ""):
        init_otel()
        self.trace_id = uuid.uuid4().hex[:12]
        self.team_id = team_id
        self.user_prompt = user_prompt
        self.start_time = time.time()
        self.spans: list[dict] = []
        self._tracer = get_tracer()
        self._active_spans: dict = {}
        self._span_data: dict[str, dict] = {}

    def start_span(self, name: str, span_type: str, parent_id: str = None, input_data: dict = None) -> str:
        span_id = uuid.uuid4().hex[:12]

        if self._tracer:
            otel_span = self._tracer.start_span(
                name=name,
                attributes={
                    "span.type": span_type,
                    "gen_ai.agent.name": name.split(":")[1] if ":" in name else name,
                    "gen_ai.operation.name": span_type,
                    **({"input." + k: str(v)[:300] for k, v in (input_data or {}).items()}),
                },
            )
            self._active_spans[span_id] = otel_span
        self._span_data[span_id] = {
            "id": span_id,
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
        }
        return span_id

    def end_span(self, span_id: str, output_data: dict = None, tokens_in: int = 0,
                 tokens_out: int = 0, model: str = "", error: str = None):
        otel_span = self._active_spans.pop(span_id, None)
        if otel_span and HAS_OTEL:
            if tokens_in or tokens_out:
                otel_span.set_attribute("gen_ai.usage.input_tokens", tokens_in)
                otel_span.set_attribute("gen_ai.usage.output_tokens", tokens_out)
            if model:
                otel_span.set_attribute("gen_ai.request.model", model)
            if output_data:
                for k, v in output_data.items():
                    otel_span.set_attribute(f"output.{k}", str(v)[:300])
            if error:
                otel_span.set_status(StatusCode.ERROR, error)
            otel_span.end()

        data = self._span_data.get(span_id)
        if data:
            data["end_time"] = datetime.utcnow()
            data["output_data"] = _truncate(output_data or {})
            data["tokens_in"] = tokens_in
            data["tokens_out"] = tokens_out
            data["model"] = model
            data["cost"] = estimate_cost(model, tokens_in, tokens_out)
            data["status"] = "error" if error else "completed"
            data["error"] = error
            self.spans.append(data)

    def save(self):
        """Persist the trace and all spans to the database."""
        elapsed_ms = (time.time() - self.start_time) * 1000
        total_tokens = sum(s["tokens_in"] + s["tokens_out"] for s in self.spans)
        total_cost = sum(s["cost"] for s in self.spans)

        session = get_session()
        try:
            trace_record = TraceModel(
                id=self.trace_id,
                team_id=self.team_id,
                user_prompt=self.user_prompt[:500],
                total_latency_ms=elapsed_ms,
                total_tokens=total_tokens,
                total_cost=total_cost,
                status="completed" if not any(s["error"] for s in self.spans) else "error",
            )
            session.add(trace_record)

            for s in self.spans:
                span = SpanModel(
                    id=s["id"],
                    trace_id=self.trace_id,
                    parent_span_id=None,
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
    result = {}
    for k, v in data.items():
        if isinstance(v, str) and len(v) > max_len:
            result[k] = v[:max_len] + "..."
        else:
            result[k] = v
    return result
