"""
LangChain callback handler for automatic LLM/tool tracing via OpenTelemetry.

Uses the OTel SDK with GenAI semantic conventions for standardized span attributes.
OpenInference auto-instrumentation handles most LangChain tracing automatically;
this handler adds custom enrichment (cost estimation, agent context) on top.
"""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from src.tracing.collector import TraceCollector


class TracingCallbackHandler(BaseCallbackHandler):
    """Enriches OTel spans with cost estimation and agent context."""

    def __init__(self, collector: TraceCollector):
        self.collector = collector
        # Maps run_id -> (span_id, model_at_start) so on_llm_end can fall back
        # to the model name already known at start time.
        self._span_map: dict[str, tuple[str, str]] = {}

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id, **kwargs):
        kw = serialized.get("kwargs", {})
        model = (
            kw.get("model_name") or kw.get("model")
            or kwargs.get("invocation_params", {}).get("model", "")
            or ""
        )
        span_id = self.collector.start_span(
            name=f"llm_call:{model}",
            span_type="llm_call",
            input_data={
                "gen_ai.request.model": model,
                "gen_ai.prompt.length": sum(len(p) for p in prompts),
                "gen_ai.prompt.count": len(prompts),
            },
        )
        self._span_map[str(run_id)] = (span_id, model)

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs):
        entry = self._span_map.pop(str(run_id), None)
        if not entry:
            return
        span_id, model_at_start = entry

        tokens_in = 0
        tokens_out = 0
        model = ""

        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            tokens_in = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
            model = (
                response.llm_output.get("model_name", "")
                or response.llm_output.get("model", "")
            )

        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    gen_info = getattr(gen, "generation_info", {}) or {}
                    if not tokens_in and gen_info.get("token_usage"):
                        tokens_in = gen_info["token_usage"].get("prompt_tokens", 0)
                        tokens_out = gen_info["token_usage"].get("completion_tokens", 0)
                    msg = getattr(gen, "message", None)
                    if msg:
                        resp_meta = getattr(msg, "response_metadata", {}) or {}
                        # Token counts — try multiple key variants (OpenAI / Anthropic)
                        if not tokens_in:
                            u = resp_meta.get("token_usage") or resp_meta.get("usage", {}) or {}
                            tokens_in = (u.get("prompt_tokens") or u.get("input_tokens") or 0)
                            tokens_out = (u.get("completion_tokens") or u.get("output_tokens") or 0)
                        # Model name — try model_name / model / OpenAI-style keys
                        if not model:
                            model = (
                                resp_meta.get("model_name") or resp_meta.get("model")
                                or resp_meta.get("system_fingerprint", "").split("-")[0]
                                or ""
                            )
                        # Anthropic usage_metadata on the message
                        um = getattr(msg, "usage_metadata", None)
                        if um and not tokens_in:
                            tokens_in = getattr(um, "input_tokens", 0) or 0
                            tokens_out = getattr(um, "output_tokens", 0) or 0

        # Fall back to the model name captured at on_llm_start if still unknown
        if not model:
            model = model_at_start

        self.collector.end_span(
            span_id, tokens_in=tokens_in, tokens_out=tokens_out, model=model,
            output_data={
                "gen_ai.usage.input_tokens": tokens_in,
                "gen_ai.usage.output_tokens": tokens_out,
                "gen_ai.request.model": model,
                "gen_ai.completion.count": len(response.generations),
            },
        )

    def on_llm_error(self, error: BaseException, *, run_id, **kwargs):
        entry = self._span_map.pop(str(run_id), None)
        if entry:
            span_id, _ = entry
            self.collector.end_span(span_id, error=str(error))

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id, **kwargs):
        tool_name = serialized.get("name", "unknown_tool")
        span_id = self.collector.start_span(
            name=f"tool:{tool_name}",
            span_type="tool_call",
            input_data={
                "gen_ai.tool.name": tool_name,
                "gen_ai.tool.type": "function",
                "input.value": input_str[:300],
            },
        )
        self._span_map[str(run_id)] = span_id

    def on_tool_end(self, output: str, *, run_id, **kwargs):
        span_id = self._span_map.pop(str(run_id), None)
        if span_id:
            self.collector.end_span(span_id, output_data={"output.value": str(output)[:300]})

    def on_tool_error(self, error: BaseException, *, run_id, **kwargs):
        span_id = self._span_map.pop(str(run_id), None)
        if span_id:
            self.collector.end_span(span_id, error=str(error))
