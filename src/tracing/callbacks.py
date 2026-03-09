"""
LangChain callback handler for automatic LLM call tracing.

Captures token usage, latency, and model info from every LLM invocation
and feeds it into the TraceCollector.
"""

from typing import Any, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from src.tracing.collector import TraceCollector


class TracingCallbackHandler(BaseCallbackHandler):
    """Captures LLM call metadata for the trace collector."""

    def __init__(self, collector: TraceCollector):
        self.collector = collector
        self._span_map: dict[str, str] = {}

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id, **kwargs):
        model = serialized.get("kwargs", {}).get("model", "unknown")
        span_id = self.collector.start_span(
            name=f"llm_call:{model}",
            span_type="llm_call",
            input_data={"prompt_length": sum(len(p) for p in prompts)},
        )
        self._span_map[str(run_id)] = span_id

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs):
        span_id = self._span_map.pop(str(run_id), None)
        if not span_id:
            return

        tokens_in = 0
        tokens_out = 0
        model = ""

        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)
            model = response.llm_output.get("model_name", "")

        self.collector.end_span(
            span_id, tokens_in=tokens_in, tokens_out=tokens_out, model=model,
            output_data={"generations": len(response.generations)},
        )

    def on_llm_error(self, error: BaseException, *, run_id, **kwargs):
        span_id = self._span_map.pop(str(run_id), None)
        if span_id:
            self.collector.end_span(span_id, error=str(error))

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id, **kwargs):
        tool_name = serialized.get("name", "unknown_tool")
        span_id = self.collector.start_span(
            name=f"tool:{tool_name}",
            span_type="tool_call",
            input_data={"input": input_str[:300]},
        )
        self._span_map[str(run_id)] = span_id

    def on_tool_end(self, output: str, *, run_id, **kwargs):
        span_id = self._span_map.pop(str(run_id), None)
        if span_id:
            self.collector.end_span(span_id, output_data={"output": str(output)[:300]})

    def on_tool_error(self, error: BaseException, *, run_id, **kwargs):
        span_id = self._span_map.pop(str(run_id), None)
        if span_id:
            self.collector.end_span(span_id, error=str(error))
