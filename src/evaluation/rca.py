"""
Root Cause Analysis engine for regression test failures.

Compares a failing trace against a baseline trace, identifies the first
divergence point, and uses an LLM to classify the root cause and provide
actionable recommendations.

Root cause categories:
- prompt_change: Agent prompt or system message caused different behavior
- model_change: Different model version produced different output
- tool_data_change: External tool returned different data
- context_overflow: Accumulated context exceeded effective window
- delegation_change: Router or supervisor made different delegation decision
- unknown: Could not determine root cause
"""

import json

from src.llm.client import get_rca_llm
from src.orchestrator import _extract_text


ROOT_CAUSE_CATEGORIES = [
    "prompt_change", "model_change", "tool_data_change",
    "context_overflow", "delegation_change", "unknown",
]


class RootCauseAnalyzer:
    async def analyze(
        self,
        failing_result: dict,
        baseline_result: dict = None,
    ) -> dict:
        trace_diff = self._compute_trace_diff(
            failing_result.get("full_trace", []),
            baseline_result.get("full_trace", []) if baseline_result else [],
        )

        cost_diff = self._compute_cost_diff(failing_result, baseline_result)

        llm_analysis = await self._llm_analyze(failing_result, baseline_result, trace_diff, cost_diff)

        result = {
            "trace_diff": trace_diff,
            "cost_diff": cost_diff,
            **llm_analysis,
        }

        try:
            self._export_to_langfuse(failing_result, result)
        except Exception:
            pass

        return result

    def _compute_trace_diff(self, failing_trace: list, baseline_trace: list) -> list:
        """Walk both traces step by step and identify divergence points."""
        diff = []
        max_len = max(len(failing_trace), len(baseline_trace))

        for i in range(max_len):
            f_step = failing_trace[i] if i < len(failing_trace) else None
            b_step = baseline_trace[i] if i < len(baseline_trace) else None

            if f_step is None:
                diff.append({
                    "step": i, "status": "missing_in_failing",
                    "baseline": _summarize_step(b_step),
                    "failing": None,
                    "diverged": True,
                })
                continue
            if b_step is None:
                diff.append({
                    "step": i, "status": "extra_in_failing",
                    "baseline": None,
                    "failing": _summarize_step(f_step),
                    "diverged": True,
                })
                continue

            diverged = False
            reasons = []

            if f_step.get("step") != b_step.get("step"):
                diverged = True
                reasons.append(f"Step type: {b_step.get('step')} -> {f_step.get('step')}")

            if f_step.get("step") == "routing" and b_step.get("step") == "routing":
                if f_step.get("selected_agent") != b_step.get("selected_agent"):
                    diverged = True
                    reasons.append(f"Agent: {b_step.get('selected_agent')} -> {f_step.get('selected_agent')}")

            if f_step.get("step") == "execution" and b_step.get("step") == "execution":
                if f_step.get("agent") != b_step.get("agent"):
                    diverged = True
                    reasons.append(f"Agent: {b_step.get('agent')} -> {f_step.get('agent')}")

                f_tools = [tc.get("tool") for tc in f_step.get("tool_calls", [])]
                b_tools = [tc.get("tool") for tc in b_step.get("tool_calls", [])]
                if f_tools != b_tools:
                    diverged = True
                    reasons.append(f"Tools: {b_tools} -> {f_tools}")

            diff.append({
                "step": i,
                "status": "diverged" if diverged else "match",
                "baseline": _summarize_step(b_step),
                "failing": _summarize_step(f_step),
                "diverged": diverged,
                "reasons": reasons,
            })

        return diff

    def _compute_cost_diff(self, failing: dict, baseline: dict = None) -> dict:
        if not baseline:
            return {
                "tokens_in_delta": 0, "tokens_out_delta": 0,
                "cost_delta": 0, "latency_delta": 0,
                "llm_calls_delta": 0, "tool_calls_delta": 0,
            }
        return {
            "tokens_in_delta": failing.get("actual_tokens_in", 0) - baseline.get("actual_tokens_in", 0),
            "tokens_out_delta": failing.get("actual_tokens_out", 0) - baseline.get("actual_tokens_out", 0),
            "cost_delta": round(failing.get("actual_cost", 0) - baseline.get("actual_cost", 0), 6),
            "latency_delta": round(failing.get("actual_latency_ms", 0) - baseline.get("actual_latency_ms", 0), 1),
            "llm_calls_delta": failing.get("actual_llm_calls", 0) - baseline.get("actual_llm_calls", 0),
            "tool_calls_delta": failing.get("actual_tool_calls", 0) - baseline.get("actual_tool_calls", 0),
        }

    async def _llm_analyze(self, failing: dict, baseline: dict, trace_diff: list, cost_diff: dict) -> dict:
        llm = get_rca_llm()

        divergence_points = [d for d in trace_diff if d.get("diverged")]
        first_div = divergence_points[0] if divergence_points else None

        baseline_summary = ""
        if baseline:
            baseline_summary = f"""BASELINE (last known-good):
- Output: {baseline.get('actual_output', '')[:400]}
- Agent: {baseline.get('actual_agent', '')}
- Tools: {baseline.get('actual_tools', [])}
- Tokens: {baseline.get('actual_tokens_in', 0)} in / {baseline.get('actual_tokens_out', 0)} out
- Cost: ${baseline.get('actual_cost', 0)}
- Latency: {baseline.get('actual_latency_ms', 0)}ms"""

        ta_summary = {
            k: v.get("passed") if isinstance(v, dict) else v
            for k, v in failing.get("trace_assertions", {}).items()
        }
        ta_str = json.dumps(ta_summary)[:200]

        prompt = f"""You are an AI agent debugging specialist. Analyze this regression test failure and determine the root cause.

TEST CASE: {failing.get('golden_case_name', '')}
PROMPT: {failing.get('prompt', '')[:300]}
EXPECTED AGENT: {failing.get('expected_agent', '')}
EXPECTED TOOLS: {failing.get('expected_tools', [])}

FAILING RUN:
- Output: {failing.get('actual_output', '')[:400]}
- Agent: {failing.get('actual_agent', '')}
- Tools: {failing.get('actual_tools', [])}
- Tokens: {failing.get('actual_tokens_in', 0)} in / {failing.get('actual_tokens_out', 0)} out
- Cost: ${failing.get('actual_cost', 0)}
- Latency: {failing.get('actual_latency_ms', 0)}ms
- Quality scores: {json.dumps(failing.get('quality_scores', {}), indent=0)[:300]}
- Trace assertions: {ta_str}

{baseline_summary}

FIRST DIVERGENCE POINT: {json.dumps(first_div, indent=0)[:400] if first_div else 'No baseline to compare'}

COST DELTA: {json.dumps(cost_diff)[:200]}

Classify the root cause into ONE of these categories:
- prompt_change: Agent prompt caused different behavior
- model_change: Model produced different output quality
- tool_data_change: External tool returned different data
- context_overflow: Context exceeded effective window
- delegation_change: Router made a different delegation decision
- unknown: Cannot determine

Respond with ONLY JSON:
{{
  "root_cause_category": "<category>",
  "divergence_point": {{"step_index": <int or null>, "description": "<what changed>"}},
  "analysis": "<detailed paragraph explaining the root cause>",
  "recommendations": ["<action 1>", "<action 2>"]
}}"""

        try:
            resp = await llm.ainvoke([{"role": "user", "content": prompt}])
            raw = _extract_text(resp.content)
            import re
            match = re.search(r'\{[\s\S]*"root_cause_category"[\s\S]*\}', raw)
            if match:
                data = json.loads(match.group())
                return {
                    "root_cause_category": data.get("root_cause_category", "unknown"),
                    "divergence_point": data.get("divergence_point", {}),
                    "analysis": data.get("analysis", ""),
                    "recommendations": data.get("recommendations", []),
                }
        except Exception:
            pass

        return {
            "root_cause_category": "unknown",
            "divergence_point": {},
            "analysis": "Could not perform automated root cause analysis.",
            "recommendations": ["Review the trace diff manually"],
        }

    def _export_to_langfuse(self, failing_result: dict, rca: dict):
        """Export the failing trace + RCA to Langfuse for external visualization."""
        from src.evaluation.integrations import get_langfuse_client
        client = get_langfuse_client()
        if not client:
            return

        trace = client.trace(
            name=f"rca:{failing_result.get('golden_case_id', '?')}",
            input={"prompt": failing_result.get("prompt", "")[:300]},
            output={"analysis": rca.get("analysis", "")[:500]},
            metadata={
                "root_cause": rca.get("root_cause_category", "unknown"),
                "golden_case": failing_result.get("golden_case_id", ""),
                "model": failing_result.get("model_used", ""),
            },
        )

        for key, score_val in failing_result.get("quality_scores", {}).items():
            if isinstance(score_val, (int, float)):
                trace.score(name=f"quality_{key}", value=float(score_val))

        trace.score(name="semantic_similarity", value=float(failing_result.get("semantic_similarity", 0)))
        client.flush()


def _summarize_step(step: dict) -> dict:
    if not step:
        return {}
    s = {"type": step.get("step", "unknown")}
    if step.get("step") == "routing":
        s["agent"] = step.get("selected_agent", "")
        s["reasoning"] = str(step.get("reasoning", ""))[:200]
    elif step.get("step") == "execution":
        s["agent"] = step.get("agent", "")
        s["tools"] = [tc.get("tool", "") for tc in step.get("tool_calls", [])]
        s["num_tools"] = len(step.get("tool_calls", []))
    elif step.get("step") == "supervisor":
        s["decision"] = step.get("decision", "")
    return s
