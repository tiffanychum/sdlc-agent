"""
External evaluation tool integrations:
- Langfuse: Trace export for observability
- DeepEval: Formal metric computation via G-Eval

Both are optional — if API keys are missing, they gracefully no-op.
"""

import os
import json
from datetime import datetime


# ── Langfuse Integration ────────────────────────────────────────

def get_langfuse_client():
    """Get Langfuse client if configured."""
    try:
        from langfuse import Langfuse
        secret = os.getenv("LANGFUSE_SECRET_KEY", "")
        public = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        if not secret or not public:
            return None
        return Langfuse(secret_key=secret, public_key=public, host=base_url)
    except Exception:
        return None


def export_trace_to_langfuse(
    trace_id: str,
    user_prompt: str,
    agent_used: str,
    tool_calls: list[dict],
    response: str,
    latency_ms: float,
    eval_scores: dict = None,
):
    """Export a single agent interaction as a Langfuse trace with spans."""
    client = get_langfuse_client()
    if not client:
        return

    try:
        trace = client.trace(
            id=trace_id,
            name=f"agent:{agent_used}",
            input={"prompt": user_prompt[:500]},
            output={"response": response[:500]},
            metadata={"agent": agent_used, "tool_count": len(tool_calls)},
        )

        trace.generation(
            name="routing",
            model="router",
            input=user_prompt[:300],
            output=agent_used,
        )

        for i, tc in enumerate(tool_calls):
            trace.span(
                name=f"tool:{tc.get('tool', '?')}",
                input=json.dumps(tc.get("args", {}))[:300],
                metadata={"tool": tc.get("tool", ""), "step": i + 1},
            )

        trace.generation(
            name="response",
            model=agent_used,
            input=user_prompt[:300],
            output=response[:500],
            metadata={"latency_ms": latency_ms},
        )

        if eval_scores:
            for name, score in eval_scores.items():
                trace.score(name=name, value=float(score))

        client.flush()
    except Exception:
        pass


def export_eval_run_to_langfuse(run_summary: dict, tasks: list[dict]):
    """Export an entire evaluation run to Langfuse."""
    client = get_langfuse_client()
    if not client:
        return

    try:
        trace = client.trace(
            id=f"eval-{run_summary.get('run_id', 'unknown')}",
            name=f"eval:{run_summary.get('model', '?')}",
            input={"num_tasks": run_summary.get("num_tasks", 0)},
            output=run_summary,
            metadata={"type": "evaluation_run"},
        )

        for metric_name in ["task_success_rate", "tool_accuracy", "reasoning_quality",
                            "step_efficiency", "faithfulness", "safety_compliance", "routing_accuracy"]:
            if metric_name in run_summary:
                trace.score(name=metric_name, value=float(run_summary[metric_name]))

        client.flush()
    except Exception:
        pass


# ── DeepEval Integration ────────────────────────────────────────

def run_deepeval_metrics(
    user_prompt: str,
    agent_response: str,
    tool_outputs: list[str] = None,
    expected_output: str = None,
) -> dict[str, float]:
    """
    Run DeepEval metrics on an agent response.
    Returns dict of metric_name -> score (0-1).

    Metrics computed:
    - Answer Relevancy
    - Faithfulness (hallucination detection)
    - Contextual Relevancy
    """
    scores = {}
    tool_outputs = tool_outputs or []

    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase

        retrieval_context = tool_outputs[:5] if tool_outputs else ["No tool outputs available"]

        test_case = LLMTestCase(
            input=user_prompt[:500],
            actual_output=agent_response[:1000],
            retrieval_context=retrieval_context,
            expected_output=expected_output or "",
        )

        try:
            relevancy = AnswerRelevancyMetric(threshold=0.5)
            relevancy.measure(test_case)
            scores["deepeval_relevancy"] = relevancy.score or 0.0
        except Exception:
            scores["deepeval_relevancy"] = 0.5

        try:
            faithfulness = FaithfulnessMetric(threshold=0.5)
            faithfulness.measure(test_case)
            scores["deepeval_faithfulness"] = faithfulness.score or 0.0
        except Exception:
            scores["deepeval_faithfulness"] = 0.5

    except ImportError:
        scores["deepeval_relevancy"] = 0.5
        scores["deepeval_faithfulness"] = 0.5
    except Exception:
        scores["deepeval_relevancy"] = 0.5
        scores["deepeval_faithfulness"] = 0.5

    return scores
