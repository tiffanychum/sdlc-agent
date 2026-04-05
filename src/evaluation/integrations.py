"""
External evaluation tool integrations:
- Langfuse: Trace export for observability
- DeepEval: Agentic evaluation metrics (standalone + trace-equivalent)

Both are optional — if API keys are missing, they gracefully no-op.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum attempts for any eval call (1 original + 2 retries).
_MAX_ATTEMPTS = 3
_RETRY_DELAY_S = 1.0

# Shared thread pool for DeepEval metric calls.
# DeepEval.measure() calls asyncio.run() internally, which cannot patch a
# running uvloop.  Running each metric in a worker thread sidesteps this:
# each thread has its own default asyncio event loop (not uvloop).
_DEEPEVAL_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="deepeval"
)


def _run_in_thread(fn, timeout: float = 180.0):
    """Execute fn() in a worker thread so asyncio.run() inside DeepEval works."""
    future = _DEEPEVAL_THREAD_POOL.submit(fn)
    return future.result(timeout=timeout)


# ── Retry helpers ─────────────────────────────────────────────────

def _retry_sync(fn, attempts: int = _MAX_ATTEMPTS, delay: float = _RETRY_DELAY_S):
    """
    Call fn() up to `attempts` times with linear back-off, running each
    attempt in a dedicated thread to avoid uvloop event-loop conflicts.
    Raises the last exception if all attempts fail.
    """
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(attempts):
        try:
            return _run_in_thread(fn)
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                logger.warning(
                    "DeepEval metric call failed (attempt %d/%d): %s: %s — retrying in %.1fs",
                    attempt + 1, attempts, type(exc).__name__, exc, delay * (attempt + 1),
                )
                time.sleep(delay * (attempt + 1))
    raise last_exc


async def _retry_async(coro_fn, attempts: int = _MAX_ATTEMPTS, delay: float = _RETRY_DELAY_S):
    """
    Call coro_fn() up to `attempts` times with linear back-off.
    Raises the last exception if all attempts fail.
    """
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(attempts):
        try:
            return await coro_fn()
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                logger.warning(
                    "Agentic eval LLM call failed (attempt %d/%d): %s: %s — retrying in %.1fs",
                    attempt + 1, attempts, type(exc).__name__, exc, delay * (attempt + 1),
                )
                await asyncio.sleep(delay * (attempt + 1))
    raise last_exc


def _fmt_error(exc: Exception, attempts: int = _MAX_ATTEMPTS) -> str:
    """Format a full error string for inclusion in report reasoning fields."""
    return f"ERROR after {attempts} attempt(s): {type(exc).__name__}: {exc}"


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
            input={"prompt": user_prompt},
            output={"response": response},
            metadata={"agent": agent_used, "tool_count": len(tool_calls)},
        )

        trace.generation(
            name="routing",
            model="router",
            input=user_prompt,
            output=agent_used,
        )

        for i, tc in enumerate(tool_calls):
            trace.span(
                name=f"tool:{tc.get('tool', '?')}",
                input=json.dumps(tc.get("args", {})),
                metadata={"tool": tc.get("tool", ""), "step": i + 1},
            )

        trace.generation(
            name="response",
            model=agent_used,
            input=user_prompt,
            output=response,
            metadata={"latency_ms": latency_ms},
        )

        if eval_scores:
            for name, score in eval_scores.items():
                if score is not None:
                    try:
                        trace.score(name=name, value=float(score))
                    except (TypeError, ValueError):
                        pass  # skip non-numeric scores (error strings)

        client.flush()
    except Exception as exc:
        logger.warning("Langfuse export failed: %s: %s", type(exc).__name__, exc)


def export_eval_run_to_langfuse(run_summary: dict):
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
                val = run_summary[metric_name]
                if val is not None:
                    try:
                        trace.score(name=metric_name, value=float(val))
                    except (TypeError, ValueError):
                        pass

        client.flush()
    except Exception as exc:
        logger.warning("Langfuse eval run export failed: %s: %s", type(exc).__name__, exc)


# ── DeepEval Environment Setup ──────────────────────────────────

def _ensure_deepeval_env():
    """Set up OpenAI-compatible env vars for DeepEval if not already set."""
    if not os.getenv("OPENAI_API_KEY"):
        poe_key = os.getenv("POE_API_KEY", "")
        base_url = os.getenv("LLM_BASE_URL", "")
        if poe_key and base_url:
            os.environ["OPENAI_API_KEY"] = poe_key
            os.environ["OPENAI_BASE_URL"] = base_url


def _strip_json_fences(text: str) -> str:
    """
    Strip markdown code-block fences from LLM output before JSON parsing.

    Claude (and some other models) wraps JSON responses in ```json ... ``` blocks
    even when instructed to return raw JSON.  DeepEval's internal pydantic validators
    call json.loads() directly on the LLM response and raise ValidationError when
    they see the fence characters.  This function normalises the output so that
    DeepEval always receives clean JSON.

    Handles:
    - ```json\\n{...}\\n```
    - ```\\n{...}\\n```
    - Leading/trailing whitespace
    """
    import re
    stripped = text.strip()
    # Remove opening fence (```json or ``` alone)
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped, flags=re.IGNORECASE)
    # Remove closing fence
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    return stripped.strip()


def _get_deepeval_model():
    """
    Return a DeepEvalBaseLLM instance that wraps our configured LLM and
    strips markdown code-block fences from every response.

    If DeepEval is not installed or the LLM cannot be initialised, returns None
    so callers can fall back to the default (env-var-based) model.
    """
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
        from openai import AsyncOpenAI, OpenAI

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("POE_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL", "") or None
        model_name = os.getenv("DEEPEVAL_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))

        class _CleanJsonLLM(DeepEvalBaseLLM):
            """Thin wrapper that strips ```json fences so DeepEval JSON parsing succeeds."""

            def load_model(self):
                return OpenAI(api_key=api_key, base_url=base_url or None)

            def generate(self, prompt: str, schema=None) -> str:  # type: ignore[override]
                client: OpenAI = self.load_model()
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                raw = response.choices[0].message.content or ""
                return _strip_json_fences(raw)

            async def a_generate(self, prompt: str, schema=None) -> str:  # type: ignore[override]
                client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                raw = response.choices[0].message.content or ""
                return _strip_json_fences(raw)

            def get_model_name(self) -> str:
                return model_name

        return _CleanJsonLLM()
    except Exception as exc:
        logger.debug("Could not build DeepEval custom LLM wrapper: %s — using default", exc)
        return None


def _extract_tool_calls_from_trace(agent_trace: list[dict]) -> list[dict]:
    """Extract flat tool call list from agent_trace structure."""
    tools = []
    for entry in (agent_trace or []):
        if entry.get("step") == "execution":
            for tc in entry.get("tool_calls", []):
                tools.append({
                    "name": tc.get("tool", "unknown"),
                    "args": tc.get("args", {}),
                    "agent": entry.get("agent", ""),
                })
    return tools


def _extract_plan_from_trace(agent_trace: list[dict]) -> str:
    """Extract plan steps from agent trace if a planner was used."""
    plan_parts: list[str] = []
    for entry in (agent_trace or []):
        if entry.get("step") == "execution" and entry.get("agent") == "planner":
            for tc in entry.get("tool_calls", []):
                if tc.get("tool") == "create_plan":
                    steps = tc.get("args", {}).get("steps", [])
                    for s in steps:
                        if isinstance(s, dict):
                            plan_parts.append(str(s.get("step", s)))
                        else:
                            plan_parts.append(str(s))
                elif tc.get("tool") == "update_plan_step":
                    step_name = tc.get("args", {}).get("step_name", "")
                    status = tc.get("args", {}).get("status", "")
                    if step_name:
                        plan_parts.append(f"[{status}] {step_name}")
    return "\n".join(plan_parts) if plan_parts else ""


def _build_execution_steps_text(agent_trace: list[dict]) -> str:
    """Build a text description of execution steps for LLM-as-judge."""
    steps = []
    for i, entry in enumerate(agent_trace or []):
        if entry.get("step") == "routing":
            steps.append(f"Step {i+1}: Routed to agent '{entry.get('selected_agent', '?')}'")
        elif entry.get("step") == "supervisor":
            steps.append(f"Step {i+1}: Supervisor decision: '{entry.get('decision', '?')}'")
        elif entry.get("step") == "execution":
            agent = entry.get("agent", "?")
            tool_calls = entry.get("tool_calls", [])
            if tool_calls:
                tool_list = ", ".join(tc.get("tool", "?") for tc in tool_calls)
                steps.append(f"Step {i+1}: Agent '{agent}' called tools: [{tool_list}]")
            else:
                steps.append(f"Step {i+1}: Agent '{agent}' executed (no tool calls)")
    return "\n".join(steps) if steps else "No execution steps recorded"


# ── DeepEval Standalone Metrics ─────────────────────────────────

def run_deepeval_standalone_metrics(
    user_prompt: str,
    agent_response: str,
    agent_trace: list[dict] = None,
    tool_outputs: list[str] = None,
) -> dict:
    """
    Run DeepEval standalone metrics on an agent response.

    Returns dict where each metric has:
      "<metric>": float score or None (None = evaluation error)
      "<metric>_reason": human-readable reason or "ERROR: ..." on failure

    Standalone metrics:
    - ToolCorrectness: Were the right tools called?
    - ArgumentCorrectness: Were tool arguments well-formed?
    - AnswerRelevancy: Does the response address the query?
    - Faithfulness: Is the response grounded in tool outputs?
    """
    _ensure_deepeval_env()
    scores: dict = {}
    tool_outputs = tool_outputs or []
    flat_tools = _extract_tool_calls_from_trace(agent_trace or [])

    try:
        from deepeval.test_case import LLMTestCase, ToolCall

        # Custom model that strips markdown fences before DeepEval JSON parsing.
        eval_model = _get_deepeval_model()

        deepeval_tool_calls = []
        for tc in flat_tools:
            args = tc.get("args", {})
            # Pass full argument values — no truncation.
            str_args = {str(k): str(v) for k, v in args.items()} if isinstance(args, dict) else {}
            deepeval_tool_calls.append(ToolCall(
                name=str(tc["name"]),
                description=f"Tool: {tc['name']} (agent: {tc.get('agent', '?')})",
                input_parameters=str_args,
            ))

        # Build retrieval context from full tool outputs.
        retrieval_context = [str(o) for o in tool_outputs] if tool_outputs else ["No tool outputs available"]

        def _make_metric(cls, **kwargs):
            """Instantiate metric with custom model when available."""
            if eval_model is not None:
                return cls(model=eval_model, **kwargs)
            return cls(**kwargs)

        # ── Tool Correctness ──
        if deepeval_tool_calls:
            try:
                from deepeval.metrics import ToolCorrectnessMetric

                def _run_tool_correctness():
                    tc_case = LLMTestCase(
                        input=user_prompt,
                        actual_output=agent_response,
                        tools_called=deepeval_tool_calls,
                        expected_tools=deepeval_tool_calls,
                    )
                    metric = _make_metric(ToolCorrectnessMetric, threshold=0.5, include_reason=True)
                    metric.measure(tc_case)
                    return metric

                metric = _retry_sync(_run_tool_correctness)
                scores["tool_correctness"] = metric.score
                scores["tool_correctness_reason"] = metric.reason or ""
            except Exception as exc:
                scores["tool_correctness"] = None
                scores["tool_correctness_reason"] = _fmt_error(exc)
                logger.error("ToolCorrectness metric failed: %s", exc)

        # ── Argument Correctness ──
        if deepeval_tool_calls:
            try:
                from deepeval.metrics import ArgumentCorrectnessMetric

                def _run_arg_correctness():
                    ac_case = LLMTestCase(
                        input=user_prompt,
                        actual_output=agent_response,
                        tools_called=deepeval_tool_calls,
                    )
                    metric = _make_metric(ArgumentCorrectnessMetric, threshold=0.5, include_reason=True)
                    metric.measure(ac_case)
                    return metric

                metric = _retry_sync(_run_arg_correctness)
                scores["argument_correctness"] = metric.score
                scores["argument_correctness_reason"] = metric.reason or ""
            except Exception as exc:
                scores["argument_correctness"] = None
                scores["argument_correctness_reason"] = _fmt_error(exc)
                logger.error("ArgumentCorrectness metric failed: %s", exc)

        # ── Answer Relevancy ──
        try:
            from deepeval.metrics import AnswerRelevancyMetric

            def _run_answer_relevancy():
                rel_case = LLMTestCase(
                    input=user_prompt,
                    actual_output=agent_response,
                    retrieval_context=retrieval_context,
                )
                metric = _make_metric(AnswerRelevancyMetric, threshold=0.5, include_reason=True)
                metric.measure(rel_case)
                return metric

            metric = _retry_sync(_run_answer_relevancy)
            scores["deepeval_relevancy"] = metric.score
            scores["deepeval_relevancy_reason"] = metric.reason or ""
        except Exception as exc:
            scores["deepeval_relevancy"] = None
            scores["deepeval_relevancy_reason"] = _fmt_error(exc)
            logger.error("AnswerRelevancy metric failed: %s", exc)

        # ── Faithfulness ──
        try:
            from deepeval.metrics import FaithfulnessMetric

            def _run_faithfulness():
                faith_case = LLMTestCase(
                    input=user_prompt,
                    actual_output=agent_response,
                    retrieval_context=retrieval_context,
                )
                metric = _make_metric(FaithfulnessMetric, threshold=0.5, include_reason=True)
                metric.measure(faith_case)
                return metric

            metric = _retry_sync(_run_faithfulness)
            scores["deepeval_faithfulness"] = metric.score
            scores["deepeval_faithfulness_reason"] = metric.reason or ""
        except Exception as exc:
            scores["deepeval_faithfulness"] = None
            scores["deepeval_faithfulness_reason"] = _fmt_error(exc)
            logger.error("Faithfulness metric failed: %s", exc)

    except ImportError as exc:
        err = _fmt_error(exc, attempts=1)
        scores.update({
            "deepeval_relevancy": None,
            "deepeval_relevancy_reason": f"ERROR: DeepEval not installed — {exc}",
            "deepeval_faithfulness": None,
            "deepeval_faithfulness_reason": f"ERROR: DeepEval not installed — {exc}",
            "tool_correctness": None,
            "tool_correctness_reason": f"ERROR: DeepEval not installed — {exc}",
            "argument_correctness": None,
            "argument_correctness_reason": f"ERROR: DeepEval not installed — {exc}",
        })
        logger.error("DeepEval import failed: %s", exc)

    return scores


# ── Agentic Trace Metrics (LLM-as-Judge) ───────────────────────

async def run_agentic_trace_metrics(
    user_prompt: str,
    agent_response: str,
    agent_trace: list[dict],
) -> dict:
    """
    Run trace-based agentic metrics using LLM-as-judge.
    Follows DeepEval's methodology for:
    - Task Completion: Did the agent accomplish the user's task?
    - Step Efficiency: Were execution steps minimal and necessary?
    - Plan Quality: Was the agent's plan logical and complete?
    - Plan Adherence: Did the agent follow its own plan?

    Scores are None on failure; reason strings begin with "ERROR:" so report
    layers can distinguish real failures from low-quality outputs.
    """
    from src.llm.client import get_judge_llm
    from src.orchestrator import _extract_text
    import re

    execution_steps = _build_execution_steps_text(agent_trace)
    plan_text = _extract_plan_from_trace(agent_trace)

    results: dict = {}
    llm = get_judge_llm()

    def _parse_score_json(raw: str, field: str = "score") -> tuple[float, str]:
        """Parse {reasoning, score} JSON from raw LLM output. Raises on failure."""
        pattern = r'\{[^}]*"' + field + r'"\s*:\s*(\d)[^}]*\}'
        match = re.search(pattern, raw, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find '{field}' in LLM output: {raw[:400]}")
        data = json.loads(match.group())
        score = max(1, min(5, data.get(field, 3)))
        return ((score - 1) / 4, data.get("reasoning", ""))

    async def _eval_task_completion() -> tuple[Optional[float], str]:
        # Full content — no truncation.
        prompt = f"""You are evaluating whether an AI agent successfully completed the user's task.

USER TASK:
{user_prompt}

AGENT EXECUTION TRACE:
{execution_steps}

AGENT FINAL OUTPUT:
{agent_response}

Evaluate:
1. What was the user's intended task/goal?
2. Did the agent's execution steps lead toward completing this task?
3. Does the final output satisfy the user's request?

Rate task completion on a scale of 1-5:
5: Task fully completed — all requirements met
4: Mostly completed — minor aspects missing
3: Partially completed — core task addressed but significant gaps
2: Barely started — some relevant actions but task largely incomplete
1: Not completed — agent failed to address the task

Respond with ONLY JSON: {{"reasoning": "<your analysis>", "score": <1-5>}}"""
        try:
            resp = await _retry_async(lambda: llm.ainvoke([{"role": "user", "content": prompt}]))
            return _parse_score_json(_extract_text(resp.content))
        except Exception as exc:
            logger.error("task_completion eval failed: %s", exc)
            return (None, _fmt_error(exc))

    async def _eval_step_efficiency() -> tuple[Optional[float], str]:
        prompt = f"""You are evaluating the efficiency of an AI agent's execution steps.

USER TASK:
{user_prompt}

EXECUTION STEPS:
{execution_steps}

Evaluate:
1. Were all steps necessary for completing the task?
2. Were there redundant or unnecessary tool calls?
3. Could the task have been completed in fewer steps?
4. Were the steps executed in a logical order?

Rate step efficiency on a scale of 1-5:
5: Optimally efficient — every step was necessary, no redundancy
4: Mostly efficient — one minor unnecessary step
3: Moderately efficient — some redundant or unnecessary steps
2: Inefficient — many unnecessary steps, poor ordering
1: Highly inefficient — majority of steps were wasteful

Respond with ONLY JSON: {{"reasoning": "<your analysis>", "score": <1-5>}}"""
        try:
            resp = await _retry_async(lambda: llm.ainvoke([{"role": "user", "content": prompt}]))
            return _parse_score_json(_extract_text(resp.content))
        except Exception as exc:
            logger.error("step_efficiency eval failed: %s", exc)
            return (None, _fmt_error(exc))

    async def _eval_plan_quality() -> tuple[Optional[float], str]:
        if not plan_text:
            return (1.0, "No plan detected in trace — metric passes by default (DeepEval convention)")

        prompt = f"""You are evaluating the quality of an AI agent's plan.

USER TASK:
{user_prompt}

AGENT'S PLAN:
{plan_text}

Evaluate:
1. Is the plan logical and well-structured?
2. Does it cover all necessary steps to complete the task?
3. Are the steps actionable and concrete?
4. Is the plan efficient (no unnecessary steps)?
5. Does the plan consider edge cases or potential failures?

Rate plan quality on a scale of 1-5:
5: Excellent plan — logical, complete, efficient, actionable
4: Good plan — covers main aspects, minor gaps
3: Adequate plan — addresses the task but lacks detail or has gaps
2: Poor plan — vague, incomplete, or illogical
1: No meaningful plan — steps are irrelevant or contradictory

Respond with ONLY JSON: {{"reasoning": "<your analysis>", "score": <1-5>}}"""
        try:
            resp = await _retry_async(lambda: llm.ainvoke([{"role": "user", "content": prompt}]))
            return _parse_score_json(_extract_text(resp.content))
        except Exception as exc:
            logger.error("plan_quality eval failed: %s", exc)
            return (None, _fmt_error(exc))

    async def _eval_plan_adherence() -> tuple[Optional[float], str]:
        if not plan_text:
            return (1.0, "No plan detected in trace — metric passes by default (DeepEval convention)")

        prompt = f"""You are evaluating whether an AI agent followed its own plan during execution.

USER TASK:
{user_prompt}

AGENT'S PLAN:
{plan_text}

ACTUAL EXECUTION STEPS:
{execution_steps}

Evaluate:
1. Did the agent follow the plan steps in order?
2. Were all planned steps actually executed?
3. Did the agent deviate from the plan? If so, was the deviation justified?
4. Were there execution steps that weren't part of the plan?

Rate plan adherence on a scale of 1-5:
5: Perfect adherence — agent followed plan exactly
4: High adherence — minor justified deviations
3: Moderate adherence — some steps skipped or reordered, partially justified
2: Low adherence — significant deviations from the plan
1: No adherence — agent ignored the plan entirely

Respond with ONLY JSON: {{"reasoning": "<your analysis>", "score": <1-5>}}"""
        try:
            resp = await _retry_async(lambda: llm.ainvoke([{"role": "user", "content": prompt}]))
            return _parse_score_json(_extract_text(resp.content))
        except Exception as exc:
            logger.error("plan_adherence eval failed: %s", exc)
            return (None, _fmt_error(exc))

    tc_score, tc_reason = await _eval_task_completion()
    se_score, se_reason = await _eval_step_efficiency()
    pq_score, pq_reason = await _eval_plan_quality()
    pa_score, pa_reason = await _eval_plan_adherence()

    results["task_completion"] = tc_score
    results["task_completion_reason"] = tc_reason
    results["step_efficiency_de"] = se_score
    results["step_efficiency_de_reason"] = se_reason
    results["plan_quality"] = pq_score
    results["plan_quality_reason"] = pq_reason
    results["plan_adherence"] = pa_score
    results["plan_adherence_reason"] = pa_reason

    return results


# ── Legacy entry point ──────────────────────────────────────────

def run_deepeval_metrics(
    user_prompt: str,
    agent_response: str,
    tool_outputs: list[str] = None,
    expected_output: str = None,
) -> dict:
    """
    Legacy entry point for basic DeepEval metrics (relevancy + faithfulness).
    Kept for backward compatibility. New code should use run_all_deepeval_metrics().

    Scores are None on failure; _reason keys contain the full error message.
    """
    _ensure_deepeval_env()
    scores = {}
    tool_outputs = tool_outputs or []

    retrieval_context = tool_outputs if tool_outputs else ["No tool outputs available"]

    try:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        eval_model = _get_deepeval_model()

        def _make_metric(cls, **kwargs):
            if eval_model is not None:
                return cls(model=eval_model, **kwargs)
            return cls(**kwargs)

        # Full content — no truncation.
        test_case = LLMTestCase(
            input=user_prompt,
            actual_output=agent_response,
            retrieval_context=retrieval_context,
            expected_output=expected_output or "",
        )

        try:
            def _run_relevancy():
                m = _make_metric(AnswerRelevancyMetric, threshold=0.5, include_reason=True)
                m.measure(test_case)
                return m
            m = _retry_sync(_run_relevancy)
            scores["deepeval_relevancy"] = m.score
            scores["deepeval_relevancy_reason"] = m.reason or ""
        except Exception as exc:
            scores["deepeval_relevancy"] = None
            scores["deepeval_relevancy_reason"] = _fmt_error(exc)
            logger.error("AnswerRelevancy (legacy) failed: %s", exc)

        try:
            def _run_faithfulness():
                m = _make_metric(FaithfulnessMetric, threshold=0.5, include_reason=True)
                m.measure(test_case)
                return m
            m = _retry_sync(_run_faithfulness)
            scores["deepeval_faithfulness"] = m.score
            scores["deepeval_faithfulness_reason"] = m.reason or ""
        except Exception as exc:
            scores["deepeval_faithfulness"] = None
            scores["deepeval_faithfulness_reason"] = _fmt_error(exc)
            logger.error("Faithfulness (legacy) failed: %s", exc)

    except ImportError as exc:
        scores["deepeval_relevancy"] = None
        scores["deepeval_relevancy_reason"] = f"ERROR: DeepEval not installed — {exc}"
        scores["deepeval_faithfulness"] = None
        scores["deepeval_faithfulness_reason"] = f"ERROR: DeepEval not installed — {exc}"
        logger.error("DeepEval import failed (legacy): %s", exc)
    except Exception as exc:
        scores["deepeval_relevancy"] = None
        scores["deepeval_relevancy_reason"] = _fmt_error(exc, attempts=1)
        scores["deepeval_faithfulness"] = None
        scores["deepeval_faithfulness_reason"] = _fmt_error(exc, attempts=1)
        logger.error("run_deepeval_metrics (legacy) failed: %s", exc)

    return scores


# ── Combined runner ─────────────────────────────────────────────

async def run_all_deepeval_metrics(
    user_prompt: str,
    agent_response: str,
    agent_trace: list[dict] = None,
    tool_outputs: list[str] = None,
) -> dict:
    """
    Run ALL DeepEval agentic metrics:
    - Standalone: ToolCorrectness, ArgumentCorrectness, AnswerRelevancy, Faithfulness
    - Trace-based: TaskCompletion, StepEfficiency, PlanQuality, PlanAdherence

    Scores are None on evaluation error; _reason keys surface the full error message.
    When no agent_trace is provided, trace-based metrics are marked as not evaluated.
    """
    all_scores: dict = {}
    agent_trace = agent_trace or []

    standalone = run_deepeval_standalone_metrics(
        user_prompt=user_prompt,
        agent_response=agent_response,
        agent_trace=agent_trace,
        tool_outputs=tool_outputs,
    )
    all_scores.update(standalone)

    if agent_trace:
        trace_scores = await run_agentic_trace_metrics(
            user_prompt=user_prompt,
            agent_response=agent_response,
            agent_trace=agent_trace,
        )
        all_scores.update(trace_scores)
    else:
        all_scores.update({
            "task_completion": None,
            "task_completion_reason": "No agent trace available — metric skipped",
            "step_efficiency_de": None,
            "step_efficiency_de_reason": "No agent trace available — metric skipped",
            "plan_quality": 1.0,
            "plan_quality_reason": "No plan detected — metric passes by default (DeepEval convention)",
            "plan_adherence": 1.0,
            "plan_adherence_reason": "No plan detected — metric passes by default (DeepEval convention)",
        })

    return all_scores
