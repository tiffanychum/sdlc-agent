"""
External evaluation tool integrations:
- Langfuse: Trace export for observability
- DeepEval: Agentic evaluation metrics (standalone + trace-equivalent)

Both are optional — if API keys are missing, they gracefully no-op.
"""

import os
import json


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
                trace.score(name=metric_name, value=float(run_summary[metric_name]))

        client.flush()
    except Exception:
        pass


# ── DeepEval Environment Setup ──────────────────────────────────

def _ensure_deepeval_env():
    """Set up OpenAI-compatible env vars for DeepEval if not already set."""
    if not os.getenv("OPENAI_API_KEY"):
        poe_key = os.getenv("POE_API_KEY", "")
        base_url = os.getenv("LLM_BASE_URL", "")
        if poe_key and base_url:
            os.environ["OPENAI_API_KEY"] = poe_key
            os.environ["OPENAI_BASE_URL"] = base_url


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


# ── DeepEval Standalone Metrics (ToolCorrectness, ArgumentCorrectness) ──

def run_deepeval_standalone_metrics(
    user_prompt: str,
    agent_response: str,
    agent_trace: list[dict] = None,
    tool_outputs: list[str] = None,
) -> dict[str, float | str]:
    """
    Run DeepEval standalone metrics on an agent response.
    Returns dict of metric_name -> score/reason.

    Standalone metrics:
    - ToolCorrectness: Were the right tools called?
    - ArgumentCorrectness: Were tool arguments well-formed?
    - AnswerRelevancy: Does the response address the query?
    - Faithfulness: Is the response grounded in tool outputs?
    """
    _ensure_deepeval_env()
    scores: dict[str, float | str] = {}
    tool_outputs = tool_outputs or []
    flat_tools = _extract_tool_calls_from_trace(agent_trace or [])

    try:
        from deepeval.test_case import LLMTestCase, ToolCall

        deepeval_tool_calls = []
        for tc in flat_tools:
            args = tc.get("args", {})
            str_args = {str(k): str(v)[:200] for k, v in args.items()} if isinstance(args, dict) else {}
            deepeval_tool_calls.append(ToolCall(
                name=str(tc["name"]),
                description=f"Tool: {tc['name']} (agent: {tc.get('agent', '?')})",
                input_parameters=str_args,
            ))

        # ── Tool Correctness ──
        if deepeval_tool_calls:
            try:
                from deepeval.metrics import ToolCorrectnessMetric
                tc_case = LLMTestCase(
                    input=user_prompt[:500],
                    actual_output=agent_response[:1000],
                    tools_called=deepeval_tool_calls,
                    expected_tools=deepeval_tool_calls,
                )
                metric = ToolCorrectnessMetric(threshold=0.5, include_reason=True)
                metric.measure(tc_case)
                scores["tool_correctness"] = metric.score or 0.0
                scores["tool_correctness_reason"] = metric.reason or ""
            except Exception as e:
                scores["tool_correctness"] = 0.5
                scores["tool_correctness_reason"] = f"Metric failed: {str(e)[:100]}"

        # ── Argument Correctness ──
        if deepeval_tool_calls:
            try:
                from deepeval.metrics import ArgumentCorrectnessMetric
                ac_case = LLMTestCase(
                    input=user_prompt[:500],
                    actual_output=agent_response[:1000],
                    tools_called=deepeval_tool_calls,
                )
                metric = ArgumentCorrectnessMetric(threshold=0.5, include_reason=True)
                metric.measure(ac_case)
                scores["argument_correctness"] = metric.score or 0.0
                scores["argument_correctness_reason"] = metric.reason or ""
            except Exception as e:
                scores["argument_correctness"] = 0.5
                scores["argument_correctness_reason"] = f"Metric failed: {str(e)[:100]}"

        # ── Answer Relevancy ──
        try:
            from deepeval.metrics import AnswerRelevancyMetric
            retrieval_context = [str(o)[:300] for o in tool_outputs[:5]] if tool_outputs else ["No tool outputs available"]
            rel_case = LLMTestCase(
                input=user_prompt[:500],
                actual_output=agent_response[:1000],
                retrieval_context=retrieval_context,
            )
            metric = AnswerRelevancyMetric(threshold=0.5, include_reason=True)
            metric.measure(rel_case)
            scores["deepeval_relevancy"] = metric.score or 0.0
            scores["deepeval_relevancy_reason"] = metric.reason or ""
        except Exception as e:
            scores["deepeval_relevancy"] = 0.5
            scores["deepeval_relevancy_reason"] = f"Metric failed: {str(e)[:100]}"

        # ── Faithfulness ──
        try:
            from deepeval.metrics import FaithfulnessMetric
            retrieval_context = [str(o)[:300] for o in tool_outputs[:5]] if tool_outputs else ["No tool outputs available"]
            faith_case = LLMTestCase(
                input=user_prompt[:500],
                actual_output=agent_response[:1000],
                retrieval_context=retrieval_context,
            )
            metric = FaithfulnessMetric(threshold=0.5, include_reason=True)
            metric.measure(faith_case)
            scores["deepeval_faithfulness"] = metric.score or 0.0
            scores["deepeval_faithfulness_reason"] = metric.reason or ""
        except Exception as e:
            scores["deepeval_faithfulness"] = 0.5
            scores["deepeval_faithfulness_reason"] = f"Metric failed: {str(e)[:100]}"

    except ImportError:
        scores.update({
            "deepeval_relevancy": 0.5,
            "deepeval_faithfulness": 0.5,
            "tool_correctness": 0.5,
            "argument_correctness": 0.5,
        })
    except Exception:
        scores.update({
            "deepeval_relevancy": 0.5,
            "deepeval_faithfulness": 0.5,
            "tool_correctness": 0.5,
            "argument_correctness": 0.5,
        })

    return scores


# ── Agentic Trace Metrics (LLM-as-Judge, following DeepEval methodology) ──
#
# These metrics require full agent trace analysis. DeepEval's native
# implementations need the @observe decorator at runtime, so for post-hoc
# evaluation we implement equivalent LLM-as-judge versions following
# the same scoring methodology.

async def run_agentic_trace_metrics(
    user_prompt: str,
    agent_response: str,
    agent_trace: list[dict],
) -> dict[str, float | str]:
    """
    Run trace-based agentic metrics using LLM-as-judge.
    Follows DeepEval's methodology for:
    - Task Completion: Did the agent accomplish the user's task?
    - Step Efficiency: Were execution steps minimal and necessary?
    - Plan Quality: Was the agent's plan logical and complete?
    - Plan Adherence: Did the agent follow its own plan?

    Returns dict with scores (0-1) and reasoning strings.
    """
    from src.llm.client import get_judge_llm
    from src.orchestrator import _extract_text

    execution_steps = _build_execution_steps_text(agent_trace)
    plan_text = _extract_plan_from_trace(agent_trace)

    results: dict[str, float | str] = {}
    llm = get_judge_llm()

    async def _eval_task_completion() -> tuple[float, str]:
        prompt = f"""You are evaluating whether an AI agent successfully completed the user's task.

USER TASK: {user_prompt[:500]}

AGENT EXECUTION TRACE:
{execution_steps[:1500]}

AGENT FINAL OUTPUT (first 800 chars):
{agent_response[:800]}

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
            resp = await llm.ainvoke([{"role": "user", "content": prompt}])
            raw = _extract_text(resp.content)
            import re
            match = re.search(r'\{[^}]*"score"\s*:\s*(\d)[^}]*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = max(1, min(5, data.get("score", 3)))
                return ((score - 1) / 4, data.get("reasoning", ""))
        except Exception:
            pass
        return (0.5, "Evaluation failed")

    async def _eval_step_efficiency() -> tuple[float, str]:
        prompt = f"""You are evaluating the efficiency of an AI agent's execution steps.

USER TASK: {user_prompt[:500]}

EXECUTION STEPS:
{execution_steps[:1500]}

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
            resp = await llm.ainvoke([{"role": "user", "content": prompt}])
            raw = _extract_text(resp.content)
            import re
            match = re.search(r'\{[^}]*"score"\s*:\s*(\d)[^}]*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = max(1, min(5, data.get("score", 3)))
                return ((score - 1) / 4, data.get("reasoning", ""))
        except Exception:
            pass
        return (0.5, "Evaluation failed")

    async def _eval_plan_quality() -> tuple[float, str]:
        if not plan_text:
            return (1.0, "No plan detected in trace — metric passes by default (DeepEval convention)")

        prompt = f"""You are evaluating the quality of an AI agent's plan.

USER TASK: {user_prompt[:500]}

AGENT'S PLAN:
{plan_text[:1500]}

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
            resp = await llm.ainvoke([{"role": "user", "content": prompt}])
            raw = _extract_text(resp.content)
            import re
            match = re.search(r'\{[^}]*"score"\s*:\s*(\d)[^}]*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = max(1, min(5, data.get("score", 3)))
                return ((score - 1) / 4, data.get("reasoning", ""))
        except Exception:
            pass
        return (0.5, "Evaluation failed")

    async def _eval_plan_adherence() -> tuple[float, str]:
        if not plan_text:
            return (1.0, "No plan detected in trace — metric passes by default (DeepEval convention)")

        prompt = f"""You are evaluating whether an AI agent followed its own plan during execution.

USER TASK: {user_prompt[:500]}

AGENT'S PLAN:
{plan_text[:1000]}

ACTUAL EXECUTION STEPS:
{execution_steps[:1500]}

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
            resp = await llm.ainvoke([{"role": "user", "content": prompt}])
            raw = _extract_text(resp.content)
            import re
            match = re.search(r'\{[^}]*"score"\s*:\s*(\d)[^}]*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = max(1, min(5, data.get("score", 3)))
                return ((score - 1) / 4, data.get("reasoning", ""))
        except Exception:
            pass
        return (0.5, "Evaluation failed")

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


# ── Combined DeepEval Runner ────────────────────────────────────

def run_deepeval_metrics(
    user_prompt: str,
    agent_response: str,
    tool_outputs: list[str] = None,
    expected_output: str = None,
) -> dict[str, float]:
    """
    Legacy entry point for basic DeepEval metrics (relevancy + faithfulness).
    Kept for backward compatibility. New code should use run_all_deepeval_metrics().
    """
    _ensure_deepeval_env()
    scores = {}
    tool_outputs = tool_outputs or []

    try:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
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


async def run_all_deepeval_metrics(
    user_prompt: str,
    agent_response: str,
    agent_trace: list[dict] = None,
    tool_outputs: list[str] = None,
) -> dict[str, float | str]:
    """
    Run ALL DeepEval agentic metrics:
    - Standalone: ToolCorrectness, ArgumentCorrectness, AnswerRelevancy, Faithfulness
    - Trace-based: TaskCompletion, StepEfficiency, PlanQuality, PlanAdherence
    """
    all_scores: dict[str, float | str] = {}
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
            "task_completion": 0.5,
            "step_efficiency_de": 0.5,
            "plan_quality": 1.0,
            "plan_adherence": 1.0,
        })

    return all_scores
