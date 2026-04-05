"""
LLM-as-Judge evaluation following the G-Eval paper methodology.

G-Eval (Liu et al., 2023) has three core innovations:
1. Auto Chain-of-Thought (CoT): Generate intermediate evaluation steps before scoring
2. Per-criterion evaluation: Score one criterion at a time to prevent interference
3. Reasoning before scoring: LLM reasons through steps before committing to a number

Token probability weighting is noted but requires logprobs API support.
"""

import asyncio
import json
import logging
import re
from typing import Optional

from src.llm.client import get_judge_llm
from src.orchestrator import _extract_text

logger = logging.getLogger(__name__)

# Maximum attempts for any LLM eval call (1 original + 2 retries).
_MAX_ATTEMPTS = 3
_RETRY_DELAY_S = 1.0


JUDGE_CRITERIA = {
    "correctness": {
        "description": "Is the response factually correct and does it accurately address the user's request?",
        "rubric": {
            5: "Perfectly correct — all facts accurate, question fully answered",
            4: "Mostly correct — minor factual issues that don't affect the core answer",
            3: "Partially correct — some accurate info but significant gaps or errors",
            2: "Mostly incorrect — fundamental errors in key claims",
            1: "Completely wrong — no accurate information",
        },
    },
    "relevance": {
        "description": "Is the response relevant to the user's request? Does it stay on topic?",
        "rubric": {
            5: "Highly relevant — directly addresses everything asked",
            4: "Mostly relevant — minor tangents but core is on-topic",
            3: "Somewhat relevant — addresses the topic but misses key aspects",
            2: "Mostly off-topic — only loosely related to the request",
            1: "Completely irrelevant — does not address the request at all",
        },
    },
    "coherence": {
        "description": "Is the response well-structured, logical, and easy to follow?",
        "rubric": {
            5: "Excellent coherence — clear structure, logical flow, easy to follow",
            4: "Good coherence — mostly well-organized with minor issues",
            3: "Acceptable — understandable but could be better organized",
            2: "Somewhat incoherent — hard to follow, jumps between ideas",
            1: "Completely incoherent — no logical structure",
        },
    },
    "tool_usage_quality": {
        "description": "Did the agent use the appropriate tools effectively? Were the tool calls necessary and well-parameterized?",
        "rubric": {
            5: "Optimal — correct tools, correct order, correct parameters, no redundancy",
            4: "Good — right tools used, minor parameter issues",
            3: "Acceptable — mostly right tools but some unnecessary calls or missing steps",
            2: "Poor — wrong tools chosen or badly parameterized",
            1: "Completely wrong — tools unrelated to the task",
        },
    },
    "completeness": {
        "description": "Does the response fully address all parts of the user's request?",
        "rubric": {
            5: "Fully complete — every part of the request addressed thoroughly",
            4: "Mostly complete — one minor aspect missing",
            3: "Partially complete — addresses main point but misses secondary aspects",
            2: "Mostly incomplete — only superficially touches the request",
            1: "Did not address the request at all",
        },
    },
}

_cot_cache: dict[str, str] = {}


# ── Retry helper ─────────────────────────────────────────────────

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
                    "Eval LLM call failed (attempt %d/%d): %s: %s — retrying in %.1fs",
                    attempt + 1, attempts, type(exc).__name__, exc, delay * (attempt + 1),
                )
                await asyncio.sleep(delay * (attempt + 1))
    raise last_exc


# ── Phase 1: Auto CoT step generation ────────────────────────────

async def _generate_evaluation_steps(criterion_name: str, criterion: dict) -> str:
    """
    Phase 1 of G-Eval: Auto-generate Chain-of-Thought evaluation steps.
    Cached so steps are only generated once per criterion per session.
    On failure after retries, falls back to a single generic step and logs the error.
    """
    if criterion_name in _cot_cache:
        return _cot_cache[criterion_name]

    rubric_text = "\n".join(f"  {score}: {desc}" for score, desc in criterion["rubric"].items())

    prompt = f"""You are an expert evaluator. Your task is to create a step-by-step evaluation procedure
for assessing an AI agent's output on the following criterion:

CRITERION: {criterion_name}
DESCRIPTION: {criterion['description']}
SCORING RUBRIC:
{rubric_text}

Generate 3-6 concise evaluation steps that a human expert would follow to assess this criterion.
Each step should be a specific, actionable check.

Format: Return ONLY a numbered list of steps, nothing else."""

    try:
        llm = get_judge_llm()
        steps = await _retry_async(
            lambda: llm.ainvoke([{"role": "user", "content": prompt}])
        )
        result = _extract_text(steps.content)
        _cot_cache[criterion_name] = result
        return result
    except Exception as exc:
        logger.error(
            "CoT step generation failed for criterion '%s' after %d attempts: %s: %s",
            criterion_name, _MAX_ATTEMPTS, type(exc).__name__, exc,
        )
        fallback = f"1. Assess whether the output meets: {criterion['description']}"
        _cot_cache[criterion_name] = fallback
        return fallback


# ── Phase 2: Score a single criterion ────────────────────────────

async def _score_single_criterion(
    criterion_name: str,
    criterion: dict,
    user_prompt: str,
    agent_response: str,
    tool_summary: str,
    output_summary: str,
) -> tuple[str, Optional[float], str]:
    """
    Phase 2 of G-Eval: Score a single criterion using the CoT evaluation steps.

    Returns (criterion_name, normalized_score_or_None, reasoning).
    Score is None when evaluation fails after all retries — callers must handle None.
    Reasoning starts with "ERROR:" on failure so report layers can flag it.
    """
    eval_steps = await _generate_evaluation_steps(criterion_name, criterion)
    rubric_text = "\n".join(f"  {score}: {desc}" for score, desc in criterion["rubric"].items())

    # Full content is passed — no truncation so the judge sees everything.
    prompt = f"""You are an expert evaluator assessing an AI agent's response.

USER REQUEST:
{user_prompt}

AGENT RESPONSE:
{agent_response}

TOOLS CALLED:
{tool_summary or "None"}

TOOL OUTPUTS:
{output_summary or "None"}

---

CRITERION: {criterion_name}
DESCRIPTION: {criterion['description']}

EVALUATION STEPS (follow these carefully):
{eval_steps}

SCORING RUBRIC:
{rubric_text}

---

First, reason through each evaluation step above. Then provide your final score.

Respond in this exact JSON format:
{{"reasoning": "<your step-by-step reasoning>", "score": <integer 1-5>}}"""

    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            llm = get_judge_llm()
            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            raw = _extract_text(response.content)

            match = re.search(r'\{[^}]*"score"\s*:\s*(\d)[^}]*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = max(1, min(5, data.get("score", 3)))
                reasoning = data.get("reasoning", "")
                return (criterion_name, (score - 1) / 4, reasoning)

            # LLM responded but JSON was unparseable — treat as a parse error, retry.
            raise ValueError(f"Could not parse score JSON from LLM response: {raw[:300]}")

        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                logger.warning(
                    "Criterion '%s' scoring failed (attempt %d/%d): %s: %s — retrying",
                    criterion_name, attempt + 1, _MAX_ATTEMPTS, type(exc).__name__, exc,
                )
                await asyncio.sleep(_RETRY_DELAY_S * (attempt + 1))

    error_msg = f"ERROR after {_MAX_ATTEMPTS} attempts: {type(last_exc).__name__}: {last_exc}"
    logger.error("Criterion '%s' scoring permanently failed: %s", criterion_name, last_exc)
    return (criterion_name, None, error_msg)


# ── Public: G-Eval scoring entry point ───────────────────────────

async def judge_response(
    user_prompt: str,
    agent_response: str,
    tool_calls: list[dict] = None,
    tool_outputs: list[str] = None,
    criteria: list[str] = None,
) -> dict:
    """
    G-Eval scoring: evaluate each criterion independently with CoT.

    Returns:
    {
        "scores":   {"correctness": 0.75, ..., "failed_criterion": null},
        "reasoning": {"correctness": "Step 1: ...", ..., "failed_criterion": "ERROR: ..."},
        "overall":  0.72,          # average of non-null scores only
        "errors":   ["failed_criterion"],  # criteria that could not be evaluated
    }
    """
    criteria = criteria or list(JUDGE_CRITERIA.keys())
    tool_calls = tool_calls or []
    tool_outputs = tool_outputs or []

    # Build full tool summary — no per-call truncation.
    tool_summary = ""
    if tool_calls:
        tool_summary = "\n".join(
            f"- {tc.get('tool', '?')}({json.dumps(tc.get('args', {}))})"
            for tc in tool_calls
        )

    # Pass all tool outputs in full.
    output_summary = "\n---\n".join(str(o) for o in tool_outputs) if tool_outputs else ""

    tasks = [
        _score_single_criterion(
            name, JUDGE_CRITERIA[name],
            user_prompt, agent_response,
            tool_summary, output_summary,
        )
        for name in criteria if name in JUDGE_CRITERIA
    ]

    results = await asyncio.gather(*tasks)

    scores = {name: score for name, score, _ in results}
    reasoning = {name: reason for name, _, reason in results}
    errors = [name for name, score, _ in results if score is None]

    valid_scores = [s for s in scores.values() if s is not None]
    overall = sum(valid_scores) / len(valid_scores) if valid_scores else None

    return {
        "scores": scores,
        "reasoning": reasoning,
        "overall": overall,
        "errors": errors,
    }


# ── Public: Trajectory evaluation ────────────────────────────────

async def judge_trajectory(
    user_prompt: str,
    trace_steps: list[dict],
    final_response: str,
) -> dict:
    """
    Evaluate the full trajectory (sequence of decisions, tool calls, parameters).
    Scores each step individually + overall trajectory quality.

    On failure, returns a result with score=None and "ERROR:" reasoning so callers
    can surface the real failure rather than treating it as a 0.5 score.
    """
    steps_desc = []
    for i, step in enumerate(trace_steps):
        if step.get("step") == "routing":
            steps_desc.append(f"Step {i+1}: Routed to agent '{step.get('selected_agent', '?')}'")
        elif step.get("step") == "execution":
            tools = step.get("tool_calls", [])
            tool_list = ", ".join(tc.get("tool", "?") for tc in tools) if tools else "none"
            steps_desc.append(f"Step {i+1}: Agent '{step.get('agent', '?')}' called tools: [{tool_list}]")
        elif step.get("step") == "supervisor":
            steps_desc.append(f"Step {i+1}: Supervisor decided: '{step.get('decision', '?')}'")

    steps_text = "\n".join(steps_desc) if steps_desc else "No steps recorded"

    # Full content — no truncation.
    prompt = f"""You are evaluating an AI agent's decision trajectory.

USER REQUEST: {user_prompt}

AGENT TRAJECTORY:
{steps_text}

FINAL RESPONSE:
{final_response}

Evaluate by reasoning through each question:
1. Was each step appropriate and necessary?
2. Were the tools called in a logical order?
3. Were there any unnecessary or redundant steps?
4. Did the trajectory lead to a correct and complete answer?

Respond with ONLY a JSON object:
{{
    "step_scores": [score1, score2, ...],
    "trajectory_score": <overall 1-5>,
    "reasoning": "<your step-by-step reasoning>"
}}
Each score is 1-5. Number of step_scores must match number of steps ({len(trace_steps)})."""

    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            llm = get_judge_llm()
            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            raw = _extract_text(response.content)

            match = re.search(r'\{[^}]*"trajectory_score"[^}]*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                t_score = max(0.0, min(1.0, (data.get("trajectory_score", 3) - 1) / 4))
                step_scores = [max(0.0, min(1.0, (s - 1) / 4)) for s in data.get("step_scores", [])]
                return {
                    "step_scores": step_scores,
                    "trajectory_score": t_score,
                    "reasoning": data.get("reasoning", ""),
                }

            raise ValueError(f"Could not parse trajectory JSON: {raw[:300]}")

        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                logger.warning(
                    "Trajectory judge failed (attempt %d/%d): %s: %s — retrying",
                    attempt + 1, _MAX_ATTEMPTS, type(exc).__name__, exc,
                )
                await asyncio.sleep(_RETRY_DELAY_S * (attempt + 1))

    error_msg = f"ERROR after {_MAX_ATTEMPTS} attempts: {type(last_exc).__name__}: {last_exc}"
    logger.error("Trajectory evaluation permanently failed: %s", last_exc)
    return {
        "step_scores": [None] * len(trace_steps),
        "trajectory_score": None,
        "reasoning": error_msg,
    }
