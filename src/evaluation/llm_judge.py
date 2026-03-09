"""
LLM-as-Judge evaluation (G-Eval approach).

Uses the LLM itself to evaluate agent outputs on multiple dimensions.
Each criterion gets a 1-5 score via a structured prompt, normalized to 0-1.

This follows the DeepEval G-Eval methodology:
- Define evaluation criteria with rubrics
- Ask the LLM to score the output
- Parse structured scores
"""

import json
import re
from src.llm.client import get_llm
from src.orchestrator import _extract_text


JUDGE_CRITERIA = {
    "correctness": {
        "description": "Is the response factually correct and does it accurately address the user's request?",
        "rubric": "5=Perfectly correct, 4=Mostly correct with minor issues, 3=Partially correct, 2=Mostly incorrect, 1=Completely wrong",
    },
    "relevance": {
        "description": "Is the response relevant to the user's request? Does it stay on topic?",
        "rubric": "5=Highly relevant, 4=Mostly relevant, 3=Somewhat relevant, 2=Mostly off-topic, 1=Completely irrelevant",
    },
    "coherence": {
        "description": "Is the response well-structured, logical, and easy to follow?",
        "rubric": "5=Excellent coherence, 4=Good coherence, 3=Acceptable, 2=Somewhat incoherent, 1=Completely incoherent",
    },
    "tool_usage_quality": {
        "description": "Did the agent use the appropriate tools effectively? Were the tool calls necessary and well-parameterized?",
        "rubric": "5=Optimal tool usage, 4=Good tool usage, 3=Acceptable, 2=Poor tool choices, 1=Completely wrong tools",
    },
    "completeness": {
        "description": "Does the response fully address all parts of the user's request?",
        "rubric": "5=Fully complete, 4=Mostly complete, 3=Partially complete, 2=Mostly incomplete, 1=Did not address the request",
    },
}


async def judge_response(
    user_prompt: str,
    agent_response: str,
    tool_calls: list[dict] = None,
    tool_outputs: list[str] = None,
    criteria: list[str] = None,
) -> dict[str, float]:
    """
    Use LLM-as-Judge to score an agent response on multiple criteria.
    Returns dict of criterion -> score (0.0 to 1.0).
    """
    criteria = criteria or list(JUDGE_CRITERIA.keys())
    tool_calls = tool_calls or []
    tool_outputs = tool_outputs or []

    tool_summary = ""
    if tool_calls:
        tool_summary = "\n".join(
            f"- {tc.get('tool', '?')}({json.dumps(tc.get('args', {}))[:100]})"
            for tc in tool_calls[:10]
        )

    output_summary = ""
    if tool_outputs:
        output_summary = "\n".join(o[:200] for o in tool_outputs[:5])

    criteria_block = "\n".join(
        f"- {name}: {JUDGE_CRITERIA[name]['description']} ({JUDGE_CRITERIA[name]['rubric']})"
        for name in criteria if name in JUDGE_CRITERIA
    )

    prompt = f"""You are an expert evaluator judging an AI agent's response quality.

USER REQUEST: {user_prompt[:500]}

AGENT RESPONSE: {agent_response[:1000]}

TOOLS CALLED:
{tool_summary or "None"}

TOOL OUTPUTS:
{output_summary or "None"}

Score the response on each criterion (1-5):
{criteria_block}

Respond with ONLY a JSON object mapping criterion names to integer scores (1-5).
Example: {{"correctness": 4, "relevance": 5, "coherence": 3, "tool_usage_quality": 4, "completeness": 4}}"""

    try:
        llm = get_llm()
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        raw = _extract_text(response.content)

        match = re.search(r'\{[^}]+\}', raw)
        if match:
            scores_raw = json.loads(match.group())
            return {k: max(0.0, min(1.0, (v - 1) / 4)) for k, v in scores_raw.items() if k in JUDGE_CRITERIA}
    except Exception:
        pass

    return {c: 0.5 for c in criteria if c in JUDGE_CRITERIA}


async def judge_trajectory(
    user_prompt: str,
    trace_steps: list[dict],
    final_response: str,
) -> dict:
    """
    Evaluate the full trajectory (sequence of decisions, tool calls, parameters).
    Scores each step individually + overall trajectory quality.

    Returns:
    {
        "step_scores": [{"step": "routing", "score": 0.8, "reasoning": "..."}, ...],
        "trajectory_score": 0.75,
        "trajectory_reasoning": "..."
    }
    """
    steps_desc = []
    for i, step in enumerate(trace_steps):
        if step.get("step") == "routing":
            steps_desc.append(f"Step {i+1}: Routed to agent '{step.get('selected_agent', '?')}'")
        elif step.get("step") == "execution":
            tools = step.get("tool_calls", [])
            tool_list = ", ".join(tc.get("tool", "?") for tc in tools) if tools else "none"
            steps_desc.append(f"Step {i+1}: Agent '{step.get('agent', '?')}' called tools: [{tool_list}]")

    steps_text = "\n".join(steps_desc) if steps_desc else "No steps recorded"

    prompt = f"""You are evaluating an AI agent's decision trajectory.

USER REQUEST: {user_prompt[:500]}

AGENT TRAJECTORY:
{steps_text}

FINAL RESPONSE (first 500 chars): {final_response[:500]}

Evaluate:
1. Was each step appropriate and necessary?
2. Were the tools called in a logical order?
3. Were there any unnecessary or redundant steps?
4. Did the trajectory lead to a correct and complete answer?

Respond with ONLY a JSON object:
{{
    "step_scores": [score1, score2, ...],
    "trajectory_score": <overall 1-5>,
    "reasoning": "<brief explanation>"
}}
Each score is 1-5. Number of step_scores must match number of steps ({len(trace_steps)})."""

    try:
        llm = get_llm()
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
    except Exception:
        pass

    return {
        "step_scores": [0.5] * len(trace_steps),
        "trajectory_score": 0.5,
        "reasoning": "Judge evaluation failed — using default score",
    }
