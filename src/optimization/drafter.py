"""Prompt-drafter: a single LLM call per optimization cycle.

The drafter is the only LLM-touched step in ``run_optimization_loop``. Everything
else — baseline collection, validation, decision tree, commit — is deterministic
Python. Keeping this narrow prevents the flow-control confusion we used to hit
when the whole optimization was an agentic loop.

Output is parsed via Pydantic so we get structured ``(new_prompt, rationale,
change_type)`` back every time — no free-form scraping.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.llm.client import get_llm


# ── Structured output ─────────────────────────────────────────────────────────


class DraftResult(BaseModel):
    """What the drafter returns. Parsed from the LLM's response."""

    new_prompt: str = Field(
        description=(
            "The complete improved system prompt text. Must be the FULL prompt, "
            "not a diff. Keep edits targeted; do not rewrite unrelated sections."
        )
    )
    rationale: str = Field(
        description=(
            "2-3 sentences explaining exactly what was changed and why, citing "
            "the specific failure pattern or low sub-metric the change addresses."
        )
    )
    change_type: Literal[
        "tool_budget",
        "read_before_write",
        "output_checklist",
        "cot_reasoning",
        "priority_tool_list",
        "role_scoping",
        "error_recovery",
        "other",
    ] = Field(
        description=(
            "Canonical category of the change — lets the UI/report group similar "
            "optimization attempts across runs."
        )
    )


# ── Drafter prompt ────────────────────────────────────────────────────────────

_DRAFTER_SYSTEM = """\
You are a prompt-engineering specialist. Your single task is to produce ONE \
targeted improvement to an agent system prompt to address a specific observed \
failure pattern in regression tests.

Guardrails:
- Output the COMPLETE new prompt, not a diff.
- Make TARGETED edits only — do not rewrite sections that are unrelated to the \
  failure. A smaller, well-justified change is always preferred.
- Do NOT invent capabilities the agent does not have. Only reference tools \
  that the agent already has access to.
- Preserve any ## section headers, ## REASONING PROTOCOL blocks, and safety \
  guardrails that already exist in the current prompt.
- If the failure pattern is unclear or the baseline shows no meaningful weakness, \
  make the MINIMAL additive change and explain the uncertainty in the rationale.

Heuristics by low metric (pick ONE lever per cycle):
- step_efficiency / step_efficiency_de low  → add a tool-call budget + \
  "state intent before each call" rule.
- tool_correctness / tool_usage low          → add a priority-ordered tool list \
  with explicit ❌ anti-patterns.
- deepeval_faithfulness low                  → add hard rule: \
  "NEVER write/edit without first read_file this session".
- deepeval_relevancy low                     → add output scoping: \
  "stick to the exact scope in the user request; no tangents".
- task_completion / plan_quality low         → add an output checklist at the \
  end of the prompt that enumerates required deliverables.
- plan_adherence low                         → add reminder to consult the \
  approved plan before every tool call.
- argument_correctness low                   → add one-line examples of the \
  correct argument shape for the tool that failed most.
"""


_DRAFTER_USER_TMPL = """\
TARGET_ROLE: {role}
TARGET_METRIC: {metric}  (threshold: {threshold})
CYCLE: {cycle}  of  {max_cycles}

--- Baseline regression evidence ---
{baseline_failures}

--- Similar past failures (few-shot memory) ---
{similar_failures}

--- Current prompt (v{version}) — edit THIS ---
{current_prompt}

--- Prior cycles this run (if any) ---
{prior_summary}

Produce the single next prompt revision targeting the ONE weakest sub-metric \
above. Return structured JSON matching the schema:

{format_instructions}
"""


# ── Public API ────────────────────────────────────────────────────────────────


async def draft_new_prompt(
    *,
    role: str,
    metric: str,
    threshold: float,
    version: str,
    current_prompt: str,
    baseline_failures: str,
    similar_failures: str,
    cycle: int,
    max_cycles: int,
    prior_summary: str,
    model: str = "claude-sonnet-4.6",
) -> DraftResult:
    """Run a single LLM call to propose a revised prompt.

    Parsed as :class:`DraftResult` via a Pydantic output parser — raises if the
    response can't be coerced. Callers should treat a failure here as a cycle
    error and fall back to keeping the baseline prompt.
    """
    parser = PydanticOutputParser(pydantic_object=DraftResult)
    user_msg = _DRAFTER_USER_TMPL.format(
        role=role,
        metric=metric,
        threshold=threshold,
        version=version,
        cycle=cycle,
        max_cycles=max_cycles,
        baseline_failures=baseline_failures.strip() or "(no baseline failures found)",
        similar_failures=similar_failures.strip() or "(no similar past failures)",
        current_prompt=current_prompt.strip(),
        prior_summary=prior_summary.strip() or "(first cycle)",
        format_instructions=parser.get_format_instructions(),
    )

    llm = get_llm(model=model, temperature=0.2)
    response = await llm.ainvoke([
        SystemMessage(content=_DRAFTER_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    raw = response.content if isinstance(response.content, str) else str(response.content)
    return parser.parse(raw)
