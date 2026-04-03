"""
Human-in-the-Loop (HITL) support for the SDLC Agent platform.

Provides four HITL checkpoint types using LangGraph's native interrupt() mechanism:
  1. Clarification Q&A   -- Agent asks user for more context
  2. Plan Review & Edit   -- Planner shows plan for user approval/editing
  3. Action Confirmation  -- Dangerous tool calls require user approval
  4. Tool Output Review   -- User reviews tool output before agent continues

Each HITL interaction pauses the graph via interrupt(), streams a hitl_request
event to the frontend, and resumes when the user submits a response via
Command(resume=...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.tools import tool as lc_tool, StructuredTool
from langgraph.types import interrupt


# ── HITL Request Types ───────────────────────────────────────────

@dataclass
class HITLRequest:
    type: str = ""
    agent: str = ""
    message: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ClarificationRequest(HITLRequest):
    type: str = "clarification"
    question: str = ""
    options: list[str] = field(default_factory=list)


@dataclass
class PlanReviewRequest(HITLRequest):
    type: str = "plan_review"
    plan: list[dict] = field(default_factory=list)


@dataclass
class ActionConfirmationRequest(HITLRequest):
    type: str = "action_confirmation"
    tool_name: str = ""
    args: dict = field(default_factory=dict)
    risk_level: str = "high"
    reason: str = ""


@dataclass
class ToolOutputReviewRequest(HITLRequest):
    type: str = "tool_review"
    tool_name: str = ""
    args: dict = field(default_factory=dict)
    output: str = ""


# ── Dangerous Tools Classification ──────────────────────────────

DANGEROUS_TOOLS = {
    "run_command",
    "run_script",
    "run_tests",
    "write_file",
    "edit_file",
    "git_commit",
    "git_branch",
    # GitHub remote operations — all require explicit user confirmation
    "github_create_repo",
    "github_remote_add",
    "github_push",
    "github_create_pr",
    # Planner write operations — always confirm before modifying Planner
    "planner_create_plan",
    "planner_create_bucket",
    "planner_create_task",
    "planner_assign_task",
    "planner_update_task",
    # Jira write operations — always confirm before creating/modifying tickets
    "jira_create_issue",
    "jira_update_issue",
    "jira_assign_issue",
    "jira_transition_issue",
    # New GitHub write operations
    "github_create_branch",
    "github_create_file",
}

REVIEWABLE_TOOLS = {
    "run_command",
    "run_script",
}


# ── ask_human Tool ──────────────────────────────────────────────

@lc_tool
def ask_human(question: str, options: str = "") -> str:
    """Ask the user a clarifying question when you need more context to proceed.
    Use this when the request is ambiguous, underspecified, or you need the user
    to choose between alternatives. Provide a clear question and optional
    comma-separated list of choices.

    Args:
        question: The clarifying question to ask the user.
        options: Optional comma-separated list of choices (e.g. "option A, option B, option C").
    """
    opts = [o.strip() for o in options.split(",") if o.strip()] if options else []
    response = interrupt(ClarificationRequest(
        agent="current",
        question=question,
        options=opts,
        message=question,
    ).to_dict())
    return response.get("answer", "") if isinstance(response, dict) else str(response)


# ── Dangerous Tool Wrapper ──────────────────────────────────────

def wrap_dangerous_tool(original: StructuredTool, agent_role: str = "") -> StructuredTool:
    """Wrap a tool so it requires user confirmation before execution."""
    original_func = original.func
    original_coroutine = original.coroutine

    def _confirm_and_run(**kwargs: Any) -> str:
        response = interrupt(ActionConfirmationRequest(
            agent=agent_role,
            tool_name=original.name,
            args=_safe_serialize_args(kwargs),
            risk_level="high" if original.name in ("run_command", "run_script") else "medium",
            reason=_risk_reason(original.name),
            message=f"Agent wants to execute: {original.name}",
        ).to_dict())

        if isinstance(response, dict) and not response.get("approved", False):
            return f"Action '{original.name}' was denied by the user."

        if original_func:
            return original_func(**kwargs)
        raise RuntimeError("Tool has no sync function")

    async def _confirm_and_run_async(**kwargs: Any) -> str:
        response = interrupt(ActionConfirmationRequest(
            agent=agent_role,
            tool_name=original.name,
            args=_safe_serialize_args(kwargs),
            risk_level="high" if original.name in ("run_command", "run_script") else "medium",
            reason=_risk_reason(original.name),
            message=f"Agent wants to execute: {original.name}",
        ).to_dict())

        if isinstance(response, dict) and not response.get("approved", False):
            return f"Action '{original.name}' was denied by the user."

        if original_coroutine:
            return await original_coroutine(**kwargs)
        if original_func:
            return original_func(**kwargs)
        raise RuntimeError("Tool has no function")

    return StructuredTool(
        name=original.name,
        description=original.description,
        args_schema=original.args_schema,
        func=_confirm_and_run,
        coroutine=_confirm_and_run_async,
    )


# ── Tool Output Review Wrapper ──────────────────────────────────

def wrap_reviewable_tool(original: StructuredTool, agent_role: str = "") -> StructuredTool:
    """Wrap a tool so its output is shown to the user for review before the agent continues."""
    original_func = original.func
    original_coroutine = original.coroutine

    def _run_and_review(**kwargs: Any) -> str:
        if original_func:
            result = original_func(**kwargs)
        else:
            raise RuntimeError("Tool has no sync function")

        review = interrupt(ToolOutputReviewRequest(
            agent=agent_role,
            tool_name=original.name,
            args=_safe_serialize_args(kwargs),
            output=str(result)[:2000],
            message=f"Review output from: {original.name}",
        ).to_dict())

        if isinstance(review, dict):
            action = review.get("action", "continue")
            if action == "modify":
                return review.get("modified_output", str(result))
            if action == "stop":
                return "Execution stopped by user after reviewing tool output."
        return str(result)

    async def _run_and_review_async(**kwargs: Any) -> str:
        if original_coroutine:
            result = await original_coroutine(**kwargs)
        elif original_func:
            result = original_func(**kwargs)
        else:
            raise RuntimeError("Tool has no function")

        review = interrupt(ToolOutputReviewRequest(
            agent=agent_role,
            tool_name=original.name,
            args=_safe_serialize_args(kwargs),
            output=str(result)[:2000],
            message=f"Review output from: {original.name}",
        ).to_dict())

        if isinstance(review, dict):
            action = review.get("action", "continue")
            if action == "modify":
                return review.get("modified_output", str(result))
            if action == "stop":
                return "Execution stopped by user after reviewing tool output."
        return str(result)

    return StructuredTool(
        name=original.name,
        description=original.description,
        args_schema=original.args_schema,
        func=_run_and_review,
        coroutine=_run_and_review_async,
    )


# ── Planner HITL Executor ──────────────────────────────────────

def make_planner_executor(role: str, built_agents: dict, exec_agent=None, agent_model: str | None = None):
    """Create a two-phase executor: plan (LLM-only) -> review -> execute (with tools).

    Phase 1: Calls the underlying LLM directly (no tools) to generate a plan.
    Phase 2: Interrupts for user review/editing of the plan.
    Phase 3: Invokes an execution agent (without HITL tool wrappers) to execute the
             approved plan, avoiding nested interrupt() issues.
    """
    execution_agent = exec_agent or built_agents[role]

    async def execute(state: dict) -> dict:
        from src.orchestrator import _ensure_messages, _extract_text
        from src.llm.client import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        msgs = _ensure_messages(state["messages"])

        # ── Phase 1: Generate plan (LLM only, no tool execution) ─────
        planning_llm = get_llm(model=agent_model if agent_model else None)
        plan_prompt = SystemMessage(content=(
            "You are a planning agent. Your job right now is ONLY to create "
            "a numbered step-by-step plan for the user's request. "
            "Do NOT execute anything yet. Just output a clear numbered plan.\n"
            "Format: one step per line, numbered 1. 2. 3. etc."
        ))
        plan_response = await planning_llm.ainvoke([plan_prompt, *msgs])
        plan_text = _extract_text(plan_response.content)
        plan_steps = _parse_plan(plan_text)

        if not plan_steps:
            plan_steps = [{"step": 1, "description": plan_text[:500], "status": "pending"}]

        # ── Phase 2: HITL — user reviews / edits the plan ────────────
        review = interrupt(PlanReviewRequest(
            agent=role,
            plan=plan_steps,
            message="Please review the proposed plan before proceeding.",
        ).to_dict())

        tool_calls = []

        if isinstance(review, dict) and review.get("approved", False):
            edited = review.get("edited_plan", plan_steps)
            plan_summary = _format_plan(edited)

            # ── Phase 3: Execute via unwrapped agent (no nested interrupts) ──
            exec_msgs = list(msgs)
            exec_msgs.append(HumanMessage(content=(
                f"Execute this approved plan step by step using your tools:\n\n"
                f"{plan_summary}\n\n"
                f"Work through each step. Use create_plan and update_plan_step "
                f"to track progress. Report results as you go."
            )))

            result = await execution_agent.ainvoke(
                {"messages": exec_msgs},
                config={"recursion_limit": 80},
            )

            for msg in result.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({"tool": tc["name"], "args": tc["args"]})
        elif isinstance(review, dict):
            feedback = review.get("feedback", "User requested changes.")
            retry_msgs = list(msgs)
            retry_msgs.append(HumanMessage(content=(
                f"The user rejected the plan and gave this feedback: {feedback}\n"
                f"Please create a revised plan (numbered steps only, don't execute)."
            )))
            retry_response = await planning_llm.ainvoke(
                [plan_prompt, *retry_msgs]
            )
            result = {"messages": retry_msgs + [retry_response]}
        else:
            result = {"messages": msgs + [plan_response]}

        return {
            "messages": result.get("messages", msgs),
            "selected_agent": role,
            "agent_trace": [{
                "step": "execution", "agent": role, "tool_calls": tool_calls,
                "num_messages": len(result.get("messages", [])),
                "hitl": "plan_review",
            }],
        }

    return execute


# ── Helper Functions ────────────────────────────────────────────

def _parse_plan(text: str) -> list[dict]:
    """Extract structured plan steps from planner output text."""
    if not text:
        return []

    steps = []
    lines = text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^(?:\d+[\.\)]\s*|[-*]\s+)(.*)', line)
        if match:
            desc = match.group(1).strip()
            if desc:
                steps.append({
                    "step": len(steps) + 1,
                    "description": desc,
                    "status": "pending",
                })

    if not steps and len(text) > 20:
        steps.append({
            "step": 1,
            "description": text[:500],
            "status": "pending",
        })

    return steps


def _format_plan(steps: list[dict]) -> str:
    """Format plan steps back into readable text."""
    return "\n".join(
        f"{s.get('step', i+1)}. {s.get('description', '')}"
        for i, s in enumerate(steps)
    )


def _safe_serialize_args(args: dict) -> dict:
    """Safely serialize tool arguments for HITL display (truncate large values)."""
    result = {}
    for k, v in args.items():
        sv = str(v)
        result[k] = sv[:500] if len(sv) > 500 else sv
    return result


def _risk_reason(tool_name: str) -> str:
    reasons = {
        "run_command": "Executes an arbitrary shell command on the system.",
        "run_script": "Runs a Python script as a subprocess.",
        "run_tests": "Executes pytest which may run arbitrary code.",
        "write_file": "Creates or overwrites a file on disk.",
        "edit_file": "Modifies an existing file.",
        "git_commit": "Stages files and creates a git commit.",
        "git_branch": "Creates or switches git branches.",
        # GitHub
        "github_create_repo": "Creates a new GitHub repository (cannot be undone without deletion).",
        "github_remote_add": "Adds a remote URL to the local git repository config.",
        "github_push": "Pushes local commits to GitHub — makes code publicly/privately visible.",
        "github_create_pr": "Opens a Pull Request on GitHub that others can see and review.",
        # Planner
        "planner_create_plan": "Creates a new Microsoft Planner plan in an M365 group.",
        "planner_create_bucket": "Creates a new bucket (Epic) in a Planner plan.",
        "planner_create_task": "Creates a new task in Microsoft Planner and optionally assigns it.",
        "planner_assign_task": "Assigns a Planner task to a developer — sends them a notification.",
        "planner_update_task": "Modifies an existing Planner task (title, progress, due date).",
        # Jira
        "jira_create_issue": "Creates a real Jira ticket (Epic/Story/Task) — visible to all project members.",
        "jira_update_issue": "Modifies an existing Jira ticket's summary, description, or priority.",
        "jira_assign_issue": "Assigns a Jira ticket to a developer — sends them a Jira notification.",
        "jira_transition_issue": "Changes the status of a Jira ticket (e.g. To Do → In Progress).",
        # New GitHub tools
        "github_create_branch": "Creates a new branch in a remote GitHub repository — affects shared codebase.",
        "github_create_file": "Commits a new or updated file directly to a GitHub branch — permanent remote commit.",
    }
    return reasons.get(tool_name, "This action modifies system state.")
