"""
Multi-agent orchestrator using LangGraph.

Implements a router-based architecture where a lightweight routing LLM
decides which specialized agent handles each request. Each agent has
access to its own set of MCP tools for a specific aspect of development.

Architecture:
    User Request → Router → [Coder | Runner | Researcher] Agent → MCP Tools → Response
"""

from typing import Literal
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

from src.llm.client import get_llm
from src.agents.definitions import build_agent, AGENT_CONFIGS
from src.agents.prompts import ROUTER_PROMPT
from src.tools.registry import (
    get_filesystem_tools,
    get_shell_tools,
    get_git_tools,
    get_web_tools,
)


def _extract_text(content) -> str:
    """Extract text from LLM response content (handles str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return " ".join(parts)
    return str(content)


def _ensure_messages(messages: list) -> list[BaseMessage]:
    """Convert dicts to proper LangChain message objects."""
    result = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            result.append(msg)
        elif isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                result.append(HumanMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            elif role == "system":
                result.append(SystemMessage(content=content))
            else:
                result.append(HumanMessage(content=content))
        else:
            result.append(HumanMessage(content=str(msg)))
    return result


class OrchestratorState(TypedDict):
    messages: list
    selected_agent: str
    agent_trace: list[dict]


async def build_orchestrator():
    """
    Build the multi-agent LangGraph.

    The graph has three phases:
    1. Router: LLM classifies the request → selects an agent
    2. Agent Execution: Selected agent runs with its MCP tools
    3. Trace Recording: Agent selection + tool calls logged for evaluation
    """
    tool_map = {
        "filesystem": await get_filesystem_tools(),
        "shell": await get_shell_tools(),
        "git": await get_git_tools(),
        "web": await get_web_tools(),
    }

    agents = {}
    for agent_id, cfg in AGENT_CONFIGS.items():
        agent_tools = []
        for group in cfg.tool_groups:
            agent_tools.extend(tool_map.get(group, []))
        agents[agent_id] = build_agent(agent_id, agent_tools)

    router_llm = get_llm()

    async def route_request(state: OrchestratorState) -> OrchestratorState:
        """Classify the user request and select the appropriate agent."""
        msgs = _ensure_messages(state["messages"])
        response = await router_llm.ainvoke([
            SystemMessage(content=ROUTER_PROMPT),
            *msgs,
        ])

        raw_content = _extract_text(response.content)
        selected = raw_content.strip().lower().strip('"\'')
        if selected not in AGENT_CONFIGS:
            selected = "coder"

        trace_entry = {
            "step": "routing",
            "selected_agent": selected,
            "reasoning": raw_content,
        }

        return {
            **state,
            "selected_agent": selected,
            "agent_trace": state.get("agent_trace", []) + [trace_entry],
        }

    async def execute_coder(state: OrchestratorState) -> OrchestratorState:
        return await _execute_agent("coder", agents["coder"], state)

    async def execute_runner(state: OrchestratorState) -> OrchestratorState:
        return await _execute_agent("runner", agents["runner"], state)

    async def execute_researcher(state: OrchestratorState) -> OrchestratorState:
        return await _execute_agent("researcher", agents["researcher"], state)

    async def _execute_agent(
        agent_id: str, agent, state: OrchestratorState,
    ) -> OrchestratorState:
        """Execute a specialized agent and record its trace."""
        msgs = _ensure_messages(state["messages"])
        result = await agent.ainvoke({"messages": msgs})

        tool_calls = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "tool": tc["name"],
                        "args": tc["args"],
                    })

        trace_entry = {
            "step": "execution",
            "agent": agent_id,
            "tool_calls": tool_calls,
            "num_messages": len(result["messages"]),
        }

        return {
            "messages": result["messages"],
            "selected_agent": agent_id,
            "agent_trace": state.get("agent_trace", []) + [trace_entry],
        }

    def select_agent(state: OrchestratorState) -> Literal["coder", "runner", "researcher"]:
        return state["selected_agent"]

    graph = StateGraph(OrchestratorState)

    graph.add_node("router", route_request)
    graph.add_node("coder", execute_coder)
    graph.add_node("runner", execute_runner)
    graph.add_node("researcher", execute_researcher)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", select_agent)
    graph.add_edge("coder", END)
    graph.add_edge("runner", END)
    graph.add_edge("researcher", END)

    return graph.compile()
