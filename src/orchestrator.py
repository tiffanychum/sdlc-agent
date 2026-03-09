"""
Dynamic multi-agent orchestrator using LangGraph.

Builds agent graphs dynamically from team configuration stored in the database.
Supports four decision strategies:
  - router_decides: LLM classifies request → routes to one agent
  - sequential: Agents run in order, each passing context to the next
  - parallel: All agents run simultaneously, results merged
  - supervisor: A supervisor agent reviews output and can re-delegate
"""

from typing import Literal
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from src.llm.client import get_llm
from src.skills.engine import build_agent_prompt
from src.tools.registry import get_all_tools


def _extract_text(content) -> str:
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


def _build_router_prompt(agents_config: list[dict]) -> str:
    """Dynamically build router prompt from team agent list."""
    agent_descs = "\n".join(
        f'- "{a["role"]}": {a["description"]}'
        for a in agents_config
    )
    agent_names = ", ".join(f'"{a["role"]}"' for a in agents_config)

    return f"""You are a routing agent. Analyze the user's request and select the best agent.

Available agents:
{agent_descs}

Respond with ONLY the agent name: {agent_names}.
If unsure, respond with "{agents_config[0]['role']}" as default."""


async def build_orchestrator_from_team(team_id: str = "default"):
    """
    Build a LangGraph orchestrator dynamically from a team's DB config.
    Supports: router_decides, sequential, parallel, supervisor strategies.
    """
    from src.db.database import get_session
    from src.db.models import Team, Agent, AgentToolMapping

    session = get_session()
    try:
        team = session.query(Team).filter_by(id=team_id).first()
        if not team:
            raise ValueError(f"Team '{team_id}' not found")

        agents_db = session.query(Agent).filter_by(team_id=team_id).all()
        if not agents_db:
            raise ValueError(f"Team '{team_id}' has no agents")

        agents_config = []
        for a in agents_db:
            tool_groups = [
                m.tool_group for m in session.query(AgentToolMapping).filter_by(agent_id=a.id).all()
            ]
            agents_config.append({
                "id": a.id,
                "name": a.name,
                "role": a.role,
                "description": a.description,
                "system_prompt": a.system_prompt,
                "tool_groups": tool_groups,
            })

        strategy = team.decision_strategy or "router_decides"
    finally:
        session.close()

    tool_map = await get_all_tools()

    built_agents = {}
    for ac in agents_config:
        agent_tools: list[StructuredTool] = []
        for group in ac["tool_groups"]:
            agent_tools.extend(tool_map.get(group, []))

        final_prompt = build_agent_prompt(ac["id"], ac["system_prompt"])
        llm = get_llm()
        built_agents[ac["role"]] = create_react_agent(
            model=llm, tools=agent_tools, prompt=final_prompt,
        )

    match strategy:
        case "router_decides":
            return _build_router_graph(agents_config, built_agents)
        case "sequential":
            return _build_sequential_graph(agents_config, built_agents)
        case "parallel":
            return _build_parallel_graph(agents_config, built_agents)
        case "supervisor":
            return _build_supervisor_graph(agents_config, built_agents)
        case _:
            return _build_router_graph(agents_config, built_agents)


def _build_router_graph(agents_config, built_agents):
    """Strategy: LLM router classifies request → delegates to one agent."""
    router_llm = get_llm()
    router_prompt = _build_router_prompt(agents_config)
    valid_roles = {a["role"] for a in agents_config}
    default_role = agents_config[0]["role"]

    async def route_request(state: OrchestratorState) -> OrchestratorState:
        msgs = _ensure_messages(state["messages"])
        response = await router_llm.ainvoke([SystemMessage(content=router_prompt), *msgs])
        raw = _extract_text(response.content)
        selected = raw.strip().lower().strip('"\'')
        if selected not in valid_roles:
            selected = default_role
        return {
            **state,
            "selected_agent": selected,
            "agent_trace": state.get("agent_trace", []) + [{
                "step": "routing", "selected_agent": selected, "reasoning": raw,
            }],
        }

    async def _make_executor(role):
        async def execute(state: OrchestratorState) -> OrchestratorState:
            msgs = _ensure_messages(state["messages"])
            result = await built_agents[role].ainvoke({"messages": msgs})
            tool_calls = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({"tool": tc["name"], "args": tc["args"]})
            return {
                "messages": result["messages"],
                "selected_agent": role,
                "agent_trace": state.get("agent_trace", []) + [{
                    "step": "execution", "agent": role, "tool_calls": tool_calls,
                    "num_messages": len(result["messages"]),
                }],
            }
        return execute

    graph = StateGraph(OrchestratorState)
    graph.add_node("router", route_request)

    import asyncio
    for ac in agents_config:
        role = ac["role"]
        executor = asyncio.get_event_loop().run_until_complete(_make_executor(role))
        graph.add_node(role, executor)
        graph.add_edge(role, END)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", lambda s: s["selected_agent"])

    return graph.compile()


def _build_sequential_graph(agents_config, built_agents):
    """Strategy: Agents run in order, each appending to the message history."""

    async def _make_step(role, step_idx):
        async def execute(state: OrchestratorState) -> OrchestratorState:
            msgs = _ensure_messages(state["messages"])
            result = await built_agents[role].ainvoke({"messages": msgs})
            tool_calls = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({"tool": tc["name"], "args": tc["args"]})
            return {
                "messages": result["messages"],
                "selected_agent": role,
                "agent_trace": state.get("agent_trace", []) + [{
                    "step": "execution", "agent": role, "tool_calls": tool_calls,
                    "num_messages": len(result["messages"]),
                }],
            }
        return execute

    graph = StateGraph(OrchestratorState)

    import asyncio
    roles = [ac["role"] for ac in agents_config]
    for i, role in enumerate(roles):
        executor = asyncio.get_event_loop().run_until_complete(_make_step(role, i))
        graph.add_node(role, executor)

    graph.add_edge(START, roles[0])
    for i in range(len(roles) - 1):
        graph.add_edge(roles[i], roles[i + 1])
    graph.add_edge(roles[-1], END)

    return graph.compile()


def _build_parallel_graph(agents_config, built_agents):
    """Strategy: All agents run, last one's output is the final response."""

    async def _make_parallel_step(role):
        async def execute(state: OrchestratorState) -> OrchestratorState:
            msgs = _ensure_messages(state["messages"])
            result = await built_agents[role].ainvoke({"messages": msgs})
            tool_calls = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({"tool": tc["name"], "args": tc["args"]})
            return {
                "messages": result["messages"],
                "selected_agent": role,
                "agent_trace": state.get("agent_trace", []) + [{
                    "step": "execution", "agent": role, "tool_calls": tool_calls,
                    "num_messages": len(result["messages"]),
                }],
            }
        return execute

    graph = StateGraph(OrchestratorState)

    import asyncio
    roles = [ac["role"] for ac in agents_config]
    for role in roles:
        executor = asyncio.get_event_loop().run_until_complete(_make_parallel_step(role))
        graph.add_node(role, executor)
        graph.add_edge(START, role)
        graph.add_edge(role, END)

    return graph.compile()


def _build_supervisor_graph(agents_config, built_agents):
    """Strategy: Supervisor reviews, delegates, and can re-delegate."""
    router_llm = get_llm()
    valid_roles = {a["role"] for a in agents_config}
    default_role = agents_config[0]["role"]

    agent_descs = "\n".join(f'- "{a["role"]}": {a["description"]}' for a in agents_config)
    agent_names = ", ".join(f'"{a["role"]}"' for a in agents_config)

    supervisor_prompt = f"""You are a supervisor agent. You decide which agent should handle the task, or if the task is complete.

Available agents:
{agent_descs}

After an agent responds, you can:
1. Respond with "DONE" if the task is complete
2. Respond with an agent name ({agent_names}) to delegate or re-delegate

Respond with ONLY "DONE" or an agent name."""

    async def supervisor_decide(state: OrchestratorState) -> OrchestratorState:
        msgs = _ensure_messages(state["messages"])
        response = await router_llm.ainvoke([SystemMessage(content=supervisor_prompt), *msgs])
        raw = _extract_text(response.content).strip().lower().strip('"\'')

        if raw == "done" or raw not in valid_roles:
            return {**state, "selected_agent": "__done__",
                    "agent_trace": state.get("agent_trace", []) + [{"step": "supervisor", "decision": "done"}]}

        return {**state, "selected_agent": raw,
                "agent_trace": state.get("agent_trace", []) + [{"step": "supervisor", "decision": raw}]}

    async def _make_worker(role):
        async def execute(state: OrchestratorState) -> OrchestratorState:
            msgs = _ensure_messages(state["messages"])
            result = await built_agents[role].ainvoke({"messages": msgs})
            tool_calls = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({"tool": tc["name"], "args": tc["args"]})
            return {
                "messages": result["messages"], "selected_agent": role,
                "agent_trace": state.get("agent_trace", []) + [{
                    "step": "execution", "agent": role, "tool_calls": tool_calls,
                    "num_messages": len(result["messages"]),
                }],
            }
        return execute

    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", supervisor_decide)

    import asyncio
    roles = [ac["role"] for ac in agents_config]
    for role in roles:
        executor = asyncio.get_event_loop().run_until_complete(_make_worker(role))
        graph.add_node(role, executor)
        graph.add_edge(role, "supervisor")

    graph.add_edge(START, "supervisor")

    def supervisor_router(state: OrchestratorState) -> str:
        sel = state.get("selected_agent", "__done__")
        if sel == "__done__" or sel not in valid_roles:
            return "__end__"
        return sel

    graph.add_conditional_edges("supervisor", supervisor_router)

    return graph.compile()


async def build_orchestrator():
    """Backward-compatible: build orchestrator from default team."""
    from src.db.database import init_db, seed_defaults
    init_db()
    seed_defaults()
    return await build_orchestrator_from_team("default")
