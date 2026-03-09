"""
Dynamic multi-agent orchestrator using LangGraph.

Builds agent graphs dynamically from team configuration stored in the database.
Supports four decision strategies:
  - router_decides: LLM classifies request -> routes to one agent
  - sequential: Agents run in order, each passing context to the next
  - parallel: All agents run simultaneously, results merged
  - supervisor: A supervisor agent reviews output and can re-delegate
"""

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


STRATEGY_INSTRUCTIONS = {
    "react": """## Decision Strategy: ReAct (Reason + Act)
Think step by step: 1) Reason about what to do next 2) Take ONE action using a tool 3) Observe the result 4) Repeat until done. Always act, never just describe.""",

    "plan_execute": """## Decision Strategy: Plan-and-Execute
ALWAYS create a plan first: 1) Use create_plan to break the task into steps 2) Execute each step one at a time 3) Use update_plan_step to track progress 4) Store intermediate results in memory.""",

    "reflexion": """## Decision Strategy: Self-Reflection (Reflexion)
After each action: 1) Review what you just did 2) Check if it's correct by reading the actual result 3) If something seems wrong, investigate before proceeding 4) Store your review findings in memory.""",

    "cot": """## Decision Strategy: Chain-of-Thought
Think through the problem thoroughly before acting: 1) List all considerations 2) Reason about the best approach 3) Explain your reasoning 4) Then execute with tools.""",
}


def _strategy_instruction(strategy: str) -> str:
    return STRATEGY_INSTRUCTIONS.get(strategy, STRATEGY_INSTRUCTIONS["react"])


class OrchestratorState(TypedDict):
    messages: list
    selected_agent: str
    agent_trace: list[dict]


def _build_router_prompt(agents_config: list[dict]) -> str:
    agent_descs = "\n".join(f'- "{a["role"]}": {a["description"]}' for a in agents_config)
    agent_names = ", ".join(f'"{a["role"]}"' for a in agents_config)
    return f"""You are a routing agent. Analyze the user's request and select the best agent.

Available agents:
{agent_descs}

Respond with ONLY the agent name: {agent_names}.
If unsure, respond with "{agents_config[0]['role']}" as default."""


def _make_agent_executor(role: str, built_agents: dict):
    """Create an executor closure for a given agent role. Synchronous factory — no async needed."""
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


async def build_orchestrator_from_team(team_id: str = "default"):
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
                "id": a.id, "name": a.name, "role": a.role,
                "description": a.description, "system_prompt": a.system_prompt,
                "tool_groups": tool_groups,
                "model": getattr(a, "model", "") or "",
                "decision_strategy": getattr(a, "decision_strategy", "react") or "react",
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
        strategy_hint = _strategy_instruction(ac["decision_strategy"])
        base_prompt = ac["system_prompt"] + (f"\n\n{strategy_hint}" if strategy_hint else "")
        final_prompt = build_agent_prompt(ac["id"], base_prompt)
        llm = get_llm(model=ac["model"] if ac["model"] else None)
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
            **state, "selected_agent": selected,
            "agent_trace": state.get("agent_trace", []) + [{
                "step": "routing", "selected_agent": selected, "reasoning": raw,
            }],
        }

    graph = StateGraph(OrchestratorState)
    graph.add_node("router", route_request)

    for ac in agents_config:
        role = ac["role"]
        graph.add_node(role, _make_agent_executor(role, built_agents))
        graph.add_edge(role, END)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", lambda s: s["selected_agent"])
    return graph.compile()


def _build_sequential_graph(agents_config, built_agents):
    graph = StateGraph(OrchestratorState)
    roles = [ac["role"] for ac in agents_config]

    for role in roles:
        graph.add_node(role, _make_agent_executor(role, built_agents))

    graph.add_edge(START, roles[0])
    for i in range(len(roles) - 1):
        graph.add_edge(roles[i], roles[i + 1])
    graph.add_edge(roles[-1], END)
    return graph.compile()


def _build_parallel_graph(agents_config, built_agents):
    graph = StateGraph(OrchestratorState)
    for ac in agents_config:
        role = ac["role"]
        graph.add_node(role, _make_agent_executor(role, built_agents))
        graph.add_edge(START, role)
        graph.add_edge(role, END)
    return graph.compile()


def _build_supervisor_graph(agents_config, built_agents):
    router_llm = get_llm()
    valid_roles = {a["role"] for a in agents_config}

    agent_descs = "\n".join(f'- "{a["role"]}": {a["description"]}' for a in agents_config)
    agent_names = ", ".join(f'"{a["role"]}"' for a in agents_config)

    supervisor_prompt = f"""You are a supervisor agent. Decide which agent handles the task, or if it's complete.

Available agents:
{agent_descs}

Respond with ONLY "DONE" or an agent name ({agent_names})."""

    async def supervisor_decide(state: OrchestratorState) -> OrchestratorState:
        msgs = _ensure_messages(state["messages"])
        response = await router_llm.ainvoke([SystemMessage(content=supervisor_prompt), *msgs])
        raw = _extract_text(response.content).strip().lower().strip('"\'')
        if raw == "done" or raw not in valid_roles:
            return {**state, "selected_agent": "__done__",
                    "agent_trace": state.get("agent_trace", []) + [{"step": "supervisor", "decision": "done"}]}
        return {**state, "selected_agent": raw,
                "agent_trace": state.get("agent_trace", []) + [{"step": "supervisor", "decision": raw}]}

    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", supervisor_decide)

    for ac in agents_config:
        role = ac["role"]
        graph.add_node(role, _make_agent_executor(role, built_agents))
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
    from src.db.database import init_db, seed_defaults
    init_db()
    seed_defaults()
    return await build_orchestrator_from_team("default")
