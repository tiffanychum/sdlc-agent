"""
Dynamic multi-agent orchestrator using LangGraph.

Builds agent graphs dynamically from team configuration stored in the database.
Supports four decision strategies:
  - router_decides: LLM classifies request -> routes to one agent
  - sequential: Agents run in order, each passing context to the next
  - parallel: All agents run simultaneously, results merged
  - supervisor: A supervisor agent reviews output and can re-delegate
  - auto: Meta-router LLM picks the best strategy for each prompt at runtime
"""

import asyncio
import logging
import operator
from collections import defaultdict
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from src.llm.client import get_llm, get_router_llm
from src.skills.engine import build_agent_prompt
from src.tools.registry import get_all_tools
from src.hitl import (
    ask_human, DANGEROUS_TOOLS, REVIEWABLE_TOOLS,
    wrap_dangerous_tool, wrap_reviewable_tool, make_planner_executor,
)

logger = logging.getLogger(__name__)

MAX_SUPERVISOR_ITERATIONS = 10

VALID_STRATEGIES = frozenset({"router_decides", "sequential", "parallel", "supervisor"})

_META_ROUTER_PROMPT = """\
You are an orchestration meta-router. Given a user task and available agents, choose the best \
multi-agent execution strategy.

Available agents:
{agent_descs}

Strategy options and when to use each:
- "router_decides" — Route to exactly ONE agent. Use when the task is a single, self-contained \
action (read a file, run a search, check git status). One agent can do it all alone.
- "sequential" — Agents run in a strict, predefined order (A then B then C). Use when the task \
has a clear, fixed pipeline with explicit ordering keywords ("first … then …", "after … do …").
- "parallel" — All agents run simultaneously on independent sub-tasks. Use when the task \
explicitly contains multiple independent parts that do not depend on each other and can run \
at the same time ("simultaneously", "at the same time", "both … and …").
- "supervisor" — A supervisor dynamically decides which agent runs next after each step. Use for \
complex tasks requiring multiple agents in an adaptive order (implement → test → deploy), where \
the next step depends on the outcome of the previous.

Decision rules (apply in order):
1. If the task is ONE clear action → "router_decides"
2. If the task uses "first … then …" or "after … write/do …" with a fixed pipeline → "sequential"
3. If the task says "simultaneously", "at the same time", or has clearly independent sub-tasks → "parallel"
4. If the task spans multiple agents and the flow is complex or adaptive → "supervisor"

User task: {prompt}

Respond with ONLY one of: router_decides, sequential, parallel, supervisor"""


async def select_strategy_auto(
    user_prompt: str,
    agents_config: list[dict],
    *,
    max_attempts: int = 3,
    base_delay: float = 2.0,
    default_strategy: str = "supervisor",
) -> str:
    """
    Meta-router: call a lightweight LLM to pick the best orchestration strategy.

    Returns one of: router_decides, sequential, parallel, supervisor.
    Retries up to *max_attempts* times with exponential back-off on transient
    provider errors.  Falls back to *default_strategy* if all attempts fail or
    the model returns an unrecognised value.
    """
    router_llm = get_router_llm()
    agent_descs = "\n".join(
        f'  - "{a["role"]}": {a.get("description", "")[:120]}' for a in agents_config
    )
    prompt = _META_ROUTER_PROMPT.format(agent_descs=agent_descs, prompt=user_prompt)

    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = await router_llm.ainvoke([SystemMessage(content=prompt)])
            raw = _extract_text(response.content).strip().lower().strip('"\'')
            # take first whitespace-delimited token in case the model adds explanation
            first_token = raw.split()[0] if raw.split() else ""
            if first_token not in VALID_STRATEGIES:
                logger.warning(
                    "Meta-router returned unrecognised strategy %r for prompt %.80s; "
                    "defaulting to %s",
                    first_token, user_prompt, default_strategy,
                )
                return default_strategy
            logger.info("Meta-router selected strategy=%r for prompt: %.80s", first_token, user_prompt)
            return first_token
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Meta-router attempt %d/%d failed (%s: %s); %s",
                attempt, max_attempts, type(exc).__name__, exc,
                "retrying…" if attempt < max_attempts else f"giving up, using '{default_strategy}'",
            )
            if attempt < max_attempts:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))

    logger.error(
        "Meta-router exhausted all %d attempts for prompt %.80s; using heuristic fallback",
        max_attempts, user_prompt,
    )
    return _heuristic_strategy(user_prompt)


def _heuristic_strategy(prompt: str) -> str:
    """
    Keyword-based strategy picker used when the LLM meta-router is unavailable.

    Decision order mirrors the _META_ROUTER_PROMPT rules:
    1. Parallel markers  → parallel
    2. Sequential markers → sequential
    3. Complex / multi-step → supervisor
    4. Everything else  → router_decides  (single-agent tasks)
    """
    text = prompt.lower()
    # Parallel signals
    if any(kw in text for kw in (
        "simultaneously", "at the same time", "in parallel", "concurrently",
        "both … and", "both and ", "two independent", "two completely independent",
    )):
        return "parallel"
    # Sequential signals
    if any(kw in text for kw in (
        "first … then", "first, then", "after that", "afterwards",
        "step 1", "step 2", "step one", "step two",
        "then based on", "then fix", "then write", "then deploy",
        "first review", "first read", "first search",
    )):
        return "sequential"
    # Complex multi-step / ambiguous → supervisor
    if any(kw in text for kw in (
        "implement", "build", "create a", "develop", "design",
        "multiple", "several", "various", "complex",
    )):
        return "supervisor"
    # Simple single-action → router_decides
    return "router_decides"


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
        return "\n".join(parts)
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


def _take_last(a: str, b: str) -> str:
    """Reducer that always keeps the most recent value."""
    return b


def _add_int(a: int, b: int) -> int:
    return a + b


class OrchestratorState(TypedDict):
    messages: Annotated[list, operator.add]
    selected_agent: Annotated[str, _take_last]
    agent_trace: Annotated[list[dict], operator.add]
    supervisor_iterations: Annotated[int, _add_int]


def _build_router_prompt(agents_config: list[dict]) -> str:
    agent_descs = "\n".join(f'- "{a["role"]}": {a["description"]}' for a in agents_config)
    agent_names = ", ".join(f'"{a["role"]}"' for a in agents_config)
    return f"""You are a routing agent. Select the SINGLE best-fit agent for the FIRST step of the task.
In a supervisor workflow, additional agents can be invoked afterward — you only choose the first.

Available agents:
{agent_descs}

Routing rules (apply in strict priority order):

## Jira operations
1. Task involves CREATING or MANAGING Jira (new projects, epics, stories, assigning tickets,
   updating status) → route to "project_manager".
2. Task involves DECOMPOSING a feature or requirement into developer tasks in Jira → "business_analyst".

## Local file access (filesystem)
3. Task asks to READ, OPEN, EXAMINE, SUMMARIZE, or INSPECT a local file or directory
   (e.g. "read README.md", "open config.py", "show me the contents of X", "summarize file Y",
   "list files in src/", "what does X.py do?") → "coder".
   NOTE: This rule takes priority over research. Reading a local file is NOT a web search.

## Research (web / external sources only)
4. Task requires SEARCHING THE WEB, fetching external URLs, finding real-time best practices,
   or answering a question that needs live internet information → "researcher".
   Do NOT route here if the task references a local file path.

## Code quality / review
5. Task says "review", "assess quality", "find bugs", "suggest improvements" for source code or
   a git diff, with NO request to also run or change anything → "reviewer".

## Source control & GitHub
6. Task involves ONLY git or GitHub operations: commit, push, create branch, open PR, list repos,
   check git status/log/diff — with no code to write → "devops".

## Testing
7. Task asks to write tests for existing code, run the test suite, or verify test results → "tester".

## Implementation
8. Task asks to write, edit, implement, or fix source code files → "coder".

## Multi-step planning / analysis
9. Task requires analyzing multiple files or areas without writing code (architecture audit,
   dependency analysis, codebase overview) → "planner".

## Defaults
- If the task starts with implementation AND also mentions testing/git: route to "coder" first
  (supervisor will invoke "tester" and "devops" in subsequent turns).
- If genuinely unclear: "planner".

Respond with ONLY the agent name from: {agent_names}."""


def _get_executor(role: str, built_agents: dict, exec_agents=None,
                   agent_model=None):
    """Return the planner HITL executor for planner roles, standard executor otherwise."""
    if role == "planner":
        exec_agent = (exec_agents or {}).get(role)
        return make_planner_executor(role, built_agents, exec_agent=exec_agent,
                                     agent_model=agent_model)
    return _make_agent_executor(role, built_agents)


def _make_agent_executor(role: str, built_agents: dict):
    """Create an executor closure for a given agent role."""
    async def execute(state: OrchestratorState) -> OrchestratorState:
        msgs = _ensure_messages(state["messages"])
        # Some models (e.g. Claude) reject requests where the conversation ends with
        # an AIMessage (treats it as "assistant prefill", which is not supported).
        # This happens in sequential/supervisor flows where a prior agent's response
        # is the last message. Append a sentinel HumanMessage to keep the API happy.
        if msgs and isinstance(msgs[-1], AIMessage):
            msgs = [*msgs, HumanMessage(content="Continue with your part of the task.")]
        result = await built_agents[role].ainvoke({"messages": msgs})
        out_messages = _ensure_messages(result.get("messages", []))
        tool_calls = []
        for msg in out_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({"tool": tc["name"], "args": tc["args"]})
        return {
            "messages": out_messages,
            "selected_agent": role,
            "agent_trace": [{
                "step": "execution", "agent": role, "tool_calls": tool_calls,
                "num_messages": len(out_messages),
            }],
            "supervisor_iterations": 0,
        }
    return execute


async def build_orchestrator_from_team(
    team_id: str = "default",
    model_override=None,
    strategy_override: str = None,
):
    # Deferred imports to avoid circular imports at module load time.
    from src.db.database import get_session
    from src.db.models import Team, Agent, AgentToolMapping

    session = get_session()
    try:
        team = session.query(Team).filter_by(id=team_id).first()
        if not team:
            raise ValueError(f"Team '{team_id}' not found")

        agents_db = session.query(Agent).filter_by(team_id=team_id).order_by(Agent.id).all()
        if not agents_db:
            raise ValueError(f"Team '{team_id}' has no agents")

        # Bulk-load all tool mappings in one query to avoid N+1 pattern.
        agent_ids = [a.id for a in agents_db]
        all_mappings = (
            session.query(AgentToolMapping)
            .filter(AgentToolMapping.agent_id.in_(agent_ids))
            .all()
        )
        mapping_by_agent: dict[str, list[str]] = defaultdict(list)
        for m in all_mappings:
            mapping_by_agent[m.agent_id].append(m.tool_group)

        agents_config = []
        for a in agents_db:
            agents_config.append({
                "id": a.id, "name": a.name, "role": a.role,
                "description": a.description, "system_prompt": a.system_prompt,
                "tool_groups": mapping_by_agent[a.id],
                "model": getattr(a, "model", "") or "",
                "decision_strategy": getattr(a, "decision_strategy", "react") or "react",
            })

        strategy = team.decision_strategy or "router_decides"

        # strategy_override wins over DB value when provided.
        # "auto" must already be resolved to a concrete strategy before this point.
        if strategy_override and strategy_override in VALID_STRATEGIES:
            strategy = strategy_override

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    tool_map = await get_all_tools()
    checkpointer = MemorySaver()

    if model_override:
        for ac in agents_config:
            ac["model"] = model_override

    built_agents = {}
    exec_agents = {}
    for ac in agents_config:
        agent_tools: list[StructuredTool] = []
        for group in ac["tool_groups"]:
            agent_tools.extend(tool_map.get(group, []))

        # Apply HITL wrappers — use elif so a tool can't be double-wrapped.
        hitl_tools: list[StructuredTool] = []
        for t in agent_tools:
            if t.name in DANGEROUS_TOOLS:
                t = wrap_dangerous_tool(t, agent_role=ac["role"])
            elif t.name in REVIEWABLE_TOOLS:
                t = wrap_reviewable_tool(t, agent_role=ac["role"])
            hitl_tools.append(t)

        # Append ask_human only if it is not already present in the tool list.
        existing_tool_names = {t.name for t in hitl_tools}
        if ask_human.name not in existing_tool_names:
            hitl_tools.append(ask_human)

        strategy_hint = _strategy_instruction(ac["decision_strategy"])
        base_prompt = (ac["system_prompt"] or "") + f"\n\n{strategy_hint}"
        final_prompt = build_agent_prompt(ac["id"], base_prompt)
        llm = get_llm(model=ac["model"] if ac["model"] else None)
        built_agents[ac["role"]] = create_react_agent(
            model=llm, tools=hitl_tools, prompt=final_prompt,
        )
        if ac["role"] == "planner":
            exec_agents[ac["role"]] = create_react_agent(
                model=llm, tools=list(agent_tools), prompt=final_prompt,
            )

    builders = {
        "router_decides": _build_router_graph,
        "sequential": _build_sequential_graph,
        "parallel": _build_parallel_graph,
        "supervisor": _build_supervisor_graph,
    }
    builder_fn = builders.get(strategy, _build_router_graph)
    return builder_fn(agents_config, built_agents, checkpointer=checkpointer,
                      exec_agents=exec_agents)


def _build_router_graph(agents_config, built_agents, checkpointer=None,
                        exec_agents=None):
    router_llm = get_router_llm()
    router_prompt = _build_router_prompt(agents_config)
    valid_roles = {a["role"] for a in agents_config}
    default_role = agents_config[0]["role"]

    async def route_request(state: OrchestratorState) -> OrchestratorState:
        msgs = _ensure_messages(state["messages"])
        response = await router_llm.ainvoke([SystemMessage(content=router_prompt), *msgs])
        raw = _extract_text(response.content)
        selected = raw.strip().lower().strip('"\'')
        if selected not in valid_roles:
            logger.warning(
                "Router returned unrecognised role %r; falling back to %r. "
                "Check router prompt and agent config.",
                selected, default_role,
            )
            selected = default_role
        return {
            "selected_agent": selected,
            "agent_trace": [{
                "step": "routing", "selected_agent": selected, "reasoning": raw,
            }],
            "supervisor_iterations": 0,
        }

    graph = StateGraph(OrchestratorState)
    graph.add_node("router", route_request)

    for ac in agents_config:
        role = ac["role"]
        executor = _get_executor(role, built_agents, exec_agents,
                                  agent_model=ac.get("model") or None)
        graph.add_node(role, executor)
        graph.add_edge(role, END)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", lambda s: s["selected_agent"])
    return graph.compile(checkpointer=checkpointer)


def _build_sequential_graph(agents_config, built_agents, checkpointer=None,
                            exec_agents=None):
    roles = [ac["role"] for ac in agents_config]
    if not roles:
        raise ValueError("Sequential strategy requires at least one agent with a role set.")

    graph = StateGraph(OrchestratorState)
    model_map = {ac["role"]: ac.get("model") or None for ac in agents_config}

    for role in roles:
        executor = _get_executor(role, built_agents, exec_agents,
                                  agent_model=model_map.get(role))
        graph.add_node(role, executor)

    graph.add_edge(START, roles[0])
    for i in range(len(roles) - 1):
        graph.add_edge(roles[i], roles[i + 1])
    graph.add_edge(roles[-1], END)
    return graph.compile(checkpointer=checkpointer)


def _build_parallel_graph(agents_config, built_agents, checkpointer=None,
                          exec_agents=None):
    """Fan-out to all agents in parallel, then fan-in to a merge node before END.

    The merge node is a no-op pass-through: LangGraph's operator.add reducer on
    `messages` and `agent_trace` already concatenates every agent's output as each
    branch completes.  The explicit fan-in node guarantees the graph waits for ALL
    branches before terminating, rather than exiting on the first completion.
    """
    graph = StateGraph(OrchestratorState)

    async def _merge(state: OrchestratorState) -> OrchestratorState:
        return {}

    graph.add_node("_merge", _merge)
    graph.add_edge("_merge", END)

    for ac in agents_config:
        role = ac["role"]
        executor = _get_executor(role, built_agents, exec_agents,
                                  agent_model=ac.get("model") or None)
        graph.add_node(role, executor)
        graph.add_edge(START, role)
        graph.add_edge(role, "_merge")

    return graph.compile(checkpointer=checkpointer)


def _build_supervisor_graph(agents_config, built_agents, checkpointer=None,
                            exec_agents=None):
    router_llm = get_router_llm()
    valid_roles = {a["role"] for a in agents_config}

    agent_descs = "\n".join(f'- "{a["role"]}": {a["description"]}' for a in agents_config)
    agent_names = ", ".join(f'"{a["role"]}"' for a in agents_config)

    supervisor_prompt = f"""You are a supervisor agent. Decide which agent handles the task, or if it's complete.

Available agents:
{agent_descs}

Respond with ONLY "DONE" or an agent name ({agent_names})."""

    async def supervisor_decide(state: OrchestratorState) -> OrchestratorState:
        iterations = state.get("supervisor_iterations", 0)
        if iterations >= MAX_SUPERVISOR_ITERATIONS:
            logger.warning(
                "Supervisor reached iteration limit (%d); forcing DONE.", MAX_SUPERVISOR_ITERATIONS
            )
            return {
                "selected_agent": "__done__",
                "agent_trace": [{"step": "supervisor", "decision": "forced_done_iteration_limit"}],
                "supervisor_iterations": 1,
            }

        msgs = _ensure_messages(state["messages"])
        # Some models (e.g. Claude) reject requests where the conversation ends with
        # an AIMessage. After an agent turn, the last message is an AIMessage, so we
        # append a sentinel HumanMessage to keep the API happy.
        if msgs and isinstance(msgs[-1], AIMessage):
            msgs = [*msgs, HumanMessage(content="What should be done next? Reply DONE or with the next agent name.")]
        response = await router_llm.ainvoke([SystemMessage(content=supervisor_prompt), *msgs])
        raw = _extract_text(response.content).strip().lower().strip('"\'')
        if raw == "done":
            return {
                "selected_agent": "__done__",
                "agent_trace": [{"step": "supervisor", "decision": "done"}],
                "supervisor_iterations": 1,
            }
        if raw not in valid_roles:
            logger.warning(
                "Supervisor returned unrecognised role %r; treating as DONE. "
                "Check supervisor prompt and agent config.",
                raw,
            )
            return {
                "selected_agent": "__done__",
                "agent_trace": [{"step": "supervisor", "decision": "done_unrecognised_role",
                                  "raw": raw}],
                "supervisor_iterations": 1,
            }
        return {
            "selected_agent": raw,
            "agent_trace": [{"step": "supervisor", "decision": raw}],
            "supervisor_iterations": 1,
        }

    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", supervisor_decide)

    for ac in agents_config:
        role = ac["role"]
        executor = _get_executor(role, built_agents, exec_agents,
                                  agent_model=ac.get("model") or None)
        graph.add_node(role, executor)
        graph.add_edge(role, "supervisor")

    graph.add_edge(START, "supervisor")

    def supervisor_router(state: OrchestratorState) -> str:
        sel = state.get("selected_agent", "__done__")
        if sel == "__done__" or sel not in valid_roles:
            return "__end__"
        return sel

    graph.add_conditional_edges("supervisor", supervisor_router)
    return graph.compile(checkpointer=checkpointer)


async def build_orchestrator():
    from src.db.database import init_db, seed_defaults
    init_db()
    seed_defaults()
    return await build_orchestrator_from_team("default")


def get_graph_config(thread_id: str, callbacks=None) -> dict:
    """Build the LangGraph config dict with thread_id for checkpointing and optional callbacks."""
    config: dict = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        config["callbacks"] = callbacks
    return config
