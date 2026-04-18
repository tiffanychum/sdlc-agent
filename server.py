"""
FastAPI server for the SDLC Agent platform.

Provides REST APIs for:
- Team & Agent configuration (CRUD)
- Skills management
- Tool registry browsing
- Chat with configurable teams
- Tracing and observability
- Multi-LLM evaluation and comparison
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.db.database import init_db, seed_defaults, update_agent_data, get_session
from src.db.models import (
    Team, Agent, AgentToolMapping, Skill, AgentSkillMapping,
    Trace, Span, EvalRun, GoldenTestCase, RegressionResult, PromptVersion,
    RagConfig as RagConfigModel, RagSource, RagQuery,
)
from src.orchestrator import build_orchestrator_from_team, _extract_text, get_graph_config
from src.tracing.collector import TraceCollector, _flush_pending_spans, estimate_cost


orchestrators: dict = {}

# System / internal node names to exclude from agent_start/agent_end detection.
# Includes orchestrator system nodes AND inner create_react_agent node names.
_GRAPH_SYSTEM_NODES = frozenset({
    # Orchestrator system nodes
    "supervisor", "supervisor_router", "router", "route_request",
    "_merge", "_synthesize", "_seq_synthesize", "_sup_synthesize",
    "LangGraph", "__start__", "__end__", "START", "END",
    # LangGraph / LangChain inner nodes (from create_react_agent internals)
    "agent", "call_model", "should_continue", "tools",
    # LangChain runnable nodes
    "RunnableSequence", "RunnableLambda", "RunnableParallel",
    "Prompt", "ChatPromptTemplate", "BaseChatModel",
})


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed_defaults()
    update_agent_data()
    from src.evaluation.golden import sync_golden_to_db
    sync_golden_to_db()
    orchestrators["default"] = await build_orchestrator_from_team("default")
    # Seed the Performance Knowledge Base in the background (idempotent).
    # This ensures perf_search is ready without blocking server startup.
    import asyncio
    async def _seed_perf_kb():
        try:
            from src.rag.performance_kb import seed_performance_kb
            result = await seed_performance_kb()
            if not result.get("skipped"):
                import logging
                logging.getLogger(__name__).info(
                    "Performance KB seeded: %d chunks from %d documents.",
                    result.get("chunks_ingested", 0), result.get("documents", 0),
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Performance KB seed failed: %s", e)
    asyncio.create_task(_seed_perf_kb())
    yield


async def _build_fresh_orchestrator(team_id: str, user_message: str | None = None):
    """Build orchestrator with the team's CURRENT strategy from DB.

    If the team's strategy is 'auto' and a user message is provided, the
    meta-router resolves the concrete strategy before building.  Returns
    (graph, resolved_strategy_or_None).
    """
    from src.db.database import get_session
    from src.db.models import Team, Agent

    session = get_session()
    try:
        team_obj = session.query(Team).filter_by(id=team_id).first()
        team_strategy = (team_obj.decision_strategy if team_obj else None) or "router_decides"
        if team_strategy == "auto" and user_message:
            agents_db = session.query(Agent).filter_by(team_id=team_id).all()
            agents_cfg = [{"role": a.role, "description": a.description or ""} for a in agents_db]
    finally:
        session.close()

    resolved_strategy = None
    strategy_override = None

    if team_strategy == "auto" and user_message:
        from src.orchestrator import select_strategy_auto
        resolved_strategy = await select_strategy_auto(user_message, agents_cfg)
        strategy_override = resolved_strategy

    graph = await build_orchestrator_from_team(team_id, strategy_override=strategy_override)
    return graph, resolved_strategy


app = FastAPI(title="SDLC Agent Platform", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Pydantic Schemas ────────────────────────────────────────────

class TeamCreate(BaseModel):
    name: str
    description: str = ""
    decision_strategy: str = "router_decides"

class TeamUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    decision_strategy: Optional[str] = None

class AgentCreate(BaseModel):
    name: str
    role: str
    description: str = ""
    system_prompt: str = ""
    tool_groups: list[str] = []
    model: str = ""
    decision_strategy: str = "react"

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    tool_groups: Optional[list[str]] = None
    model: Optional[str] = None
    decision_strategy: Optional[str] = None
    prompt_version: Optional[str] = None

class SkillCreate(BaseModel):
    name: str
    description: str = ""
    instructions: str
    trigger_pattern: str = ""

class SkillUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    trigger_pattern: Optional[str] = None

class ChatRequest(BaseModel):
    message: str

class ChatStreamRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ChatResumeRequest(BaseModel):
    thread_id: str
    hitl_response: dict

class EvalRequest(BaseModel):
    team_id: str = "default"
    use_llm_judge: bool = True
    use_deepeval: bool = False

class EvalCompareRequest(BaseModel):
    team_id: str = "default"
    model_configs: list[dict]


class GoldenCaseCreate(BaseModel):
    id: str
    name: str
    prompt: str
    expected_agent: str = ""
    expected_tools: list[str] = []
    expected_output_keywords: list[str] = []
    expected_delegation_pattern: list[str] = []
    quality_thresholds: dict = {}
    max_llm_calls: int = 15
    max_tool_calls: int = 10
    max_tokens: int = 8000
    max_latency_ms: int = 120000
    complexity: str = "quick"
    version: str = "1.0"
    reference_output: str = ""


class GoldenCaseUpdate(BaseModel):
    name: Optional[str] = None
    prompt: Optional[str] = None
    expected_agent: Optional[str] = None
    expected_tools: Optional[list[str]] = None
    expected_output_keywords: Optional[list[str]] = None
    expected_delegation_pattern: Optional[list[str]] = None
    quality_thresholds: Optional[dict] = None
    max_llm_calls: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_tokens: Optional[int] = None
    max_latency_ms: Optional[int] = None
    complexity: Optional[str] = None
    version: Optional[str] = None
    reference_output: Optional[str] = None


class RegressionRunRequest(BaseModel):
    team_id: str = "default"
    case_ids: Optional[list[str]] = None
    model: Optional[str] = None
    prompt_version: str = "v1"
    prompt_versions_by_role: Optional[dict] = None  # e.g. {"coder": "v3", "planner": "v2"}
    baseline_run_id: Optional[str] = None


# RCARequest removed — RCA tab deleted, replaced by A/B Comparison


class PromptVersionCreate(BaseModel):
    version_label: str
    description: str = ""
    agent_prompts: dict = {}
    team_strategy: str = ""


class PromptVersionUpdate(BaseModel):
    version_label: Optional[str] = None
    description: Optional[str] = None
    agent_prompts: Optional[dict] = None
    team_strategy: Optional[str] = None


AVAILABLE_MODELS = [
    # ── OpenAI ──────────────────────────────────────────────────
    {"id": "gpt-4o-mini",             "name": "GPT-4o Mini",            "provider": "OpenAI",     "tier": "router", "thinking": False},
    {"id": "gpt-4o",                  "name": "GPT-4o",                 "provider": "OpenAI",     "tier": "agent",  "thinking": False},
    {"id": "gpt-5-mini",              "name": "GPT-5 Mini",             "provider": "OpenAI",     "tier": "agent",  "thinking": False},
    {"id": "gpt-5.3-codex",           "name": "GPT-5.3 Codex",         "provider": "OpenAI",     "tier": "agent",  "thinking": False},
    # ── Anthropic — use Poe model names (not Anthropic-native IDs) ──
    {"id": "claude-haiku-3",          "name": "Claude Haiku 3",        "provider": "Anthropic",  "tier": "agent",  "thinking": False},
    {"id": "Claude-Haiku-3.5",        "name": "Claude Haiku 3.5",      "provider": "Anthropic",  "tier": "agent",  "thinking": False},
    {"id": "Claude-Sonnet-4",         "name": "Claude Sonnet 4",       "provider": "Anthropic",  "tier": "agent",  "thinking": False},
    {"id": "claude-opus-4.5",         "name": "Claude Opus 4.5",       "provider": "Anthropic",  "tier": "agent",  "thinking": False},
    {"id": "Claude-Opus-4",           "name": "Claude Opus 4",         "provider": "Anthropic",  "tier": "agent",  "thinking": False},
    # ── Google — ⚠️ may trigger safety blocks ──────────────────
    {"id": "gemini-2.5-flash-lite",   "name": "Gemini 2.5 Flash Lite ⚠️", "provider": "Google", "tier": "agent",  "thinking": False, "unstable": True},
    {"id": "gemini-3-flash",          "name": "Gemini 3 Flash ⚠️",     "provider": "Google",     "tier": "agent",  "thinking": False, "unstable": True},
    # ── Other providers ─────────────────────────────────────────
    {"id": "llama-3.1-8b-cs",         "name": "Llama 3.1 8B",          "provider": "Meta",       "tier": "agent",  "thinking": False},
    {"id": "grok-4.1-fast-reasoning", "name": "Grok 4.1 Fast",         "provider": "xAI",        "tier": "agent",  "thinking": False},
    {"id": "Grok-3-Mini",             "name": "Grok 3 Mini",            "provider": "xAI",        "tier": "agent",  "thinking": False},
]


# ── Teams API ───────────────────────────────────────────────────

@app.get("/api/teams")
def list_teams():
    session = get_session()
    teams = session.query(Team).all()
    result = []
    for t in teams:
        agents = session.query(Agent).filter_by(team_id=t.id).all()
        result.append({
            "id": t.id, "name": t.name, "description": t.description,
            "decision_strategy": t.decision_strategy,
            "agents_count": len(agents),
            "created_at": t.created_at.isoformat() if t.created_at else None,
        })
    session.close()
    return result

@app.post("/api/teams")
def create_team(data: TeamCreate):
    session = get_session()
    team = Team(name=data.name, description=data.description, decision_strategy=data.decision_strategy)
    session.add(team)
    session.commit()
    tid = team.id
    session.close()
    return {"id": tid, "status": "created"}

@app.get("/api/teams/{team_id}")
def get_team(team_id: str):
    session = get_session()
    team = session.query(Team).filter_by(id=team_id).first()
    if not team:
        session.close()
        raise HTTPException(404, "Team not found")
    agents = session.query(Agent).filter_by(team_id=team_id).all()
    agents_data = []
    for a in agents:
        tools = [m.tool_group for m in session.query(AgentToolMapping).filter_by(agent_id=a.id).all()]
        skills = [
            {"id": sm.skill.id, "name": sm.skill.name}
            for sm in session.query(AgentSkillMapping).filter_by(agent_id=a.id).all()
        ]
        agents_data.append({
            "id": a.id, "name": a.name, "role": a.role,
            "description": a.description, "system_prompt": a.system_prompt,
            "tool_groups": tools, "skills": skills,
            "model": getattr(a, "model", "") or "",
            "decision_strategy": getattr(a, "decision_strategy", "react") or "react",
        })
    result = {
        "id": team.id, "name": team.name, "description": team.description,
        "decision_strategy": team.decision_strategy, "agents": agents_data,
    }
    session.close()
    return result

@app.put("/api/teams/{team_id}")
def update_team(team_id: str, data: TeamUpdate):
    session = get_session()
    team = session.query(Team).filter_by(id=team_id).first()
    if not team:
        session.close()
        raise HTTPException(404, "Team not found")
    if data.name is not None: team.name = data.name
    if data.description is not None: team.description = data.description
    if data.decision_strategy is not None: team.decision_strategy = data.decision_strategy
    session.commit()
    session.close()
    return {"status": "updated"}

@app.delete("/api/teams/{team_id}")
def delete_team(team_id: str):
    session = get_session()
    team = session.query(Team).filter_by(id=team_id).first()
    if not team:
        session.close()
        raise HTTPException(404, "Team not found")
    session.delete(team)
    session.commit()
    session.close()
    return {"status": "deleted"}


# ── Agents API ──────────────────────────────────────────────────

@app.post("/api/teams/{team_id}/agents")
def create_agent(team_id: str, data: AgentCreate):
    session = get_session()
    if not session.query(Team).filter_by(id=team_id).first():
        session.close()
        raise HTTPException(404, "Team not found")
    agent = Agent(team_id=team_id, name=data.name, role=data.role,
                  description=data.description, system_prompt=data.system_prompt,
                  model=data.model, decision_strategy=data.decision_strategy)
    session.add(agent)
    session.flush()
    for tg in data.tool_groups:
        session.add(AgentToolMapping(agent_id=agent.id, tool_group=tg))
    session.commit()
    aid = agent.id
    session.close()
    return {"id": aid, "status": "created"}

@app.put("/api/agents/{agent_id}")
def update_agent(agent_id: str, data: AgentUpdate):
    session = get_session()
    agent = session.query(Agent).filter_by(id=agent_id).first()
    if not agent:
        session.close()
        raise HTTPException(404, "Agent not found")
    if data.name is not None: agent.name = data.name
    if data.role is not None: agent.role = data.role
    if data.description is not None: agent.description = data.description
    if data.system_prompt is not None: agent.system_prompt = data.system_prompt
    if data.model is not None: agent.model = data.model
    if data.decision_strategy is not None: agent.decision_strategy = data.decision_strategy
    if data.prompt_version is not None:
        try:
            agent.prompt_version = data.prompt_version
        except Exception:
            pass  # column may not exist in old schema
    if data.tool_groups is not None:
        session.query(AgentToolMapping).filter_by(agent_id=agent_id).delete()
        for tg in data.tool_groups:
            session.add(AgentToolMapping(agent_id=agent_id, tool_group=tg))
    session.commit()
    session.close()
    return {"status": "updated"}

@app.delete("/api/agents/{agent_id}")
def delete_agent(agent_id: str):
    session = get_session()
    agent = session.query(Agent).filter_by(id=agent_id).first()
    if not agent:
        session.close()
        raise HTTPException(404, "Agent not found")
    session.delete(agent)
    session.commit()
    session.close()
    return {"status": "deleted"}


# ── Skills API ──────────────────────────────────────────────────

@app.get("/api/skills")
def list_skills():
    session = get_session()
    skills = session.query(Skill).all()
    result = [{"id": s.id, "name": s.name, "description": s.description,
               "instructions": s.instructions, "trigger_pattern": s.trigger_pattern} for s in skills]
    session.close()
    return result

@app.post("/api/skills")
def create_skill(data: SkillCreate):
    session = get_session()
    skill = Skill(name=data.name, description=data.description,
                  instructions=data.instructions, trigger_pattern=data.trigger_pattern)
    session.add(skill)
    session.commit()
    sid = skill.id
    session.close()
    return {"id": sid, "status": "created"}

@app.put("/api/skills/{skill_id}")
def update_skill(skill_id: str, data: SkillUpdate):
    session = get_session()
    skill = session.query(Skill).filter_by(id=skill_id).first()
    if not skill:
        session.close()
        raise HTTPException(404, "Skill not found")
    if data.name is not None: skill.name = data.name
    if data.description is not None: skill.description = data.description
    if data.instructions is not None: skill.instructions = data.instructions
    if data.trigger_pattern is not None: skill.trigger_pattern = data.trigger_pattern
    session.commit()
    session.close()
    return {"status": "updated"}

@app.delete("/api/skills/{skill_id}")
def delete_skill(skill_id: str):
    session = get_session()
    skill = session.query(Skill).filter_by(id=skill_id).first()
    if not skill:
        session.close()
        raise HTTPException(404, "Skill not found")
    session.delete(skill)
    session.commit()
    session.close()
    return {"status": "deleted"}

@app.put("/api/agents/{agent_id}/skills")
def assign_skills(agent_id: str, skill_ids: list[str]):
    session = get_session()
    session.query(AgentSkillMapping).filter_by(agent_id=agent_id).delete()
    for sid in skill_ids:
        session.add(AgentSkillMapping(agent_id=agent_id, skill_id=sid))
    session.commit()
    session.close()
    return {"status": "updated"}


# ── Tools API ───────────────────────────────────────────────────

@app.get("/api/tools")
async def list_tools():
    from src.tools.registry import get_all_tools
    tool_map = await get_all_tools()
    result = {}
    for group, tools in tool_map.items():
        result[group] = [{"name": t.name, "description": t.description} for t in tools]
    return result


# ── Chat API ────────────────────────────────────────────────────

@app.post("/api/teams/{team_id}/chat")
async def chat(team_id: str, request: ChatRequest):
    if team_id not in orchestrators:
        orchestrators[team_id] = await build_orchestrator_from_team(team_id)

    collector = TraceCollector(team_id=team_id, user_prompt=request.message)
    from src.tracing.callbacks import TracingCallbackHandler
    tracing_cb = TracingCallbackHandler(collector)
    routing_span = collector.start_span("routing", "routing", input_data={"prompt": request.message[:200]})

    from langgraph.types import Command as LGCommand

    legacy_thread = uuid.uuid4().hex[:12]
    graph = orchestrators[team_id]
    lg_config = {"configurable": {"thread_id": legacy_thread}, "callbacks": [tracing_cb]}

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": request.message}],
         "selected_agent": "", "agent_trace": []},
        config=lg_config,
    )

    for _ in range(10):
        state = await graph.aget_state(lg_config)
        if not state.next:
            break
        result = await graph.ainvoke(
            LGCommand(resume={"approved": True, "action": "continue", "answer": "proceed"}),
            config=lg_config,
        )
    else:
        result = (await graph.aget_state(lg_config)).values

    agent_used = result.get("selected_agent", "unknown")
    collector.end_span(routing_span, output_data={"selected_agent": agent_used})

    # Extract model/token metadata from LangChain response messages
    llm_model = ""
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for msg in result.get("messages", []):
        meta = getattr(msg, "response_metadata", None) or {}
        usage = meta.get("token_usage") or meta.get("usage", {})
        if usage:
            total_prompt_tokens += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        if not llm_model and meta.get("model_name"):
            llm_model = meta["model_name"]
        um = getattr(msg, "usage_metadata", None)
        if um:
            total_prompt_tokens += getattr(um, "input_tokens", 0) or 0
            total_completion_tokens += getattr(um, "output_tokens", 0) or 0

    agent_trace = result.get("agent_trace", [])
    tool_calls = []
    all_agents_used = []
    for entry in agent_trace:
        if entry.get("step") == "execution":
            agent_name = entry.get("agent", "unknown")
            if agent_name not in all_agents_used:
                all_agents_used.append(agent_name)

            # Use per-agent timing from the orchestrator when available, otherwise
            # fall back to a synthetic span (start≈end) that at least names the agent.
            entry_latency_ms = entry.get("latency_ms")
            entry_tokens_in = entry.get("tokens_in") or 0
            entry_tokens_out = entry.get("tokens_out") or 0
            entry_model = entry.get("model") or llm_model or ""

            agent_span = collector.start_span(f"agent:{agent_name}", "agent_execution",
                                              input_data={"agent": agent_name})
            for tc in entry.get("tool_calls", []):
                tool_calls.append({**tc, "agent": agent_name})
                span_id = collector.start_span(f"tool:{tc['tool']}", "tool_call",
                                               input_data={"args": str(tc.get("args", {}))[:200], "agent": agent_name})
                collector.end_span(span_id, output_data={"result": "completed"})

            # Patch the span's start_time so it reflects actual agent latency.
            # This makes per-agent latency percentiles accurate in TraceAnalyzer.
            if entry_latency_ms and entry_latency_ms > 0:
                from datetime import datetime as _dt, timedelta as _td
                _end_t = _dt.utcnow()
                _start_t = _end_t - _td(milliseconds=entry_latency_ms)
                if agent_span in collector._span_data:
                    collector._span_data[agent_span]["start_time"] = _start_t

            collector.end_span(agent_span,
                               output_data={"tool_calls": len(entry.get("tool_calls", []))},
                               model=entry_model,
                               tokens_in=entry_tokens_in or total_prompt_tokens,
                               tokens_out=entry_tokens_out or total_completion_tokens)
        elif entry.get("step") == "routing":
            pass  # already captured above
        elif entry.get("step") == "supervisor":
            sup_span = collector.start_span("supervisor_decision", "supervisor",
                                            input_data={"decision": entry.get("decision", "")})
            collector.end_span(sup_span, output_data={"decision": entry.get("decision", "")})

    agents_label = " > ".join(all_agents_used) if all_agents_used else agent_used

    last_msg = result["messages"][-1]
    response = _extract_text(last_msg.content if hasattr(last_msg, "content") else str(last_msg))

    tool_outputs = []
    for msg in result.get("messages", []):
        if hasattr(msg, "content") and hasattr(msg, "type") and msg.type == "tool":
            tool_outputs.append(str(msg.content)[:300])

    collector.save()

    # Always persist agent_used and agent_response — this must not be gated on eval success
    _session = get_session()
    try:
        _tr = _session.query(Trace).filter_by(id=collector.trace_id).first()
        if _tr:
            _tr.agent_used = agents_label
            _tr.agent_response = response[:2000]
            _tr.tool_calls_json = agent_trace
            _session.commit()
    except Exception:
        pass
    finally:
        _session.close()

    # Save quick rule-based eval scores (best-effort — does not block agent_used)
    try:
        from src.evaluation.metrics import TaskMetric, ToolCallMetric
        quick_task = TaskMetric(
            task_id=collector.trace_id, scenario_name="chat",
            prompt=request.message, expected_agent=agent_used,
            actual_agent=agent_used, completed=True,
            expected_tools=[tc.get("tool", "") for tc in tool_calls],
            final_response=response, tool_outputs=tool_outputs,
        )
        for tc in tool_calls:
            quick_task.tool_calls.append(ToolCallMetric(
                tool_name=tc.get("tool", ""), arguments=tc.get("args", {}),
                was_correct=True,
            ))
        quick_scores = {
            "tool_accuracy": round(quick_task.tool_call_accuracy, 3),
            "step_efficiency": round(quick_task.step_efficiency, 3),
            "faithfulness": round(quick_task.hallucination_score, 3),
            "safety": round(quick_task.safety_score, 3),
            "reasoning_quality": round(quick_task.reasoning_quality, 3),
        }

        session = get_session()
        tr = session.query(Trace).filter_by(id=collector.trace_id).first()
        if tr:
            tr.eval_scores = quick_scores
            tr.eval_status = "quick"
            session.commit()
        session.close()
    except Exception:
        pass

    try:
        from src.evaluation.integrations import export_trace_to_langfuse
        export_trace_to_langfuse(
            trace_id=collector.trace_id, user_prompt=request.message,
            agent_used=agent_used, tool_calls=tool_calls,
            response=response, latency_ms=collector.to_dict()["total_latency_ms"],
        )
    except Exception:
        pass

    return {
        "response": response, "agent_used": agents_label,
        "tool_calls": tool_calls, "agent_trace": agent_trace,
        "trace": collector.to_dict(),
    }

@app.post("/api/teams/{team_id}/rebuild")
async def rebuild_team(team_id: str):
    """Rebuild orchestrator after config changes."""
    orchestrators[team_id] = await build_orchestrator_from_team(team_id)
    return {"status": "rebuilt"}


# ── SSE Streaming Chat ──────────────────────────────────────────

def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/api/teams/{team_id}/chat/stream")
async def chat_stream(team_id: str, request: ChatStreamRequest):
    """SSE streaming chat endpoint with HITL support.

    Streams real-time events: agent_start, tool_start, tool_end, trace_span,
    hitl_request, response, done. Supports LangGraph interrupt/resume via thread_id.

    On every NEW conversation (no thread_id supplied) the orchestrator is rebuilt
    from the DB so Studio changes take effect without a server restart.
    For 'auto' strategy teams, the meta-router resolves the concrete strategy
    using the first message and emits a 'strategy_selected' SSE event.
    """
    is_new_thread = not request.thread_id

    resolved_strategy = None
    if is_new_thread:
        # Always rebuild for new conversations — picks up latest Studio config.
        graph, resolved_strategy = await _build_fresh_orchestrator(team_id, request.message)
        orchestrators[team_id] = graph
    elif team_id not in orchestrators:
        graph, _ = await _build_fresh_orchestrator(team_id)
        orchestrators[team_id] = graph

    graph = orchestrators[team_id]
    thread_id = request.thread_id or uuid.uuid4().hex[:12]

    span_queue: asyncio.Queue = asyncio.Queue()

    def on_span_event(event_type: str, span_data: dict):
        span_queue.put_nowait({"event": event_type, "span": span_data})

    collector = TraceCollector(team_id=team_id, user_prompt=request.message,
                               on_span_event=on_span_event)
    from src.tracing.callbacks import TracingCallbackHandler
    tracing_cb = TracingCallbackHandler(collector)
    routing_span = collector.start_span("routing", "routing",
                                        input_data={"prompt": request.message[:200]})

    config = get_graph_config(thread_id, callbacks=[tracing_cb])
    initial_input = {
        "messages": [{"role": "user", "content": request.message}],
        "selected_agent": "", "agent_trace": [],
    }

    async def event_generator():
        yield _sse_event("thread_id", {"thread_id": thread_id})

        # Emit resolved strategy immediately so the frontend can show it.
        if resolved_strategy:
            yield _sse_event("strategy_selected", {
                "strategy": resolved_strategy, "from": "auto",
            })

        current_agent = ""
        try:
            async for event in graph.astream_events(initial_input, config=config, version="v2"):
                kind = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data", {})

                # Detect agent node start.  Use langgraph_node from metadata which
                # always reflects the *actual graph node name*, avoiding noise from
                # inner chain/tool events fired within the same node.
                meta = event.get("metadata", {})
                lg_node = meta.get("langgraph_node", "")
                if kind == "on_chain_start" and lg_node and lg_node not in _GRAPH_SYSTEM_NODES and not lg_node.startswith("_") and name == lg_node:
                    # name == lg_node ensures we only fire once for the node entry point,
                    # not for every inner chain invocation within that node.
                    current_agent = lg_node
                    yield _sse_event("agent_start", {"agent": lg_node})

                elif kind == "on_tool_start":
                    tool_input = data.get("input", {})
                    if isinstance(tool_input, dict):
                        safe_input = {k: str(v)[:200] for k, v in tool_input.items()}
                    else:
                        safe_input = {"input": str(tool_input)[:200]}
                    yield _sse_event("tool_start", {
                        "agent": current_agent, "tool": name,
                        "args": safe_input,
                    })

                elif kind == "on_tool_end":
                    output = data.get("output", "")
                    yield _sse_event("tool_end", {
                        "agent": current_agent, "tool": name,
                        "output_preview": str(output)[:300],
                    })

                elif kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk:
                        content = getattr(chunk, "content", "")
                        if content and isinstance(content, str):
                            yield _sse_event("llm_token", {
                                "agent": current_agent, "token": content,
                            })

                elif kind == "on_chain_end" and lg_node and lg_node not in _GRAPH_SYSTEM_NODES and not lg_node.startswith("_") and name == lg_node:
                    output = data.get("output", {})
                    ended_agent = (
                        (output.get("selected_agent") if isinstance(output, dict) else None)
                        or current_agent
                    )
                    if ended_agent and ended_agent != "__done__":
                        yield _sse_event("agent_end", {"agent": ended_agent})

                while not span_queue.empty():
                    sq = span_queue.get_nowait()
                    yield _sse_event("trace_span", sq)

        except Exception as e:
            yield _sse_event("error", {"message": str(e)[:500]})
            while not span_queue.empty():
                sq = span_queue.get_nowait()
                yield _sse_event("trace_span", sq)
            collector.save()
            yield _sse_event("done", {"thread_id": thread_id, "partial": True})
            return

        collector.end_span(routing_span, output_data={"selected_agent": current_agent})

        while not span_queue.empty():
            sq = span_queue.get_nowait()
            yield _sse_event("trace_span", sq)

        state = await graph.aget_state(config)
        if state.next:
            interrupt_value = {}
            if state.tasks:
                for task in state.tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        interrupt_value = task.interrupts[0].value
                        break
            interrupt_value["thread_id"] = thread_id
            yield _sse_event("hitl_request", interrupt_value)
        else:
            full_state = state.values
            agent_used = full_state.get("selected_agent", "unknown")
            agent_trace = full_state.get("agent_trace", [])
            messages = full_state.get("messages", [])

            all_agents = []
            tool_calls = []
            for entry in agent_trace:
                if entry.get("step") == "execution":
                    an = entry.get("agent", "unknown")
                    if an not in all_agents:
                        all_agents.append(an)
                    for tc in entry.get("tool_calls", []):
                        tool_calls.append({**tc, "agent": an})

            response_text = ""
            if messages:
                last = messages[-1]
                response_text = _extract_text(
                    last.content if hasattr(last, "content") else str(last)
                )

            collector.save()

            yield _sse_event("response", {
                "content": response_text,
                "agent_used": " > ".join(all_agents) if all_agents else agent_used,
                "tool_calls": tool_calls,
                "agent_trace": agent_trace,
                "trace": collector.to_dict(),
            })
            yield _sse_event("done", {})

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/teams/{team_id}/chat/resume")
async def chat_resume(team_id: str, request: ChatResumeRequest):
    """Resume an interrupted chat session after HITL response.

    The graph is resumed from the checkpointed state using the same thread_id.
    Returns an SSE stream for the remainder of execution.
    """
    if team_id not in orchestrators:
        orchestrators[team_id] = await build_orchestrator_from_team(team_id)

    graph = orchestrators[team_id]
    thread_id = request.thread_id

    from langgraph.types import Command

    span_queue: asyncio.Queue = asyncio.Queue()

    def on_span_event(event_type: str, span_data: dict):
        span_queue.put_nowait({"event": event_type, "span": span_data})

    collector = TraceCollector(team_id=team_id, user_prompt="[HITL resume]",
                               on_span_event=on_span_event)
    from src.tracing.callbacks import TracingCallbackHandler
    tracing_cb = TracingCallbackHandler(collector)
    config = get_graph_config(thread_id, callbacks=[tracing_cb])

    async def event_generator():
        yield _sse_event("resumed", {"thread_id": thread_id})

        current_agent = ""
        try:
            async for event in graph.astream_events(
                Command(resume=request.hitl_response), config=config, version="v2"
            ):
                kind = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data", {})

                meta = event.get("metadata", {})
                lg_node = meta.get("langgraph_node", "")
                if kind == "on_chain_start" and lg_node and lg_node not in _GRAPH_SYSTEM_NODES and not lg_node.startswith("_") and name == lg_node:
                    current_agent = lg_node
                    yield _sse_event("agent_start", {"agent": lg_node})
                elif kind == "on_tool_start":
                    tool_input = data.get("input", {})
                    safe_input = {k: str(v)[:200] for k, v in tool_input.items()} if isinstance(tool_input, dict) else {"input": str(tool_input)[:200]}
                    yield _sse_event("tool_start", {"agent": current_agent, "tool": name, "args": safe_input})
                elif kind == "on_tool_end":
                    yield _sse_event("tool_end", {"agent": current_agent, "tool": name, "output_preview": str(data.get("output", ""))[:300]})
                elif kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk:
                        content = getattr(chunk, "content", "")
                        if content and isinstance(content, str):
                            yield _sse_event("llm_token", {"agent": current_agent, "token": content})
                elif kind == "on_chain_end" and lg_node and lg_node not in _GRAPH_SYSTEM_NODES and not lg_node.startswith("_") and name == lg_node:
                    output = data.get("output", {})
                    ended_agent = (output.get("selected_agent") if isinstance(output, dict) else None) or current_agent
                    if ended_agent and ended_agent != "__done__":
                        yield _sse_event("agent_end", {"agent": ended_agent})

                while not span_queue.empty():
                    sq = span_queue.get_nowait()
                    yield _sse_event("trace_span", sq)

        except Exception as e:
            yield _sse_event("error", {"message": str(e)[:500]})
            while not span_queue.empty():
                sq = span_queue.get_nowait()
                yield _sse_event("trace_span", sq)
            collector.save()
            yield _sse_event("done", {"thread_id": thread_id, "partial": True})
            return

        while not span_queue.empty():
            sq = span_queue.get_nowait()
            yield _sse_event("trace_span", sq)

        state = await graph.aget_state(config)
        if state.next:
            interrupt_value = {}
            if state.tasks:
                for task in state.tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        interrupt_value = task.interrupts[0].value
                        break
            interrupt_value["thread_id"] = thread_id
            yield _sse_event("hitl_request", interrupt_value)
        else:
            full_state = state.values
            agent_used = full_state.get("selected_agent", "unknown")
            agent_trace = full_state.get("agent_trace", [])
            messages = full_state.get("messages", [])

            all_agents = []
            tool_calls = []
            for entry in agent_trace:
                if entry.get("step") == "execution":
                    an = entry.get("agent", "unknown")
                    if an not in all_agents:
                        all_agents.append(an)
                    for tc in entry.get("tool_calls", []):
                        tool_calls.append({**tc, "agent": an})

            response_text = ""
            if messages:
                last = messages[-1]
                response_text = _extract_text(
                    last.content if hasattr(last, "content") else str(last)
                )

            collector.save()

            yield _sse_event("response", {
                "content": response_text,
                "agent_used": " > ".join(all_agents) if all_agents else agent_used,
                "tool_calls": tool_calls,
                "agent_trace": agent_trace,
                "trace": collector.to_dict(),
            })
            yield _sse_event("done", {})

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Traces API ──────────────────────────────────────────────────

@app.get("/api/traces")
def list_traces(limit: int = 50, offset: int = 0):
    session = get_session()
    traces = session.query(Trace).order_by(Trace.created_at.desc()).offset(offset).limit(limit).all()
    result = []
    for t in traces:
        spans = session.query(Span).filter_by(trace_id=t.id).order_by(Span.start_time).all()
        spans_data = [{
            "id": s.id, "name": s.name, "span_type": s.span_type,
            "model": s.model or "",
            "input_data": s.input_data or {}, "output_data": s.output_data or {},
            "tokens_in": s.tokens_in or 0, "tokens_out": s.tokens_out or 0,
            "cost": round(s.cost or 0, 6), "status": s.status or "completed",
            "start_time": s.start_time.isoformat() if s.start_time else None,
            "end_time": s.end_time.isoformat() if s.end_time else None,
        } for s in spans]
        agent_trace = getattr(t, "tool_calls_json", []) or []
        result.append({
            "id": t.id, "team_id": t.team_id, "user_prompt": t.user_prompt,
            "agent_used": getattr(t, "agent_used", "") or "",
            "agent_response": (getattr(t, "agent_response", "") or ""),
            "agent_trace": agent_trace,
            "tool_calls": agent_trace,
            "spans": spans_data,
            "total_latency_ms": round(t.total_latency_ms, 1),
            "total_tokens": t.total_tokens, "total_cost": round(t.total_cost, 6),
            "status": t.status,
            "eval_scores": getattr(t, "eval_scores", {}) or {},
            "eval_status": getattr(t, "eval_status", "pending") or "pending",
            "created_at": t.created_at.isoformat() if t.created_at else None,
        })
    session.close()
    return result

@app.get("/api/traces/stats")
def trace_stats(days: int = 30, team_id: str = None):
    session = get_session()
    cutoff = datetime.utcnow() - timedelta(days=days)
    q = session.query(Trace).filter(Trace.created_at >= cutoff)
    if team_id:
        q = q.filter(Trace.team_id == team_id)
    traces = q.all()

    if not traces:
        session.close()
        return {"total_runs": 0, "failures": 0, "avg_latency_ms": 0,
                "total_cost": 0, "total_tokens": 0, "success_rate": 0,
                "p50_latency_ms": 0, "p95_latency_ms": 0, "p99_latency_ms": 0,
                "avg_tokens": 0, "daily": {}}

    total = len(traces)
    failures = sum(1 for t in traces if t.status == "error")
    latencies = sorted([t.total_latency_ms for t in traces])
    costs = [t.total_cost for t in traces]
    tokens = [t.total_tokens for t in traces]

    def percentile(data, p):
        if not data: return 0
        k = (len(data) - 1) * p / 100
        f, c = int(k), int(k) + 1
        if c >= len(data): return data[-1]
        return data[f] + (k - f) * (data[c] - data[f])

    daily = {}
    for t in traces:
        day = t.created_at.strftime("%Y-%m-%d") if t.created_at else "unknown"
        daily.setdefault(day, {"runs": 0, "cost": 0, "tokens": 0, "latency": []})
        daily[day]["runs"] += 1
        daily[day]["cost"] += t.total_cost
        daily[day]["tokens"] += t.total_tokens
        daily[day]["latency"].append(t.total_latency_ms)

    for d in daily.values():
        d["avg_latency"] = round(sum(d["latency"]) / len(d["latency"]), 1) if d["latency"] else 0
        del d["latency"]

    result = {
        "total_runs": total, "failures": failures,
        "success_rate": round((total - failures) / total, 3) if total else 0,
        "avg_latency_ms": round(sum(latencies) / total, 1),
        "p50_latency_ms": round(percentile(latencies, 50), 1),
        "p95_latency_ms": round(percentile(latencies, 95), 1),
        "p99_latency_ms": round(percentile(latencies, 99), 1),
        "total_cost": round(sum(costs), 4),
        "total_tokens": sum(tokens),
        "avg_tokens": round(sum(tokens) / total),
        "daily": daily,
    }
    session.close()
    return result


# ── Prompt Registry API ─────────────────────────────────────────

@app.get("/api/prompts/versions")
async def prompt_versions(role: str = None):
    """List all prompt versions. Optionally filter by role."""
    try:
        from src.prompts.registry import get_registry
        reg = get_registry()
        if role:
            return {"role": role, "versions": reg.list_versions(role)}
        roles = reg.list_all_roles()
        return {
            "roles": {r: reg.list_versions(r) for r in roles}
        }
    except Exception as e:
        return {"error": str(e), "roles": {}}


@app.get("/api/prompts/text")
async def prompt_text(role: str, version: str = "latest"):
    """Return the full prompt text for a given role + version."""
    try:
        from src.prompts.registry import get_registry
        reg = get_registry()
        text_val = reg.get_prompt(role, version)
        if text_val is None:
            return {"error": f"No prompt found for role={role} version={version}", "text": ""}
        return {"role": role, "version": version, "text": text_val}
    except Exception as e:
        return {"error": str(e), "text": ""}


@app.get("/api/prompts/ab-compare")
async def prompt_ab_compare(role: str, version_a: str = "v1", version_b: str = "v2"):
    """
    Compare two prompt versions across ALL golden tests using stored regression results.
    Returns per-test pass/fail, metric scores, cost, and latency for both versions,
    plus aggregate metric deltas and a recommendation.
    """
    try:
        from src.db.database import get_session
        from sqlalchemy import text
        import json as _json

        session = get_session()
        try:
            def fetch_results(version: str) -> dict:
                rows = session.execute(text("""
                    SELECT golden_case_id, golden_case_name, overall_pass,
                           quality_scores, deepeval_scores,
                           actual_latency_ms, actual_cost, actual_tool_calls,
                           actual_agent
                    FROM regression_results
                    WHERE actual_agent LIKE :role
                      AND prompt_version = :version
                    ORDER BY created_at DESC
                """), {"role": f"%{role}%", "version": version}).fetchall()

                by_test: dict[str, dict] = {}
                for r in rows:
                    gid = r[0]
                    if gid in by_test:
                        continue  # keep most recent
                    try:
                        qs = _json.loads(r[3]) if isinstance(r[3], str) else (r[3] or {})
                    except Exception:
                        qs = {}
                    try:
                        ds = _json.loads(r[4]) if isinstance(r[4], str) else (r[4] or {})
                    except Exception:
                        ds = {}
                    by_test[gid] = {
                        "name": r[1] or gid,
                        "pass": bool(r[2]),
                        "quality_scores": qs,
                        "deepeval_scores": ds,
                        "all_scores": {**qs, **ds},
                        "latency_ms": r[5] or 0,
                        "cost": r[6] or 0.0,
                        "tool_calls": r[7] or 0,
                        "agent": r[8] or "",
                    }
                return by_test

            data_a = fetch_results(version_a)
            data_b = fetch_results(version_b)
        finally:
            session.close()

        # All test IDs in either version
        all_ids = sorted(set(data_a.keys()) | set(data_b.keys()))
        if not all_ids:
            return {"error": f"No regression data for role={role} with versions {version_a} and {version_b}"}

        # Per-test comparison
        test_rows = []
        for gid in all_ids:
            ra = data_a.get(gid)
            rb = data_b.get(gid)
            row: dict = {"golden_case_id": gid, "name": (ra or rb or {}).get("name", gid)}
            row["a_pass"] = ra["pass"] if ra else None
            row["b_pass"] = rb["pass"] if rb else None
            row["a_latency_ms"] = ra["latency_ms"] if ra else None
            row["b_latency_ms"] = rb["latency_ms"] if rb else None
            row["a_cost"] = ra["cost"] if ra else None
            row["b_cost"] = rb["cost"] if rb else None
            row["a_tool_calls"] = ra["tool_calls"] if ra else None
            row["b_tool_calls"] = rb["tool_calls"] if rb else None
            # Per-metric scores for this test
            all_metric_keys = set()
            if ra:
                all_metric_keys.update(ra["all_scores"].keys())
            if rb:
                all_metric_keys.update(rb["all_scores"].keys())
            row["metrics"] = {}
            for m in all_metric_keys:
                row["metrics"][m] = {
                    "a": ra["all_scores"].get(m) if ra else None,
                    "b": rb["all_scores"].get(m) if rb else None,
                }
            test_rows.append(row)

        # Aggregate metric deltas across all shared tests
        shared = [gid for gid in all_ids if gid in data_a and gid in data_b]
        metric_agg: dict[str, dict] = {}
        for gid in shared:
            all_m = set(data_a[gid]["all_scores"].keys()) | set(data_b[gid]["all_scores"].keys())
            for m in all_m:
                va = data_a[gid]["all_scores"].get(m)
                vb = data_b[gid]["all_scores"].get(m)
                if va is not None and vb is not None:
                    try:
                        va_f, vb_f = float(va), float(vb)
                    except (TypeError, ValueError):
                        continue
                    entry = metric_agg.setdefault(m, {"a_vals": [], "b_vals": []})
                    entry["a_vals"].append(va_f)
                    entry["b_vals"].append(vb_f)

        metric_summary: dict[str, dict] = {}
        for m, vals in metric_agg.items():
            a_avg = sum(vals["a_vals"]) / len(vals["a_vals"])
            b_avg = sum(vals["b_vals"]) / len(vals["b_vals"])
            delta = b_avg - a_avg
            metric_summary[m] = {
                "a_avg": round(a_avg, 3),
                "b_avg": round(b_avg, 3),
                "delta": round(delta, 3),
                "delta_pct": round(delta / a_avg * 100, 1) if a_avg else 0,
                "improved": delta > 0.005,
                "regressed": delta < -0.005,
            }

        # Pass rate comparison
        a_pass = sum(1 for d in data_a.values() if d["pass"]) / len(data_a) if data_a else 0
        b_pass = sum(1 for d in data_b.values() if d["pass"]) / len(data_b) if data_b else 0
        a_avg_cost = sum(d["cost"] for d in data_a.values()) / len(data_a) if data_a else 0
        b_avg_cost = sum(d["cost"] for d in data_b.values()) / len(data_b) if data_b else 0
        a_avg_lat = sum(d["latency_ms"] for d in data_a.values()) / len(data_a) if data_a else 0
        b_avg_lat = sum(d["latency_ms"] for d in data_b.values()) / len(data_b) if data_b else 0

        # Recommendation
        improved_metrics = sum(1 for v in metric_summary.values() if v["improved"])
        regressed_metrics = sum(1 for v in metric_summary.values() if v["regressed"])
        pass_delta_pp = (b_pass - a_pass) * 100
        if pass_delta_pp >= 5 or (improved_metrics > regressed_metrics and pass_delta_pp >= 0):
            rec = f"✓ {version_b} is better: pass rate {a_pass*100:.1f}% → {b_pass*100:.1f}% (+{pass_delta_pp:.1f}pp), {improved_metrics} metrics improved"
        elif pass_delta_pp <= -5:
            rec = f"✗ {version_b} is worse: pass rate dropped {pass_delta_pp:.1f}pp. Revert to {version_a}."
        else:
            rec = f"≈ Inconclusive: pass rate Δ{pass_delta_pp:+.1f}pp, {improved_metrics} better / {regressed_metrics} worse metrics. Manual review suggested."

        return {
            "role": role,
            "version_a": version_a,
            "version_b": version_b,
            "summary": {
                "a_tests": len(data_a),
                "b_tests": len(data_b),
                "shared_tests": len(shared),
                "a_pass_rate": round(a_pass, 3),
                "b_pass_rate": round(b_pass, 3),
                "pass_rate_delta_pp": round(pass_delta_pp, 1),
                "a_avg_cost": round(a_avg_cost, 6),
                "b_avg_cost": round(b_avg_cost, 6),
                "a_avg_latency_ms": round(a_avg_lat, 0),
                "b_avg_latency_ms": round(b_avg_lat, 0),
                "improved_metrics": improved_metrics,
                "regressed_metrics": regressed_metrics,
            },
            "recommendation": rec,
            "metrics": metric_summary,
            "tests": test_rows,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/prompts/diff")
async def prompt_diff(role: str, version_old: str = "v1", version_new: str = "v2"):
    """Return a unified diff between two prompt versions for a role."""
    try:
        import difflib
        from src.prompts.registry import get_registry
        reg = get_registry()
        old_p = reg.get_prompt(role, version_old) or ""
        new_p = reg.get_prompt(role, version_new) or ""
        if not old_p:
            return {"error": f"Version {version_old} not found for role={role}"}
        if not new_p:
            return {"error": f"Version {version_new} not found for role={role}"}
        diff = list(difflib.unified_diff(
            old_p.splitlines(keepends=True),
            new_p.splitlines(keepends=True),
            fromfile=f"{role}/{version_old}",
            tofile=f"{role}/{version_new}",
            n=3,
        ))
        return {
            "role": role,
            "version_old": version_old,
            "version_new": version_new,
            "diff": "".join(diff),
            "lines_added": sum(1 for l in diff if l.startswith("+")),
            "lines_removed": sum(1 for l in diff if l.startswith("-")),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/prompts/optimize")
async def prompt_optimize(request: Request):
    """
    Non-streaming optimization endpoint (kept for backward compat).
    Prefer /api/prompts/optimize/stream for real-time trajectory.
    """
    try:
        body = await request.json()
        role = body.get("role", "coder")
        metric = body.get("metric", "step_efficiency")
        threshold = float(body.get("threshold", 0.7))
        model = body.get("model", "claude-sonnet-4.6")
        version = body.get("version")  # target prompt version to optimize (optional)

        prompt = _build_optimize_prompt(role, metric, threshold, version)

        from src.orchestrator import build_orchestrator_from_team
        orchestrator = await build_orchestrator_from_team(
            team_id="default", model_override=model
        )
        result = await orchestrator.ainvoke({
            "messages": [{"role": "user", "content": prompt}],
            "agent_trace": [],
            "completed_steps": [],
            "supervisor_iterations": 0,
        })
        last_msg = result["messages"][-1]
        response_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        agents = list(dict.fromkeys([
            e.get("agent") for e in result.get("agent_trace", [])
            if e.get("step") == "execution"
        ]))
        return {
            "status": "completed",
            "role": role,
            "metric": metric,
            "agents_used": agents,
            "response": response_text[:4000],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _build_optimize_prompt(role: str, metric: str, threshold: float, version: str | None) -> str:
    """Build the user prompt for the PromptOptimizer agent."""
    version_clause = f"target prompt version: {version}" if version else "the latest available version"
    return (
        f"Run a full prompt optimization loop for the **{role}** agent ({version_clause}). "
        f"Target metric: **{metric}** (improvement threshold: {threshold}). "
        f"Follow the SETUP → LOOP (up to 3 cycles) → FINAL OUTPUT protocol exactly as specified in your instructions. "
        f"Bootstrap note: if there are no regression records yet for role={role} version={version or 'latest'}, "
        f"run a baseline regression first using golden_ids=['golden_001','golden_004','golden_005','golden_006','golden_021'] "
        f"before starting the analysis. "
        f"Report the full iteration summary, root cause, changes per cycle, and final diff."
    )


@app.post("/api/prompts/optimize/stream")
async def prompt_optimize_stream(request: Request):
    """
    SSE streaming version of the PromptOptimizer.
    Emits real-time events: agent_start, tool_start, tool_end, thinking, version_registered,
    regression_result, response, done.

    Body: {"role": "coder", "metric": "step_efficiency", "threshold": 0.7,
           "model": "claude-sonnet-4.6", "version": "v4" (optional)}
    """
    body = await request.json()
    role = body.get("role", "coder")
    metric = body.get("metric", "step_efficiency")
    threshold = float(body.get("threshold", 0.7))
    model = body.get("model", "claude-sonnet-4.6")
    version = body.get("version")

    user_prompt = _build_optimize_prompt(role, metric, threshold, version)

    from src.orchestrator import build_orchestrator_from_team
    from langgraph.graph.state import CompiledStateGraph

    orchestrator = await build_orchestrator_from_team(
        team_id="default", model_override=model
    )
    thread_id = uuid.uuid4().hex[:12]
    config = get_graph_config(thread_id)
    initial_input = {
        "messages": [{"role": "user", "content": user_prompt}],
        "agent_trace": [],
        "completed_steps": [],
        "supervisor_iterations": 0,
    }

    async def event_generator():
        yield _sse_event("thread_id", {"thread_id": thread_id})
        yield _sse_event("optimize_start", {
            "role": role, "metric": metric, "threshold": threshold,
            "version": version or "latest", "model": model,
        })

        current_agent = ""
        token_buf = ""
        try:
            async for event in orchestrator.astream_events(
                initial_input, config=config, version="v2"
            ):
                kind = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data", {})
                meta = event.get("metadata", {})
                lg_node = meta.get("langgraph_node", "")

                if kind == "on_chain_start" and lg_node and lg_node not in _GRAPH_SYSTEM_NODES and not lg_node.startswith("_") and name == lg_node:
                    current_agent = lg_node
                    yield _sse_event("agent_start", {"agent": lg_node})

                elif kind == "on_tool_start":
                    tool_input = data.get("input", {})
                    safe_input = {k: str(v)[:300] for k, v in (tool_input.items() if isinstance(tool_input, dict) else {}.items())}
                    yield _sse_event("tool_start", {
                        "agent": current_agent, "tool": name, "args": safe_input,
                    })
                    # Special event for version registration so UI can highlight it
                    if name == "register_prompt_version":
                        yield _sse_event("version_registering", {
                            "role": safe_input.get("role", role),
                            "rationale": safe_input.get("rationale", "")[:200],
                        })

                elif kind == "on_tool_end":
                    output = str(data.get("output", ""))
                    yield _sse_event("tool_end", {
                        "agent": current_agent, "tool": name,
                        "output_preview": output[:400],
                    })
                    # Special event when regression subset completes
                    if name == "run_regression_subset":
                        yield _sse_event("regression_result", {
                            "tool_output": output[:600],
                        })
                    elif name == "register_prompt_version":
                        # Extract version label from tool output e.g. "✓ Registered coder prompt as v5."
                        import re
                        m = re.search(r'as\s+(v\d+)', output)
                        if m:
                            yield _sse_event("version_registered", {
                                "role": role, "version": m.group(1),
                            })

                elif kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk:
                        token = chunk.content if hasattr(chunk, "content") else ""
                        if token:
                            token_buf += token
                            if len(token_buf) >= 40 or token.endswith(("\n", ".")):
                                yield _sse_event("thinking", {"delta": token_buf, "agent": current_agent})
                                token_buf = ""

        except Exception as e:
            yield _sse_event("error", {"message": str(e)})
            return

        if token_buf:
            yield _sse_event("thinking", {"delta": token_buf, "agent": current_agent})

        yield _sse_event("done", {"status": "completed", "role": role, "metric": metric})

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── TraceAnalyzer Performance Report ────────────────────────────
# IMPORTANT: these specific routes must be declared BEFORE the
# /api/traces/{trace_id} wildcard, otherwise FastAPI matches
# "performance-report" as a trace_id (route ordering matters).

@app.get("/api/traces/performance-report")
async def traces_performance_report(days: int = 7, team_id: str = None):
    """
    Deep performance analytics computed by TraceAnalyzer.

    Returns per-agent and per-tool latency percentiles (p50/p95/p99), tool
    failure rates, context window utilisation trend, cost breakdown by agent
    and model, and z-score-based anomaly detection on historical spans.

    Query params:
        days     — look-back window (default: 7)
        team_id  — filter to a specific team (default: all teams)
    """
    try:
        from src.tracing.analyzer import TraceAnalyzer
        analyzer = TraceAnalyzer(team_id=team_id or None)
        report = analyzer.full_report(days=days)
        return report
    except Exception as e:
        return {"error": str(e), "agent_latency_percentiles": {}, "tool_failure_rates": {},
                "cost_breakdown": {"summary": {}}, "performance_anomalies": []}


@app.get("/api/traces/performance-report/anomalies")
async def traces_anomalies(days: int = 7, z_threshold: float = 2.5, team_id: str = None):
    """
    Return only the z-score anomaly list from TraceAnalyzer.
    Useful for surfacing as alerts in the monitoring dashboard.
    """
    try:
        from src.tracing.analyzer import TraceAnalyzer
        analyzer = TraceAnalyzer(team_id=team_id or None)
        return {"anomalies": analyzer.performance_anomalies(days=days, z_threshold=z_threshold)}
    except Exception as e:
        return {"anomalies": [], "error": str(e)}


@app.get("/api/traces/performance-report/regression-insights")
async def regression_insights(days: int = 30):
    """
    Aggregate DeepEval + G-Eval scores across all RegressionResult rows.

    Returns: worst-scoring metrics, costliest/slowest golden tests, agent
    roles that appear most in failed delegation patterns, and per-test
    pass rates. Intended for the Performance Analysis dashboard.
    """
    try:
        from src.tracing.analyzer import TraceAnalyzer
        analyzer = TraceAnalyzer()
        return analyzer.regression_metric_insights(days=days)
    except Exception as e:
        return {
            "metric_averages": {}, "worst_metrics": [], "costliest_tests": [],
            "slowest_tests": [], "failed_agent_patterns": {},
            "pass_rate_by_test": {}, "summary": {"total_runs": 0},
            "error": str(e),
        }


@app.get("/api/traces/performance-report/ab-compare")
async def ab_compare_runs(run_a: str, run_b: str):
    """
    Side-by-side comparison of two eval runs (A/B analysis).

    Computes per-metric deltas, flags significant changes (>=0.1 DeepEval,
    >=0.3 G-Eval), and returns radar-chart-ready normalised data plus a
    plain-English recommendation.

    Query params:
        run_a  — ID of the baseline eval run
        run_b  — ID of the new eval run
    """
    try:
        from src.tracing.analyzer import TraceAnalyzer
        analyzer = TraceAnalyzer()
        return analyzer.ab_compare(run_a, run_b)
    except Exception as e:
        return {
            "run_a": {"id": run_a}, "run_b": {"id": run_b},
            "metrics": {}, "performance": {}, "radar_data": [],
            "recommendation": "Error running comparison.",
            "test_comparison": [], "error": str(e),
        }


@app.get("/api/traces/{trace_id}")
def get_trace(trace_id: str):
    session = get_session()
    trace = session.query(Trace).filter_by(id=trace_id).first()
    if not trace:
        session.close()
        raise HTTPException(404, "Trace not found")
    spans = session.query(Span).filter_by(trace_id=trace_id).order_by(Span.start_time).all()
    result = {
        "id": trace.id, "team_id": trace.team_id, "user_prompt": trace.user_prompt,
        "total_latency_ms": round(trace.total_latency_ms, 1),
        "total_tokens": trace.total_tokens, "total_cost": round(trace.total_cost, 6),
        "status": trace.status,
        "created_at": trace.created_at.isoformat() if trace.created_at else None,
        "spans": [{
            "id": s.id, "parent_span_id": s.parent_span_id, "name": s.name,
            "span_type": s.span_type,
            "start_time": s.start_time.isoformat() if s.start_time else None,
            "end_time": s.end_time.isoformat() if s.end_time else None,
            "input_data": s.input_data, "output_data": s.output_data,
            "tokens_in": s.tokens_in, "tokens_out": s.tokens_out,
            "cost": round(s.cost, 6), "model": s.model,
            "status": s.status, "error": s.error,
        } for s in spans],
    }
    session.close()
    return result


@app.get("/api/otel/spans/stats")
def otel_span_stats(days: int = 30):
    """OTel span-level analytics with GenAI semantic convention breakdowns."""
    session = get_session()
    cutoff = datetime.utcnow() - timedelta(days=days)

    spans = session.query(Span).join(Trace).filter(Trace.created_at >= cutoff).all()
    traces = session.query(Trace).filter(Trace.created_at >= cutoff).all()

    if not spans:
        session.close()
        return {"total_spans": 0, "by_type": {}, "by_model": {}, "token_flow": [],
                "cost_breakdown": [], "latency_by_type": {}, "error_spans": 0}

    by_type: dict = {}
    by_model: dict = {}
    errors = 0
    for s in spans:
        st = s.span_type or "unknown"
        by_type.setdefault(st, {"count": 0, "tokens_in": 0, "tokens_out": 0, "cost": 0.0, "latency_ms": []})
        by_type[st]["count"] += 1
        by_type[st]["tokens_in"] += s.tokens_in or 0
        by_type[st]["tokens_out"] += s.tokens_out or 0
        by_type[st]["cost"] += s.cost or 0
        if s.start_time and s.end_time:
            dur = (s.end_time - s.start_time).total_seconds() * 1000
            by_type[st]["latency_ms"].append(dur)
        if s.status == "error":
            errors += 1
        if s.model:
            by_model.setdefault(s.model, {"count": 0, "tokens_in": 0, "tokens_out": 0, "cost": 0.0})
            by_model[s.model]["count"] += 1
            by_model[s.model]["tokens_in"] += s.tokens_in or 0
            by_model[s.model]["tokens_out"] += s.tokens_out or 0
            by_model[s.model]["cost"] += s.cost or 0

    for st_data in by_type.values():
        lats = st_data.pop("latency_ms")
        st_data["avg_latency_ms"] = round(sum(lats) / len(lats), 1) if lats else 0
        st_data["cost"] = round(st_data["cost"], 6)

    for m_data in by_model.values():
        m_data["cost"] = round(m_data["cost"], 6)

    token_flow = []
    for t in sorted(traces, key=lambda x: x.created_at or datetime.min)[-30:]:
        trace_spans = [s for s in spans if s.trace_id == t.id]
        t_in = sum(s.tokens_in or 0 for s in trace_spans)
        t_out = sum(s.tokens_out or 0 for s in trace_spans)
        if not t_in and t.total_tokens:
            t_in = t.total_tokens // 2
            t_out = t.total_tokens - t_in
        token_flow.append({
            "id": t.id[:6],
            "tokens_in": t_in,
            "tokens_out": t_out,
            "cost": round(t.total_cost or 0, 6),
            "latency_ms": round(t.total_latency_ms or 0, 1),
            "time": t.created_at.isoformat() if t.created_at else "",
        })

    session.close()
    return {
        "total_spans": len(spans),
        "total_traces": len(traces),
        "error_spans": errors,
        "by_type": by_type,
        "by_model": by_model,
        "token_flow": token_flow,
    }


# ── Trace Evaluation ────────────────────────────────────────────

@app.post("/api/traces/evaluate")
async def evaluate_traces():
    """Run G-Eval + DeepEval on traces that haven't been fully evaluated."""
    import json as _json
    import copy
    from sqlalchemy.orm.attributes import flag_modified

    session = get_session()
    pending = session.query(Trace).filter(
        Trace.eval_status.in_(["pending", "quick"]),
        Trace.agent_response != "",
        Trace.agent_response != None,
    ).order_by(Trace.created_at.desc()).limit(10).all()

    evaluated = 0
    errors = []
    for tr in pending:
        existing = copy.deepcopy(tr.eval_scores) if tr.eval_scores else {}
        if isinstance(existing, str):
            existing = _json.loads(existing) if existing else {}

        # ── G-Eval (LLM-as-Judge with CoT) ──
        try:
            from src.evaluation.llm_judge import judge_response
            geval_result = await judge_response(
                user_prompt=tr.user_prompt or "",
                agent_response=tr.agent_response or "",
                tool_calls=tr.tool_calls_json or [],
            )
            geval_scores = geval_result.get("scores", {})
            geval_reasoning = geval_result.get("reasoning", {})
            existing["geval_scores"] = geval_scores
            existing["geval_reasoning"] = geval_reasoning
            existing["geval_overall"] = geval_result.get("overall", 0.5)
            for k, v in geval_scores.items():
                existing[f"judge_{k}"] = v
        except Exception as e:
            errors.append(f"geval:{tr.id}:{str(e)[:100]}")

        # ── DeepEval (All Agentic Metrics) ──
        try:
            from src.evaluation.integrations import run_all_deepeval_metrics

            tool_outputs = []
            for span_row in session.query(Span).filter_by(trace_id=tr.id).all():
                if span_row.output_data:
                    for v in span_row.output_data.values():
                        if v:
                            tool_outputs.append(str(v)[:300])

            agent_trace = tr.tool_calls_json if isinstance(tr.tool_calls_json, list) else []

            deepeval_scores = await run_all_deepeval_metrics(
                user_prompt=tr.user_prompt or "",
                agent_response=tr.agent_response or "",
                agent_trace=agent_trace,
                tool_outputs=tool_outputs[:5],
            )
            existing["deepeval_scores"] = deepeval_scores
        except Exception as e:
            existing["deepeval_scores"] = {
                "deepeval_relevancy": 0.5, "deepeval_faithfulness": 0.5,
                "tool_correctness": 0.5, "argument_correctness": 0.5,
                "task_completion": 0.5, "step_efficiency_de": 0.5,
                "plan_quality": 0.5, "plan_adherence": 0.5,
            }
            errors.append(f"deepeval:{tr.id}:{str(e)[:100]}")

        tr.eval_scores = existing
        tr.eval_status = "evaluated"
        flag_modified(tr, "eval_scores")
        evaluated += 1

    session.commit()
    total_pending = session.query(Trace).filter(Trace.eval_status.in_(["pending", "quick"])).count()
    session.close()
    return {"evaluated": evaluated, "remaining": total_pending, "errors": errors}


# ── Evaluation API ──────────────────────────────────────────────

@app.get("/api/eval/runs")
def list_eval_runs():
    session = get_session()
    runs = session.query(EvalRun).order_by(EvalRun.created_at.desc()).all()
    result = []
    for r in runs:
        rj = r.results_json or {}
        result.append({
            "id": r.id, "model": r.model, "prompt_version": r.prompt_version,
            "team_id": r.team_id, "num_tasks": r.num_tasks,
            "task_success_rate": rj.get("task_success_rate", r.task_completion_rate or 0),
            "tool_accuracy": rj.get("tool_accuracy", r.avg_tool_call_accuracy or 0),
            "reasoning_quality": rj.get("reasoning_quality", 0),
            "step_efficiency": rj.get("step_efficiency", 0),
            "faithfulness": rj.get("faithfulness", 0),
            "safety_compliance": rj.get("safety_compliance", 0),
            "routing_accuracy": rj.get("routing_accuracy", r.routing_accuracy or 0),
            "failure_recovery": rj.get("failure_recovery", 0),
            "avg_latency_ms": rj.get("avg_latency_ms", r.avg_latency_ms),
            "total_cost": r.total_cost or 0,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "results_json": rj,
        })
    session.close()
    return result

@app.get("/api/eval/runs/{run_id}")
def get_eval_run(run_id: str):
    session = get_session()
    run = session.query(EvalRun).filter_by(id=run_id).first()
    if not run:
        session.close()
        raise HTTPException(404, "Eval run not found")
    rj = run.results_json or {}
    import json, os
    tasks_detail = []
    results_dir = "eval/results"
    for fname in os.listdir(results_dir) if os.path.exists(results_dir) else []:
        if fname.startswith(f"eval_{run_id}"):
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)
                tasks_detail = data.get("tasks", [])
                break
    result = {
        "id": run.id, "model": run.model, "prompt_version": run.prompt_version,
        "num_tasks": run.num_tasks,
        "task_success_rate": rj.get("task_success_rate", 0),
        "tool_accuracy": rj.get("tool_accuracy", 0),
        "reasoning_quality": rj.get("reasoning_quality", 0),
        "step_efficiency": rj.get("step_efficiency", 0),
        "faithfulness": rj.get("faithfulness", 0),
        "safety_compliance": rj.get("safety_compliance", 0),
        "routing_accuracy": rj.get("routing_accuracy", 0),
        "avg_latency_ms": rj.get("avg_latency_ms"),
        "tasks": tasks_detail,
    }
    session.close()
    return result

@app.post("/api/eval/run")
async def run_eval(request: EvalRequest):
    from src.evaluation.evaluator import AgentEvaluator
    from src.evaluation.scenarios import FAST_SCENARIOS

    evaluator = AgentEvaluator(
        use_llm_judge=request.use_llm_judge,
        use_deepeval=request.use_deepeval,
    )
    run = await evaluator.run_evaluation(
        scenarios=FAST_SCENARIOS, team_id=request.team_id, skip_init=True,
    )
    return run.summary()

@app.post("/api/eval/compare")
async def compare_models(request: EvalCompareRequest):
    from src.evaluation.evaluator import AgentEvaluator
    evaluator = AgentEvaluator()
    runs = await evaluator.run_comparison(
        model_configs=request.model_configs,
        team_id=request.team_id,
    )
    return [r.summary() for r in runs]

@app.get("/api/eval/compare/{run_a}/{run_b}")
def compare_runs(run_a: str, run_b: str):
    """Compare two eval runs across all 7 CLASSic metrics using results_json."""
    session = get_session()
    a = session.query(EvalRun).filter_by(id=run_a).first()
    b = session.query(EvalRun).filter_by(id=run_b).first()
    session.close()
    if not a or not b:
        raise HTTPException(404, "Run not found")

    rj_a = a.results_json or {}
    rj_b = b.results_json or {}

    COMPARE_KEYS = [
        "task_success_rate", "tool_accuracy", "reasoning_quality", "step_efficiency",
        "faithfulness", "safety_compliance", "routing_accuracy",
    ]

    DB_FALLBACKS = {
        "task_success_rate": "task_completion_rate",
        "tool_accuracy": "avg_tool_call_accuracy",
        "routing_accuracy": "routing_accuracy",
    }

    comparison = {}
    for key in COMPARE_KEYS:
        va = rj_a.get(key)
        if va is None:
            fallback = DB_FALLBACKS.get(key)
            va = getattr(a, fallback, 0) if fallback else 0
        va = va or 0

        vb = rj_b.get(key)
        if vb is None:
            fallback = DB_FALLBACKS.get(key)
            vb = getattr(b, fallback, 0) if fallback else 0
        vb = vb or 0

        delta = vb - va
        comparison[key] = {"before": va, "after": vb, "delta": round(delta, 3), "regression": delta < -0.05}

    meta_a = {"id": a.id, "model": a.model, "prompt_version": a.prompt_version,
              "num_tasks": a.num_tasks, "created_at": a.created_at.isoformat() if a.created_at else None}
    meta_b = {"id": b.id, "model": b.model, "prompt_version": b.prompt_version,
              "num_tasks": b.num_tasks, "created_at": b.created_at.isoformat() if b.created_at else None}

    regressions = [k for k, v in comparison.items() if v["regression"]]
    return {
        "run_a": run_a, "run_b": run_b,
        "meta_a": meta_a, "meta_b": meta_b,
        "comparison": comparison,
        "regressions": regressions,
        "pass": len(regressions) == 0,
    }


# ── Models API ──────────────────────────────────────────────────

@app.get("/api/models")
def list_available_models():
    return AVAILABLE_MODELS


@app.get("/api/config/llm")
def get_llm_config():
    """Return current LLM defaults so the UI can display the active default model."""
    from src.config import config as app_config
    default_model = app_config.llm.model
    # Find a human-readable name from AVAILABLE_MODELS, fall back to the raw id
    name = next((m["name"] for m in AVAILABLE_MODELS if m["id"] == default_model), default_model)
    return {
        "default_model": default_model,
        "default_model_name": name,
        "judge_model": app_config.llm.judge_model,
        "router_model": app_config.llm.router_model,
    }


# ── Prompt Versions API ────────────────────────────────────────

@app.get("/api/prompt-versions")
def list_prompt_versions():
    """
    Return all prompt versions from the PromptRegistry (PromptVersionEntry rows).
    Groups entries by version label so the regression page dropdown shows one option
    per version label (e.g. v1, v2, v3) with a description of which roles changed.
    Also falls back to legacy PromptVersion snapshot rows if any exist.
    """
    from src.db.models import PromptVersionEntry
    session = get_session()
    try:
        # Collect all PromptVersionEntry rows grouped by version label
        entries = (
            session.query(PromptVersionEntry)
            .order_by(PromptVersionEntry.version, PromptVersionEntry.created_at)
            .all()
        )

        # Group by version label across roles → one dropdown option per version
        from collections import defaultdict
        by_version: dict[str, dict] = defaultdict(lambda: {
            "roles": [], "cot_roles": [], "optimizer_roles": [],
            "created_at": None, "latest_rationale": "",
        })
        for e in entries:
            v = by_version[e.version]
            v["roles"].append(e.role)
            if e.cot_enhanced:
                v["cot_roles"].append(e.role)
            if e.created_by == "optimizer":
                v["optimizer_roles"].append(e.role)
            if v["created_at"] is None or (e.created_at and e.created_at.isoformat() > v["created_at"]):
                v["created_at"] = e.created_at.isoformat() if e.created_at else None
            if e.rationale:
                v["latest_rationale"] = e.rationale[:120]

        result = []
        for version_label, info in sorted(by_version.items()):
            roles_str = ", ".join(sorted(set(info["roles"])))
            tags = []
            if info["cot_roles"]:
                tags.append("CoT")
            if info["optimizer_roles"]:
                tags.append("optimizer")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            desc = f"{len(set(info['roles']))} roles ({roles_str}){tag_str}"
            if info["latest_rationale"]:
                desc += f" — {info['latest_rationale']}"
            result.append({
                "id": version_label,
                "version_label": version_label,
                "description": desc,
                "roles": sorted(set(info["roles"])),
                "cot_enhanced_roles": info["cot_roles"],
                "optimizer_generated_roles": info["optimizer_roles"],
                "is_active": True,
                "created_at": info["created_at"],
            })

        # Also include legacy PromptVersion snapshot rows if any
        legacy = session.query(PromptVersion).order_by(PromptVersion.created_at.desc()).all()
        legacy_labels = {r["version_label"] for r in result}
        for p in legacy:
            if p.version_label not in legacy_labels:
                result.append({
                    "id": p.id,
                    "version_label": p.version_label,
                    "description": p.description or "(legacy snapshot)",
                    "roles": [],
                    "is_active": p.is_active,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                })

        return result
    finally:
        session.close()


@app.post("/api/prompt-versions")
def create_prompt_version(data: PromptVersionCreate):
    session = get_session()
    existing = session.query(PromptVersion).filter_by(version_label=data.version_label).first()
    if existing:
        session.close()
        raise HTTPException(409, f"Version '{data.version_label}' already exists")
    pv = PromptVersion(
        version_label=data.version_label,
        description=data.description,
        agent_prompts=data.agent_prompts,
        team_strategy=data.team_strategy,
    )
    session.add(pv)
    session.commit()
    pid = pv.id
    session.close()
    return {"id": pid, "status": "created"}


@app.put("/api/prompt-versions/{pv_id}")
def update_prompt_version(pv_id: str, data: PromptVersionUpdate):
    session = get_session()
    pv = session.query(PromptVersion).filter_by(id=pv_id).first()
    if not pv:
        session.close()
        raise HTTPException(404, "Prompt version not found")
    if data.version_label is not None:
        pv.version_label = data.version_label
    if data.description is not None:
        pv.description = data.description
    if data.agent_prompts is not None:
        pv.agent_prompts = data.agent_prompts
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(pv, "agent_prompts")
    if data.team_strategy is not None:
        pv.team_strategy = data.team_strategy
    session.commit()
    session.close()
    return {"status": "updated"}


@app.delete("/api/prompt-versions/{pv_id}")
def delete_prompt_version(pv_id: str):
    session = get_session()
    pv = session.query(PromptVersion).filter_by(id=pv_id).first()
    if not pv:
        session.close()
        raise HTTPException(404, "Prompt version not found")
    session.delete(pv)
    session.commit()
    session.close()
    return {"status": "deleted"}


@app.get("/api/prompt-versions/current")
def get_current_prompts():
    """Get the current agent prompts from the default team for editing."""
    session = get_session()
    agents = session.query(Agent).filter_by(team_id="default").all()
    prompts = {a.role: a.system_prompt for a in agents}
    team = session.query(Team).filter_by(id="default").first()
    strategy = team.decision_strategy if team else "router_decides"
    session.close()
    return {"agent_prompts": prompts, "team_strategy": strategy}


# ── Golden Dataset API ──────────────────────────────────────────

@app.get("/api/golden")
def list_golden_cases():
    session = get_session()
    cases = session.query(GoldenTestCase).order_by(GoldenTestCase.id).all()
    result = [{
        "id": c.id, "name": c.name, "prompt": c.prompt,
        "expected_agent": c.expected_agent, "expected_tools": c.expected_tools or [],
        "expected_output_keywords": c.expected_output_keywords or [],
        "expected_delegation_pattern": c.expected_delegation_pattern or [],
        "quality_thresholds": c.quality_thresholds or {},
        "max_llm_calls": c.max_llm_calls, "max_tool_calls": c.max_tool_calls,
        "max_tokens": c.max_tokens, "max_latency_ms": c.max_latency_ms,
        "complexity": c.complexity, "version": c.version,
        "reference_output": c.reference_output, "is_active": c.is_active,
        "strategy": getattr(c, "strategy", None),
        "expected_strategy": getattr(c, "expected_strategy", None),
    } for c in cases]
    session.close()
    return result


@app.post("/api/golden")
def create_golden_case(data: GoldenCaseCreate):
    session = get_session()
    existing = session.query(GoldenTestCase).filter_by(id=data.id).first()
    if existing:
        session.close()
        raise HTTPException(409, f"Golden case '{data.id}' already exists")
    case = GoldenTestCase(
        id=data.id, name=data.name, prompt=data.prompt,
        expected_agent=data.expected_agent, expected_tools=data.expected_tools,
        expected_output_keywords=data.expected_output_keywords,
        expected_delegation_pattern=data.expected_delegation_pattern,
        quality_thresholds=data.quality_thresholds,
        max_llm_calls=data.max_llm_calls, max_tool_calls=data.max_tool_calls,
        max_tokens=data.max_tokens, max_latency_ms=data.max_latency_ms,
        complexity=data.complexity, version=data.version,
        reference_output=data.reference_output,
    )
    session.add(case)
    session.commit()
    session.close()
    _sync_db_to_json()
    return {"id": data.id, "status": "created"}


@app.put("/api/golden/{case_id}")
def update_golden_case(case_id: str, data: GoldenCaseUpdate):
    session = get_session()
    case = session.query(GoldenTestCase).filter_by(id=case_id).first()
    if not case:
        session.close()
        raise HTTPException(404, "Golden case not found")
    for field in [
        "name", "prompt", "expected_agent", "expected_tools",
        "expected_output_keywords", "expected_delegation_pattern",
        "quality_thresholds", "max_llm_calls", "max_tool_calls",
        "max_tokens", "max_latency_ms", "complexity", "version", "reference_output",
    ]:
        val = getattr(data, field, None)
        if val is not None:
            setattr(case, field, val)
    session.commit()
    session.close()
    _sync_db_to_json()
    return {"status": "updated"}


@app.delete("/api/golden/{case_id}")
def delete_golden_case(case_id: str):
    session = get_session()
    case = session.query(GoldenTestCase).filter_by(id=case_id).first()
    if not case:
        session.close()
        raise HTTPException(404, "Golden case not found")
    case.is_active = False
    session.commit()
    session.close()
    _sync_db_to_json()
    return {"status": "deactivated"}


@app.post("/api/golden/sync")
def sync_golden():
    from src.evaluation.golden import sync_golden_to_db
    sync_golden_to_db()
    return {"status": "synced"}


def _sync_db_to_json():
    """Write active golden cases back to JSON file for version control."""
    session = get_session()
    cases = session.query(GoldenTestCase).filter_by(is_active=True).order_by(GoldenTestCase.id).all()
    data = [{
        "id": c.id, "name": c.name, "prompt": c.prompt,
        "expected_agent": c.expected_agent, "expected_tools": c.expected_tools or [],
        "expected_output_keywords": c.expected_output_keywords or [],
        "expected_delegation_pattern": c.expected_delegation_pattern or [],
        "quality_thresholds": c.quality_thresholds or {},
        "max_llm_calls": c.max_llm_calls, "max_tool_calls": c.max_tool_calls,
        "max_tokens": c.max_tokens, "max_latency_ms": c.max_latency_ms,
        "complexity": c.complexity, "version": c.version,
        "reference_output": c.reference_output,
    } for c in cases]
    session.close()
    from src.evaluation.golden import save_golden_to_json
    save_golden_to_json(data)


# ── Regression Testing API ──────────────────────────────────────

@app.post("/api/regression/run")
async def run_regression(request: RegressionRunRequest):
    from src.evaluation.regression import RegressionRunner
    runner = RegressionRunner(
        model=request.model,
        prompt_version=request.prompt_version,
        team_id=request.team_id,
        prompt_versions_by_role=request.prompt_versions_by_role,
    )
    result = await runner.run(
        case_ids=request.case_ids,
        baseline_run_id=request.baseline_run_id,
    )
    return result


@app.get("/api/regression/runs")
def list_regression_runs():
    """List eval runs that have associated regression results."""
    session = get_session()
    run_ids = session.query(RegressionResult.run_id).distinct().all()
    run_ids = [r[0] for r in run_ids]
    runs = session.query(EvalRun).filter(EvalRun.id.in_(run_ids)).order_by(EvalRun.created_at.desc()).all()
    result = []
    for r in runs:
        cases = session.query(RegressionResult).filter_by(run_id=r.id).all()
        case_count = len(cases)
        passed_count = sum(1 for c in cases if c.overall_pass)
        case_ids = [c.golden_case_id for c in cases]
        rj = r.results_json or {}
        result.append({
            "id": r.id, "model": r.model, "prompt_version": r.prompt_version,
            "num_cases": case_count, "passed": passed_count, "failed": case_count - passed_count,
            "pass_rate": round(passed_count / max(case_count, 1), 3),
            "avg_latency_ms": rj.get("avg_latency_ms", r.avg_latency_ms),
            "total_cost": r.total_cost or 0,
            "regressions": rj.get("regressions", {}),
            "case_ids": case_ids,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    session.close()
    return result


@app.get("/api/regression/results/{run_id}")
def get_regression_results(run_id: str):
    session = get_session()
    run = session.query(EvalRun).filter_by(id=run_id).first()
    if not run:
        session.close()
        raise HTTPException(404, "Run not found")
    rows = session.query(RegressionResult).filter_by(run_id=run_id).all()
    rj = run.results_json or {}
    results = [{
        "id": r.id, "golden_case_id": r.golden_case_id,
        "golden_case_name": r.golden_case_name,
        "actual_output": r.actual_output, "actual_agent": r.actual_agent,
        "actual_tools": r.actual_tools or [],
        "actual_delegation_pattern": r.actual_delegation_pattern or [],
        "actual_llm_calls": r.actual_llm_calls, "actual_tool_calls": r.actual_tool_calls,
        "actual_tokens_in": r.actual_tokens_in, "actual_tokens_out": r.actual_tokens_out,
        "actual_latency_ms": r.actual_latency_ms, "actual_cost": r.actual_cost,
        "semantic_similarity": r.semantic_similarity,
        "quality_scores": r.quality_scores or {},
        "deepeval_scores": r.deepeval_scores or {},
        "trace_assertions": r.trace_assertions or {},
        "eval_reasoning": r.eval_reasoning or {},
        "cost_regression": r.cost_regression, "latency_regression": r.latency_regression,
        "quality_regression": r.quality_regression, "trace_regression": r.trace_regression,
        "overall_pass": r.overall_pass,
        "model_used": r.model_used, "prompt_version": r.prompt_version,
        "expected_strategy": getattr(r, "expected_strategy", None),
        "actual_strategy": getattr(r, "actual_strategy", None),
        "router_prompt_version": getattr(r, "router_prompt_version", None),
        "created_at": r.created_at.isoformat() if r.created_at else None,
    } for r in rows]
    session.close()
    return {
        "run_id": run_id,
        "model": run.model,
        "prompt_version": run.prompt_version,
        "summary": rj,
        "results": results,
    }


@app.get("/api/regression/results/{run_id}/{case_id}")
def get_regression_case_detail(run_id: str, case_id: str):
    session = get_session()
    r = session.query(RegressionResult).filter_by(run_id=run_id, golden_case_id=case_id).first()
    if not r:
        session.close()
        raise HTTPException(404, "Case result not found")

    golden = session.query(GoldenTestCase).filter_by(id=case_id).first()
    golden_data = {}
    if golden:
        golden_data = {
            "id": golden.id, "name": golden.name, "prompt": golden.prompt,
            "expected_agent": golden.expected_agent, "expected_tools": golden.expected_tools or [],
            "expected_delegation_pattern": golden.expected_delegation_pattern or [],
            "quality_thresholds": golden.quality_thresholds or {},
            "reference_output": golden.reference_output,
            "max_llm_calls": golden.max_llm_calls, "max_tool_calls": golden.max_tool_calls,
            "max_tokens": golden.max_tokens, "max_latency_ms": golden.max_latency_ms,
        }

    result = {
        "golden_case": golden_data,
        "result": {
            "id": r.id, "golden_case_id": r.golden_case_id,
            "golden_case_name": r.golden_case_name,
            "actual_output": r.actual_output, "actual_agent": r.actual_agent,
            "actual_tools": r.actual_tools or [],
            "actual_delegation_pattern": r.actual_delegation_pattern or [],
            "full_trace": r.full_trace or [], "span_data": r.span_data or [],
            "actual_llm_calls": r.actual_llm_calls, "actual_tool_calls": r.actual_tool_calls,
            "actual_tokens_in": r.actual_tokens_in, "actual_tokens_out": r.actual_tokens_out,
            "actual_latency_ms": r.actual_latency_ms, "actual_cost": r.actual_cost,
            "semantic_similarity": r.semantic_similarity,
            "quality_scores": r.quality_scores or {},
            "deepeval_scores": r.deepeval_scores or {},
            "trace_assertions": r.trace_assertions or {},
            "eval_reasoning": r.eval_reasoning or {},
            "cost_regression": r.cost_regression, "latency_regression": r.latency_regression,
            "quality_regression": r.quality_regression, "trace_regression": r.trace_regression,
            "overall_pass": r.overall_pass,
            "rca_analysis": r.rca_analysis,
            "model_used": r.model_used, "prompt_version": r.prompt_version,
            "expected_strategy": getattr(r, "expected_strategy", None),
            "actual_strategy": getattr(r, "actual_strategy", None),
            "router_prompt_version": getattr(r, "router_prompt_version", None),
        },
    }
    session.close()
    return result


@app.get("/api/regression/ab/options")
def regression_ab_options(golden_id: str):
    """
    Returns distinct regression runs available for a given golden test case,
    ordered newest-first. Each entry includes run_id, model, prompt_version,
    created_at, and overall_pass so the UI can populate two run-picker dropdowns.
    """
    session = get_session()
    try:
        rows = (
            session.query(RegressionResult)
            .filter(RegressionResult.golden_case_id == golden_id)
            .order_by(RegressionResult.created_at.desc())
            .all()
        )
        # Deduplicate by run_id (keep first = most recent per run)
        seen = set()
        options = []
        for r in rows:
            if r.run_id in seen:
                continue
            seen.add(r.run_id)
            options.append({
                "run_id": r.run_id,
                "model": r.model_used or "unknown",
                "prompt_version": r.prompt_version or "v1",
                "router_prompt_version": getattr(r, "router_prompt_version", None),
                "actual_strategy": getattr(r, "actual_strategy", None),
                "overall_pass": r.overall_pass,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            })
        return {"golden_id": golden_id, "options": options}
    finally:
        session.close()


@app.get("/api/regression/ab")
def regression_ab_compare(
    golden_id: str,
    run_id_a: Optional[str] = None,
    run_id_b: Optional[str] = None,
    model_a: Optional[str] = None,
    model_b: Optional[str] = None,
    version_a: Optional[str] = None,
    version_b: Optional[str] = None,
):
    """
    A/B comparison for a golden test case.
    Prefer run_id_a / run_id_b (exact run selection); falls back to
    model_a/b + version_a/b filters to pick the most recent matching result.
    """
    import json as _json
    session = get_session()
    try:
        golden = session.query(GoldenTestCase).filter_by(id=golden_id).first()
        expected_output = golden.reference_output if golden else ""
        golden_prompt = golden.prompt if golden else ""
        golden_name = golden.name if golden else golden_id

        def fetch_side(run_id: Optional[str], model: Optional[str], version: Optional[str]):
            q = session.query(RegressionResult).filter(
                RegressionResult.golden_case_id == golden_id
            )
            if run_id:
                q = q.filter(RegressionResult.run_id == run_id)
            else:
                if model:
                    q = q.filter(RegressionResult.model_used == model)
                if version:
                    q = q.filter(RegressionResult.prompt_version == version)
            r = q.order_by(RegressionResult.created_at.desc()).first()
            if not r:
                return None
            # Build per-agent trajectory from full_trace
            trace = r.full_trace or []
            per_agent: dict[str, dict] = {}
            for ev in trace:
                agent = ev.get("agent", "unknown")
                if agent not in per_agent:
                    per_agent[agent] = {"tools": [], "llm_calls": 0, "tool_calls": 0}
                step = ev.get("step", "")
                if step == "tool_call":
                    per_agent[agent]["tools"].append(ev.get("tool", ev.get("name", "")))
                    per_agent[agent]["tool_calls"] += 1
                elif step == "execution":
                    per_agent[agent]["llm_calls"] += 1

            # DeepEval scores + reasons split
            ds_raw = r.deepeval_scores or {}
            de_scores: dict[str, dict] = {}
            for k, v in ds_raw.items():
                if k.endswith("_reason"):
                    metric = k[: -len("_reason")]
                    de_scores.setdefault(metric, {})["reason"] = v
                else:
                    de_scores.setdefault(k, {})["score"] = v

            return {
                "run_id": r.run_id,
                "model": r.model_used,
                "prompt_version": r.prompt_version,
                "actual_strategy": getattr(r, "actual_strategy", None),
                "actual_output": r.actual_output or "",
                "actual_agent": r.actual_agent or "",
                "actual_delegation_pattern": r.actual_delegation_pattern or [],
                "actual_tools": r.actual_tools or [],
                "actual_llm_calls": r.actual_llm_calls,
                "actual_tool_calls": r.actual_tool_calls,
                "actual_latency_ms": r.actual_latency_ms,
                "actual_cost": r.actual_cost,
                "semantic_similarity": r.semantic_similarity,
                "overall_pass": r.overall_pass,
                "per_agent": per_agent,
                "deepeval": de_scores,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }

        side_a = fetch_side(run_id_a, model_a, version_a)
        side_b = fetch_side(run_id_b, model_b, version_b)
    finally:
        session.close()

    if not side_a and not side_b:
        return {
            "golden_id": golden_id,
            "golden_name": golden_name,
            "golden_prompt": golden_prompt,
            "expected_output": expected_output,
            "error": "No regression results found for these filters. Run the tests first.",
            "side_a": None, "side_b": None,
        }

    # Compute DeepEval metric diffs
    all_metrics = set()
    if side_a:
        all_metrics.update(side_a["deepeval"].keys())
    if side_b:
        all_metrics.update(side_b["deepeval"].keys())

    metric_diff: dict[str, dict] = {}
    for m in sorted(all_metrics):
        a_score = (side_a["deepeval"].get(m, {}) or {}).get("score") if side_a else None
        b_score = (side_b["deepeval"].get(m, {}) or {}).get("score") if side_b else None
        a_reason = (side_a["deepeval"].get(m, {}) or {}).get("reason", "") if side_a else ""
        b_reason = (side_b["deepeval"].get(m, {}) or {}).get("reason", "") if side_b else ""
        try:
            delta = round(float(b_score) - float(a_score), 3) if a_score is not None and b_score is not None else None
        except (TypeError, ValueError):
            delta = None
        metric_diff[m] = {
            "a_score": a_score, "b_score": b_score,
            "delta": delta,
            "improved": delta is not None and delta > 0.02,
            "regressed": delta is not None and delta < -0.02,
            "a_reason": a_reason, "b_reason": b_reason,
        }

    return {
        "golden_id": golden_id,
        "golden_name": golden_name,
        "golden_prompt": golden_prompt,
        "expected_output": expected_output,
        "side_a": side_a,
        "side_b": side_b,
        "metric_diff": metric_diff,
    }


# ── WebSocket (fallback with HITL support) ──────────────────────

@app.websocket("/ws/chat/{team_id}")
async def ws_chat(websocket: WebSocket, team_id: str):
    await websocket.accept()
    if team_id not in orchestrators:
        orchestrators[team_id] = await build_orchestrator_from_team(team_id)

    graph = orchestrators[team_id]
    thread_id = uuid.uuid4().hex[:12]
    await websocket.send_json({"type": "thread_id", "thread_id": thread_id})

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "message")
            config = get_graph_config(thread_id)

            if msg_type == "resume":
                from langgraph.types import Command
                graph_input = Command(resume=data.get("hitl_response", {}))
            else:
                message = data.get("message", "")
                graph_input = {
                    "messages": [{"role": "user", "content": message}],
                    "selected_agent": "", "agent_trace": [],
                }

            current_agent = ""
            async for event in graph.astream_events(graph_input, config=config, version="v2"):
                kind = event.get("event", "")
                name = event.get("name", "")
                ev_data = event.get("data", {})

                if kind == "on_chain_start" and "agent" in name.lower():
                    current_agent = name.replace("agent:", "").strip() or name
                    await websocket.send_json({"type": "agent_start", "agent": current_agent})
                elif kind == "on_tool_start":
                    await websocket.send_json({"type": "tool_start", "agent": current_agent, "tool": name})
                elif kind == "on_tool_end":
                    await websocket.send_json({"type": "tool_end", "agent": current_agent, "tool": name,
                                               "output_preview": str(ev_data.get("output", ""))[:300]})
                elif kind == "on_chat_model_stream":
                    chunk = ev_data.get("chunk")
                    if chunk:
                        content = getattr(chunk, "content", "")
                        if content and isinstance(content, str):
                            await websocket.send_json({"type": "llm_token", "agent": current_agent, "token": content})

            state = await graph.aget_state(config)
            if state.next:
                interrupt_value = {}
                if state.tasks:
                    for task in state.tasks:
                        if hasattr(task, "interrupts") and task.interrupts:
                            interrupt_value = task.interrupts[0].value
                            break
                interrupt_value["thread_id"] = thread_id
                await websocket.send_json({"type": "hitl_request", **interrupt_value})
            else:
                full_state = state.values
                last = full_state.get("messages", [])[-1] if full_state.get("messages") else None
                response = _extract_text(last.content if last and hasattr(last, "content") else str(last)) if last else ""
                await websocket.send_json({
                    "type": "response", "content": response,
                    "agent_used": full_state.get("selected_agent", "unknown"),
                })
                await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass


# ── RAG Pipeline Endpoints ─────────────────────────────────────────────────


class RagConfigCreate(BaseModel):
    name: str
    description: str = ""
    embedding_model: str = "openai/text-embedding-3-small"
    vector_store: str = "chroma"
    llm_model: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_strategy: str = "recursive"
    retrieval_strategy: str = "similarity"
    top_k: int = 5
    mmr_lambda: float = 0.5
    multi_query_n: int = 3
    system_prompt: Optional[str] = None
    reranker: str = "none"


class RagSourceCreate(BaseModel):
    source_type: str    # text | file | url
    content: str
    label: str = ""


class RagChatRequest(BaseModel):
    query: str
    config_id: str
    auto_evaluate: bool = False
    reranker_override: Optional[str] = None   # override reranker for this query only


class RagEvalRequest(BaseModel):
    config_id: str
    samples: list[dict]   # [{query, expected_answer}]
    thresholds: Optional[dict] = None


def _cfg_to_pipeline_config(cfg_row: RagConfigModel, reranker_override: str | None = None):
    from src.rag.pipeline import RAGConfig
    return RAGConfig(
        config_id=cfg_row.id,
        name=cfg_row.name,
        embedding_model=cfg_row.embedding_model,
        vector_store=cfg_row.vector_store,
        llm_model=cfg_row.llm_model,
        chunk_size=cfg_row.chunk_size,
        chunk_overlap=cfg_row.chunk_overlap,
        chunk_strategy=cfg_row.chunk_strategy,
        retrieval_strategy=cfg_row.retrieval_strategy,
        top_k=cfg_row.top_k,
        mmr_lambda=cfg_row.mmr_lambda or 0.5,
        multi_query_n=cfg_row.multi_query_n or 3,
        system_prompt=cfg_row.system_prompt,
        reranker=reranker_override if reranker_override is not None else (cfg_row.reranker or "none"),
    )


def _cfg_row_to_dict(r: RagConfigModel) -> dict:
    return {
        "id": r.id,
        "name": r.name,
        "description": r.description,
        "embedding_model": r.embedding_model,
        "vector_store": r.vector_store,
        "llm_model": r.llm_model,
        "chunk_size": r.chunk_size,
        "chunk_overlap": r.chunk_overlap,
        "chunk_strategy": r.chunk_strategy,
        "retrieval_strategy": r.retrieval_strategy,
        "top_k": r.top_k,
        "mmr_lambda": r.mmr_lambda,
        "multi_query_n": r.multi_query_n,
        "system_prompt": r.system_prompt,
        "reranker": r.reranker or "none",
        "is_active": r.is_active,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "sources": [
            {
                "id": s.id,
                "source_type": s.source_type,
                "content": s.content[:120],
                "label": s.label,
                "chunks_count": s.chunks_count,
                "tokens_estimated": s.tokens_estimated,
                "status": s.status,
                "error_message": s.error_message,
                "ingested_at": s.ingested_at.isoformat() if s.ingested_at else None,
            }
            for s in (r.sources or [])
        ],
    }


@app.get("/api/rag/models")
def list_rag_models():
    """Return available embedding models, vector store options, and reranker models."""
    from src.rag.embeddings import EMBEDDING_MODELS
    from src.rag.vectorstore import VECTOR_STORES
    from src.rag.reranker import RERANKER_MODELS
    return {
        "embedding_models": {k: v for k, v in EMBEDDING_MODELS.items()},
        "vector_stores": VECTOR_STORES,
        "chunk_strategies": ["recursive", "fixed", "semantic", "code"],
        "retrieval_strategies": ["similarity", "mmr", "multi_query", "hybrid"],
        "reranker_models": RERANKER_MODELS,
    }


@app.get("/api/rag/configs")
def list_rag_configs():
    session = get_session()
    try:
        rows = session.query(RagConfigModel).order_by(RagConfigModel.created_at.desc()).all()
        return [_cfg_row_to_dict(r) for r in rows]
    finally:
        session.close()


@app.post("/api/rag/configs", status_code=201)
def create_rag_config(body: RagConfigCreate):
    session = get_session()
    try:
        row = RagConfigModel(
            name=body.name,
            description=body.description,
            embedding_model=body.embedding_model,
            vector_store=body.vector_store,
            llm_model=body.llm_model,
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
            chunk_strategy=body.chunk_strategy,
            retrieval_strategy=body.retrieval_strategy,
            top_k=body.top_k,
            mmr_lambda=body.mmr_lambda,
            multi_query_n=body.multi_query_n,
            system_prompt=body.system_prompt,
            reranker=body.reranker,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _cfg_row_to_dict(row)
    finally:
        session.close()


@app.get("/api/rag/configs/{config_id}")
def get_rag_config(config_id: str):
    session = get_session()
    try:
        row = session.query(RagConfigModel).filter_by(id=config_id).first()
        if not row:
            raise HTTPException(404, "RAG config not found")
        return _cfg_row_to_dict(row)
    finally:
        session.close()


@app.put("/api/rag/configs/{config_id}")
def update_rag_config(config_id: str, body: RagConfigCreate):
    session = get_session()
    try:
        row = session.query(RagConfigModel).filter_by(id=config_id).first()
        if not row:
            raise HTTPException(404, "RAG config not found")
        for field, val in body.model_dump().items():
            setattr(row, field, val)
        session.commit()
        # Evict cached pipeline so new config takes effect
        from src.rag.pipeline import evict_pipeline
        evict_pipeline(config_id)
        session.refresh(row)
        return _cfg_row_to_dict(row)
    finally:
        session.close()


@app.delete("/api/rag/configs/{config_id}", status_code=204)
def delete_rag_config(config_id: str):
    session = get_session()
    try:
        row = session.query(RagConfigModel).filter_by(id=config_id).first()
        if not row:
            raise HTTPException(404, "RAG config not found")
        session.delete(row)
        session.commit()
        from src.rag.pipeline import evict_pipeline
        evict_pipeline(config_id)
    finally:
        session.close()


@app.post("/api/rag/configs/{config_id}/ingest")
async def ingest_source(config_id: str, body: RagSourceCreate):
    """Ingest a new document source into the RAG pipeline."""
    session = get_session()
    try:
        cfg_row = session.query(RagConfigModel).filter_by(id=config_id).first()
        if not cfg_row:
            raise HTTPException(404, "RAG config not found")

        src_row = RagSource(
            config_id=config_id,
            source_type=body.source_type,
            content=body.content,
            label=body.label or body.content[:60],
            status="ingesting",
        )
        session.add(src_row)
        session.commit()
        src_id = src_row.id
        cfg_config = _cfg_to_pipeline_config(cfg_row)
    finally:
        session.close()

    # Run ingestion asynchronously; update status when done
    async def _do_ingest():
        from src.rag.pipeline import get_pipeline
        sess2 = get_session()
        try:
            pipeline = get_pipeline(cfg_config)
            result = await pipeline.ingest(body.source_type, body.content)
            src = sess2.query(RagSource).filter_by(id=src_id).first()
            if src:
                src.chunks_count = result.get("chunks", 0)
                src.tokens_estimated = result.get("tokens_estimated", 0)
                src.status = "ingested"
                src.ingested_at = datetime.utcnow()
            sess2.commit()
        except Exception as e:
            src = sess2.query(RagSource).filter_by(id=src_id).first()
            if src:
                src.status = "error"
                src.error_message = str(e)
            sess2.commit()
        finally:
            sess2.close()

    asyncio.create_task(_do_ingest())
    return {"source_id": src_id, "status": "ingesting"}


@app.delete("/api/rag/configs/{config_id}/sources/{source_id}", status_code=204)
def delete_source(config_id: str, source_id: str):
    session = get_session()
    try:
        src = session.query(RagSource).filter_by(id=source_id, config_id=config_id).first()
        if not src:
            raise HTTPException(404, "Source not found")
        session.delete(src)
        session.commit()
    finally:
        session.close()


@app.post("/api/rag/chat")
async def rag_chat(body: RagChatRequest):
    """Query the RAG pipeline and return an answer with citations."""
    import uuid as _uuid_mod
    session = get_session()
    try:
        cfg_row = session.query(RagConfigModel).filter_by(id=body.config_id).first()
        if not cfg_row:
            raise HTTPException(404, "RAG config not found")
        cfg_config = _cfg_to_pipeline_config(cfg_row)
    finally:
        session.close()

    # Apply reranker override: update config in-place (keeps same config_id/collection)
    if body.reranker_override is not None:
        import dataclasses as _dc
        cfg_config = _dc.replace(cfg_config, reranker=body.reranker_override)

    from src.rag.pipeline import get_pipeline, evict_pipeline
    # If reranker override differs from cached pipeline, rebuild temporarily
    if body.reranker_override is not None:
        evict_pipeline(cfg_config.config_id)
    pipeline = get_pipeline(cfg_config)
    if pipeline.chunk_count() == 0:
        raise HTTPException(400, "No documents ingested yet. Please add data sources first.")

    trace_id = _uuid_mod.uuid4().hex[:12]
    response = await pipeline.query(body.query)

    # Persist query log + OTel spans
    sess2 = get_session()
    query_id = None
    try:
        q_row = RagQuery(
            config_id=body.config_id,
            query=body.query,
            answer=response.answer,
            citations=[
                {"source": c.source, "chunk_index": c.chunk_index, "total_chunks": c.total_chunks,
                 "page": c.page, "score": c.score, "snippet": c.snippet}
                for c in response.citations
            ],
            strategy_used=response.strategy_used,
            chunks_retrieved=response.chunks_retrieved,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            latency_ms=response.latency_ms,
            eval_status="pending",
            trace_id=trace_id,
        )
        sess2.add(q_row)

        # Flush OTel spans (rag.ingest / rag.retrieve / rag.generate / rag.embed)
        # and persist them under a synthetic Trace record so they appear in the
        # OTel monitoring dashboard and evaluation page.
        otel_spans = _flush_pending_spans(trace_id)
        if otel_spans:
            rag_trace = Trace(
                id=trace_id,
                team_id=None,
                user_prompt=body.query[:500],
                agent_used="rag",
                agent_response=response.answer[:500],
                total_latency_ms=response.latency_ms,
                total_tokens=response.tokens_in + response.tokens_out,
                total_cost=sum(s.get("cost", 0) for s in otel_spans),
                status="completed",
            )
            sess2.add(rag_trace)
            seen_ids: set = set()
            for s in otel_spans:
                sid = s.get("id", _uuid_mod.uuid4().hex[:12])
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)
                # Assign rag.* span_types for visibility in OTel stats
                raw_name = s.get("name", "")
                span_type = s.get("span_type", "unknown")
                if raw_name.startswith("rag."):
                    span_type = raw_name  # e.g. "rag.retrieve", "rag.generate"
                sess2.add(Span(
                    id=sid,
                    trace_id=trace_id,
                    parent_span_id=s.get("parent_span_id"),
                    name=raw_name,
                    span_type=span_type,
                    start_time=s.get("start_time"),
                    end_time=s.get("end_time"),
                    input_data=s.get("input_data", {}),
                    output_data=s.get("output_data", {}),
                    tokens_in=s.get("tokens_in", 0),
                    tokens_out=s.get("tokens_out", 0),
                    cost=s.get("cost", 0.0),
                    model=s.get("model", ""),
                    status=s.get("status", "completed"),
                    error=s.get("error"),
                ))

        sess2.commit()
        query_id = q_row.id
    finally:
        sess2.close()

    # Kick off async DeepEval evaluation (non-blocking)
    if query_id and body.auto_evaluate:
        _bg_tasks: set = set()
        async def _auto_eval():
            await _run_rag_eval_for_query(query_id, cfg_config, response)
        t = asyncio.create_task(_auto_eval())
        _bg_tasks.add(t)
        t.add_done_callback(_bg_tasks.discard)

    return {
        "query_id": query_id,
        "trace_id": trace_id,
        "answer": response.answer,
        "citations": [
            {"source": c.source, "chunk_index": c.chunk_index, "total_chunks": c.total_chunks,
             "page": c.page, "score": c.score, "snippet": c.snippet}
            for c in response.citations
        ],
        "strategy_used": response.strategy_used,
        "chunks_retrieved": response.chunks_retrieved,
        "tokens_in": response.tokens_in,
        "tokens_out": response.tokens_out,
        "latency_ms": response.latency_ms,
    }


# ── A/B Compare endpoint ────────────────────────────────────────────────────


class RagComparePane(BaseModel):
    config_id: str
    retrieval_strategy_override: Optional[str] = None
    reranker_override: Optional[str] = None
    llm_model_override: Optional[str] = None


class RagCompareRequest(BaseModel):
    query: str
    pane_a: RagComparePane
    pane_b: RagComparePane
    auto_evaluate: bool = True


async def _run_pane_query(
    pane: RagComparePane,
    query: str,
    run_eval: bool,
) -> dict:
    """Run a single RAG query for one compare pane and return result + eval scores."""
    session = get_session()
    try:
        cfg_row = session.query(RagConfigModel).filter_by(id=pane.config_id).first()
        if not cfg_row:
            return {"error": f"Config {pane.config_id} not found"}
        cfg_config = _cfg_to_pipeline_config(
            cfg_row,
            reranker_override=pane.reranker_override,
        )
        if pane.retrieval_strategy_override:
            import dataclasses as _dc
            cfg_config = _dc.replace(cfg_config, retrieval_strategy=pane.retrieval_strategy_override)
        if pane.llm_model_override:
            import dataclasses as _dc
            cfg_config = _dc.replace(cfg_config, llm_model=pane.llm_model_override)
    finally:
        session.close()

    from src.rag.pipeline import get_pipeline, evict_pipeline
    # Evict cache if any override changes the effective config
    if pane.reranker_override is not None or pane.retrieval_strategy_override:
        evict_pipeline(cfg_config.config_id)
    pipeline = get_pipeline(cfg_config)
    if pipeline.chunk_count() == 0:
        return {"error": "No documents ingested for this config."}

    response = await pipeline.query(query)

    eval_scores: dict = {}
    if run_eval:
        try:
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import (
                AnswerRelevancyMetric, FaithfulnessMetric,
                ContextualRelevancyMetric, ContextualPrecisionMetric,
                ContextualRecallMetric,
            )
            from src.evaluation.integrations import _get_deepeval_model
            judge = _get_deepeval_model()
            ctx = [c.snippet for c in response.citations]
            tc = LLMTestCase(
                input=query,
                actual_output=response.answer,
                expected_output=response.answer,
                retrieval_context=ctx,
            )
            metrics_to_run = [
                ("answer_relevancy",     AnswerRelevancyMetric(threshold=0.5, model=judge, include_reason=True)),
                ("faithfulness",         FaithfulnessMetric(threshold=0.5, model=judge, include_reason=True)),
                ("contextual_relevancy", ContextualRelevancyMetric(threshold=0.5, model=judge, include_reason=True)),
                ("contextual_precision", ContextualPrecisionMetric(threshold=0.5, model=judge, include_reason=True)),
                ("contextual_recall",    ContextualRecallMetric(threshold=0.5, model=judge, include_reason=True)),
            ]
            for name, metric in metrics_to_run:
                for attempt in range(2):
                    try:
                        await metric.a_measure(tc)
                        eval_scores[name] = {
                            "score": float(metric.score or 0),
                            "passed": bool(metric.is_successful()),
                            "reason": metric.reason or "",
                        }
                        break
                    except Exception as me:
                        if attempt == 1:
                            eval_scores[name] = {"score": 0.0, "passed": False, "reason": f"ERROR: {me}"}
        except Exception as e:
            eval_scores = {"error": str(e)}

    return {
        "answer": response.answer,
        "citations": [
            {"source": c.source, "chunk_index": c.chunk_index, "score": c.score, "snippet": c.snippet}
            for c in response.citations
        ],
        "strategy_used": response.strategy_used,
        "reranker_used": cfg_config.reranker,
        "chunks_retrieved": response.chunks_retrieved,
        "tokens_in": response.tokens_in,
        "tokens_out": response.tokens_out,
        "latency_ms": response.latency_ms,
        "eval_scores": eval_scores,
    }


@app.post("/api/rag/compare")
async def rag_compare(body: RagCompareRequest):
    """Run the same query on two pipeline configurations in parallel.

    Returns answers, citations, latency, and optionally DeepEval metrics for
    each pane so the frontend can render a side-by-side radar comparison.
    """
    result_a, result_b = await asyncio.gather(
        _run_pane_query(body.pane_a, body.query, body.auto_evaluate),
        _run_pane_query(body.pane_b, body.query, body.auto_evaluate),
        return_exceptions=True,
    )
    if isinstance(result_a, Exception):
        result_a = {"error": str(result_a)}
    if isinstance(result_b, Exception):
        result_b = {"error": str(result_b)}
    return {"pane_a": result_a, "pane_b": result_b, "query": body.query}


async def _run_rag_eval_for_query(query_id: str, cfg_config, rag_response) -> None:
    """
    Background task: run all 5 DeepEval RAG metrics on a stored query.

    Metrics:
      - answer_relevancy      (no ground truth needed)
      - faithfulness          (no ground truth needed)
      - contextual_relevancy  (no ground truth needed)
      - contextual_precision  (uses answer as ground-truth proxy)
      - contextual_recall     (uses answer as ground-truth proxy)
    """
    import logging as _log
    _rl = _log.getLogger("rag_eval")

    sess = get_session()
    try:
        q_row = sess.query(RagQuery).filter_by(id=query_id).first()
        if not q_row:
            return
        q_row.eval_status = "running"
        sess.commit()
    finally:
        sess.close()

    try:
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            ContextualRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
        )
        from src.evaluation.integrations import _get_deepeval_model

        judge = _get_deepeval_model()
        answer = rag_response.answer if hasattr(rag_response, "answer") else ""
        query_text = rag_response.query if hasattr(rag_response, "query") else ""
        # Use full snippet text for richer evaluation context
        ctx = [c.snippet for c in rag_response.citations] if hasattr(rag_response, "citations") else []

        if not query_text:
            sess_q = get_session()
            try:
                r = sess_q.query(RagQuery).filter_by(id=query_id).first()
                query_text = r.query if r else ""
            finally:
                sess_q.close()

        # For Contextual Precision / Recall we need an expected_output.
        # Without a ground-truth dataset we use the model's own answer as a proxy —
        # this is the standard approach for live RAG evaluations where ground truth
        # is unavailable.
        tc = LLMTestCase(
            input=query_text,
            actual_output=answer,
            expected_output=answer,      # proxy for precision/recall when no ground truth
            retrieval_context=ctx,
        )

        metrics_to_run = [
            ("answer_relevancy",     AnswerRelevancyMetric(    threshold=0.5, model=judge, include_reason=True)),
            ("faithfulness",         FaithfulnessMetric(       threshold=0.5, model=judge, include_reason=True)),
            ("contextual_relevancy", ContextualRelevancyMetric(threshold=0.5, model=judge, include_reason=True)),
            ("contextual_precision", ContextualPrecisionMetric(threshold=0.5, model=judge, include_reason=True)),
            ("contextual_recall",    ContextualRecallMetric(   threshold=0.5, model=judge, include_reason=True)),
        ]

        scores: dict = {}
        for name, metric in metrics_to_run:
            for attempt in range(2):
                try:
                    await metric.a_measure(tc)
                    scores[name] = {
                        "score": float(metric.score or 0),
                        "passed": bool(metric.is_successful()),
                        "reason": metric.reason or "",
                    }
                    _rl.info("RAG eval [%s] %s=%.2f", query_id, name, metric.score or 0)
                    break
                except Exception as me:
                    if attempt == 1:
                        _rl.warning("RAG eval [%s] metric '%s' failed after 2 attempts: %s", query_id, name, me)
                        scores[name] = {"score": 0.0, "passed": False, "reason": f"ERROR: {me}"}

        sess2 = get_session()
        try:
            q = sess2.query(RagQuery).filter_by(id=query_id).first()
            if q:
                q.eval_scores = scores
                q.eval_status = "done"
            sess2.commit()
        finally:
            sess2.close()

    except Exception as e:
        _rl.error("RAG eval failed for %s: %s", query_id, e, exc_info=True)
        sess3 = get_session()
        try:
            q = sess3.query(RagQuery).filter_by(id=query_id).first()
            if q:
                q.eval_status = "error"
                q.eval_error = str(e)
            sess3.commit()
        finally:
            sess3.close()


def _query_to_dict(r: RagQuery) -> dict:
    return {
        "id": r.id,
        "query": r.query,
        "answer": r.answer,
        "citations": r.citations or [],
        "strategy_used": r.strategy_used,
        "chunks_retrieved": r.chunks_retrieved,
        "tokens_in": r.tokens_in,
        "tokens_out": r.tokens_out,
        "latency_ms": r.latency_ms,
        "eval_scores": r.eval_scores,
        "eval_status": r.eval_status or "pending",
        "eval_error": r.eval_error,
        "trace_id": r.trace_id,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


@app.get("/api/rag/configs/{config_id}/history")
def get_rag_history(config_id: str, limit: int = 200):
    """Return past queries for a RAG config, oldest first (for chat replay)."""
    session = get_session()
    try:
        rows = (
            session.query(RagQuery)
            .filter_by(config_id=config_id)
            .order_by(RagQuery.created_at.asc())
            .limit(limit)
            .all()
        )
        return [_query_to_dict(r) for r in rows]
    finally:
        session.close()


@app.delete("/api/rag/configs/{config_id}/history", status_code=204)
def clear_rag_history(config_id: str):
    """Delete all chat history for a RAG pipeline."""
    session = get_session()
    try:
        session.query(RagQuery).filter_by(config_id=config_id).delete()
        session.commit()
    finally:
        session.close()


@app.delete("/api/rag/queries/{query_id}", status_code=204)
def delete_rag_query(query_id: str):
    """Delete a single RAG query from history."""
    session = get_session()
    try:
        q = session.query(RagQuery).filter_by(id=query_id).first()
        if q:
            session.delete(q)
            session.commit()
    finally:
        session.close()


@app.get("/api/rag/queries/{query_id}")
def get_rag_query(query_id: str):
    """Get a single RAG query with its evaluation scores."""
    session = get_session()
    try:
        q = session.query(RagQuery).filter_by(id=query_id).first()
        if not q:
            raise HTTPException(404, "Query not found")
        return _query_to_dict(q)
    finally:
        session.close()


@app.get("/api/rag/traces/{trace_id}")
def get_rag_trace(trace_id: str):
    """Return OTel spans for a RAG query identified by trace_id."""
    session = get_session()
    try:
        spans = session.query(Span).filter(Span.trace_id == trace_id).order_by(Span.start_time).all()
        return [
            {
                "span_id": s.span_id,
                "parent_span_id": s.parent_span_id,
                "name": s.name,
                "span_type": s.span_type,
                "start_time": s.start_time.isoformat() if s.start_time else None,
                "end_time": s.end_time.isoformat() if s.end_time else None,
                "duration_ms": s.duration_ms,
                "status": s.status,
                "model": s.model,
                "tokens_in": s.tokens_in,
                "tokens_out": s.tokens_out,
                "cost": s.cost,
                "attributes": s.attributes,
                "error": s.error,
            }
            for s in spans
        ]
    finally:
        session.close()


@app.post("/api/rag/queries/{query_id}/evaluate")
async def evaluate_rag_query(query_id: str):
    """Trigger DeepEval evaluation for a stored query (re-run or first run)."""
    session = get_session()
    try:
        q = session.query(RagQuery).filter_by(id=query_id).first()
        if not q:
            raise HTTPException(404, "Query not found")
        cfg_row = session.query(RagConfigModel).filter_by(id=q.config_id).first()
        if not cfg_row:
            raise HTTPException(404, "RAG config not found")
        cfg_config = _cfg_to_pipeline_config(cfg_row)
        q_data = _query_to_dict(q)
    finally:
        session.close()

    # Build a mock RAGResponse from stored data
    from types import SimpleNamespace
    from src.rag.pipeline import Citation
    mock_resp = SimpleNamespace(
        query=q_data["query"],
        answer=q_data["answer"],
        citations=[
            Citation(
                source=c.get("source", ""),
                chunk_index=c.get("chunk_index", 0),
                total_chunks=c.get("total_chunks", 1),
                page=c.get("page"),
                score=c.get("score", 0.0),
                snippet=c.get("snippet", ""),
            )
            for c in (q_data["citations"] or [])
        ],
    )
    asyncio.create_task(_run_rag_eval_for_query(query_id, cfg_config, mock_resp))
    return {"status": "evaluating", "query_id": query_id}


@app.post("/api/rag/evaluate")
async def evaluate_rag(body: RagEvalRequest):
    """Run DeepEval RAG metrics on a set of question-answer samples."""
    session = get_session()
    try:
        cfg_row = session.query(RagConfigModel).filter_by(id=body.config_id).first()
        if not cfg_row:
            raise HTTPException(404, "RAG config not found")
        cfg_config = _cfg_to_pipeline_config(cfg_row)
    finally:
        session.close()

    from src.rag.pipeline import get_pipeline
    from src.rag.evaluation import RAGEvalSample, batch_evaluate

    pipeline = get_pipeline(cfg_config)
    samples = [
        RAGEvalSample(
            query=s["query"],
            expected_answer=s.get("expected_answer", ""),
            actual_answer=s.get("actual_answer", ""),
        )
        for s in body.samples
    ]
    results = await batch_evaluate(pipeline, samples, thresholds=body.thresholds)
    return [
        {
            "query": r.sample.query,
            "actual_answer": r.sample.actual_answer,
            "expected_answer": r.sample.expected_answer,
            "retrieved_contexts": r.sample.retrieved_contexts,
            "overall_pass": r.overall_pass,
            "avg_score": r.avg_score,
            "latency_ms": r.latency_ms,
            "error": r.error,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "passed": m.passed,
                    "threshold": m.threshold,
                    "reason": m.reason,
                }
                for m in r.metrics
            ],
        }
        for r in results
    ]


@app.get("/api/rag/stats")
def rag_stats(days: int = 30):
    """Aggregate stats for all RAG queries: latency, tokens, eval scores."""
    from datetime import timedelta as _td
    session = get_session()
    try:
        cutoff = datetime.utcnow() - _td(days=days)
        queries = session.query(RagQuery).filter(RagQuery.created_at >= cutoff).all()
        if not queries:
            return {
                "total_queries": 0, "queries_with_eval": 0, "pending_eval": 0,
                "avg_latency_ms": 0, "avg_tokens": 0, "avg_scores": {},
                "strategy_breakdown": {}, "daily": [],
            }

        total = len(queries)
        done = [q for q in queries if q.eval_status == "done" and q.eval_scores]
        pending = sum(1 for q in queries if q.eval_status in ("pending", "running"))

        latencies = [q.latency_ms for q in queries if q.latency_ms]
        tokens = [(q.tokens_in or 0) + (q.tokens_out or 0) for q in queries]

        # Average per-metric DeepEval scores across all evaluated queries
        metric_sums: dict = {}
        metric_counts: dict = {}
        for q in done:
            for k, v in q.eval_scores.items():
                score = v.get("score", 0) if isinstance(v, dict) else 0
                metric_sums[k] = metric_sums.get(k, 0) + score
                metric_counts[k] = metric_counts.get(k, 0) + 1
        avg_scores = {k: round(metric_sums[k] / metric_counts[k], 3) for k in metric_sums}

        # Strategy breakdown
        strategy_cnt: dict = {}
        for q in queries:
            s = q.strategy_used or "similarity"
            strategy_cnt[s] = strategy_cnt.get(s, 0) + 1

        # Daily buckets (last 30 days)
        daily: dict = {}
        for q in queries:
            day = q.created_at.strftime("%Y-%m-%d") if q.created_at else "unknown"
            daily.setdefault(day, {"queries": 0, "latency_sum": 0, "tokens": 0})
            daily[day]["queries"] += 1
            daily[day]["latency_sum"] += q.latency_ms or 0
            daily[day]["tokens"] += (q.tokens_in or 0) + (q.tokens_out or 0)

        daily_list = sorted([
            {"date": d, "queries": v["queries"],
             "avg_latency": round(v["latency_sum"] / v["queries"], 1) if v["queries"] else 0,
             "tokens": v["tokens"]}
            for d, v in daily.items()
        ], key=lambda x: x["date"])

        return {
            "total_queries": total,
            "queries_with_eval": len(done),
            "pending_eval": pending,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "avg_tokens": round(sum(tokens) / len(tokens)) if tokens else 0,
            "avg_scores": avg_scores,
            "strategy_breakdown": strategy_cnt,
            "daily": daily_list,
        }
    finally:
        session.close()


@app.get("/api/rag/queries")
def list_rag_queries(days: int = 30, limit: int = 100):
    """Return recent RAG queries across all configs (for evaluation page)."""
    from datetime import timedelta as _td
    session = get_session()
    try:
        cutoff = datetime.utcnow() - _td(days=days)
        rows = (
            session.query(RagQuery)
            .filter(RagQuery.created_at >= cutoff)
            .order_by(RagQuery.created_at.desc())
            .limit(limit)
            .all()
        )
        return [_query_to_dict(r) for r in rows]
    finally:
        session.close()


if __name__ == "__main__":
    import uvicorn
    # Only watch actual source code dirs — exclude workspace dirs where agents write files.
    # This prevents the server from reloading (and losing checkpoints) when the coder
    # agent writes files to tests/, src/agents/, etc.
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src", "."],   # watch both src/ and server.py itself
        reload_includes=["*.py"],
        reload_excludes=[
            "tests/*", "*.pyc", "__pycache__/*",
            "*.db", "*.sqlite", "*.log",
            "frontend/*", "node_modules/*",
        ],
    )
