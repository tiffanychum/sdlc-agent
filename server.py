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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.db.database import init_db, seed_defaults, get_session
from src.db.models import (
    Team, Agent, AgentToolMapping, Skill, AgentSkillMapping,
    Trace, Span, EvalRun, GoldenTestCase, RegressionResult, PromptVersion,
)
from src.orchestrator import build_orchestrator_from_team, _extract_text, get_graph_config
from src.tracing.collector import TraceCollector


orchestrators: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed_defaults()
    from src.evaluation.golden import sync_golden_to_db
    sync_golden_to_db()
    orchestrators["default"] = await build_orchestrator_from_team("default")
    yield


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
    baseline_run_id: Optional[str] = None


class RCARequest(BaseModel):
    run_id: str
    case_id: str
    baseline_run_id: Optional[str] = None


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
    {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "OpenAI", "tier": "router"},
    {"id": "gpt-4o", "name": "GPT-4o", "provider": "OpenAI", "tier": "agent"},
    {"id": "gpt-5-mini", "name": "GPT-5 Mini", "provider": "OpenAI", "tier": "agent"},
    {"id": "gpt-5-mini-2025-08-07", "name": "GPT-5 Mini (2025-08-07)", "provider": "OpenAI", "tier": "agent"},
    {"id": "gpt-5.3-codex", "name": "GPT-5.3 Codex", "provider": "OpenAI", "tier": "agent"},
    {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "provider": "Anthropic", "tier": "agent"},
    {"id": "claude-sonnet-4.6", "name": "Claude Sonnet 4.6", "provider": "Anthropic", "tier": "agent"},
    {"id": "deepseek-r1", "name": "DeepSeek R1", "provider": "DeepSeek", "tier": "judge/rca"},
    {"id": "gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash Lite", "provider": "Google", "tier": "agent"},
    {"id": "gemini-3-flash", "name": "Gemini 3 Flash", "provider": "Google", "tier": "agent"},
    {"id": "llama-3.1-8b-cs", "name": "Llama 3.1 8B", "provider": "Meta", "tier": "agent"},
    {"id": "mistral-small-3", "name": "Mistral Small 3", "provider": "Mistral", "tier": "agent"},
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
            agent_span = collector.start_span(f"agent:{agent_name}", "agent_execution",
                                              input_data={"agent": agent_name})
            for tc in entry.get("tool_calls", []):
                tool_calls.append({**tc, "agent": agent_name})
                span_id = collector.start_span(f"tool:{tc['tool']}", "tool_call",
                                               input_data={"args": str(tc.get("args", {}))[:200], "agent": agent_name})
                collector.end_span(span_id, output_data={"result": "completed"})
            collector.end_span(agent_span, output_data={"tool_calls": len(entry.get("tool_calls", []))},
                               model=llm_model, tokens_in=total_prompt_tokens, tokens_out=total_completion_tokens)
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

    # Save full agent_trace + quick rule-based eval
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
            tr.agent_used = agents_label
            tr.agent_response = response[:2000]
            tr.tool_calls_json = agent_trace
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
    """
    if team_id not in orchestrators:
        orchestrators[team_id] = await build_orchestrator_from_team(team_id)

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

        current_agent = ""
        try:
            async for event in graph.astream_events(initial_input, config=config, version="v2"):
                kind = event.get("event", "")
                name = event.get("name", "")
                tags = event.get("tags", [])
                data = event.get("data", {})

                if kind == "on_chain_start" and "agent" in name.lower():
                    agent_name = name.replace("agent:", "").replace("Agent", "").strip() or name
                    current_agent = agent_name
                    yield _sse_event("agent_start", {"agent": agent_name})

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

                elif kind == "on_chain_end" and "agent" in name.lower():
                    yield _sse_event("agent_end", {"agent": current_agent})

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

                if kind == "on_chain_start" and "agent" in name.lower():
                    current_agent = name.replace("agent:", "").strip() or name
                    yield _sse_event("agent_start", {"agent": current_agent})
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
                elif kind == "on_chain_end" and "agent" in name.lower():
                    yield _sse_event("agent_end", {"agent": current_agent})

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


# ── Prompt Versions API ────────────────────────────────────────

@app.get("/api/prompt-versions")
def list_prompt_versions():
    session = get_session()
    pvs = session.query(PromptVersion).order_by(PromptVersion.created_at.desc()).all()
    result = [{
        "id": p.id, "version_label": p.version_label,
        "description": p.description,
        "agent_prompts": p.agent_prompts or {},
        "team_strategy": p.team_strategy,
        "is_active": p.is_active,
        "created_at": p.created_at.isoformat() if p.created_at else None,
    } for p in pvs]
    session.close()
    return result


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
        },
    }
    session.close()
    return result


@app.get("/api/regression/diff/{run_a}/{run_b}/{case_id}")
def regression_trace_diff(run_a: str, run_b: str, case_id: str):
    """Side-by-side trace comparison for a specific golden test case across two runs."""
    session = get_session()
    r_a = session.query(RegressionResult).filter_by(run_id=run_a, golden_case_id=case_id).first()
    r_b = session.query(RegressionResult).filter_by(run_id=run_b, golden_case_id=case_id).first()
    session.close()
    if not r_a or not r_b:
        raise HTTPException(404, "One or both case results not found")

    from src.evaluation.rca import RootCauseAnalyzer
    analyzer = RootCauseAnalyzer()
    trace_diff = analyzer._compute_trace_diff(
        r_b.full_trace or [],
        r_a.full_trace or [],
    )
    cost_diff = analyzer._compute_cost_diff(
        {"actual_tokens_in": r_b.actual_tokens_in, "actual_tokens_out": r_b.actual_tokens_out,
         "actual_cost": r_b.actual_cost, "actual_latency_ms": r_b.actual_latency_ms,
         "actual_llm_calls": r_b.actual_llm_calls, "actual_tool_calls": r_b.actual_tool_calls},
        {"actual_tokens_in": r_a.actual_tokens_in, "actual_tokens_out": r_a.actual_tokens_out,
         "actual_cost": r_a.actual_cost, "actual_latency_ms": r_a.actual_latency_ms,
         "actual_llm_calls": r_a.actual_llm_calls, "actual_tool_calls": r_a.actual_tool_calls},
    )

    def _result_dict(r):
        return {
            "actual_output": r.actual_output, "actual_agent": r.actual_agent,
            "actual_tools": r.actual_tools or [], "actual_delegation_pattern": r.actual_delegation_pattern or [],
            "actual_llm_calls": r.actual_llm_calls, "actual_tool_calls": r.actual_tool_calls,
            "actual_tokens_in": r.actual_tokens_in, "actual_tokens_out": r.actual_tokens_out,
            "actual_latency_ms": r.actual_latency_ms, "actual_cost": r.actual_cost,
            "semantic_similarity": r.semantic_similarity,
            "quality_scores": r.quality_scores or {},
            "deepeval_scores": r.deepeval_scores or {},
            "trace_assertions": r.trace_assertions or {},
            "eval_reasoning": r.eval_reasoning or {},
            "overall_pass": r.overall_pass,
            "model_used": r.model_used,
        }

    return {
        "case_id": case_id,
        "run_a": {"id": run_a, **_result_dict(r_a)},
        "run_b": {"id": run_b, **_result_dict(r_b)},
        "trace_diff": trace_diff,
        "cost_diff": cost_diff,
    }


@app.post("/api/regression/rca")
async def run_rca(request: RCARequest):
    session = get_session()
    result = session.query(RegressionResult).filter_by(
        run_id=request.run_id, golden_case_id=request.case_id
    ).first()
    if not result:
        session.close()
        raise HTTPException(404, "Case result not found")

    failing = {
        "golden_case_id": result.golden_case_id,
        "golden_case_name": result.golden_case_name,
        "prompt": "",
        "actual_output": result.actual_output,
        "actual_agent": result.actual_agent,
        "actual_tools": result.actual_tools or [],
        "actual_tokens_in": result.actual_tokens_in,
        "actual_tokens_out": result.actual_tokens_out,
        "actual_cost": result.actual_cost,
        "actual_latency_ms": result.actual_latency_ms,
        "actual_llm_calls": result.actual_llm_calls,
        "actual_tool_calls": result.actual_tool_calls,
        "quality_scores": result.quality_scores or {},
        "trace_assertions": result.trace_assertions or {},
        "full_trace": result.full_trace or [],
        "model_used": result.model_used,
    }

    golden = session.query(GoldenTestCase).filter_by(id=request.case_id).first()
    if golden:
        failing["prompt"] = golden.prompt
        failing["expected_agent"] = golden.expected_agent
        failing["expected_tools"] = golden.expected_tools or []

    baseline = None
    if request.baseline_run_id:
        b = session.query(RegressionResult).filter_by(
            run_id=request.baseline_run_id, golden_case_id=request.case_id
        ).first()
        if b:
            baseline = {
                "actual_output": b.actual_output, "actual_agent": b.actual_agent,
                "actual_tools": b.actual_tools or [],
                "actual_tokens_in": b.actual_tokens_in, "actual_tokens_out": b.actual_tokens_out,
                "actual_cost": b.actual_cost, "actual_latency_ms": b.actual_latency_ms,
                "actual_llm_calls": b.actual_llm_calls, "actual_tool_calls": b.actual_tool_calls,
                "quality_scores": b.quality_scores or {},
                "trace_assertions": b.trace_assertions or {},
                "full_trace": b.full_trace or [],
            }

    session.close()

    from src.evaluation.rca import RootCauseAnalyzer
    analyzer = RootCauseAnalyzer()
    rca_result = await analyzer.analyze(failing, baseline)

    session = get_session()
    r = session.query(RegressionResult).filter_by(
        run_id=request.run_id, golden_case_id=request.case_id
    ).first()
    if r:
        r.rca_analysis = rca_result
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(r, "rca_analysis")
        session.commit()
    session.close()

    return rca_result


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
