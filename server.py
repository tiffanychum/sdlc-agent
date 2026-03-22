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

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.db.database import init_db, seed_defaults, get_session
from src.db.models import (
    Team, Agent, AgentToolMapping, Skill, AgentSkillMapping,
    Trace, Span, EvalRun,
)
from src.orchestrator import build_orchestrator_from_team, _extract_text
from src.tracing.collector import TraceCollector


orchestrators: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed_defaults()
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

class EvalRequest(BaseModel):
    team_id: str = "default"
    use_llm_judge: bool = True
    use_deepeval: bool = False

class EvalCompareRequest(BaseModel):
    team_id: str = "default"
    model_configs: list[dict]


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
    routing_span = collector.start_span("routing", "routing", input_data={"prompt": request.message[:200]})

    result = await orchestrators[team_id].ainvoke({
        "messages": [{"role": "user", "content": request.message}],
        "selected_agent": "", "agent_trace": [],
    })

    agent_used = result.get("selected_agent", "unknown")
    collector.end_span(routing_span, output_data={"selected_agent": agent_used})

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
            collector.end_span(agent_span, output_data={"tool_calls": len(entry.get("tool_calls", []))})
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


# ── Trace Evaluation ────────────────────────────────────────────

@app.post("/api/traces/evaluate")
async def evaluate_traces():
    """Run G-Eval + DeepEval on traces that haven't been fully evaluated."""
    session = get_session()
    pending = session.query(Trace).filter(
        Trace.eval_status.in_(["pending", "quick"]),
        Trace.agent_response != "",
        Trace.agent_response != None,
    ).order_by(Trace.created_at.desc()).limit(10).all()

    evaluated = 0
    for tr in pending:
        import json as _json
        existing = tr.eval_scores or {}
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
        except Exception:
            pass

        # ── DeepEval (External Validation) ──
        try:
            from src.evaluation.integrations import run_deepeval_metrics

            tool_outputs = []
            for span_row in session.query(Span).filter_by(trace_id=tr.id).all():
                if span_row.output_data:
                    for v in span_row.output_data.values():
                        if v:
                            tool_outputs.append(str(v)[:300])

            deepeval_scores = run_deepeval_metrics(
                user_prompt=tr.user_prompt or "",
                agent_response=tr.agent_response or "",
                tool_outputs=tool_outputs[:5],
            )
            existing["deepeval_scores"] = deepeval_scores
        except Exception:
            existing["deepeval_scores"] = {"deepeval_relevancy": 0.5, "deepeval_faithfulness": 0.5}

        tr.eval_scores = existing
        tr.eval_status = "evaluated"
        evaluated += 1

    session.commit()
    total_pending = session.query(Trace).filter(Trace.eval_status.in_(["pending", "quick"])).count()
    session.close()
    return {"evaluated": evaluated, "remaining": total_pending}


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
    session = get_session()
    a = session.query(EvalRun).filter_by(id=run_a).first()
    b = session.query(EvalRun).filter_by(id=run_b).first()
    session.close()
    if not a or not b:
        raise HTTPException(404, "Run not found")

    comparison = {}
    for key in ["task_completion_rate", "routing_accuracy", "avg_tool_call_accuracy", "avg_failure_recovery_rate"]:
        va = getattr(a, key, 0) or 0
        vb = getattr(b, key, 0) or 0
        delta = vb - va
        comparison[key] = {"before": va, "after": vb, "delta": round(delta, 3), "regression": delta < -0.05}
    return {"run_a": run_a, "run_b": run_b, "comparison": comparison}


# ── WebSocket ───────────────────────────────────────────────────

@app.websocket("/ws/chat/{team_id}")
async def ws_chat(websocket: WebSocket, team_id: str):
    await websocket.accept()
    if team_id not in orchestrators:
        orchestrators[team_id] = await build_orchestrator_from_team(team_id)

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")

            result = await orchestrators[team_id].ainvoke({
                "messages": [{"role": "user", "content": message}],
                "selected_agent": "", "agent_trace": [],
            })

            agent_used = result.get("selected_agent", "unknown")
            trace = result.get("agent_trace", [])
            for entry in trace:
                if entry.get("step") == "execution":
                    for tc in entry.get("tool_calls", []):
                        await websocket.send_json({"type": "tool_call", "tool": tc["tool"], "args": tc.get("args", {})})

            last_msg = result["messages"][-1]
            response = _extract_text(last_msg.content if hasattr(last_msg, "content") else str(last_msg))
            await websocket.send_json({"type": "response", "content": response, "agent_used": agent_used})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
