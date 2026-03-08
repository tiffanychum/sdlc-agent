"""
FastAPI server for the SDLC Agent.

Provides REST and WebSocket endpoints for:
- Chat with the multi-agent system
- Agent configuration listing
- Evaluation pipeline execution
- MCP server health checks
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.orchestrator import build_orchestrator
from src.agents.definitions import AGENT_CONFIGS
from src.evaluation.evaluator import AgentEvaluator


orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    orchestrator = await build_orchestrator()
    yield


app = FastAPI(
    title="SDLC Agent API",
    description="General-purpose coding agent with MCP tool integration",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    agent_used: str
    tool_calls: list[dict]


class EvalResponse(BaseModel):
    summary: dict
    num_tasks: int


@app.get("/api/agents")
async def list_agents():
    return {
        agent_id: {
            "name": cfg.name,
            "role": cfg.role,
            "description": cfg.description,
            "tool_groups": cfg.tool_groups,
        }
        for agent_id, cfg in AGENT_CONFIGS.items()
    }


@app.get("/api/health/mcp")
async def mcp_health():
    from src.mcp_servers.filesystem_server import list_tools as fs_list
    from src.mcp_servers.shell_server import list_tools as shell_list
    from src.mcp_servers.git_server import list_tools as git_list
    from src.mcp_servers.web_server import list_tools as web_list

    servers = {}
    for name, list_fn in [
        ("filesystem", fs_list), ("shell", shell_list),
        ("git", git_list), ("web", web_list),
    ]:
        try:
            tools = await list_fn()
            servers[name] = {"status": "healthy", "tools": len(tools)}
        except Exception as e:
            servers[name] = {"status": "unhealthy", "error": str(e)}

    return {"servers": servers}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = await orchestrator.ainvoke({
        "messages": [{"role": "user", "content": request.message}],
        "selected_agent": "",
        "agent_trace": [],
    })

    agent_used = result.get("selected_agent", "unknown")
    trace = result.get("agent_trace", [])
    tool_calls = []
    for entry in trace:
        if entry.get("step") == "execution":
            tool_calls = entry.get("tool_calls", [])

    last_msg = result["messages"][-1]
    response = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    return ChatResponse(response=response, agent_used=agent_used, tool_calls=tool_calls)


@app.post("/api/eval", response_model=EvalResponse)
async def run_evaluation():
    evaluator = AgentEvaluator()
    run = await evaluator.run_evaluation()
    return EvalResponse(summary=run.summary(), num_tasks=len(run.tasks))


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")

            result = await orchestrator.ainvoke({
                "messages": [{"role": "user", "content": message}],
                "selected_agent": "",
                "agent_trace": [],
            })

            agent_used = result.get("selected_agent", "unknown")
            trace = result.get("agent_trace", [])
            tool_calls = []
            for entry in trace:
                if entry.get("step") == "execution":
                    tool_calls = entry.get("tool_calls", [])

            for tc in tool_calls:
                await websocket.send_json({
                    "type": "tool_call",
                    "tool": tc["tool"],
                    "args": tc.get("args", {}),
                })

            last_msg = result["messages"][-1]
            response = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

            await websocket.send_json({
                "type": "response",
                "content": response,
                "agent_used": agent_used,
            })

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
