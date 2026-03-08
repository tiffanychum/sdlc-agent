# SDLC Agent — General-Purpose Coding Assistant

A multi-agent coding assistant (similar to [Manus](https://manus.im) and [Claude Code](https://claude.ai)) that can read/write code, run commands, search the web, and manage git — all through natural language. Built with **LangGraph** for agent orchestration and **MCP (Model Context Protocol)** for standardized agent-to-tool communication.

## What It Does

Ask it anything a developer would do:

```
You: "Read main.py and explain the architecture"
→ Coder Agent reads the file via Filesystem MCP → explains the code

You: "Run the tests and tell me what's failing"
→ Runner Agent executes pytest via Shell MCP → reports failures

You: "How do I use LangGraph's create_react_agent?"
→ Researcher Agent searches the web via Web MCP → fetches docs → summarizes

You: "Create a new branch, add a docstring to config.py, and commit"
→ Coder Agent uses Git MCP + Filesystem MCP → branch, edit, commit
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      User Request                             │
│            "Run the tests and fix any failures"               │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                     Router Agent (LLM)                        │
│                                                               │
│   Classifies request → selects the best agent                 │
│   "This involves running tests → route to Runner"             │
└──────────┬─────────────────┬─────────────────┬───────────────┘
           │                 │                 │
           ▼                 ▼                 ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────────┐
│  Coder Agent   │ │  Runner Agent  │ │ Researcher Agent   │
│                │ │                │ │                    │
│ Read code      │ │ Run commands   │ │ Search the web     │
│ Write code     │ │ Execute tests  │ │ Fetch docs         │
│ Edit files     │ │ Build projects │ │ Look up errors     │
│ Search code    │ │ Run scripts    │ │ Read API refs      │
│ Git operations │ │                │ │                    │
└───────┬────────┘ └───────┬────────┘ └──────────┬─────────┘
        │                  │                      │
        ▼                  ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│                  MCP Communication Layer                      │
│                                                               │
│  Standardized tool discovery, schema validation,              │
│  invocation, error handling, and observability                 │
└──────┬──────────────┬──────────────┬──────────────┬──────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│ Filesystem │ │   Shell    │ │    Git     │ │    Web     │
│ MCP Server │ │ MCP Server │ │ MCP Server │ │ MCP Server │
│            │ │            │ │            │ │            │
│ read_file  │ │ run_command│ │ git_status │ │ fetch_url  │
│ write_file │ │ run_script │ │ git_diff   │ │ web_search │
│ edit_file  │ │ run_tests  │ │ git_log    │ │ check_url  │
│ search     │ │            │ │ git_commit │ │            │
│ find       │ │            │ │ git_branch │ │            │
│ list_dir   │ │            │ │ git_show   │ │            │
└────────────┘ └────────────┘ └────────────┘ └────────────┘
```

### How It Works — Step by Step

1. **User sends a request** (e.g., "Find the bug in auth.py and fix it")
2. **Router Agent** — A lightweight LLM call classifies the request and picks the best agent
3. **Specialized Agent** — Uses the ReAct pattern (Reasoning + Acting) via LangGraph to decide which tools to call
4. **MCP Layer** — Handles tool discovery (what tools exist?), schema validation (are the args correct?), invocation (execute the tool), and error handling (what if it fails?)
5. **MCP Servers** — Execute the actual operations against the filesystem, shell, git, or web
6. **Agent synthesizes** the tool results into a coherent response
7. **Trace is recorded** — Every routing decision and tool call is logged for the evaluation pipeline

### Why MCP (Model Context Protocol)?

[MCP](https://modelcontextprotocol.io/) is the open standard (created by Anthropic) for connecting AI agents to external tools. Instead of hardcoding tool integrations, MCP gives us:

| Benefit | How |
|---------|-----|
| **Tool Discovery** | Agents dynamically discover available tools and their schemas at runtime |
| **Standardized Interface** | Same request/response format for all tools, regardless of what they do |
| **Error Propagation** | Consistent error handling from tools back to agents |
| **Observability** | Every tool call is recorded with inputs, outputs, and success/failure |
| **Extensibility** | Add a new tool by deploying a new MCP server — zero agent code changes |

This is the same protocol used by Claude, Cursor, and other AI coding tools.

## Project Structure

```
sdlc-agent/
├── main.py                          # CLI: chat, eval, health check
├── server.py                        # FastAPI server: REST + WebSocket
├── requirements.txt
├── pyproject.toml
│
├── src/
│   ├── config.py                    # Environment + app configuration
│   │
│   ├── llm/
│   │   └── client.py                # LLM client (OpenAI-compatible API)
│   │
│   ├── agents/
│   │   ├── prompts.py               # System prompts per agent role
│   │   └── definitions.py           # Agent configs + factory functions
│   │
│   ├── tools/
│   │   └── registry.py              # MCP tools → LangChain bridge
│   │
│   ├── mcp_servers/
│   │   ├── filesystem_server.py     # Read, write, edit, search files
│   │   ├── shell_server.py          # Execute commands, run tests
│   │   ├── git_server.py            # Git status, diff, commit, branch
│   │   └── web_server.py            # Fetch URLs, web search
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # Metric definitions
│   │   ├── scenarios.py             # Test scenarios
│   │   ├── evaluator.py             # Evaluation pipeline runner
│   │   └── reporter.py              # Rich console reports
│   │
│   └── orchestrator.py              # LangGraph multi-agent router
│
├── tests/
│   ├── test_mcp_communication.py    # E2E tests for MCP layer
│   └── test_evaluation.py           # Unit tests for eval pipeline
│
└── eval/
    └── results/                     # Persisted evaluation results
```

## MCP Servers — The Tool Layer

### Filesystem MCP Server
The core tool for any coding agent:

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with optional line range |
| `write_file` | Create or overwrite files |
| `edit_file` | Precise string replacement (like sed) — rejects ambiguous matches |
| `list_directory` | List directory contents (recursive optional) |
| `search_files` | Grep-like pattern search across codebase |
| `find_files` | Find files by glob pattern |

**Safety**: All paths are sandboxed to the workspace root. Directory traversal (`../../etc/passwd`) is blocked.

### Shell MCP Server
Command execution for builds, tests, and system operations:

| Tool | Description |
|------|-------------|
| `run_command` | Execute any shell command with timeout enforcement |
| `run_script` | Run a Python script file |
| `run_tests` | Execute pytest with verbose output and pattern matching |

**Safety**: Destructive commands (`rm -rf /`, `mkfs`, etc.) are blocked. Timeout enforcement prevents hangs.

### Git MCP Server
Version control for code changes:

| Tool | Description |
|------|-------------|
| `git_status` | Show modified, staged, and untracked files |
| `git_diff` | View changes (staged/unstaged, between commits) |
| `git_log` | Commit history (oneline or detailed) |
| `git_commit` | Stage and commit changes |
| `git_branch` | List, create, or switch branches |
| `git_show` | View commit details |

### Web MCP Server
Information gathering from the internet:

| Tool | Description |
|------|-------------|
| `fetch_url` | Read web pages as clean text (HTML → text conversion) |
| `web_search` | Search the web (DuckDuckGo, swap with Google/Brave API) |
| `check_url` | Verify URL reachability and headers |

## Evaluation Pipeline

### What It Measures

| Metric | Description |
|--------|-------------|
| **Task Completion Rate** | % of scenarios completed successfully |
| **Routing Accuracy** | % of requests routed to the correct agent |
| **Tool-Call Accuracy** | % of tool invocations matching expected tools |
| **Failure Recovery Rate** | % of errors where the agent recovered gracefully |
| **Latency** | End-to-end time per task |

### How It Works

```python
from src.evaluation.evaluator import AgentEvaluator

# Run evaluation
evaluator = AgentEvaluator(model="gpt-5.3-codex", prompt_version="v1")
run = await evaluator.run_evaluation()

print(run.summary())
# {
#   "task_completion_rate": 0.818,
#   "routing_accuracy": 0.909,
#   "avg_tool_call_accuracy": 0.85,
#   "avg_failure_recovery_rate": 1.0,
#   "avg_latency_ms": 1250.3
# }
```

### Regression Detection

Compare runs before/after prompt or model changes:

```python
# Baseline
run_v1 = await evaluator.run_evaluation()

# After modifying prompts
evaluator_v2 = AgentEvaluator(prompt_version="v2")
run_v2 = await evaluator_v2.run_evaluation()

# Compare — flags any metric that dropped >5%
comparison = AgentEvaluator.compare_runs(run_v1, run_v2)
# {"task_completion_rate": {"before": 0.82, "after": 0.73, "regression": True}}
```

### Evaluation Scenarios

11 scenarios covering:
- **Routing**: Does the router pick the right agent?
- **Tool selection**: Does the agent call the right tools?
- **Multi-step**: Can the agent chain operations (search → read → edit)?
- **Failure recovery**: Does the agent handle errors gracefully?
- **Cross-server**: Workflows spanning filesystem + shell + git

## Testing

### Run All Tests

```bash
# Full test suite
pytest tests/ -v

# MCP communication E2E tests only
pytest tests/test_mcp_communication.py -v

# Evaluation pipeline unit tests only
pytest tests/test_evaluation.py -v
```

### Test Coverage

| Test Class | What It Validates |
|------------|-------------------|
| `TestFilesystemMCPServer` | File CRUD, search, edit precision, path security, observability |
| `TestShellMCPServer` | Command execution, stderr capture, timeouts, safety blocks |
| `TestWebMCPServer` | URL fetching, JSON handling, error recovery |
| `TestCrossServerWorkflows` | Multi-server flows: read → run, write → execute, search → read |
| `TestTaskMetrics` | Metric calculations: accuracy, recovery rate, latency |
| `TestEvalRunMetrics` | Aggregated run metrics, summary generation |
| `TestRegressionDetection` | Before/after comparison, regression flagging |
| `TestEvalScenarios` | Scenario well-formedness, uniqueness, coverage |

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI-compatible API key (Poe, OpenAI, etc.)

### Installation

```bash
cd sdlc-agent
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API key
```

### Usage

```bash
# Interactive chat
python main.py chat

# Run evaluation pipeline
python main.py eval

# MCP server health check
python main.py test-mcp

# Start REST API server
python server.py
# Docs at http://localhost:8000/docs
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/agents` | List available agents |
| `GET` | `/api/health/mcp` | MCP server health check |
| `POST` | `/api/chat` | Send a message |
| `POST` | `/api/eval` | Run evaluation pipeline |
| `WS` | `/ws/chat` | Real-time streaming chat |

## Design Decisions

### Why Multi-Agent over Single Agent?
Specialized agents with focused tool sets produce better results than one agent with 20+ tools. The router adds minimal latency (~200ms) but significantly improves tool selection accuracy — the coder never accidentally runs `rm` and the runner never tries to edit files.

### Why MCP over Direct Tool Calls?
MCP provides a standardized protocol that decouples agents from tools. Adding a new capability (e.g., Docker, database, Slack) requires only a new MCP server — the agents and orchestrator don't change. This is the same architecture used by production AI coding tools.

### Why LangGraph over LangChain AgentExecutor?
LangGraph provides explicit, debuggable control flow as a state graph. Each agent is a node with conditional routing edges. This makes the system testable, observable, and easy to extend with new agents.

### Why Evaluation Pipeline?
AI agents are non-deterministic. Without systematic evaluation, prompt changes or model swaps can silently degrade performance. The evaluation pipeline catches regressions before they reach users.

## License

MIT
