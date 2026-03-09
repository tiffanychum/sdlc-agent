# SDLC Agent — Multi-Agent Coding Platform

A full-stack multi-agent platform for automating software development workflows. Features a **5-agent team** with per-agent decision strategies, **MCP (Model Context Protocol)** for standardized tool integration, an **industry-standard evaluation pipeline** with LLM-as-Judge scoring, and a **Next.js dashboard** for configuration, monitoring, and observability.

Built with LangGraph, MCP, FastAPI, SQLite, DeepEval, Langfuse, and Next.js.

## Architecture

```
                         User Request
                              │
                              ▼
                    ┌───────────────────┐
                    │   Router (LLM)    │  Classifies intent,
                    │                   │  selects best agent
                    └─────────┬─────────┘
                              │
          ┌───────────┬───────┼───────┬────────────┐
          ▼           ▼       ▼       ▼            ▼
     ┌─────────┐ ┌────────┐ ┌────────┐ ┌─────────┐ ┌──────────┐
     │ Coder   │ │ Runner │ │Research│ │ Planner │ │ Reviewer │
     │ (ReAct) │ │(ReAct) │ │(ReAct) │ │(Plan+Ex)│ │(Reflexn) │
     └────┬────┘ └───┬────┘ └───┬────┘ └────┬────┘ └─────┬────┘
          │          │          │           │            │
          ▼          ▼          ▼           ▼            ▼
     ┌──────────────────────────────────────────────────────┐
     │              MCP Communication Layer                  │
     │  Tool discovery · Schema validation · Error handling  │
     └──┬─────────┬──────────┬──────────┬──────────┬───────┘
        ▼         ▼          ▼          ▼          ▼
   Filesystem   Shell       Git        Web      Memory
   (6 tools)  (3 tools)  (6 tools)  (3 tools)  (6 tools)
```

## How the Agent Team Works

### 5 Specialized Agents

Each agent has a specific role, decision strategy, MCP tool set, and skills:

| Agent | Strategy | Tools | Role |
|-------|----------|-------|------|
| **Coder** | ReAct | filesystem, git | Reads/writes code, navigates codebases, manages version control |
| **Runner** | ReAct | shell | Executes commands, runs tests, builds projects |
| **Researcher** | ReAct | web | Searches documentation, fetches API references, finds solutions |
| **Planner** | Plan-and-Execute | memory, filesystem | Creates structured task plans, tracks progress, stores context |
| **Reviewer** | Reflexion | filesystem, shell, git, memory | Reviews code quality, runs tests, reflects on correctness |

### Per-Agent Decision Strategies

Each agent uses the decision strategy best suited to its role:

| Strategy | How It Works | Best For |
|----------|-------------|----------|
| **ReAct** | Think → Act → Observe → Repeat | Quick, focused tasks (reading files, running commands) |
| **Plan-and-Execute** | Create a numbered plan first, then execute step by step | Complex multi-step tasks that need decomposition |
| **Reflexion** | After each action, reflect on whether it was correct | Quality-critical tasks (code review, verification) |
| **Chain-of-Thought** | Reason thoroughly before taking any action | Tasks requiring careful analysis |

### MCP vs Skills — Clear Separation

Following the [Claude Code principle](https://docs.anthropic.com/): **"MCP gives power. Skills control power."**

**MCP Tools** (what agents CAN do):
- 24 tools across 5 MCP servers (filesystem, shell, git, web, memory)
- Each tool has a JSON Schema defining its parameters
- Agents discover tools dynamically at runtime via the MCP protocol

**Skills** (HOW agents SHOULD behave):
- Instruction-only injections appended to the agent's system prompt
- No tools, no execution — just behavioral guidance
- 7 default skills: Code Review, Git Conventions, Error Recovery, Plan First, Security Awareness, Documentation Citation, Test-Driven Approach
- Skills can be assigned to specific agents and auto-triggered by prompt patterns

### Routing: How Requests Reach the Right Agent

1. User sends a request (e.g., "Create a plan for refactoring the config module")
2. A lightweight **Router LLM call** classifies the intent and selects the best agent
3. The selected agent runs with its assigned tools and strategy
4. Every step is traced (routing decision, tool calls, response) for evaluation

### LLM Configuration

The platform uses any **OpenAI-compatible API** as its LLM backbone. Each agent can use a different model:

```
# .env — Global default
LLM_BASE_URL=https://api.poe.com/v1
LLM_MODEL=gpt-5-nano

# Per-agent model selection (via Studio UI)
# Coder → gemini-2.5-flash-lite
# Runner → llama-3.1-8b-cs
# Researcher → gpt-4o-mini-search
```

Supported models: `gemini-2.5-flash-lite`, `llama-3.1-8b-cs`, `mistral-small-3`, `grok-4.1-fast-reasoning`, `claude-haiku-3`, `gpt-4o-mini-search`, or any model available through the configured API.

## Evaluation System

### 7 Industry-Standard Metrics

| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Task Success Rate** | Did the agent complete the task accurately? | Keyword matching against expected outputs |
| **Tool Accuracy** | Did it select the right tools with correct parameters? | Compare actual vs expected tool calls |
| **Reasoning Quality** | Were the intermediate steps coherent and logical? | Composite: routing correct + tools used + task completed |
| **Step Efficiency** | Did it complete with minimum necessary steps? | Ratio of expected min steps to actual steps |
| **Faithfulness** | Is the output grounded in tool data, not hallucinated? | Keyword overlap between tool outputs and response |
| **Safety/Compliance** | Any PII leakage or prompt injection in output? | Regex patterns for SSN, credit cards, emails, injection markers |
| **Routing Accuracy** | Was the request sent to the correct agent? | Compare actual agent vs expected agent |

### Evaluation Approaches

The evaluation system uses multiple complementary approaches:

**1. Rule-Based Metrics**
- Automated checks for task completion, tool accuracy, step efficiency, safety
- Fast, deterministic, no LLM cost
- Run on every evaluation

**2. LLM-as-Judge (G-Eval)**
- Uses the LLM itself to score agent responses on 5 criteria:
  - Correctness, Relevance, Coherence, Tool Usage Quality, Completeness
- Each criterion has a rubric (1-5 scale), normalized to 0-1
- Following the [DeepEval G-Eval methodology](https://docs.confident-ai.com/)

**3. Trajectory/Step Evaluation**
- Scores each step individually (routing decision, each tool call, final response)
- Assesses whether steps were necessary, in logical order, and non-redundant
- Returns per-step scores + overall trajectory quality + reasoning explanation

**4. DeepEval Integration**
- Answer Relevancy metric (is the response relevant to the input?)
- Faithfulness metric (is the response grounded in tool outputs?)
- Formal evaluation via the DeepEval SDK

**5. Langfuse Integration**
- Auto-exports every chat trace as a Langfuse trace with spans
- Auto-exports evaluation runs with metric scores
- Enables production observability at [cloud.langfuse.com](https://cloud.langfuse.com)

### 16 Evaluation Scenarios

Organized by complexity to test different agent capabilities:

| Complexity | Count | What It Tests | Example |
|-----------|-------|---------------|---------|
| **Quick** | 5 | Single-step routing accuracy | "Read main.py and explain what it does" |
| **Medium** | 5 | 2-3 step tool selection | "Find where AgentConfig is defined, then read and explain it" |
| **Complex** | 6 | 4+ step multi-tool workflows | "Explore project structure, read 3 key files, write a summary" |

### Regression Detection

Compare evaluation runs before/after prompt or model changes:

```python
comparison = AgentEvaluator.compare_runs(run_v1, run_v2)
# Flags any metric that dropped by >5% as a regression
# {"task_success_rate": {"before": 1.0, "after": 0.8, "regression": True}}
```

### Multi-LLM Comparison

Run the same scenarios with different models and compare side-by-side:

```python
configs = [
    {"model": "gemini-2.5-flash-lite"},
    {"model": "gpt-4o-mini-search"},
    {"model": "claude-haiku-3"},
]
results = await evaluator.run_comparison(configs)
```

## Observability & Tracing

Every agent interaction is traced following the **OpenTelemetry trace/span model**:

- **Trace**: One complete user request → response cycle
- **Spans**: Individual operations (routing decision, LLM call, tool invocation)
- Each span records: input/output data, tokens, cost, latency, status

Traces are persisted to SQLite and exported to Langfuse for production monitoring.

### Monitoring Dashboard

The dashboard shows per-team metrics:
- Application Runs, Failures, Pass Rate, Total Cost
- Task Quality trends (success, tool accuracy, safety over time)
- Operational Efficiency (latency P50/P95/P99, tokens, cost per run)
- Recent trace list with latency and status

## Project Structure

```
sdlc-agent/
├── main.py                          # CLI: chat, eval, health check
├── server.py                        # FastAPI: 25+ REST endpoints
├── frontend/                        # Next.js dashboard (4 pages)
│
├── src/
│   ├── config.py                    # Environment configuration
│   ├── orchestrator.py              # LangGraph multi-agent router + 4 strategies
│   │
│   ├── agents/
│   │   ├── prompts.py               # Per-agent system prompts + strategy instructions
│   │   └── definitions.py           # Agent configs + factory
│   │
│   ├── tools/
│   │   └── registry.py              # MCP → LangChain bridge (JSON Schema → Pydantic)
│   │
│   ├── mcp_servers/
│   │   ├── filesystem_server.py     # 6 tools: read, write, edit, search, find, list
│   │   ├── shell_server.py          # 3 tools: command, script, tests
│   │   ├── git_server.py            # 6 tools: status, diff, log, commit, branch, show
│   │   ├── web_server.py            # 3 tools: fetch, search, check
│   │   └── memory_server.py         # 6 tools: store, retrieve, list, delete, plan, update
│   │
│   ├── skills/
│   │   └── engine.py                # Skill injection + trigger matching
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # 7 metrics + PII/injection detection
│   │   ├── scenarios.py             # 16 scenarios (quick/medium/complex)
│   │   ├── evaluator.py             # Pipeline: rule-based + LLM-judge + trajectory
│   │   ├── llm_judge.py             # G-Eval: 5-criteria LLM scoring + trajectory eval
│   │   ├── integrations.py          # DeepEval + Langfuse wrappers
│   │   └── reporter.py              # Rich console reports
│   │
│   ├── tracing/
│   │   ├── collector.py             # OpenTelemetry-style trace/span recording
│   │   └── callbacks.py             # LangChain callback handler
│   │
│   └── db/
│       ├── models.py                # SQLAlchemy: teams, agents, skills, traces, evals
│       └── database.py              # SQLite connection + auto-migration + seeding
│
├── tests/
│   ├── test_mcp_communication.py    # 28 E2E tests for MCP layer
│   └── test_evaluation.py           # 16 unit tests for eval pipeline
│
└── frontend/src/app/
    ├── page.tsx                      # Agent Studio: team config, agents, skills, tools
    ├── chat/page.tsx                 # Chat + real-time trace inspector
    ├── monitoring/page.tsx           # Monitoring: metrics, charts, efficiency
    └── evaluation/page.tsx           # Evaluation: run, compare, detail view
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- An OpenAI-compatible API key

### Installation

```bash
git clone https://github.com/tiffanychum/sdlc-agent.git
cd sdlc-agent

# Backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys

# Frontend
cd frontend
npm install
```

### Configuration (.env)

```bash
POE_API_KEY=your_api_key           # LLM API key
LLM_BASE_URL=https://api.poe.com/v1
LLM_MODEL=gpt-5-nano              # Default model
GITHUB_TOKEN=your_github_token     # Optional: for GitHub MCP server

# Optional: evaluation integrations
DEEPEVAL_KEY=your_deepeval_key
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### Running

```bash
# Terminal 1 — Backend API
source venv/bin/activate
python server.py
# API at http://localhost:8000/docs

# Terminal 2 — Frontend
cd frontend
npm run dev
# UI at http://localhost:3000

# Or use the CLI
python main.py chat      # Interactive chat
python main.py eval      # Run evaluation
python main.py test-mcp  # MCP health check
```

### Example Prompts to Test

```
# Simple (1-2 steps)
"Read main.py and explain the architecture"
"Run the tests and tell me the results"

# Medium (2-3 steps)
"Find where decision strategies are defined and list them"
"Check git status and show me the diff"

# Complex (4+ steps, long trajectory)
"Create a plan for refactoring the config module"
"Review recent git changes for issues"
"Explore the project, read 3 key files, write a summary"
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/teams` | List teams |
| `POST` | `/api/teams` | Create team |
| `GET` | `/api/teams/{id}` | Team detail with agents |
| `POST` | `/api/teams/{id}/agents` | Add agent to team |
| `PUT` | `/api/agents/{id}` | Update agent (model, strategy, prompt) |
| `GET` | `/api/skills` | List skills |
| `POST` | `/api/skills` | Create skill |
| `PUT` | `/api/agents/{id}/skills` | Assign skills |
| `GET` | `/api/tools` | List all MCP tools (24 total) |
| `POST` | `/api/teams/{id}/chat` | Chat with a team |
| `GET` | `/api/traces` | List traces |
| `GET` | `/api/traces/stats` | Aggregated metrics |
| `POST` | `/api/eval/run` | Run evaluation |
| `GET` | `/api/eval/runs` | List eval runs |
| `GET` | `/api/eval/runs/{id}` | Eval detail (per-request breakdown) |
| `GET` | `/api/eval/compare/{a}/{b}` | Regression comparison |

## Design Decisions

### Why Multi-Agent over Single Agent?
Following the principle of **"do the simplest thing that works"** — each agent has a focused tool set (3-12 tools) instead of one agent with 24+ tools. The router adds ~200ms but significantly improves tool selection accuracy. Specialized agents also allow per-agent strategies (Planner uses Plan-Execute while Coder uses ReAct).

### Why MCP over Direct Tool Calls?
MCP provides a standardized protocol that decouples agents from tools. Adding a new capability (Docker, database, Slack) requires only a new MCP server — the agents and orchestrator don't change. This is the same architecture used by Claude, Cursor, and other production AI tools.

### Why Per-Agent Decision Strategies?
Different tasks require different reasoning approaches. A code reader benefits from quick ReAct loops, while a project planner needs structured Plan-and-Execute, and a code reviewer benefits from self-reflection (Reflexion). One-size-fits-all strategies leave performance on the table.

### Why LLM-as-Judge + Rule-Based Evaluation?
Rule-based metrics (tool accuracy, step efficiency, safety) are fast and deterministic. LLM-as-Judge (G-Eval) catches nuanced quality issues (coherence, completeness) that rules miss. Using both gives comprehensive coverage without over-relying on either approach.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph |
| Tool Protocol | MCP (Model Context Protocol) |
| LLM Client | LangChain OpenAI (any OpenAI-compatible API) |
| Backend API | FastAPI + Uvicorn |
| Database | SQLite + SQLAlchemy |
| Evaluation | Custom + DeepEval + Langfuse |
| Frontend | Next.js 16 + Tailwind CSS + Recharts |
| Testing | pytest (44 tests) |

## License

MIT
