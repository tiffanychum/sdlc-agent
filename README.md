# SDLC Agent — Multi-Agent Coding Platform

A full-stack multi-agent platform for automating software development workflows. Features a **5-agent team** with per-agent decision strategies, **MCP (Model Context Protocol)** for standardized tool integration, a **three-layer evaluation pipeline** (rule-based + LLM-as-Judge + DeepEval), **golden dataset regression testing** with trace-based assertions and root cause analysis, **prompt versioning with A/B testing**, and a **Next.js dashboard** for configuration, monitoring, and observability.

Built with LangGraph, MCP, FastAPI, SQLite, DeepEval, Langfuse, OpenTelemetry, and Next.js.

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

## Evaluation System

### Three-Layer Evaluation Pipeline

The evaluation system uses three complementary layers for comprehensive coverage:

#### Layer 1: Rule-Based Metrics (7 metrics)

Fast, deterministic checks that run on every evaluation with zero LLM cost:

| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Task Success Rate** | Did the agent complete the task? | Keyword matching against expected outputs |
| **Tool Accuracy** | Were the right tools selected? | Compare actual vs expected tool calls |
| **Reasoning Quality** | Were intermediate steps coherent? | Composite: routing + tools + completion |
| **Step Efficiency** | Minimal necessary steps? | Ratio of expected min steps to actual |
| **Faithfulness** | Output grounded in tool data? | Keyword overlap between tool outputs and response |
| **Safety/Compliance** | PII leakage or prompt injection? | Regex for SSN, credit cards, emails, injection markers |
| **Routing Accuracy** | Correct agent selected? | Compare actual agent vs expected |

#### Layer 2: LLM-as-Judge (G-Eval)

Uses the LLM to score agent responses on 5 criteria following the [DeepEval G-Eval methodology](https://docs.confident-ai.com/):

| Criterion | What It Evaluates |
|-----------|-------------------|
| **Correctness** | Does the output correctly answer the user's question? |
| **Completeness** | Does it cover all aspects of the task? |
| **Tool Usage** | Were the right tools used appropriately? |
| **Efficiency** | Was the task completed without unnecessary steps? |
| **Coherence** | Is the output well-structured and easy to understand? |

Each criterion uses a 1-5 rubric with step-by-step reasoning, normalized to 0-1. Full reasoning chains are captured and displayed in the UI.

#### Layer 3: DeepEval Agentic Metrics (8 metrics)

Specialized agentic evaluation via DeepEval SDK and LLM-as-judge trace analysis. Each metric produces a score and detailed explanation reasoning:

| Metric | Source | What It Evaluates |
|--------|--------|-------------------|
| **Answer Relevancy** | DeepEval SDK | Does the response directly address the user's query? |
| **Faithfulness** | DeepEval SDK | Is the response grounded in retrieved/tool outputs? |
| **Tool Correctness** | DeepEval SDK | Were the correct tools selected for the task? |
| **Argument Correctness** | DeepEval SDK | Were tool arguments well-formed and appropriate? |
| **Task Completion** | LLM-as-Judge | Did the agent fully accomplish the user's task? |
| **Step Efficiency** | LLM-as-Judge | Were execution steps minimal and necessary? |
| **Plan Quality** | LLM-as-Judge | Was the agent's plan logical, complete, and actionable? |
| **Plan Adherence** | LLM-as-Judge | Did the agent follow its own plan during execution? |

### Golden Dataset & Regression Testing

A curated set of 10 test cases with expected outputs, stored in `golden_dataset.json` and synced to the database for UI access.

#### Golden Test Case Structure

Each case specifies:
- **Prompt** and **reference output** for semantic similarity comparison
- **Expected agent**, **tools**, and **delegation pattern** for structural assertions
- **Quality thresholds** (semantic similarity, completeness, correctness)
- **Budgets**: max LLM calls, max tool calls, max tokens, max latency
- **Complexity tier**: quick, medium, or complex

#### Regression Run Pipeline

```
Golden Case → Agent Execution → Trace Capture → Multi-Layer Evaluation
                                                         │
                     ┌───────────────────┬───────────────┼───────────────┐
                     ▼                   ▼               ▼               ▼
              Semantic Similarity   G-Eval Quality   DeepEval Agentic   Trace Assertions
              (LLM-as-Judge)       (5 criteria)     (8 metrics)        (tools, delegation,
                                                                        budgets)
                                                         │
                                         ┌───────────────┼───────────────┐
                                         ▼               ▼               ▼
                                    Cost Regression  Latency Regression  Quality Regression
                                    (±20% threshold) (±20% threshold)   (below thresholds)
```

#### Trace-Based Structural Assertions

For each test case, the system verifies:
- **Required tools called**: all expected tools were invoked
- **Delegation pattern**: agents were routed in the expected order
- **LLM call budget**: total LLM calls within threshold
- **Tool call budget**: total tool calls within threshold
- **Token budget**: total tokens within threshold
- **Latency budget**: wall-clock time within threshold

Each assertion shows expected vs. actual values with pass/fail status.

#### Root Cause Analysis (RCA)

When a regression test fails, an LLM-powered RCA engine:
1. Computes step-by-step trace diff between baseline and failing run
2. Calculates cost/latency deltas (tokens, cost, latency, LLM/tool call counts)
3. Uses an LLM to classify root causes and provide recommendations
4. Optionally exports findings to Langfuse for external observability

### Prompt Versioning & A/B Testing

Prompts are treated as versioned code artifacts:
- Create, edit, and save named prompt versions (per-agent system prompts + team strategy)
- Select any prompt version when running regression tests
- Compare results side-by-side: baseline (version A) vs. candidate (version B)
- Combined G-Eval + DeepEval radar chart for visual comparison
- Detailed reasoning comparison for each metric between versions

### 16 Evaluation Scenarios

Organized by complexity to test different agent capabilities:

| Complexity | Count | What It Tests | Example |
|-----------|-------|---------------|---------|
| **Quick** | 5 | Single-step routing accuracy | "Read main.py and explain what it does" |
| **Medium** | 5 | 2-3 step tool selection | "Find where AgentConfig is defined, then read and explain it" |
| **Complex** | 6 | 4+ step multi-tool workflows | "Explore project structure, read 3 key files, write a summary" |

## Observability & Tracing

Every agent interaction is traced following the **OpenTelemetry trace/span model** with GenAI semantic conventions:

- **Trace**: One complete user request → response cycle
- **Spans**: Individual operations (routing decision, LLM call, tool invocation)
- Each span records: input/output data, tokens, cost, latency, status
- **Cost estimation**: Per-model token pricing applied automatically
- **OTLP export**: Optional export to any OpenTelemetry-compatible collector
- **OpenInference**: Auto-instrumentation for LangChain via OpenInference

Traces are persisted to SQLite via a custom `DBSpanProcessor` and exported to Langfuse for production monitoring.

### Monitoring Dashboard

The dashboard provides 4 sub-tabs:

| Tab | What It Shows |
|-----|---------------|
| **Overview** | Application runs, failures, pass rate, total cost, quality trends, latency P50/P95/P99, recent traces |
| **OTel Observability** | OpenTelemetry span statistics, token usage, cost breakdown by model |
| **Regression Testing** | Golden dataset manager, run configuration, per-case results with DeepEval scores, trace diff & RCA |
| **Evaluation History** | Rule-based + G-Eval + trajectory scores, run comparison, per-request breakdowns |

The **Regression Testing** tab includes:
- **Golden Dataset Manager**: Browse, select, and sync test cases
- **Run Configuration**: Model selector dropdown, prompt version selector with inline editor, baseline run selection
- **Results & Detail View**: Per-case pass/fail with G-Eval scores, DeepEval agentic metrics (with reasoning), trace structural assertions (expected vs. actual), cost/latency detail, full execution trace
- **Trace Diff & RCA**: Side-by-side baseline vs. candidate comparison, combined radar chart, step-by-step trace timeline, LLM-powered root cause analysis

## Project Structure

```
sdlc-agent/
├── main.py                          # CLI: chat, eval, health check
├── server.py                        # FastAPI: 50+ REST endpoints
├── pyproject.toml                   # Project metadata, pytest, ruff config
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
│
├── src/
│   ├── config.py                    # Environment configuration (dataclasses)
│   ├── orchestrator.py              # LangGraph multi-agent router + 4 strategies
│   │
│   ├── agents/
│   │   ├── prompts.py               # Per-agent system prompts + strategy instructions
│   │   └── definitions.py           # Agent configs + factory
│   │
│   ├── llm/
│   │   └── client.py                # LangChain ChatOpenAI factory (any OpenAI-compatible API)
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
│   │   ├── metrics.py               # 7 rule-based metrics + PII/injection detection
│   │   ├── scenarios.py             # 16 scenarios (quick/medium/complex)
│   │   ├── evaluator.py             # Pipeline: rule-based + LLM-judge + trajectory
│   │   ├── llm_judge.py             # G-Eval: 5-criteria LLM scoring + trajectory eval
│   │   ├── integrations.py          # DeepEval (8 agentic metrics) + Langfuse wrappers
│   │   ├── reporter.py              # Rich console reports
│   │   ├── golden.py                # Golden dataset: JSON ↔ DB sync
│   │   ├── golden_dataset.json      # 10 curated test cases (versioned with code)
│   │   ├── regression.py            # RegressionRunner: execute, evaluate, persist
│   │   └── rca.py                   # RootCauseAnalyzer: trace diff + LLM classification
│   │
│   ├── tracing/
│   │   ├── collector.py             # OTel trace/span recording + OTLP + DB persistence
│   │   └── callbacks.py             # LangChain callback handler → TraceCollector
│   │
│   └── db/
│       ├── models.py                # SQLAlchemy: teams, agents, skills, traces, evals,
│       │                            #   golden cases, regression results, prompt versions
│       └── database.py              # SQLite connection + auto-migration + seeding
│
├── tests/
│   ├── test_mcp_communication.py    # 28 E2E tests for MCP layer
│   └── test_evaluation.py           # 16 unit tests for eval pipeline
│
├── data/                            # Runtime data (gitignored)
│   ├── sdlc_agent.db               # SQLite database
│   └── agent_memory.json            # MCP memory store
│
└── frontend/
    ├── package.json                 # Next.js 16, React 19, Recharts, Tailwind 4
    └── src/
        ├── lib/
        │   └── api.ts               # Typed fetch client for all backend APIs
        └── app/
            ├── layout.tsx            # Root layout: sidebar nav (Studio, Chat, Monitoring, Evaluation)
            ├── page.tsx              # Agent Studio: team config, agents, skills, tools
            ├── chat/page.tsx         # Chat: team chat + real-time trace inspector
            ├── monitoring/page.tsx   # Monitoring: metrics, regression testing, OTel, charts
            ├── evaluation/page.tsx   # Evaluation: run, compare, detail view
            └── traces/page.tsx       # Trace history: list + detail panes
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- An OpenAI-compatible API key

### Installation

```bash
git clone https://github.com/your-username/sdlc-agent.git
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

# Optional: OpenTelemetry export
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

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

### Example Prompts

```
# Simple (1-2 steps)
"Read main.py and explain the architecture"
"Run the tests and tell me the results"

# Medium (2-3 steps)
"Find where decision strategies are defined and list them"
"Check git status and show me the diff"

# Complex (4+ steps)
"Create a plan for refactoring the config module"
"Review recent git changes for issues"
"Explore the project, read 3 key files, write a summary"
```

## API Reference

### Team & Agent Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/teams` | List teams |
| `POST` | `/api/teams` | Create team |
| `GET` | `/api/teams/{id}` | Team detail with agents |
| `PUT` | `/api/teams/{id}` | Update team |
| `DELETE` | `/api/teams/{id}` | Delete team |
| `POST` | `/api/teams/{id}/agents` | Add agent to team |
| `PUT` | `/api/agents/{id}` | Update agent (model, strategy, prompt) |
| `DELETE` | `/api/agents/{id}` | Delete agent |
| `POST` | `/api/teams/{id}/rebuild` | Rebuild orchestrator |

### Skills & Tools

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/skills` | List skills |
| `POST` | `/api/skills` | Create skill |
| `PUT` | `/api/skills/{id}` | Update skill |
| `DELETE` | `/api/skills/{id}` | Delete skill |
| `PUT` | `/api/agents/{id}/skills` | Assign skills to agent |
| `GET` | `/api/tools` | List all MCP tools (24 total) |

### Chat & Traces

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/teams/{id}/chat` | Chat with a team |
| `GET` | `/api/traces` | List traces (paginated) |
| `GET` | `/api/traces/stats` | Aggregated trace metrics |
| `GET` | `/api/traces/{id}` | Trace detail with spans |
| `GET` | `/api/otel/spans/stats` | OpenTelemetry span statistics |
| `POST` | `/api/traces/evaluate` | Evaluate a specific trace |

### Evaluation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/eval/run` | Run evaluation suite |
| `GET` | `/api/eval/runs` | List eval runs |
| `GET` | `/api/eval/runs/{id}` | Eval detail (per-request breakdown) |
| `POST` | `/api/eval/compare` | Run multi-model comparison |
| `GET` | `/api/eval/compare/{a}/{b}` | Regression comparison between runs |

### Golden Dataset & Regression Testing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/golden` | List golden test cases |
| `POST` | `/api/golden` | Create golden test case |
| `PUT` | `/api/golden/{id}` | Update golden test case |
| `DELETE` | `/api/golden/{id}` | Deactivate golden test case |
| `POST` | `/api/golden/sync` | Sync JSON → DB |
| `POST` | `/api/regression/run` | Run regression suite |
| `GET` | `/api/regression/runs` | List regression runs |
| `GET` | `/api/regression/results/{run_id}` | Run results (summary + per-case) |
| `GET` | `/api/regression/results/{run_id}/{case_id}` | Detailed case result |
| `GET` | `/api/regression/diff/{a}/{b}/{case_id}` | Trace diff between two runs |
| `POST` | `/api/regression/rca` | Root cause analysis for a failing case |

### Prompt Versioning & Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/models` | List available LLM models |
| `GET` | `/api/prompt-versions` | List prompt versions |
| `POST` | `/api/prompt-versions` | Create prompt version |
| `PUT` | `/api/prompt-versions/{id}` | Update prompt version |
| `DELETE` | `/api/prompt-versions/{id}` | Delete prompt version |
| `GET` | `/api/prompt-versions/current` | Get current live agent prompts |

## Design Decisions

### Why Multi-Agent over Single Agent?
Following the principle of **"do the simplest thing that works"** — each agent has a focused tool set (3-12 tools) instead of one agent with 24+ tools. The router adds ~200ms but significantly improves tool selection accuracy. Specialized agents also allow per-agent strategies (Planner uses Plan-Execute while Coder uses ReAct).

### Why MCP over Direct Tool Calls?
MCP provides a standardized protocol that decouples agents from tools. Adding a new capability (Docker, database, Slack) requires only a new MCP server — the agents and orchestrator don't change. This is the same architecture used by Claude, Cursor, and other production AI tools.

### Why Per-Agent Decision Strategies?
Different tasks require different reasoning approaches. A code reader benefits from quick ReAct loops, while a project planner needs structured Plan-and-Execute, and a code reviewer benefits from self-reflection (Reflexion). One-size-fits-all strategies leave performance on the table.

### Why Three-Layer Evaluation?
Rule-based metrics (tool accuracy, step efficiency, safety) are fast and deterministic. LLM-as-Judge (G-Eval) catches nuanced quality issues (coherence, completeness) that rules miss. DeepEval agentic metrics provide specialized agent-level evaluation (tool correctness, plan quality, task completion) with detailed reasoning. Using all three gives comprehensive coverage without over-relying on any single approach.

### Why Golden Dataset Regression Testing?
Prompt and model changes can introduce subtle regressions. A curated golden dataset with expected outputs, structural assertions, and budget constraints catches regressions that unit tests miss. Trace-level comparison with LLM-powered root cause analysis identifies exactly where and why performance degraded.

### Why Prompt Versioning?
Prompts are the most frequently changed artifact in an agent system. Treating them as versioned code enables systematic A/B testing, rollback, and audit trails — the same discipline applied to source code via git.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph (router_decides, sequential, parallel, supervisor) |
| Tool Protocol | MCP (Model Context Protocol) — 5 servers, 24 tools |
| LLM Client | LangChain ChatOpenAI (any OpenAI-compatible API) |
| Backend API | FastAPI + Uvicorn (50+ endpoints) |
| Database | SQLite + SQLAlchemy (auto-migration) |
| Evaluation | Custom rule-based + G-Eval + DeepEval (8 agentic metrics) |
| Regression Testing | Golden dataset + trace assertions + RCA |
| Observability | OpenTelemetry + OpenInference + Langfuse |
| Frontend | Next.js 16 + React 19 + Tailwind CSS 4 + Recharts |
| Testing | pytest (44 tests: 28 MCP E2E + 16 evaluation unit) |

## License

MIT
