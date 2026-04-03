# SDLC Agent — Multi-Agent Coding Platform

A full-stack multi-agent platform for automating software development workflows. Features a **5-agent team** with per-agent decision strategies, **Human-in-the-Loop (HITL)** checkpoints with plan review and action confirmation, **MCP (Model Context Protocol)** for standardized tool integration, a **three-layer evaluation pipeline** (rule-based + LLM-as-Judge + DeepEval), **golden dataset regression testing** with trace-based assertions and root cause analysis, **real-time trace inspection** via SSE streaming, **prompt versioning with A/B testing**, and a **Next.js dashboard** with conversation management, live markdown rendering, and observability.

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
     │            HITL Checkpoints (4 types)                │
     │  Clarification · Plan Review · Action Confirm · Tool │
     │  Review — interrupt() / Command(resume=) via SSE     │
     └──────────────────────────────────────────────────────┘
          │          │          │           │            │
          ▼          ▼          ▼           ▼            ▼
     ┌──────────────────────────────────────────────────────┐
     │              MCP Communication Layer                  │
     │  Tool discovery · Schema validation · Error handling  │
     └──┬─────────┬──────────┬──────────┬──────────┬───────┘
        ▼         ▼          ▼          ▼          ▼
   Filesystem   Shell       Git        Web      Memory   Planner   GitHub    Jira
   (6 tools)  (3 tools)  (6 tools)  (3 tools)  (6 tools) (varies)  (varies)  (varies)
```

## How the Agent Team Works

### 5 Specialized Agents

Each agent has a specific role, decision strategy, MCP tool set, and skills:

| Agent | Strategy | Tools | Role | Tool Call Limit |
|-------|----------|-------|------|----------------|
| **Coder** | ReAct | filesystem, shell, git, github, jira, memory (typical) | Code, tests, local git, GitHub API, Jira read/fetch, end-to-end SDLC when routed | 14 (prompt budget; tune in DB) |
| **Runner** | ReAct | shell | Single-task executor — runs ONE command or test suite and reports output | 5 |
| **Researcher** | ReAct | web | Searches documentation, fetches API references, synthesizes findings | 6 |
| **Planner** | Plan-and-Execute | memory, filesystem | Multi-step coordinator for tasks requiring two or more distinct actions | 15 |
| **Reviewer** | Reflexion | filesystem, shell, git, memory | Reviews code/git diffs for quality and bugs; runs commands only when explicitly asked | 8 |

### Per-Agent Decision Strategies

| Strategy | How It Works | Best For |
|----------|-------------|----------|
| **ReAct** | Think → Act → Observe → Repeat | Quick, focused tasks (reading files, running commands) |
| **Plan-and-Execute** | Create a numbered plan first, then execute step by step | Complex multi-step tasks that need decomposition |
| **Reflexion** | After each action, reflect on whether it was correct | Quality-critical tasks (code review, verification) |
| **Chain-of-Thought** | Reason thoroughly before taking any action | Tasks requiring careful analysis |

### Agent Prompt Design

All agent system prompts follow a structured six-section format that enforces disciplined tool use:

```
## Role          — what the agent is responsible for
## Hard Constraints — non-negotiable per-task tool call limits and anti-patterns
## Tool Selection Rules — priority-ordered rules for choosing the right tool
## Execution Loop — strategy-specific reasoning loop (ReAct / Plan-and-Execute / Reflexion)
## Output        — required output format
## Error Recovery — what to do when tools fail
```

**Hard constraints** embedded directly in each prompt prevent runaway tool loops:

| Agent | Max Tool Calls | Key Prohibitions |
|-------|---------------|-----------------|
| Coder | 8 | Never read the same file twice; `search_files` before `list_directory` |
| Runner | 5 | One command per goal; never re-run to verify — read the output |
| Researcher | 6 | Never repeat a search query; fetch URL only if snippet is insufficient |
| Planner | 15 | Max 5 plan steps; max 3 `memory_retrieve` calls; never write files unless task says so |
| Reviewer | 8 | Never call `run_tests` / `run_command` unless task says "run" or "verify" |

Prompts are stored in the database and configurable via the Studio UI. On every startup, `patch_agent_prompts()` in `src/db/database.py` applies the latest prompt version to any existing team in the database — no manual DB reset required when prompts are updated.

### Human-in-the-Loop (HITL)

The system supports four HITL checkpoint types, powered by LangGraph's `interrupt()` / `Command(resume=)` with a `MemorySaver` checkpointer:

| Checkpoint | When It Triggers | What the User Sees |
|------------|-----------------|-------------------|
| **Clarification Q&A** | Agent calls `ask_human` tool to gather context | Question with optional quick-reply buttons + free-text input |
| **Plan Review** | Planner generates a plan (two-phase: plan first, then execute) | Editable step list with reorder, add, remove; approve or request changes |
| **Action Confirmation** | Dangerous tools (`run_command`, `write_file`, `git_commit`, etc.) | Tool name, arguments, risk level; allow or deny |
| **Tool Output Review** | Reviewable tools (`run_command`, `run_script`) complete | Tool output with option to continue, edit, or stop |

The **Planner** uses a two-phase architecture:
1. **Phase 1 — Plan**: LLM generates a numbered plan (no tool execution)
2. **Phase 2 — Review**: HITL interrupt for user review/editing
3. **Phase 3 — Execute**: Approved plan is executed by an unwrapped agent (no nested HITL interrupts)

During regression testing, all HITL checkpoints are auto-approved to allow unattended execution.

### MCP vs Skills — Clear Separation

Following the [Claude Code principle](https://docs.anthropic.com/): **"MCP gives power. Skills control power."**

**MCP Tools** (what agents CAN do):
- MCP tool groups include filesystem, shell, git, web, memory, planner (Microsoft Graph), **github**, **jira** — assign per agent in Studio
- Each tool has a JSON Schema defining its parameters
- Agents discover tools dynamically at runtime via the MCP protocol

**Skills** (HOW agents SHOULD behave):
- Instruction-only injections appended to the agent's system prompt
- No tools, no execution — just behavioral guidance
- 7 default skills: Code Review, Git Conventions, Error Recovery, Plan First, Security Awareness, Documentation Citation, Test-Driven Approach
- Skills can be assigned to specific agents and auto-triggered by prompt patterns

### Routing: How Requests Reach the Right Agent

1. User sends a request (e.g., "Create a plan for refactoring the config module")
2. A lightweight **Router LLM call** classifies the intent using priority-ordered routing rules (see `src/orchestrator.py`), including:
   - Jira fetch + implement + push/PR → **coder**
   - GitHub branch/file/PR workflows → **coder**
   - Jira project/issue *management* (create/assign), no coding → **project_manager** / **business_analyst** when configured
   - Multi-step tasks spanning domains (without GitHub/Jira coding) → **planner**
   - "Review / assess / find bugs" without execution → **reviewer**
   - Single command or test run → **runner**
   - Read / write / edit code (single action) → **coder**
   - Web search or external documentation → **researcher**
3. The selected agent runs with its assigned tools and strategy
4. HITL checkpoints may pause execution for user input
5. Every step is traced (routing decision, tool calls, HITL pauses, response) for evaluation

### Multi-Model Architecture

The platform uses any **OpenAI-compatible API** and supports **role-based model selection** — different models for agents, evaluation judges, and root cause analysis:

```
# .env — Model configuration
LLM_BASE_URL=https://api.poe.com/v1
LLM_MODEL=gpt-5                    # Global default / fallback
LLM_JUDGE_MODEL=deepseek-r1        # G-Eval scoring, semantic similarity, DeepEval trace metrics
LLM_RCA_MODEL=deepseek-r1          # Root cause analysis on regression failures
LLM_ROUTER_MODEL=gpt-4o-mini       # Request routing and supervisor decisions (lightweight)
```

| Tier | Purpose | Default Model | Why |
|------|---------|---------------|-----|
| **Agent (Planner/Coder)** | Complex tool calling, code generation, multi-step plans | `claude-sonnet-4.6` | Best tool-calling accuracy (91.9% tau2-bench), strong code quality |
| **Agent (Runner/Reviewer)** | Test execution, simple tool use, code review | `gemini-3-flash` | Cost-effective, competitive quality for simpler tasks |
| **Agent (Researcher)** | Web search, summarization | Global default | Synthesis-focused, configurable per team |
| **Router / Supervisor** | Intent classification, agent selection | `gpt-4o-mini` | Lightweight, fast, cheap — routing doesn't need frontier reasoning |
| **LLM-as-a-Judge** | G-Eval quality scoring, semantic similarity, DeepEval trace metrics | `deepseek-r1` | Strong reasoning for consistent evaluation scoring |
| **RCA** | Root cause analysis on failed regression cases | `deepseek-r1` | Deep reasoning model for multi-step trace debugging |

Per-agent models are stored in the database and configurable via the Studio UI. The `get_llm()`, `get_judge_llm()`, `get_rca_llm()`, and `get_router_llm()` factory functions in `src/llm/client.py` route to the appropriate model.

## Chat Interface

The chat page features a three-panel layout:

### Conversation Sidebar (left)
- **Conversation history**: Switch between past chats with full message + trace preservation
- **Create / delete**: Start new conversations or remove old ones
- Titles auto-derived from first user message, sorted by last activity

### Chat Panel (center)
- **Cursor-style thinking box**: While the model streams, a collapsible grey box shows reasoning tokens in real time
- **Thinking history**: After the turn completes, the same content is stored on the assistant message and can be reopened via **Show thinking** (so it is not lost when streaming ends)
- **Markdown rendering**: All assistant messages rendered with `react-markdown` (headings, code blocks, tables, lists, links) — styled during streaming, not after
- **Session model override**: Dropdown next to the team selector; choice is persisted in the browser and sent as optional `model` on `POST /api/teams/{id}/chat/stream` (overrides per-agent DB model for that request path when the backend applies it)
- **HITL widgets**: Inline interactive widgets for clarification, plan review, action confirmation, and tool output review
- **Stop button**: Red stop button replaces Send during processing; aborts the SSE stream immediately
- **Clear**: Clears messages within the current conversation without deleting it

### Trace Inspector (right)
- **Live trajectory**: Agent and tool execution flow shown as pills — format: `agent/tool_name` with pulsing indicators for active steps
- **Summary cards**: Elapsed time, token count, cost — persists after query completes
- **Spans grouped by agent**: Collapsible agent sections with per-agent token/cost totals and individual span details
- **HITL pause indicator**: Orange banner when waiting for user response
- **After refresh**: The last trace summary is restored from `localStorage` so the panel is not empty on reload
- Live trace data remains visible after the query finishes (no disappearing on completion)

### Streaming Architecture

```
Backend (FastAPI)          SSE Stream              Frontend (Next.js)
─────────────────    ─────────────────────    ──────────────────────
astream_events() →   thread_id                → setThreadId
                     agent_start / agent_end  → trajectory pills
                     tool_start / tool_end    → trajectory + status
                     llm_token               → thinking box
                     trace_span              → live span tree
                     hitl_request            → HITL widget
                     response                → markdown message
                     done                    → finalize trace
```

Two SSE endpoints:
- `POST /api/teams/{id}/chat/stream` — Initial query; JSON body may include `message`, `thread_id`, and optional `model`
- `POST /api/teams/{id}/chat/resume` — Resume after HITL pause

## Evaluation System

### Three-Layer Evaluation Pipeline

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

A curated set of **golden test cases** (including integration scenarios such as GitHub branch/file/PR, Jira issue workflows, and end-to-end Jira → code → pytest → PR) stored in `src/evaluation/golden_dataset.json` and synced to the database for UI access.

**Standalone runners** for heavy integration cases: `tests/golden_tests/run_golden_019.py` (Jira), `run_golden_020.py` (GitHub), `run_golden_021.py` (full SDLC). They resolve the repo root automatically; run: `python tests/golden_tests/run_golden_021.py` from the project root.

**Version 3.0 context:** budget constraints were recalibrated to match structured prompts — token caps looser for thinking models, LLM call limits tighter where appropriate. Reviewer cases align with “read only when asked” behavior.

#### Golden Test Case Structure

Each case specifies:
- **Prompt** and **reference output** for semantic similarity comparison
- **Expected agent**, **tools**, and **delegation pattern** for structural assertions
- **Quality thresholds** (semantic similarity, completeness, correctness)
- **Budgets**: max LLM calls, max tool calls, max tokens, max latency
- **Complexity tier**: quick, medium, or complex

#### Regression Run Pipeline

```
Golden Case → Agent Execution → HITL Auto-Approve → Trace Capture → Multi-Layer Evaluation
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

During regression runs, all HITL checkpoints are auto-approved with sensible defaults (plans approved as-is, actions allowed, tools continued, clarifications answered with "proceed with best judgment") to enable fully unattended execution.

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
- **Real-time streaming**: Spans are emitted via SSE as they start/end, enabling live trace inspection
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
├── server.py                        # FastAPI: 50+ REST + SSE endpoints
├── pyproject.toml                   # Project metadata, pytest, ruff config
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
│
├── src/
│   ├── config.py                    # Environment configuration (dataclasses)
│   ├── orchestrator.py              # LangGraph multi-agent router + 4 strategies
│   │                                #   + MemorySaver checkpointer + HITL integration
│   ├── hitl.py                      # Human-in-the-Loop: 4 checkpoint types,
│   │                                #   ask_human tool, dangerous/reviewable wrappers,
│   │                                #   two-phase planner executor
│   │
│   ├── llm/
│   │   └── client.py                # LangChain ChatOpenAI factory (any OpenAI-compatible API)
│   │
│   ├── tools/
│   │   └── registry.py              # MCP → LangChain bridge (JSON Schema → Pydantic)
│   │
│   ├── mcp_servers/
│   │   ├── filesystem_server.py     # 6 tools; workspace sandbox + optional absolute paths
│   │   ├── shell_server.py          # 3 tools: command, script, tests
│   │   ├── git_server.py            # 6 tools: status, diff, log, commit, branch, show
│   │   ├── web_server.py            # 3 tools: fetch, search, check
│   │   ├── memory_server.py         # 6 tools: store, retrieve, list, delete, plan, update
│   │   ├── planner_server.py       # Microsoft Planner / Graph (optional)
│   │   ├── github_server.py         # GitHub REST: repo, branch, file, PR, …
│   │   └── jira_server.py           # Jira REST: issues, transitions, assign, …
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
│   │   ├── golden_dataset.json      # Curated regression cases (synced to DB)
│   │   ├── regression.py            # RegressionRunner: execute, HITL auto-approve,
│   │   │                            #   evaluate, persist
│   │   └── rca.py                   # RootCauseAnalyzer: trace diff + LLM classification
│   │
│   ├── tracing/
│   │   ├── collector.py             # OTel trace/span recording + OTLP + DB persistence
│   │   │                            #   + real-time on_span_event callback for SSE
│   │   └── callbacks.py             # LangChain callback handler → TraceCollector
│   │
│   └── db/
│       ├── models.py                # SQLAlchemy: teams, agents, skills, traces, evals,
│       │                            #   golden cases, regression results, prompt versions
│       └── database.py              # SQLite connection + auto-migration + seeding
│                                    #   + patch_agent_prompts() runtime prompt migration
│
├── tests/
│   ├── golden_tests/                # Standalone golden integration runners (019–021, …)
│   ├── test_filesystem_absolute_path.py  # Absolute-path filesystem MCP (env-gated)
│   ├── test_mcp_communication.py    # E2E tests for MCP layer
│   └── test_evaluation.py           # Unit tests for eval pipeline
│
├── data/                            # Runtime data (gitignored)
│   ├── sdlc_agent.db               # SQLite database
│   └── agent_memory.json            # MCP memory store
│
└── frontend/
    ├── package.json                 # Next.js 16, React 19, Recharts, Tailwind 4,
    │                                #   react-markdown, remark-gfm
    └── src/
        ├── lib/
        │   └── api.ts               # Typed fetch + SSE client for all backend APIs
        └── app/
            ├── layout.tsx            # Root layout: sidebar nav (Studio, Chat, Monitoring, Evaluation)
            ├── page.tsx              # Agent Studio: team config, agents, skills, tools
            ├── chat/page.tsx         # Chat: conversation sidebar, HITL widgets,
            │                         #   Cursor-style thinking box, live trace inspector
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
LLM_MODEL=gpt-5                   # Default agent model (fallback)

# Role-specific models (optional — falls back to LLM_MODEL if unset)
LLM_JUDGE_MODEL=deepseek-r1       # Evaluation: G-Eval, semantic similarity, trace metrics
LLM_RCA_MODEL=deepseek-r1         # Root cause analysis on regression failures
LLM_ROUTER_MODEL=gpt-4o-mini      # Request routing / supervisor (lightweight)

# Filesystem MCP
# AGENT_WORKSPACE=/path/to/project   # Optional: workspace root for relative paths (defaults to cwd)
# AGENT_ALLOW_ABSOLUTE_PATHS=1       # Optional: allow read/write outside workspace (use with care)

GITHUB_TOKEN=your_github_token     # GitHub MCP (repo, branch, file, PR)
JIRA_BASE_URL=https://your.atlassian.net
JIRA_EMAIL=you@example.com
JIRA_API_TOKEN=your_jira_token     # Jira MCP

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
# Dev reload watches only ./src (see server.py) so agent-written files under tests/ etc.
# do not restart the server and break in-flight HITL checkpoints.

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

# Complex (4+ steps, triggers HITL)
"Create a plan for refactoring the config module"    → Plan Review HITL
"Review recent git changes for issues"               → Action Confirmation HITL
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
| `GET` | `/api/tools` | List all MCP tools (by group) |

### Chat (SSE Streaming + HITL)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/teams/{id}/chat` | Synchronous chat (auto-resumes HITL) |
| `POST` | `/api/teams/{id}/chat/stream` | SSE stream: real-time events (tokens, spans, HITL) |
| `POST` | `/api/teams/{id}/chat/resume` | Resume SSE stream after HITL pause |

### Traces

| Method | Endpoint | Description |
|--------|----------|-------------|
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
| `POST` | `/api/regression/run` | Run regression suite (HITL auto-approved) |
| `GET` | `/api/regression/runs` | List regression runs |
| `GET` | `/api/regression/results/{run_id}` | Run results (summary + per-case) |
| `GET` | `/api/regression/results/{run_id}/{case_id}` | Detailed case result |
| `GET` | `/api/regression/diff/{a}/{b}/{case_id}` | Trace diff between two runs |
| `POST` | `/api/regression/rca` | Root cause analysis for a failing case |

### Prompt Versioning & Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/models` | List available LLM models (id, name, provider, tier; Studio + Chat dropdowns) |
| `GET` | `/api/config/llm` | Resolved defaults: `default_model`, `default_model_name`, `judge_model`, `router_model` |
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

### Why Human-in-the-Loop?
Production agent systems need guardrails. HITL checkpoints give users control over dangerous actions (shell commands, file writes, git operations) and plan approval before execution. The two-phase planner ensures plans are reviewed before any tools are called. During automated testing, HITL is auto-approved to maintain unattended execution.

### Why SSE over WebSocket for Streaming?
SSE is simpler (HTTP-based, auto-reconnect, one-way), sufficient for server→client streaming, and works through proxies. The granular event types (`agent_start`, `tool_start`, `llm_token`, `trace_span`, `hitl_request`) give the frontend full control over what to render and when. WebSocket is available as a fallback for bidirectional needs.

### Why Three-Layer Evaluation?
Rule-based metrics (tool accuracy, step efficiency, safety) are fast and deterministic. LLM-as-Judge (G-Eval) catches nuanced quality issues (coherence, completeness) that rules miss. DeepEval agentic metrics provide specialized agent-level evaluation (tool correctness, plan quality, task completion) with detailed reasoning. Using all three gives comprehensive coverage without over-relying on any single approach.

### Why Golden Dataset Regression Testing?
Prompt and model changes can introduce subtle regressions. A curated golden dataset with expected outputs, structural assertions, and budget constraints catches regressions that unit tests miss. Trace-level comparison with LLM-powered root cause analysis identifies exactly where and why performance degraded.

### Why Prompt Versioning?
Prompts are the most frequently changed artifact in an agent system. Treating them as versioned code enables systematic A/B testing, rollback, and audit trails — the same discipline applied to source code via git.

### Why `patch_agent_prompts()` on Startup?
Agent prompts evolve frequently. Rather than forcing users to drop and re-seed the database every time a prompt is updated, `patch_agent_prompts()` runs on every startup and applies the latest prompt to any existing agent record. This makes prompt updates zero-friction for anyone with a running deployment — the next server restart picks them up automatically.

### Why restrict Uvicorn reload to `src/`?
With `reload=True`, watching the whole repo causes a restart whenever the agent writes a file (e.g. under `tests/`). That resets the in-memory LangGraph checkpointer and the next `chat/resume` fails. Limiting reload to `src/` keeps developer ergonomics while preserving HITL sessions during agent-driven file writes.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph (router_decides, sequential, parallel, supervisor) + MemorySaver |
| Human-in-the-Loop | LangGraph interrupt() / Command(resume=), 4 checkpoint types |
| Tool Protocol | MCP (Model Context Protocol) — 5 servers, 24 tools |
| LLM Client | LangChain ChatOpenAI (any OpenAI-compatible API) |
| Backend API | FastAPI + Uvicorn (50+ REST + SSE endpoints) |
| Database | SQLite + SQLAlchemy (auto-migration) |
| Evaluation | Custom rule-based + G-Eval + DeepEval (8 agentic metrics) |
| Regression Testing | Golden dataset + trace assertions + HITL auto-approve + RCA |
| Observability | OpenTelemetry + OpenInference + Langfuse + real-time SSE spans |
| Frontend | Next.js 16 + React 19 + Tailwind CSS 4 + Recharts + react-markdown |
| Testing | pytest (MCP E2E, evaluation unit, filesystem absolute-path tests, golden runners) |

## License

MIT
