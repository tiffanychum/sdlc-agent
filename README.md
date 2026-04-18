# SDLC Agent

A full-stack **multi-agent platform** for automating software development workflows — built on LangGraph, MCP, FastAPI, and Next.js.

![Studio](docs/screenshots/studio.png)

---

## Overview

SDLC Agent orchestrates a **seven-role agent team** (Coder, Runner, Researcher, Planner, Reviewer, Project Manager, Business Analyst) behind a lightweight LLM router. Each agent operates with a focused tool set, a strategy tailored to its role (ReAct, Plan-and-Execute, Reflexion), and structured prompts with hard tool-call budgets.

Key capabilities:
- **Multi-strategy orchestration** — router_decides, sequential, parallel, supervisor, and **auto** (LLM selects best strategy)
- **Human-in-the-Loop (HITL)** — plan review, action confirmation, clarification, and tool output review via SSE interrupts
- **RAG pipeline** — configurable embedding model, vector DB, chunking, and retrieval strategy with a chat interface
- **Three-layer evaluation** — rule-based metrics → LLM-as-Judge (G-Eval) → DeepEval agentic metrics
- **Golden dataset regression testing** — trace-level assertions, prompt versioning, A/B comparison, and LLM-powered root cause analysis
- **Full observability** — OpenTelemetry spans, real-time trace inspector, monitoring dashboard

---

## Architecture

```
                         User Request
                              │
                              ▼
                  ┌─────────────────────┐
                  │  Meta-Router (LLM)  │  Selects strategy + target agent(s)
                  └──────────┬──────────┘
                             │
      ┌──────────────────────┼──────────────────────┐
      │           Orchestration Strategies           │
      │  router_decides · sequential · parallel ·    │
      │  supervisor · auto (LLM-chosen)              │
      └──────────────────────┬──────────────────────┘
                             │
   ┌───────┬──────┬──────────┼──────────┬──────────┬──────────────┐
   ▼       ▼      ▼          ▼          ▼          ▼              ▼
Coder  Runner  Researcher  Planner  Reviewer  Project Mgr  Business Analyst
ReAct  ReAct    ReAct     Plan+Ex  Reflexion  Plan+Ex        ReAct
   │       │      │          │          │          │              │
   └───────┴──────┴──────────┴──────────┴──────────┴──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  HITL Checkpoints (4 types) │
              │  Plan Review · Action Confirm│
              │  Clarification · Tool Review │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │           MCP Tool Layer              │
         │  filesystem · shell · git · web ·     │
         │  memory · github · jira · planner     │
         └───────────────────────────────────────┘
```

---

## Agent Team

| Agent | Strategy | Tools | Role |
|-------|----------|-------|------|
| **Coder** | ReAct | filesystem, shell, git, github, jira, memory | Code, tests, full SDLC |
| **Runner** | ReAct | shell | Execute a single command or test suite |
| **Researcher** | ReAct | web | Search docs, fetch references, synthesize findings |
| **Planner** | Plan-and-Execute | memory, filesystem | Multi-step coordination with plan review HITL |
| **Reviewer** | Reflexion | filesystem, shell, git, memory | Code/diff review with self-reflection |
| **Project Manager** | Plan-and-Execute | jira, memory | Jira project & issue management with approval gates |
| **Business Analyst** | ReAct | jira, memory | Decompose requirements into Jira tasks |

All prompts are stored in the database, editable via Studio, and sync automatically from `src/agents/prompts.py` on every server start — no manual DB reset needed.

---

## UI Pages

### Agent Studio
Configure teams, agents, skills, tool groups, and per-agent models.

![Studio](docs/screenshots/studio.png)

### Chat
Three-panel layout: conversation sidebar · streaming chat with HITL widgets and thinking box · live trace inspector.

![Chat](docs/screenshots/chat.png)

### RAG Pipeline
Configurable pipeline (embedding model, vector DB, chunking, retrieval strategy) with a minimalist chat interface, citation links, DeepEval scores, and a side-by-side compare mode.

![RAG](docs/screenshots/rag.png)

### Monitoring
Overview metrics, OTel span statistics, token/cost breakdown by model and agent, latency percentiles.

![Monitoring](docs/screenshots/monitoring.png)

### Regression Testing
Golden dataset manager, run configuration (model + strategy + prompt version), per-case results with overlapping radar chart comparison, full agent trajectory, and LLM-powered root cause analysis.

![Regression](docs/screenshots/regression.png)

### Evaluation
Agent and RAG evaluation tabs — G-Eval quality scores, DeepEval agentic/RAG metrics with reasoning, all-request breakdown.

![Evaluation](docs/screenshots/evaluation.png)

---

## Evaluation Pipeline

```
Agent Response
      │
      ├─► Layer 1 · Rule-Based (instant, zero LLM cost)
      │     tool accuracy · step efficiency · routing accuracy
      │     faithfulness · safety/PII · task success
      │
      ├─► Layer 2 · G-Eval / LLM-as-Judge
      │     correctness · completeness · tool usage
      │     efficiency · coherence  (1–5 rubric → 0–1)
      │
      └─► Layer 3 · DeepEval (8 agentic metrics)
            answer relevancy · faithfulness · tool correctness
            argument correctness · task completion · step efficiency
            plan quality · plan adherence
```

**RAG evaluation** uses 5 DeepEval metrics: Answer Relevancy, Faithfulness, Contextual Relevancy, Contextual Precision, Contextual Recall — scored on every chat query and visible in the right panel.

### Regression Testing

Each golden test case specifies: prompt, reference output, expected agent + tools + delegation pattern, quality thresholds, and budgets (LLM calls, tool calls, tokens, latency).

```
Golden Case ──► Agent Execution ──► HITL Auto-Approve ──► Trace Capture
                                                               │
                              ┌────────────────────────────────┤
                              ▼                ▼               ▼
                       Semantic Similarity  G-Eval       DeepEval Agentic
                       (LLM-as-Judge)      (5 criteria)  (8 metrics)
                                                               │
                              ┌────────────────────────────────┤
                              ▼                ▼               ▼
                       Cost Regression   Latency Regression  Quality Regression
```

- **Trace diff**: side-by-side run comparison with overlapping radar chart (blue = Run A, orange = Run B)
- **Root Cause Analysis**: LLM classifies failure cause from trace diff and cost/latency deltas
- **Prompt versioning**: create/edit named versions and A/B test them across regression runs

---

## RAG Pipeline

```
Data Sources (PDF, URL, local files)
          │
          ▼
     Chunking (LangChain RecursiveCharacterTextSplitter / CharacterTextSplitter)
          │
          ▼
     Embedding (OpenRouter embedding models)
          │
          ▼
     Vector Store (ChromaDB / FAISS / in-memory)
          │
          ├─► Retrieval Strategy
          │     dense · MMR rerank · BM25 hybrid (Reciprocal Rank Fusion) · multi-query
          │
          ▼
     LLM Generation + Citation extraction
          │
          ▼
     DeepEval (5 RAG metrics) + OTel spans (rag.ingest / rag.embed / rag.retrieve / rag.generate / rag.evaluate)
```

---

## Getting Started

### Prerequisites

- Python 3.11+, Node.js 18+
- An OpenAI-compatible API (e.g. [Poe](https://poe.com), OpenAI, OpenRouter)

### Install

```bash
git clone https://github.com/your-username/sdlc-agent.git
cd sdlc-agent

# Backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your keys

# Frontend
cd frontend && npm install
```

### Configure `.env`

```bash
# LLM (Poe / OpenAI-compatible)
POE_API_KEY=your_key
LLM_BASE_URL=https://api.poe.com/v1
LLM_MODEL=gpt-5                    # default agent model
LLM_JUDGE_MODEL=deepseek-r1        # G-Eval + DeepEval scoring
LLM_RCA_MODEL=deepseek-r1          # root cause analysis
LLM_ROUTER_MODEL=gpt-4o-mini       # routing (lightweight)

# RAG embeddings (OpenRouter)
OPENROUTER_KEY=your_key

# Optional integrations
GITHUB_TOKEN=...
JIRA_BASE_URL=https://your.atlassian.net
JIRA_EMAIL=you@example.com
JIRA_API_TOKEN=...
DEEPEVAL_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
```

### Run

```bash
# Terminal 1 — backend
source venv/bin/activate && python server.py
# → http://localhost:8000/docs

# Terminal 2 — frontend
cd frontend && npm run dev
# → http://localhost:3000
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph (4 strategies + MemorySaver checkpointer) |
| HITL | LangGraph `interrupt()` / `Command(resume=)` over SSE |
| Tool protocol | MCP — filesystem, shell, git, web, memory, github, jira, planner |
| RAG | LangChain chunkers + ChromaDB/FAISS + rank_bm25 |
| LLM client | LangChain `ChatOpenAI` (any OpenAI-compatible API) |
| Backend | FastAPI + Uvicorn (50+ REST + SSE endpoints) |
| Database | SQLite + SQLAlchemy (auto-migration + prompt sync on startup) |
| Evaluation | Rule-based + G-Eval + DeepEval (8 agentic + 5 RAG metrics) |
| Observability | OpenTelemetry + OpenInference + Langfuse + real-time SSE spans |
| Frontend | Next.js 16 · React 19 · Tailwind CSS 4 · Recharts · react-markdown |
| Testing | pytest — MCP E2E, eval unit, golden integration runners |

---

## Project Structure

```
sdlc-agent/
├── server.py                  # FastAPI: 50+ REST + SSE endpoints
├── main.py                    # CLI: chat, eval, test-mcp
├── src/
│   ├── config.py              # Environment configuration
│   ├── orchestrator.py        # LangGraph multi-agent router + 5 strategies
│   ├── hitl.py                # HITL: 4 checkpoint types + two-phase planner
│   ├── llm/client.py          # LLM factory (agent / judge / rca / router)
│   ├── tools/registry.py      # MCP → LangChain bridge
│   ├── mcp_servers/           # filesystem · shell · git · web · memory
│   │                          # github · jira · planner (MS 365)
│   ├── agents/prompts.py      # Single source of truth for all agent prompts
│   ├── skills/engine.py       # Skill injection + trigger matching
│   ├── rag/
│   │   ├── pipeline.py        # RAG pipeline: chunk, embed, retrieve, generate
│   │   ├── chunker.py         # LangChain-backed chunking strategies
│   │   └── evaluation.py      # DeepEval RAG metrics (5 metrics)
│   ├── evaluation/
│   │   ├── metrics.py         # 7 rule-based metrics
│   │   ├── llm_judge.py       # G-Eval (5 criteria)
│   │   ├── integrations.py    # DeepEval (8 agentic metrics) + Langfuse
│   │   ├── regression.py      # RegressionRunner + HITL auto-approve
│   │   ├── golden_dataset.json# Curated test cases
│   │   └── rca.py             # Root cause analysis
│   ├── tracing/
│   │   ├── collector.py       # OTel spans + OTLP + DB persistence
│   │   └── callbacks.py       # LangChain → TraceCollector
│   └── db/
│       ├── models.py          # SQLAlchemy ORM models
│       └── database.py        # Connection + auto-migration + seeding
├── tests/                     # pytest suites + golden integration runners
├── docs/screenshots/          # UI screenshots (auto-captured)
└── frontend/src/app/
    ├── page.tsx               # Agent Studio
    ├── chat/                  # Chat with HITL + trace inspector
    ├── rag/                   # RAG chat + compare + pipeline config
    ├── monitoring/            # OTel metrics + overview dashboard
    ├── regression/            # Regression testing + trace diff + RCA
    └── evaluation/            # Agent + RAG evaluation history
```

---

## Design Decisions

**Multi-agent over single agent** — each agent has a focused tool set rather than one monolithic agent with everything. The router adds ~200ms but significantly improves tool selection accuracy and keeps prompts small.

**MCP for tools** — standardized protocol decouples agents from tools. Adding Docker, Slack, or a database requires only a new MCP server; agents and orchestrator don't change.

**Per-agent strategies** — ReAct for quick focused tasks, Plan-and-Execute for multi-step coordination, Reflexion for quality-critical review. One-size-fits-all leaves performance on the table.

**Three-layer evaluation** — rule-based catches structural failures instantly at zero LLM cost; G-Eval catches nuanced quality issues; DeepEval provides specialized agentic scoring with detailed reasoning. All three together give comprehensive, non-redundant coverage.

**Prompt versioning** — prompts are the most frequently changed artifact in an agent system. Treating them as versioned code with A/B regression testing applies the same discipline as git for source code.

**SSE over WebSocket** — HTTP-based, auto-reconnect, works through proxies, sufficient for server→client streaming. Granular event types (`agent_start`, `tool_start`, `llm_token`, `trace_span`, `hitl_request`) give the frontend full rendering control.

**Reload restricted to `src/`** — watching the whole repo restarts the server whenever an agent writes a file (e.g. under `tests/`), which resets the in-memory LangGraph checkpointer and breaks in-flight HITL sessions.

---

## License

MIT
