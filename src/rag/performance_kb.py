"""
Performance Knowledge Base — agentic workflow performance playbooks.

Seeds a dedicated ChromaDB collection with domain-specific content for
diagnosing and improving multi-agent system performance. Content is drawn
from industrial evaluation frameworks (DeepEval, LangSmith, AgentBench,
G-Eval methodology) adapted to this SDLC agent system's architecture.

Coverage:
  - Agent latency attribution (P50/P95/P99 breakdown by component)
  - Cost drivers (token inflation, model tier selection, verbosity)
  - Routing accuracy diagnosis (supervisor misdelegation patterns)
  - DeepEval metric patterns (what low scores mean per agent type)
  - A/B regression comparison methodology
  - Tool failure patterns and recovery strategies
  - Context window management (saturation, RAG-based mitigation)
  - Regression baseline guide (threshold setting, regression detection)

Usage:
    from src.rag.performance_kb import seed_performance_kb, PERF_KB_CONFIG_ID
    await seed_performance_kb()   # idempotent — skips if already seeded
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

PERF_KB_CONFIG_ID = "perf-kb-v1"
PERF_KB_NAME = "Agentic Performance Playbooks"

# ── Playbook documents ─────────────────────────────────────────────────────────

PERFORMANCE_DOCUMENTS: list[dict] = [

    # ── 1. Agent Latency Attribution ──────────────────────────────────────────
    {
        "source": "playbook://agent-latency-attribution",
        "content": """\
# Playbook: Agent Latency Attribution

## Overview
Agent call latency has three distinct components. Misidentifying the dominant
component leads to wrong optimisations (e.g. switching models when the real
bottleneck is tool output size).

## Latency Decomposition
```
total_latency = context_build_ms + llm_call_ms + tool_execution_ms
```

### Component 1 — Context Build Time (5–15% of total)
- Assembling conversation history, system prompt, handoff context
- Grows with conversation length: each prior message adds tokens
- Symptom: latency increases linearly across multi-agent chains
- Fix: truncate or summarize prior messages; use handoff context compression

### Component 2 — LLM Call Time (50–70% of total)
- Time for the model to generate output (first token + generation)
- Grows with: (a) output length, (b) model size, (c) thinking/reasoning mode
- Symptom: latency consistent per agent regardless of input size
- Diagnosis: compare latency between thinking models (Claude + extended_thinking)
  and non-thinking models — thinking adds 5-30s per call
- Fix: reserve thinking models for planner/reviewer; use fast models for coder/tester

### Component 3 — Tool Execution Time (15–45% of total)
- Waiting for tool responses: file I/O, web fetch, git operations, Jira API
- Symptom: latency spikes when specific tools are called (web_search ≫ read_file)
- Common offenders by agent:
  | Agent    | Slow tool         | Typical latency |
  |----------|-------------------|-----------------|
  | coder    | read_file (>50KB) | 2–8s per call   |
  | researcher | web_search      | 3–10s per call  |
  | devops   | git_commit        | 1–5s per call   |
  | tester   | run_tests (pytest)| 5–60s per run   |

## P50/P95/P99 Interpretation
- P50 (median): typical experience, dominated by LLM call time
- P95: captures occasional slow web fetches + large file reads
- P99: exposes worst cases — cold starts, API rate limits, long test suites
- If P99 >> 3× P50: high variance → tool timeouts or context overflow
- If P99 ≈ P50: consistent but slow → LLM model selection issue

## SLO Targets (recommended defaults)
| Agent Role | P50 target | P95 target | P99 target |
|------------|------------|------------|------------|
| researcher | < 15s      | < 30s      | < 60s      |
| coder      | < 20s      | < 45s      | < 90s      |
| tester     | < 30s      | < 90s      | < 180s     |
| planner    | < 25s      | < 60s      | < 120s     |
| devops     | < 15s      | < 40s      | < 80s      |

## Quick Diagnosis Checklist
1. Is p99 > 3× p50? → tool timeout or rate limit issue
2. Is latency growing linearly with conversation turns? → context inflation
3. Is latency consistent but high? → model tier too large
4. Spike only for specific golden tests? → that test's tools are slow
""",
    },

    # ── 2. Cost Attribution ───────────────────────────────────────────────────
    {
        "source": "playbook://cost-attribution",
        "content": """\
# Playbook: Agent Cost Attribution

## Cost Formula
```
cost = (tokens_in × input_price + tokens_out × output_price) / 1_000_000
```

## Primary Cost Drivers (ranked by impact)

### Driver 1 — Input Token Inflation (40–60% of cost)
The most common cost driver in multi-agent systems. Each agent receives
the FULL conversation history as context.

Causes:
- Long tool output included verbatim (e.g. full file content, full git diff)
- Prior agent responses not summarised before handoff
- Repeated reads of the same large file across agent turns
- System prompts are long (300–800 tokens each)

Detection: `tokens_in` grows > 2× between agent 1 and agent 3 in a chain
Fix: Handoff context compression — summarise prior outputs to < 200 tokens each

### Driver 2 — Model Tier Selection (2–10× cost multiplier)
Using Claude Opus or GPT-4o for every agent is the single largest unnecessary cost.

Model cost tiers (approximate USD per 1M tokens in+out):
| Model tier          | Example models        | Relative cost |
|---------------------|-----------------------|---------------|
| Frontier (thinking) | Claude Opus, o1       | 100×          |
| Premium             | Claude Sonnet, GPT-4o | 15–25×        |
| Mid-tier            | Claude Haiku, GPT-4o-mini | 3–5×      |
| Fast/cheap          | Gemini Flash          | 1×            |

Recommended model assignment:
- planner, reviewer: Premium (complex reasoning required)
- coder: Mid-tier (code generation doesn't need frontier reasoning)
- tester, devops: Mid-tier or Fast
- researcher: Mid-tier (web synthesis)
- project_manager: Fast (structured Jira JSON output)

### Driver 3 — Output Token Verbosity (20–35% of cost)
Agents that produce long explanatory text before/after tool calls.

Detection: `tokens_out / tokens_in` ratio > 0.4 (healthy: 0.1–0.25)
Fix: Add to agent prompt — "Be concise. State conclusions, not reasoning steps."

### Driver 4 — Excessive Tool Calls (5–15% of cost)
Each tool call result is appended to context, inflating subsequent LLM calls.

Detection: tool_calls > max_tool_calls threshold in golden test config
Common offenders:
- Coder reading the same file multiple times (max: 1 read per file)
- Researcher fetching every URL (max: 2 fetches per task)
- Planner calling list_directory > 2 times

## Cost Regression Detection
Flag a regression when:
- actual_cost > expected_baseline × 1.5 (50% over baseline)
- tokens_in > 8000 for single-agent tasks
- tokens_in > 20000 for multi-agent chains

## Cost Optimisation Priority
1. Compress handoff context (save 30–50% on multi-agent chains)
2. Downgrade model for coder/tester/devops (save 2–5× per run)
3. Cap tool call verbosity via prompt (save 15–25% on output tokens)
4. Use RAG for large file reads instead of full read_file (save 40–70% tokens)
""",
    },

    # ── 3. Routing Accuracy Diagnosis ────────────────────────────────────────
    {
        "source": "playbook://routing-accuracy-diagnosis",
        "content": """\
# Playbook: Supervisor Routing Accuracy Diagnosis

## What Is Routing Accuracy?
Routing accuracy = fraction of supervisor decisions that selected the correct
agent for each step of a multi-agent task. A score below 0.85 indicates the
supervisor is misdelegating, causing agents to run in wrong order or be skipped.

## Common Misdelegation Patterns

### Pattern 1 — Premature DONE
Symptom: supervisor returns DONE after 1–2 agents when 4–5 were needed
Cause: the supervisor inferred task completion from a positive-sounding agent
       response, even though some deliverables (tests, git push) weren't done
Detection in regression: actual_delegation_pattern shorter than expected_delegation_pattern
Fix: Add explicit completion criteria to supervisor prompt:
     "DONE only when: plan created ∧ code written ∧ tests passed ∧ git pushed"

### Pattern 2 — Wrong First Agent
Symptom: coder or devops called before planner for complex multi-step tasks
Cause: the task prompt contains code keywords ("implement", "create file"),
       causing the supervisor to skip straight to coder
Detection: first entry in actual_delegation_pattern ≠ "planner"
Fix: ReAct step tracking — derive required_steps from task keywords before
     first LLM routing call (prevents LLM from skipping planner)

### Pattern 3 — Agent with Zero Tool Calls
Symptom: an agent is invoked but makes no tool calls (zero tool call count)
Cause: agent's system prompt constraints conflicted with the task, or the
       handoff context made the agent think the task was already done
Detection: any execution entry in agent_trace with tool_calls == []
Fix: Check agent prompt for over-restrictive constraints; verify handoff
     context doesn't claim the task is complete

### Pattern 4 — Devops Running Tests
Symptom: devops agent calls run_command with pytest
Cause: devops prompt allowed run_command broadly; tester was skipped
Detection: "pytest" in devops tool call args, OR tester not in delegation pattern
Fix: Add hard constraint to devops prompt:
     "NEVER use run_command to run pytest, jest, or any test framework"

### Pattern 5 — Planner Writing Files
Symptom: planner agent calls write_file
Cause: planner had write_file in its tool group; task wording said "create"
Detection: "write_file" in planner's tool_calls
Fix: Physical enforcement — give planner only filesystem_read tool group

## Routing Accuracy Thresholds
| Score   | Interpretation |
|---------|----------------|
| ≥ 0.95  | Excellent — supervisor reliably routes correctly |
| 0.85–0.95 | Good — occasional wrong agent but self-corrects |
| 0.70–0.85 | Needs work — systematic misdelegation patterns |
| < 0.70  | Critical — supervisor prompt needs redesign |

## Diagnosis Steps
1. Count tasks where delegation pattern matches expected pattern exactly
2. Identify the most common mismatch type (premature DONE vs wrong order vs skipped)
3. Check if mismatch is model-specific (some models follow instructions better)
4. Review supervisor prompt for the specific rule that was violated
""",
    },

    # ── 4. DeepEval Metric Patterns ───────────────────────────────────────────
    {
        "source": "playbook://deepeval-metric-patterns",
        "content": """\
# Playbook: DeepEval Metric Patterns for SDLC Agents

## Metric Overview
DeepEval provides 5 key metrics for RAG and agent evaluation, each measuring
a different dimension of quality. Low scores mean different things for different
agent types.

## Metric 1 — Answer Relevancy (threshold: 0.7)
Measures: does the response directly address what was asked?
Low score symptoms:
- Agent went off-topic (started with background info, never answered the question)
- Agent delegated to the wrong sub-task (e.g. coder was asked to review but wrote new code)
- Response too brief — answered a different simpler question
Per-agent patterns:
- Researcher low: web_search returned off-topic results; didn't synthesise well
- Coder low: wrote boilerplate instead of the specific logic requested
- Reviewer low: gave generic style feedback instead of addressing the specific code

## Metric 2 — Faithfulness (threshold: 0.7)
Measures: are all claims in the response grounded in the retrieved/tool context?
Low score symptoms:
- Agent hallucinated file paths, function names, or API endpoints not in the code
- Researcher cited information not present in the fetched URLs
- Coder invented method signatures instead of reading the source first
Per-agent patterns:
- Coder low: didn't read source files before writing; guessed APIs → wrong code
- Researcher low: synthesised beyond what the sources actually said
- Reviewer low: claimed bugs exist that aren't in the actual code

Diagnosis: check if read_file was called BEFORE write_file in coder's tool_calls
Fix: enforce "always read the source file before writing" in agent prompt

## Metric 3 — Contextual Recall (threshold: 0.7)
Measures: did the response cover all the relevant information from retrieved context?
Low score symptoms:
- Agent addressed only part of the request (e.g. wrote API but ignored the UI requirement)
- Multi-step task where later steps were silently dropped
- Truncated output due to hitting max_tokens limit
Per-agent patterns:
- Coder low: created only some of the required files
- Tester low: wrote tests for happy path only, missed edge cases
- Planner low: plan missing steps that were in the user's requirements

## Metric 4 — Hallucination (threshold: < 0.3, lower is better)
Measures: rate of factual statements not supported by context
Low score (high hallucination) symptoms:
- Planner invented non-existent files in its architecture review
- DevOps referenced a git branch that doesn't exist
- Coder called methods that don't exist in the imported libraries
Per-agent patterns:
- Most common in planner (reads partial context, extrapolates)
- Second most common in researcher (over-summarises web content)
Fix: add reflexion loop — agent re-reads its own output and checks facts

## Metric 5 — G-Eval Scores (scale 1–5, threshold: 3.5)
Sub-scores measured: correctness, relevance, coherence, tool_usage_quality,
completeness.

| Sub-score         | What it measures | Common low-score cause |
|-------------------|------------------|------------------------|
| correctness       | Factual accuracy | Hallucination (see above) |
| relevance         | On-topic response | Wrong agent routed |
| coherence         | Logical structure | Truncated output, prompt confusion |
| tool_usage_quality| Right tool, right args | N+1 reads, wrong tool selected |
| completeness      | All requirements met | Premature stop, missing steps |

## Metric Correlation Patterns
- Low faithfulness AND low correctness → agent is hallucinating heavily
- Low recall AND low completeness → agent stopping too early (max tool calls hit)
- Low relevance AND wrong delegation pattern → routing accuracy issue, not agent quality
- All metrics low → model too small for the task complexity
""",
    },

    # ── 5. A/B Regression Comparison ─────────────────────────────────────────
    {
        "source": "playbook://agent-ab-comparison",
        "content": """\
# Playbook: A/B Regression Run Comparison

## When to Use A/B Comparison
Compare two eval runs when:
- Changing the LLM model (e.g. claude-sonnet → gpt-4o)
- Modifying an agent prompt (before/after prompt engineering)
- Changing team strategy (router_decides → supervisor)
- Updating tool access (adding/removing tool groups)

## Reading a Metric Delta Table
```
Metric              Run A   Run B   Delta   Significant?
correctness         3.8     4.2     +0.4    YES (>0.3 threshold)
faithfulness        0.72    0.68    -0.04   NO  (<0.1 threshold)
answer_relevancy    0.81    0.85    +0.04   NO
tool_usage_quality  3.4     4.1     +0.7    YES
avg_cost_usd        0.0052  0.0031  -40%    COST WIN
avg_latency_ms      28000   19000   -32%    LATENCY WIN
```

Rules for interpreting deltas:
- G-Eval score delta ≥ 0.3: meaningful quality change
- DeepEval metric delta ≥ 0.1: meaningful quality change
- Cost delta > 20%: investigate model tier or token inflation
- Latency delta > 15%: investigate model size or tool changes
- Any pass_rate delta > 10%: critical — investigate root cause

## Radar Chart Interpretation
A radar chart overlaying two runs shows:
- Symmetric shape: balanced quality, no specific weakness
- Run B bulges on tool_usage_quality but shrinks on faithfulness:
  → more tool calls made, but hallucination rate increased
  → likely the new prompt encouraged more tool use but less fact-checking
- Run B uniformly smaller: model regression or harder test subset
- Run B uniformly larger: genuine improvement across all dimensions

## Statistical Significance for Agent Evaluations
Unlike A/B tests for UI features, agent evaluations have small sample sizes
(typically 10–30 golden tests). Standard t-tests are unreliable at N<30.

Recommended approach:
- Use effect size (Cohen's d) instead of p-values
  - d < 0.2: negligible difference
  - d 0.2–0.5: small but possibly meaningful
  - d > 0.5: large, definitely meaningful
- For cost/latency: 20% delta is the practical significance threshold
- For quality scores: 0.3/5.0 delta (6%) is the practical threshold

## Common A/B Pitfalls
1. Comparing runs with different golden test subsets (different N)
   → Always compare same set of golden test IDs
2. One run has more timed-out tests (inflating pass rate of other)
   → Check actual_latency_ms outliers, exclude or normalise
3. Confounding: model changed AND strategy changed simultaneously
   → Change one variable at a time for clean comparison
4. Novelty effect: first run with new model may perform differently
   → Run each config 3× and average

## Decision Framework
| Scenario | Recommendation |
|---|---|
| Run B better on quality AND cost | Ship Run B config |
| Run B better on quality, worse on cost | Evaluate cost/quality tradeoff |
| Run B worse on faithfulness only | Fix: add "read before write" prompt |
| Run B better on cost only | Acceptable only if quality delta < 0.2 |
| All metrics roughly equal | Keep current (simpler is better) |
""",
    },

    # ── 6. Tool Failure Patterns ──────────────────────────────────────────────
    {
        "source": "playbook://tool-failure-patterns",
        "content": """\
# Playbook: Tool Failure Patterns and Recovery Strategies

## Tool Failure Rate Benchmarks
Healthy tool failure rates for an SDLC agent system:
| Tool          | Healthy failure rate | Warning | Critical |
|---------------|---------------------|---------|---------|
| read_file     | < 2%                | 2–5%    | > 5%    |
| write_file    | < 1%                | 1–3%    | > 3%    |
| web_search    | < 5%                | 5–15%   | > 15%   |
| fetch_url     | < 10%               | 10–25%  | > 25%   |
| run_tests     | < 15%               | 15–30%  | > 30%   |
| git_commit    | < 3%                | 3–8%    | > 8%    |
| jira_*        | < 5%                | 5–10%   | > 10%   |

## Pattern 1 — read_file Truncation (most common)
Symptom: response contains "⚠️ FILE TRUNCATED" warning
Cause: file > 100KB threshold; full content not returned
Impact: agent makes decisions based on partial information → low faithfulness
Recovery: use RAG-based read with query parameter:
  read_file(path, query="what you're looking for")
Prevention: planner and coder should detect truncation and retry with query=

## Pattern 2 — fetch_url 403/429 (frequent for researcher)
Symptom: "FETCH FAILED: 403 Forbidden" or "429 Too Many Requests"
Cause: website blocks bots, or rate limiting
Impact: researcher hallucinates or uses stale snippet data
Recovery: immediately fall back to web_search on same topic
Prevention: check_url before fetch_url; add exponential backoff for 429s

## Pattern 3 — run_tests Failure (expected, not a tool error)
Symptom: exit code 1, test failures reported
Cause: this is EXPECTED — tests failing means code has bugs
Impact: if tester doesn't retry, devops pushes broken code
Recovery: tester should fix test or report failure to coder for code fix
Prevention: max 2 fix cycles (write → run → fix → run); escalate if still failing

## Pattern 4 — git_commit Pre-Hook Rejection
Symptom: "pre-commit hook failed" or "gpg signing failed"
Cause: repo has pre-commit hooks (linting, signing) not satisfied
Impact: devops fails to commit, leaves work uncommitted
Recovery: run lint/format first (run_command "black src/ && flake8 src/")
Prevention: devops prompt should check git_status before committing

## Pattern 5 — Jira Authentication Failure
Symptom: "401 Unauthorized" or "403 Forbidden" from Jira API
Cause: JIRA_API_TOKEN or JIRA_EMAIL env var not set / expired
Impact: project_manager fails to create tickets
Recovery: report configuration error to user; do not retry
Detection: error contains "Unauthorized" AND tool is jira_*

## Cascading Failure Pattern
When one tool fails, subsequent agents often fail too:
```
Coder: write_file fails (disk full) → incomplete code
Tester: read_file on missing file → 0 tests written
DevOps: git_commit on empty change → empty commit
→ All 3 agents fail in cascade
```
Recovery: supervisor should detect that coder had 0 successful tool_calls
and re-route to coder before proceeding to tester/devops.

## Tool Failure Rate Spike Diagnosis
If tool failure rate spikes suddenly:
1. check run_command for "permission denied" → file permission issue
2. check web tools for "connection timeout" → network issue
3. check git tools for "authentication failed" → expired token
4. check jira/github tools for "rate limit" → API quota exhausted
""",
    },

    # ── 7. Context Window Management ─────────────────────────────────────────
    {
        "source": "playbook://context-window-management",
        "content": """\
# Playbook: Context Window Management for Multi-Agent Systems

## The Context Inflation Problem
In a multi-agent chain (planner → coder → tester → devops), each subsequent
agent receives ALL prior messages as context. By the time devops runs:

```
planner output:    ~2,000 tokens
coder output:      ~4,000 tokens (code + explanations)
tester output:     ~1,500 tokens (test results)
+ system prompts:  ~3,000 tokens (4 agents × ~750 tokens each)
+ original task:   ~500 tokens
= devops context:  ~11,000 tokens (before devops says anything)
```

With a 128K context window (GPT-4o), this is only 8.5% usage — fine.
With a 200K window (Claude), this is 5.5% — fine.
But at 10 agents or long file reads, context can reach 50–80% utilisation.

## Warning Thresholds
| Context utilisation | Status | Action |
|---------------------|--------|--------|
| < 30%              | Healthy | No action needed |
| 30–60%             | Warning | Monitor, consider handoff compression |
| 60–80%             | Degraded | Apply handoff compression now |
| > 80%              | Critical | Model may truncate; quality degrades |

## Handoff Context Compression (Primary Fix)
Instead of passing all prior messages, compress to a structured summary:

```
ORIGINAL REQUEST: Build a FastAPI task manager with tests

WORKFLOW CONTEXT (completed):
  • planner: used [list_directory, read_file] — created 5-step plan
  • coder: used [write_file×4] — wrote main.py, index.html, requirements.txt, tests/test_main.py
  • tester: used [run_tests] — 3/3 tests passed (exit code 0)

YOUR TASK: You are devops. Commit and push code, create GitHub repo.
```

This handoff context is ~100 tokens vs ~7,500 tokens of raw conversation.

## RAG for Large File Reads (Secondary Fix)
When an agent needs to read a large file (> 50KB), full reads are expensive:
- Full read of server.py (2800 lines): ~8,000 tokens
- RAG-targeted read with query="authentication endpoints": ~800 tokens
  → 90% token reduction

Pattern: read_file(path, query="specific thing you need")

## When Context Utilisation Is High
Signs the model is context-saturated:
1. Answer quality drops on later agents in the chain
2. Agent "forgets" instructions from the system prompt
3. Agent repeats work already done by prior agents
4. Coherence score in G-Eval drops from 4.0+ to 2.5–3.0

Fixes in priority order:
1. Enable handoff context compression (src/orchestrator.py _build_handoff_context)
2. Use RAG for file reads (read_file with query=)
3. Limit tool output to first 2000 chars
4. Switch to model with larger context window
5. Reduce number of agents in chain (merge closely-related roles)
""",
    },

    # ── 8. Regression Baseline Guide ─────────────────────────────────────────
    {
        "source": "playbook://regression-baseline-guide",
        "content": """\
# Playbook: Regression Baseline Guide for SDLC Agent Evaluation

## What Is a Regression Baseline?
A baseline is the set of quality/performance metrics from a known-good
eval run. Future runs are compared against it. A regression is when any
metric drops below the baseline by more than its tolerance.

## Recommended Quality Thresholds

### G-Eval Sub-scores (scale 1–5)
| Sub-score         | Minimum passing | Regression delta |
|-------------------|-----------------|------------------|
| correctness       | 3.5             | -0.5             |
| relevance         | 3.5             | -0.5             |
| coherence         | 3.5             | -0.5             |
| tool_usage_quality| 3.5             | -0.5             |
| completeness      | 3.5             | -0.5             |

### DeepEval Metrics (scale 0–1)
| Metric            | Minimum passing | Regression delta |
|-------------------|-----------------|------------------|
| answer_relevancy  | 0.70            | -0.10            |
| faithfulness      | 0.70            | -0.10            |
| contextual_recall | 0.65            | -0.10            |
| hallucination     | < 0.30          | +0.10            |
| semantic_similarity | 0.60          | -0.10            |

### Performance Metrics
| Metric            | Regression condition |
|-------------------|---------------------|
| actual_latency_ms | > baseline × 1.5 (50% increase) |
| actual_cost       | > baseline × 1.5 (50% increase) |
| actual_tool_calls | > max_tool_calls threshold |
| actual_llm_calls  | > max_llm_calls threshold |

## Metric Consistently Below Threshold: Root Cause Guide

### faithfulness consistently low (< 0.70)
1. Check: coder calling write_file before read_file?
2. Check: researcher synthesising beyond source content?
3. Fix: add "read before write" rule; add reflexion loop

### answer_relevancy consistently low (< 0.70)
1. Check: is the task being routed to the right agent?
2. Check: is agent addressing the task or doing adjacent work?
3. Fix: improve router prompt or agent's task interpretation

### completeness consistently low (< 3.5)
1. Check: is max_tool_calls being hit mid-task?
2. Check: is agent stopping after first sub-task?
3. Fix: increase max_tool_calls; add "don't stop until all steps done" to prompt

### tool_usage_quality consistently low (< 3.5)
1. Check: N+1 read_file calls (reading same file multiple times)?
2. Check: wrong tool for job (read_file vs search_files)?
3. Fix: add explicit tool selection rules to agent prompt

## A/B Baseline Comparison Process
1. Run current config → save as baseline eval_run
2. Make ONE change (model / prompt / strategy)
3. Run same golden tests → new eval_run
4. Compare: actual_cost, actual_latency_ms, quality_scores, deepeval_scores
5. Accept change if: quality_delta > -0.1 AND (cost_delta < 0 OR latency_delta < 0)

## Setting Baselines for New Golden Tests
For each new golden test:
1. Run it 3 times with the current best config
2. Take the MINIMUM quality scores as thresholds (conservative baseline)
3. Take the MAXIMUM latency/cost as upper bounds
4. Set max_tool_calls = observed_tool_calls + 2 (give headroom)
""",
    },
]


# ── Seed function ──────────────────────────────────────────────────────────────

async def seed_performance_kb(force: bool = False) -> dict:
    """
    Create and seed the Agentic Performance Knowledge Base RAG pipeline.
    Idempotent: skips if KB already has documents unless force=True.
    """
    from src.db.database import get_session
    from src.db.models import RagConfig
    from src.rag.pipeline import RAGConfig, get_pipeline

    session = get_session()
    try:
        existing = session.query(RagConfig).filter_by(id=PERF_KB_CONFIG_ID).first()
        if existing is None:
            existing = RagConfig(
                id=PERF_KB_CONFIG_ID,
                name=PERF_KB_NAME,
                description=(
                    "Playbooks for diagnosing and improving multi-agent system performance: "
                    "latency attribution, cost drivers, routing accuracy, DeepEval metric "
                    "patterns, A/B comparison methodology, tool failure patterns, "
                    "context window management, and regression baseline setting."
                ),
                embedding_model="mistralai/codestral-embed-2505",
                vector_store="chroma",
                chunk_size=800,
                chunk_overlap=100,
                chunk_strategy="recursive",
                retrieval_strategy="hybrid",
                top_k=4,
                mmr_lambda=0.7,
                system_prompt=(
                    "You are an agentic systems performance expert. Answer questions about "
                    "multi-agent latency, cost, routing accuracy, evaluation metrics, A/B "
                    "comparison, tool failures, and regression detection using the provided "
                    "playbooks. Always cite your sources and give actionable recommendations."
                ),
                is_active=True,
            )
            session.add(existing)
            session.commit()
            logger.info("Created Agentic Performance KB config: %s", PERF_KB_CONFIG_ID)
    finally:
        session.close()

    cfg = RAGConfig(
        config_id=PERF_KB_CONFIG_ID,
        name=PERF_KB_NAME,
        embedding_model="mistralai/codestral-embed-2505",
        vector_store="chroma",
        chunk_size=800,
        chunk_overlap=100,
        chunk_strategy="recursive",
        retrieval_strategy="hybrid",
        top_k=4,
        mmr_lambda=0.7,
        system_prompt=(
            "You are an agentic systems performance expert. Answer questions about "
            "multi-agent latency, cost, routing accuracy, evaluation metrics, A/B "
            "comparison, tool failures, and regression detection using the provided "
            "playbooks. Always cite your sources and give actionable recommendations."
        ),
    )
    pipeline = get_pipeline(cfg)

    if pipeline.chunk_count() > 0 and not force:
        logger.info(
            "Agentic Performance KB already seeded (%d chunks). Use force=True to re-seed.",
            pipeline.chunk_count(),
        )
        return {
            "config_id": PERF_KB_CONFIG_ID,
            "chunks_ingested": 0,
            "skipped": True,
            "existing_chunks": pipeline.chunk_count(),
        }

    total_chunks = 0
    for doc in PERFORMANCE_DOCUMENTS:
        try:
            result = await pipeline.ingest("text", doc["content"])
            total_chunks += result.get("chunks", 0)
            logger.info("Seeded %s → %d chunks", doc["source"], result.get("chunks", 0))
        except Exception as e:
            logger.warning("Failed to seed %s: %s", doc["source"], e)

    logger.info(
        "Agentic Performance KB seeded: %d documents → %d total chunks",
        len(PERFORMANCE_DOCUMENTS), total_chunks,
    )
    return {
        "config_id": PERF_KB_CONFIG_ID,
        "chunks_ingested": total_chunks,
        "skipped": False,
        "documents": len(PERFORMANCE_DOCUMENTS),
    }
