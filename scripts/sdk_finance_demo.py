"""
End-to-end SDK demo for the Finance Team.

Walks through every SDK surface the user asked for:

    1. ``HubClient.auto_create`` — stand up the team + 6 agents
       (Market / News / Social / Fundamentals / Risk Analysts + Trader).
    2. ``hub.tools.list_groups`` — confirm the finance MCP tool groups
       are registered (finance_market / finance_news / finance_fundamentals).
    3. ``hub.prompts.register`` + ``set_agent_version`` — push versioned
       prompts for every finance agent.
    4. ``hub.rag.create_config`` + ``ingest_paths`` — seed a RAG corpus
       from a directory of finance research notes.
    5. ``hub.regression.run`` — execute the dedicated ``finance_v1``
       golden set with full DeepEval scoring.
    6. ``hub.optimizer.run`` (dry-run) — show the prompt-optimization
       loop for the Market Analyst.
    7. ``hub.chat.send`` — issue a comprehensive trade-thesis query and
       inspect the trace + per-trace evaluation in one shot.

Run from the repo root:

    # Stand the team up + show registry/prompt/rag wiring (no LLM calls yet)
    PYTHONPATH=. python -m scripts.sdk_finance_demo

    # ALSO run regression on the finance set (real LLM calls)
    PYTHONPATH=. python -m scripts.sdk_finance_demo --run-regression

    # ALSO issue a live trade-thesis chat against the running backend
    PYTHONPATH=. python -m scripts.sdk_finance_demo --run-query \
        "Should I buy NVDA today? Give a thesis with conviction and risks."

The Studio UI will reflect everything this script produces — the team
appears in the team dropdown, agents are visible in `/`, prompts in
`/prompts`, RAG sources in `/rag`, regression results in `/regression`,
and chat traces in `/monitoring`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.agents.finance_prompts import (
    FINANCE_AGENT_PROMPTS,
    FINANCE_PROMPT_VERSION,
)
from src.sdk import HubClient


# ── Constants ────────────────────────────────────────────────────────────

FINANCE_TEAM_ID = "finance_team"
FINANCE_TEAM_NAME = "Finance Team"
FINANCE_DEFAULT_MODEL = "claude-sonnet-4.6"

# Default config — wired into ``team.config_json``. The Coordinator will
# fan out to the four research analysts in parallel, then route to Trader.
FINANCE_TEAM_CONFIG: dict[str, Any] = {
    "dataset_groups": ["finance_v1"],
    "parallel_coordinator": {
        "parallel_pool": [
            "market_analyst",
            "news_analyst",
            "social_analyst",
            "fundamentals_analyst",
            "risk_analyst",
        ],
        "finalizer_role": "trader",
        "max_parallel": 5,
    },
}


def _agent_def(
    aid: str,
    name: str,
    description: str,
    tool_groups: list[str],
    *,
    decision_strategy: str = "react",
) -> dict[str, Any]:
    prompt_text, _ = FINANCE_AGENT_PROMPTS[aid]
    return {
        "id":               aid,
        "name":             name,
        "role":             aid,
        "description":      description,
        "system_prompt":    prompt_text,
        "model":            FINANCE_DEFAULT_MODEL,
        "decision_strategy": decision_strategy,
        "prompt_version":   FINANCE_PROMPT_VERSION,
        "tool_groups":      tool_groups,
    }


FINANCE_AGENTS: list[dict[str, Any]] = [
    _agent_def(
        "market_analyst",
        "Market Analyst",
        "Reads OHLCV + technical indicators (MACD, RSI, Bollinger).",
        ["finance_market"],
    ),
    _agent_def(
        "news_analyst",
        "News Analyst",
        "Surfaces ticker + macro headlines and assigns sentiment.",
        ["finance_news"],
    ),
    _agent_def(
        "social_analyst",
        "Social Analyst",
        "Approximates retail sentiment from news headlines as a proxy.",
        ["finance_news"],
    ),
    _agent_def(
        "fundamentals_analyst",
        "Fundamentals Analyst",
        "Pulls income / balance / cash-flow statements + key ratios.",
        ["finance_fundamentals"],
    ),
    _agent_def(
        "risk_analyst",
        "Risk Analyst",
        "Volatility + idiosyncratic + macro risk overlay; runs in parallel "
        "with the other analysts and produces an independent risk matrix.",
        ["finance_market", "finance_news"],
    ),
    _agent_def(
        "trader",
        "Trader",
        "Synthesizes analyst + risk reports into a strict-JSON trade decision.",
        [],   # Trader synthesises text; no tool calls needed
    ),
]


# ── Helpers ──────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    print()
    print("═" * 78)
    print(title)
    print("═" * 78)


def _truncate(s: Any, n: int = 90) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


# ── Steps ────────────────────────────────────────────────────────────────


def step_create_team() -> HubClient:
    _section("1. Auto-create Finance Team (HubClient.auto_create)")
    hub = HubClient.auto_create(
        team_id=FINANCE_TEAM_ID,
        name=FINANCE_TEAM_NAME,
        description="Multi-analyst trading team backed by yfinance + stockstats.",
        decision_strategy="parallel_coordinator",
        config_json=FINANCE_TEAM_CONFIG,
        agents=FINANCE_AGENTS,
    )
    print(f"  ✓ {hub}")
    payload = hub.teams.get()
    if payload:
        print(f"  strategy: {payload['decision_strategy']}")
        print(f"  agents:   {len(payload['agents'])}")
        for a in payload["agents"]:
            print(
                f"    · {a['id']:<22} model={a['model']:<22} "
                f"tools={a['tool_groups']}"
            )
    return hub


def step_show_tools(hub: HubClient) -> None:
    _section("2. Tool Registry (hub.tools.list_groups)")
    groups = hub.tools.list_groups()
    finance_keys = sorted(k for k in groups if k.startswith("finance_"))
    if not finance_keys:
        print("  ! finance MCP groups not registered — check src/tools/registry.py")
        return
    for g in finance_keys:
        tools = groups[g]
        print(f"  {g:<25} {len(tools):>2} tools  →  {', '.join(tools)}")


def step_register_prompts(hub: HubClient) -> None:
    _section("3. Prompt Registry (hub.prompts.register)")
    for role, (prompt_text, _) in FINANCE_AGENT_PROMPTS.items():
        new_version = hub.prompts.register(
            role=role,
            prompt_text=prompt_text,
            rationale=(
                "Initial Finance Team prompts — Tauric-inspired structure with "
                "explicit tool budgets, source-grounding, and a fixed Markdown "
                "schema downstream agents can parse deterministically."
            ),
            scope="team",
        )
        hub.prompts.set_agent_version(role, new_version)
        print(f"  registered  {role:<22} → {new_version}  (active on agent)")
    print()
    print("  Per-role version history (team-scoped, latest first):")
    for role in FINANCE_AGENT_PROMPTS.keys():
        versions = hub.prompts.list_versions(role)
        latest = versions[0] if versions else None
        if latest:
            print(
                f"    · {role:<22} latest={latest['version']:<5} "
                f"by={latest['created_by']}"
            )


def step_seed_rag(hub: HubClient) -> None:
    _section("4. RAG (hub.rag.create_config + ingest_paths)")
    cfg = hub.rag.create_config(
        config_id="finance_research_v1",
        name="Finance Team — Research Corpus",
        description="Analyst notes, sector primers, earnings transcripts.",
        embedding_model="openai/text-embedding-3-small",
        vector_store="chroma",
        chunk_strategy="parent_child",
        chunk_size=900,
        chunk_overlap=180,
        retrieval_strategy="hybrid",
        top_k=6,
        reranker="bge-reranker-base",
    )
    print(f"  ✓ config_id={cfg['id']}  strategy={cfg['retrieval_strategy']}")
    print(
        f"    chunk_strategy={cfg['chunk_strategy']}  "
        f"reranker={cfg['reranker']}  vector_store={cfg['vector_store']}"
    )

    # Best-effort ingest from a sample directory if it exists in the repo.
    sample_dirs = [
        Path("data/finance_corpus"),
        Path("docs/finance"),
    ]
    targets = [str(p) for p in sample_dirs if p.exists()]
    if targets:
        summary = hub.rag.ingest_paths(cfg["id"], paths=targets)
        print(
            f"  ingest_paths: ingested={summary['files_ingested']} "
            f"skipped={summary['files_skipped']} "
            f"chunks={summary['total_chunks']}"
        )
    else:
        print(
            "  (no data/finance_corpus or docs/finance directory present — "
            "skipping ingest, RAG config still registered for the UI)"
        )


def step_regression(hub: HubClient, run: bool) -> None:
    _section("5. Regression (hub.regression — finance_v1 dataset_group)")
    cases = hub.regression.list_cases()
    print(f"  Visible cases for this team: {len(cases)}")
    for c in cases:
        print(
            f"    · {c['id']:<8} [{c['dataset_group']}]  "
            f"{_truncate(c['name'], 60)}"
        )
    if not run:
        print()
        print("  (skipping live run — pass --run-regression to execute)")
        return
    print()
    print("  Running regression — this calls real LLM endpoints, may take minutes…")
    summary = hub.regression.run(prompt_version=FINANCE_PROMPT_VERSION)
    if "error" in summary:
        print(f"  ! {summary['error']}")
        return
    print(f"  run_id   : {summary.get('eval_run_id')}")
    print(f"  passed   : {summary.get('passed')}/{summary.get('total')}")
    print(f"  pass_rate: {summary.get('pass_rate')}")


def step_optimizer(hub: HubClient, run: bool) -> None:
    _section("6. Prompt Optimizer (hub.optimizer.run, dry-run)")
    if not run:
        print(
            "  (skipping — pass --run-optimizer to drive a single dry-run cycle "
            "for the market_analyst role)"
        )
        return
    report = hub.optimizer.run(
        role="market_analyst",
        metric="step_efficiency",
        threshold=0.7,
        dry_run=True,
        max_cycles=1,
        model=FINANCE_DEFAULT_MODEL,
    )
    print(f"  classification:    {report.get('classification')}")
    print(f"  baseline avg:      {report.get('baseline_metric_avg')}")
    print(f"  best attempt avg:  {report.get('best_metric_avg')}")
    print(f"  commit_status:     {report.get('commit_status')}")


def step_chat(hub: HubClient, query: str | None) -> None:
    _section("7. Chat + per-trace inspection (hub.chat.send)")
    if not query:
        print(
            "  (skipping — pass --run-query 'your question' to issue a live "
            "chat against the running FastAPI backend)"
        )
        return
    print(f"  POST → /api/teams/{hub.team_id}/chat")
    try:
        result = hub.chat.send(query, transport="http")
    except Exception as exc:
        print(f"  ! chat failed: {exc}")
        print("    (start the backend with `uvicorn server:app --port 8000` first)")
        return
    print(f"  ← agent_used = {result.get('agent_used')}")
    trace_id = (result.get("trace") or {}).get("trace_id")
    print(f"  ← trace_id   = {trace_id}")
    print(f"  ← response   : {_truncate(result.get('response'), 100)}")

    if not trace_id:
        return

    print("\n  Single-trace performance + evaluation:")
    trace_payload = hub.reports.trace(trace_id) or {}
    if trace_payload:
        print(
            f"    latency: {trace_payload.get('latency_ms')}ms  "
            f"tokens_in={trace_payload.get('tokens_in')} "
            f"tokens_out={trace_payload.get('tokens_out')}"
        )
    import asyncio
    eval_payload = asyncio.run(hub.eval.run(trace_id))
    print(f"\n  DeepEval payload (truncated):")
    print(json.dumps(eval_payload, indent=2)[:600])


# ── Entrypoint ───────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="SDK demo for the Finance Team.")
    parser.add_argument("--run-regression", action="store_true",
                        help="Run the finance_v1 regression suite (real LLM calls).")
    parser.add_argument("--run-optimizer", action="store_true",
                        help="Run a single dry-run optimization cycle for market_analyst.")
    parser.add_argument("--run-query", type=str, default=None,
                        help="Issue a live chat against the running backend.")
    args = parser.parse_args()

    hub = step_create_team()
    step_show_tools(hub)
    step_register_prompts(hub)
    step_seed_rag(hub)
    step_regression(hub, run=args.run_regression)
    step_optimizer(hub, run=args.run_optimizer)
    step_chat(hub, query=args.run_query)
    print()
    print("All done. Open the UI to see the team / agents / prompts / RAG / regression.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
