"""
End-to-end SDK demo for the seeded "SDLC 2.0" team.

What this script demonstrates
-----------------------------
1. **Connecting** — `HubClient(team="SDLC 2.0")` resolves to team_id
   ``sdlc_2_0`` and every subsequent call is automatically scoped.
2. **Monitoring** — quick KPI tiles + recent traces (mirrors
   `/monitoring`'s Overview tab).
3. **OTel Observability** — model usage, throughput, top errors, and
   per-model error breakdown (mirrors the OTel Observability tab AFTER
   the team-filter + model-consolidation fixes).
4. **Performance Analysis** — per-agent latency percentiles, cost
   breakdown (mirrors `/regression > Performance Analysis`).
5. **Anomalies** — z-score latency outliers.
6. **Single-Query Inspection** — pick ONE trace by id and pull *only*
   its performance + evaluation data (the most common SDK use-case
   when wiring CI checks or per-run dashboards).

Run from the repo root with the venv activated:

    # Full demo, no extra LLM cost
    python -m scripts.sdk_sdlc20_demo

    # Inspect a specific trace
    python -m scripts.sdk_sdlc20_demo --trace-id 8fd4f1b6c7…

    # ALSO call DeepEval on that single trace (real LLM calls)
    python -m scripts.sdk_sdlc20_demo --trace-id 8fd4f1b6c7… --evaluate

    # Run a brand-new agentic call end-to-end and score all 8 DeepEval metrics
    # on the resulting trace (requires the FastAPI backend on :8000).
    python -m scripts.sdk_sdlc20_demo \
        --run-query "Plan and then execute: read README.md and summarize in 3 bullets"

The output is plain text so you can paste it into a notebook or a
report. Everything you see here is also available in the UI when you
select "SDLC 2.0" in the team dropdown.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from typing import Any

from src.sdk import HubClient


def _truncate(value: Any, n: int = 80) -> str:
    s = str(value)
    return s if len(s) <= n else s[: n - 1] + "…"


def _section(title: str) -> None:
    print()
    print("═" * 78)
    print(title)
    print("═" * 78)


def _sub(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 70 - len(title)))


def _summarize_single_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Derive a compact performance summary from a single trace.

    Uses ONLY the fields returned by ``hub.reports.trace(trace_id)`` —
    no extra DB queries — so the same code can be reused inside a CI
    check or a custom per-trace dashboard.
    """
    spans = trace.get("spans") or []
    by_type: dict[str, int] = defaultdict(int)
    by_tool: dict[str, int] = defaultdict(int)
    by_agent: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "latency_ms": 0.0, "tokens": 0, "cost": 0.0}
    )
    error_spans: list[dict[str, Any]] = []

    for s in spans:
        by_type[s.get("span_type") or "unknown"] += 1
        if s.get("span_type") == "tool_call":
            tool = (s.get("name") or "").replace("tool:", "").replace("tool_call:", "")
            by_tool[tool or "unknown"] += 1
        if s.get("span_type") == "agent_execution":
            agent = (s.get("name") or "").replace("agent_execution:", "").replace("agent:", "")
            bucket = by_agent[agent or "unknown"]
            bucket["calls"] += 1
            bucket["latency_ms"] += s.get("duration_ms") or 0
            bucket["tokens"] += (s.get("tokens_in") or 0) + (s.get("tokens_out") or 0)
            bucket["cost"] += s.get("cost") or 0
        if s.get("status") == "error" and s.get("error"):
            error_spans.append({
                "name": s.get("name"),
                "span_type": s.get("span_type"),
                "error": str(s.get("error"))[:200],
            })

    return {
        "trace_id": trace["id"],
        "status": trace.get("status"),
        "agent_used": trace.get("agent_used"),
        "total_latency_ms": trace.get("total_latency_ms"),
        "total_tokens": trace.get("total_tokens"),
        "total_cost": trace.get("total_cost"),
        "span_count": len(spans),
        "spans_by_type": dict(by_type),
        "tools_called": dict(sorted(by_tool.items(), key=lambda kv: -kv[1])),
        "agents": {a: {**v, "latency_ms": round(v["latency_ms"], 1)}
                   for a, v in by_agent.items()},
        "error_spans": error_spans,
        "eval_status": trace.get("eval_status"),
        "eval_scores": trace.get("eval_scores") or {},
    }


def _print_single_trace_summary(summary: dict[str, Any]) -> None:
    print(f"Trace ID         : {summary['trace_id']}")
    print(f"Status           : {summary['status']}")
    print(f"Agent (router)   : {summary['agent_used'] or '—'}")
    print(f"Total latency    : {(summary['total_latency_ms'] or 0):.0f} ms")
    print(f"Total tokens     : {summary['total_tokens'] or 0:,}")
    print(f"Total cost       : ${(summary['total_cost'] or 0):.4f}")
    print(f"Span count       : {summary['span_count']}")
    print(f"Eval status      : {summary['eval_status']}")

    if summary["spans_by_type"]:
        print("\n  Spans by type:")
        for stype, count in sorted(summary["spans_by_type"].items(), key=lambda kv: -kv[1]):
            print(f"    {stype:<22} {count}")

    if summary["agents"]:
        print("\n  Per-agent breakdown (only agent_execution spans):")
        for agent, d in summary["agents"].items():
            print(
                f"    {agent:<14} calls={int(d['calls']):>2}  "
                f"latency={d['latency_ms']:>7.0f}ms  "
                f"tokens={int(d['tokens']):>6,}  "
                f"cost=${d['cost']:.4f}"
            )

    if summary["tools_called"]:
        print("\n  Top tools used:")
        for tool, count in list(summary["tools_called"].items())[:10]:
            print(f"    {tool:<22} {count}")

    if summary["error_spans"]:
        print(f"\n  Error spans on this trace ({len(summary['error_spans'])}):")
        for e in summary["error_spans"][:5]:
            print(f"    [{e['span_type']:<14}] {e['name']}")
            print(f"      → {_truncate(e['error'], 110)}")


# DeepEval's 8 agentic metrics — kept in fixed order so the table is
# stable across runs and the user can see the full set even when some
# were silently skipped (e.g. tool_correctness when no tool was called).
_DEEPEVAL_METRICS: list[tuple[str, str]] = [
    # (key, group)
    ("deepeval_relevancy",     "standalone"),
    ("deepeval_faithfulness",  "standalone"),
    ("tool_correctness",       "standalone"),
    ("argument_correctness",   "standalone"),
    ("task_completion",        "trace-based"),
    ("step_efficiency_de",     "trace-based"),
    ("plan_quality",           "trace-based"),
    ("plan_adherence",         "trace-based"),
]


def _eval_metric_status(scores: dict[str, Any], key: str) -> tuple[str, str, str]:
    """Return ``(status_icon, score_str, reason)`` for one DeepEval metric.

    Status icons:
      ✓ — scored numerically
      ✗ — attempted but failed (reason starts with "ERROR")
      ⊝ — skipped by design (no trace / no tools / etc.)
      ⊘ — never attempted (key absent from result dict)
    """
    if key not in scores:
        return ("⊘", "—", "not attempted (likely missing precondition)")
    value = scores.get(key)
    reason = str(scores.get(f"{key}_reason") or "")
    if isinstance(value, (int, float)):
        return ("✓", f"{float(value):.3f}", reason)
    if reason.startswith("ERROR"):
        return ("✗", "—", reason)
    return ("⊝", "—", reason or "skipped")


def _run_real_query(backend_url: str, team_id: str, prompt: str) -> str | None:
    """POST a chat request to the running FastAPI backend and return the
    new trace_id. Used by ``--run-query`` to produce a fresh agentic trace
    that is rich enough to exercise all 8 DeepEval metrics.
    """
    try:
        import httpx  # local import — only needed for this code path
    except ImportError:
        print("  ERROR: httpx is required for --run-query. `pip install httpx`")
        return None

    url = f"{backend_url.rstrip('/')}/api/teams/{team_id}/chat"
    print(f"  POST {url}")
    print(f"  prompt: {_truncate(prompt, 100)}")
    try:
        resp = httpx.post(url, json={"message": prompt}, timeout=300.0)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  ERROR contacting backend: {exc}")
        print(f"  Make sure FastAPI is running:  uvicorn server:app --port 8000")
        return None

    body = resp.json()
    trace = body.get("trace") or {}
    trace_id = trace.get("trace_id")
    print(f"  ← agent_used={body.get('agent_used')!r}")
    print(f"  ← tool_calls={len(body.get('tool_calls') or [])}, "
          f"agent_trace_steps={len(body.get('agent_trace') or [])}")
    print(f"  ← response: {_truncate(body.get('response'), 100)}")
    print(f"  ← new trace_id = {trace_id}")
    return trace_id


def _print_deepeval_table(deepeval: dict[str, Any]) -> None:
    """Render the 8-metric DeepEval table (icon · score · reason)."""
    scored = sum(
        1 for k, _ in _DEEPEVAL_METRICS
        if isinstance(deepeval.get(k), (int, float))
    )
    print(
        f"  DeepEval — {scored}/{len(_DEEPEVAL_METRICS)} metrics scored numerically"
        "  (✓=scored  ✗=errored  ⊝=skipped  ⊘=not attempted)"
    )
    last_group = ""
    for key, group in _DEEPEVAL_METRICS:
        if group != last_group:
            print(f"\n  [{group}]")
            last_group = group
        icon, score_str, reason = _eval_metric_status(deepeval, key)
        print(f"    {icon} {key:<24} {score_str:>6}   {_truncate(reason, 90)}")


def _print_eval_scores(scores: dict[str, Any]) -> None:
    """Pretty-print the trace's eval_scores dict (DeepEval + any extras)."""
    if not scores:
        print("  (none — trace has not been evaluated yet)")
        return

    deepeval = scores.get("deepeval_scores") or {}
    if deepeval:
        _print_deepeval_table(deepeval)

    other = {k: v for k, v in scores.items() if k != "deepeval_scores"}
    if other:
        print("\n  Other persisted scores (rule-based / semantic similarity):")
        for k, v in other.items():
            if isinstance(v, dict):
                print(f"    {k}:")
                for kk, vv in v.items():
                    print(f"      {kk}: {vv}")
            else:
                print(f"    {k}: {v}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="SDLC 2.0", help='team name (default: "SDLC 2.0")')
    parser.add_argument("--days", type=int, default=30, help="look-back window")
    parser.add_argument(
        "--trace-id",
        default=None,
        help="full or short prefix of a trace id to inspect in section 6. "
             "If omitted, the most recent COMPLETED trace for the team is used.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run hub.eval.run() on the selected trace. This makes real "
             "DeepEval LLM calls and persists scores back to the trace.",
    )
    parser.add_argument(
        "--run-query",
        default=None,
        metavar="PROMPT",
        help="Send PROMPT to the running backend (POST /api/teams/<id>/chat) "
             "to produce a brand-new agentic trace for this team, then use "
             "that trace as the target for sections 6+. Implies --evaluate. "
             "Requires the FastAPI backend on http://127.0.0.1:8000.",
    )
    parser.add_argument(
        "--backend",
        default="http://127.0.0.1:8000",
        help="Base URL of the FastAPI backend (only used with --run-query).",
    )
    args = parser.parse_args()
    if args.run_query:
        args.evaluate = True  # always score the freshly-produced trace

    print(f"Connecting SDK to team: {args.team!r}")
    hub = HubClient(team=args.team)
    print(f"Resolved team_id = {hub.team_id}")

    # ╭──────────────────────────────────────────────────────────────────────╮
    # │  1.  Monitoring — Overview tab equivalent                           │
    # ╰──────────────────────────────────────────────────────────────────────╯
    _section("1. Monitoring — Overview")
    stats = hub.reports.stats()
    print("hub.reports.stats():")
    print(json.dumps(stats, indent=2, default=str))
    err_rate = (
        stats["errored"] / stats["total_traces"]
        if stats["total_traces"] else 0
    )
    print(
        f"\n  → headline error rate: {err_rate * 100:.1f}% "
        f"({stats['errored']} / {stats['total_traces']} requests)"
    )

    _sub("Most recent 5 traces (hub.reports.list_traces(limit=5))")
    traces = hub.reports.list_traces(limit=5)
    print(f"{len(traces)} trace(s):")
    for t in traces:
        deepeval = (t.get("eval_scores") or {}).get("deepeval_scores") or {}
        numeric = {k: v for k, v in deepeval.items() if isinstance(v, (int, float))}
        de = ", ".join(f"{k}={v:.2f}" for k, v in list(numeric.items())[:2]) or "—"
        print(
            f"  {t['id'][:8]}  "
            f"{t.get('agent_used') or '—':<11}  "
            f"{t['total_latency_ms']:>7.0f}ms  "
            f"{t['total_tokens']:>7} tok  "
            f"${t['total_cost']:>7.4f}  "
            f"{t['status']:<9} "
            f"deepeval=[{de}]  "
            f"{_truncate(t.get('user_prompt'), 36)}"
        )

    # ╭──────────────────────────────────────────────────────────────────────╮
    # │  2.  OTel Observability — what the UI tab now shows                 │
    # ╰──────────────────────────────────────────────────────────────────────╯
    _section(f"2. OTel Observability (hub.reports.otel_stats(days={args.days}))")
    otel = hub.reports.otel_stats(days=args.days)
    print(f"  total_spans      : {otel['total_spans']:,}")
    print(f"  total_traces     : {otel['total_traces']:,}")
    print(f"  error_spans      : {otel['error_spans']:,}   (real exceptions only)")
    print(f"  hitl_pause_spans : {otel.get('hitl_pause_spans', 0):,}   (awaiting user confirmation, not failures)")

    _sub("Per-Model Performance (consolidated)")
    print(
        f"  {'Model':<22} {'Calls':>6} {'Tokens In':>11} {'Tokens Out':>11} "
        f"{'Cost':>10} {'Tok/sec':>8} {'AvgLat':>8} {'ErrRate':>8} {'HITL%':>7}"
    )
    by_model = sorted(
        otel.get("by_model", {}).items(),
        key=lambda kv: -kv[1].get("cost", 0.0),
    )
    for model, d in by_model:
        aliases = d.get("raw_aliases") or []
        merged = f" [{len(aliases)}]" if len(aliases) > 1 else ""
        print(
            f"  {model + merged:<22} "
            f"{d['count']:>6,} "
            f"{d['tokens_in']:>11,} "
            f"{d['tokens_out']:>11,} "
            f"${d['cost']:>9.4f} "
            f"{d.get('tokens_per_sec', 0):>8.1f} "
            f"{d.get('avg_latency_ms', 0):>7.0f}ms "
            f"{d.get('error_rate', 0) * 100:>6.1f}% "
            f"{d.get('hitl_pause_rate', 0) * 100:>6.1f}%"
        )
        if len(aliases) > 1:
            print(f"      ↳ aliases merged: {', '.join(aliases)}")

    _sub("Top 10 Errors  (HITL pauses excluded — they're not failures)")
    if not otel.get("top_errors"):
        print("  (none — all status=error spans were HITL pauses awaiting confirmation)")
    for i, e in enumerate(otel.get("top_errors", []), start=1):
        models = ", ".join(e.get("models", [])) or "—"
        msg = e["message"][:78] + ("…" if len(e["message"]) > 78 else "")
        print(
            f"  {i:>2}. ({e['count']:>4})  {msg:<78}  models=[{models}]"
        )

    _sub("Top 5 Errors per Model  (HITL pauses excluded)")
    for model, errs in (otel.get("errors_by_model") or {}).items():
        total = sum(e["count"] for e in errs)
        print(f"  {model}  ({total} errored span(s))")
        for e in errs:
            print(f"    ({e['count']:>4})  {e['message']}")

    # ╭──────────────────────────────────────────────────────────────────────╮
    # │  3.  Performance Analysis — what /regression shows                  │
    # ╰──────────────────────────────────────────────────────────────────────╯
    _section(f"3. Performance Analysis (hub.reports.performance_report(days={args.days}))")
    report = hub.reports.performance_report(days=args.days)
    print("Top-level keys:", sorted(report.keys()))

    _sub("Agent latency percentiles")
    al = report.get("agent_latency_percentiles") or {}
    for agent, pct in al.items():
        print(
            f"  {agent:<14} "
            f"p50={pct.get('p50', 0):>7.0f}ms  "
            f"p95={pct.get('p95', 0):>7.0f}ms  "
            f"p99={pct.get('p99', 0):>7.0f}ms  "
            f"count={pct.get('count', 0)}"
        )

    cb_summary = (report.get("cost_breakdown") or {}).get("summary") or {}
    if cb_summary:
        _sub("Cost summary (from analyzer)")
        print(json.dumps(cb_summary, indent=2, default=str))

    cb_models = (report.get("cost_breakdown") or {}).get("by_model") or {}
    if cb_models:
        _sub("Cost breakdown by model (analyzer view)")
        for model, d in list(cb_models.items())[:8]:
            print(
                f"  {model:<24} "
                f"${d.get('total_cost_usd', 0):>10.4f}  "
                f"calls={d.get('calls', 0):>5}  "
                f"avg_cost=${d.get('avg_cost_per_call', 0):>9.5f}"
            )

    # ╭──────────────────────────────────────────────────────────────────────╮
    # │  4.  Anomalies                                                      │
    # ╰──────────────────────────────────────────────────────────────────────╯
    _section(f"4. Anomalies (hub.reports.anomalies(days={args.days}, z_threshold=2.5))")
    anoms = hub.reports.anomalies(days=args.days, z_threshold=2.5)
    print(f"{len(anoms)} anomaly(ies):")
    for a in anoms[:10]:
        print(
            f"  [{a.get('severity', '?'):<6}] "
            f"agent={a.get('agent_used', '—'):<10} "
            f"latency={a.get('latency_ms', 0):>7.0f}ms "
            f"z={a.get('z_score', 0):>5.2f}  "
            f"{_truncate(a.get('reason'), 60)}"
        )

    # ╭──────────────────────────────────────────────────────────────────────╮
    # │  6.  Single-Query Inspection                                        │
    # │     Pull performance + evaluation for ONE trace.                    │
    # ╰──────────────────────────────────────────────────────────────────────╯
    _section("6. Single-Query Inspection — performance + evaluation for ONE trace")

    # 6a) Resolve which trace to inspect.
    if args.run_query:
        target_id = _run_real_query(args.backend, hub.team_id, args.run_query)
        if not target_id:
            return 1
    elif args.trace_id:
        target_id: str | None = args.trace_id
        # If a short prefix was passed, expand it against this team's traces.
        if len(target_id) < 32:
            matches = [t for t in hub.reports.list_traces(limit=200) if t["id"].startswith(target_id)]
            if not matches:
                print(f"  No trace found in team {hub.team_id!r} starting with {target_id!r}.")
                return 1
            target_id = matches[0]["id"]
            print(f"  Resolved short prefix {args.trace_id!r} → {target_id}")
    else:
        completed = hub.reports.list_traces(limit=20, status="completed")
        if not completed:
            print("  No completed traces in this team yet.")
            return 0
        target_id = completed[0]["id"]
        print(f"  No --trace-id given; picking most-recent COMPLETED trace: {target_id}")

    # 6b) Performance — hub.reports.trace(trace_id) returns the trace plus
    #     its full ordered span list. From that we derive a per-agent +
    #     per-tool summary without hitting any extra endpoint.
    _sub("hub.reports.trace(trace_id) — single-query performance")
    trace = hub.reports.trace(target_id)
    if trace is None:
        print(f"  Trace {target_id!r} not found in team {hub.team_id!r}.")
        return 1

    print(f"  user_prompt: {_truncate(trace.get('user_prompt'), 100)}")
    print(f"  agent_response: {_truncate(trace.get('agent_response'), 100)}")
    print()
    summary = _summarize_single_trace(trace)
    _print_single_trace_summary(summary)

    # 6c) Evaluation — already-persisted DeepEval scores.
    _sub("Stored evaluation scores (trace.eval_scores)")
    _print_eval_scores(summary["eval_scores"])

    # 6d) On-demand evaluation — only if the user opted in. This makes
    #     real LLM calls, so it's gated behind a flag to keep the demo
    #     cheap by default.
    if args.evaluate:
        _sub("hub.eval.run(trace_id) — on-demand DeepEval (real LLM calls)")
        result = asyncio.run(hub.eval.run(target_id))
        print(f"  evaluated  : {result.get('evaluated')}")
        if result.get("error"):
            print(f"  error      : {result['error']}")
        scores = result.get("deepeval_scores") or {}
        if scores:
            print()
            _print_deepeval_table(scores)
        # Re-fetch to confirm the trace now reports eval_status="evaluated".
        refreshed = hub.reports.trace(target_id) or {}
        print(f"\n  trace.eval_status after run() = {refreshed.get('eval_status')!r}")
    else:
        print("\n  (Skipped on-demand DeepEval. Re-run with --evaluate to score this")
        print("   trace. The flag triggers real LLM calls and persists fresh scores.)")

    _section("UI deep-link")
    print(hub.reports.ui_url())
    print(
        "\nOpen this URL in the dashboard, or pick 'SDLC 2.0' in the team\n"
        "selector at the top of /monitoring, /regression, /evaluation —\n"
        "the data shown there will exactly match this script's output."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
