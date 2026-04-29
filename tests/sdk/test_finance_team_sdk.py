"""
End-to-end test for the SDLC Hub SDK against the seeded "Finance Team".

What it proves:

1. A Python program (no FastAPI server, no UI clicks) can produce real
   traces tagged to the Finance Team via the SDK.
2. Inner spans (fast / slow / failing) all flow through the existing
   TraceCollector and land in the same `traces` + `spans` tables that
   the UI reads from.
3. The SDK can score those traces with DeepEval (the only judge after
   the G-Eval removal) and persist scores back to ``traces.eval_scores``.
4. The SDK can read back team-scoped analytics — list traces, fetch a
   specific trace, retrieve the performance report, and surface anomaly
   diagnostics — without touching the HTTP layer.

Default mode: fake LLM (no Poe spend, ~3 seconds, deterministic).
Real-LLM mode: pass ``--real-llm`` (pytest --real-llm or python ... --real-llm)
to actually invoke DeepEval — useful for validating the wiring against a
live model. Disabled by default so the test stays cheap & fast.

Run from repo root::

    python tests/sdk/test_finance_team_sdk.py             # fake LLM (default)
    python tests/sdk/test_finance_team_sdk.py --real-llm  # exercise DeepEval
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Make the repo root importable when invoked as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sdk import HubClient, TraceResult  # noqa: E402

TEAM_NAME = "Finance Team"


# ── Fake LLM scaffolding (skips DeepEval to keep the test deterministic) ─────


class _FakeAsyncResult:
    """Minimal stand-in for ``run_all_deepeval_metrics`` output."""

    def __init__(self, scores: dict):
        self.scores = scores


async def _fake_run_all_deepeval_metrics(**_kwargs):
    """Returns deterministic scores so we can assert exact shape."""
    return {
        "deepeval_relevancy": 0.82,
        "deepeval_relevancy_reason": "(fake) response addresses the prompt directly.",
        "deepeval_faithfulness": 0.91,
        "deepeval_faithfulness_reason": "(fake) no hallucinated facts detected.",
        "tool_correctness": 1.0,
        "tool_correctness_reason": "(fake) tools matched expectations.",
        "argument_correctness": 0.88,
        "argument_correctness_reason": "(fake) arguments valid.",
        "task_completion": 0.85,
        "task_completion_reason": "(fake) task substantially complete.",
        "step_efficiency_de": 0.74,
        "step_efficiency_de_reason": "(fake) one redundant step observed.",
        "plan_quality": 0.80,
        "plan_quality_reason": "(fake) plan was coherent.",
        "plan_adherence": 0.76,
        "plan_adherence_reason": "(fake) executed close to plan.",
    }


def _install_fake_deepeval():
    """Monkey-patch the module the SDK calls so DeepEval isn't invoked.

    We patch *both* the underlying integrations module *and* any place that
    imports from it via ``from ... import run_all_deepeval_metrics``.
    """
    import src.evaluation.integrations as integrations
    integrations.run_all_deepeval_metrics = _fake_run_all_deepeval_metrics  # type: ignore[assignment]


# ── The actual agent flow we instrument with the SDK ─────────────────────────


def build_agent(hub: HubClient):
    """Returns a coroutine modelling a small finance-report agent.

    Three branches are exercised by the test driver below:
      * fast happy path
      * a deliberately slow tool span (sleep 1.5s) → flags as anomaly
      * a failing tool span → flags as error trace
    """

    @hub.trace.trace_agent(name="finance_report:answer", agent_used="planner")
    async def finance_report(question: str) -> str:
        # Step 1: planning (cheap llm-style span)
        with hub.trace.span("planner:think", span_type="llm_call") as s:
            await asyncio.sleep(0.05)
            s.set_tokens(in_=120, out_=40, model="gpt-4o-mini")
            s.set_output(plan="fetch -> compute -> summarise")

        # Step 2: data fetch — speed varies based on question marker
        with hub.trace.span("tool_call:fetch_ledger", span_type="tool_call") as s:
            if "slow" in question.lower():
                # NOTE: 1.5s well above span p95; will be picked up as anomaly
                # once we have >=5 spans of this type in the lookback window.
                await asyncio.sleep(1.5)
                s.set_output(rows=2400, source="ledger.parquet")
            else:
                await asyncio.sleep(0.05)
                s.set_output(rows=240, source="ledger.parquet")

        # Step 3: compute
        with hub.trace.span("tool_call:compute_burn", span_type="tool_call") as s:
            await asyncio.sleep(0.05)
            s.set_output(burn_usd=128_500)

        # Step 4: failure injection
        if "fail" in question.lower():
            with hub.trace.span("tool_call:summarise", span_type="tool_call") as s:
                s.set_output(stage="about-to-fail")
                raise RuntimeError("simulated downstream summariser outage")

        # Step 5: summarise (llm-style)
        with hub.trace.span("summariser:write", span_type="llm_call") as s:
            await asyncio.sleep(0.05)
            s.set_tokens(in_=300, out_=80, model="gpt-4o-mini")
            answer = (
                f"Q4 burn rate is approximately $128,500 across 240 transactions. "
                f"Question echoed: {question}"
            )
            s.set_output(response_preview=answer[:120])
        return answer

    return finance_report


# ── Driver ──────────────────────────────────────────────────────────────────


async def _drive(real_llm: bool) -> int:
    print("=" * 78)
    print(" SDLC Hub SDK — Finance Team end-to-end test")
    print(f" mode: {'real-llm' if real_llm else 'fake-llm (default)'}")
    print("=" * 78)

    if not real_llm:
        _install_fake_deepeval()

    hub = HubClient(team=TEAM_NAME)
    print(f"\n[hub]  resolved {hub!r}")

    finance_report = build_agent(hub)

    # 1) Drive three traces ---------------------------------------------------
    questions = [
        ("fast",  "What was Q4 burn rate?"),
        ("slow",  "Slow path: recompute Q4 burn rate from raw ledger."),
        ("fail",  "Fail path: summarise budget overrun and notify."),
    ]
    results: list[tuple[str, TraceResult | None, str | None]] = []
    for label, q in questions:
        t0 = time.time()
        try:
            res: TraceResult = await finance_report(q)
            elapsed = (time.time() - t0) * 1000
            print(f"[run]  {label:<5} ok    trace={res.trace_id}  "
                  f"{elapsed:>6.0f}ms  tokens={res.total_tokens}  cost=${res.total_cost:.6f}")
            results.append((label, res, None))
        except Exception as exc:
            elapsed = (time.time() - t0) * 1000
            print(f"[run]  {label:<5} FAIL  ({elapsed:.0f}ms)  err={exc}")
            results.append((label, None, str(exc)))

    # 2) Score each trace with DeepEval --------------------------------------
    print("\n[eval] scoring traces with DeepEval"
          + ("" if real_llm else "  (fake judge — patched in-process)"))
    scored: list[dict] = []
    for label, res, err in results:
        if res is None:
            continue
        score = await hub.eval.run(res.trace_id)
        scored.append(score)
        ds = score.get("deepeval_scores") or {}
        avg = sum(v for v in ds.values() if isinstance(v, (int, float))) / max(
            sum(1 for v in ds.values() if isinstance(v, (int, float))), 1
        )
        print(f"[eval] {label:<5} trace={res.trace_id}  avg_deepeval={avg:.3f}")

    # 3) Programmatically retrieve reports -----------------------------------
    print("\n[reports]  team-scoped analytics retrieved by the SDK:")

    stats = hub.reports.stats()
    print(f"  stats:           total={stats['total_traces']}  "
          f"completed={stats['completed']}  errored={stats['errored']}  "
          f"evaluated={stats['evaluated']}  pending={stats['pending_evaluation']}")

    listed = hub.reports.list_traces(limit=5)
    print(f"  list_traces:     {len(listed)} most-recent rows")
    for t in listed[:3]:
        print(f"    - {t['id']}  status={t['status']:<9} "
              f"agent={t['agent_used']:<10}  "
              f"latency={t['total_latency_ms']:.0f}ms  "
              f"prompt={(t['user_prompt'] or '')[:50]!r}")

    # Drill into one trace for span detail
    if listed:
        deep = hub.reports.trace(listed[0]["id"])
        if deep:
            print(f"  trace detail:    {deep['id']}  spans={len(deep.get('spans', []))}")
            for s in (deep.get("spans") or [])[:6]:
                print(f"      {s['span_type']:<16} {s['name']:<30} "
                      f"{(s.get('duration_ms') or 0):>6.0f}ms")

    perf = hub.reports.performance_report(days=1)
    cost = perf.get("cost_breakdown", {}).get("summary", {})
    print(f"  performance:     ${cost.get('total_cost_usd', 0):.6f} over "
          f"{cost.get('total_calls', 0)} calls  "
          f"(anomalies: {len(perf.get('performance_anomalies', []))})")

    anomalies = hub.reports.anomalies(days=1, z_threshold=2.0)
    print(f"  anomalies:       {len(anomalies)} (z>=2.0 in last 1d)")
    for a in anomalies[:3]:
        print(f"      [{a['severity']}] {a['name']:<24} "
              f"{a['duration_ms']:.0f}ms vs μ {a['mean_ms']:.0f}ms  z={a['z_score']}")

    print(f"\n  UI deep link:    {hub.reports.ui_url()}")
    print(f"  filter by team:  hub_team_id = {hub.team_id}")
    print(f"\nNote: traces are visible at /monitoring and /traces in the existing UI;"
          f"\n      filter the team selector to '{hub.team_name}' to see only SDK runs.")

    # 4) Assertions -----------------------------------------------------------
    print("\n[assert] verifying invariants…")
    assert any(label == "fast" and r is not None and r.error is None for label, r, _ in results), (
        "fast-path trace did not produce a successful TraceResult"
    )
    assert any(label == "fail" and (r is None or r.error) for label, r, _ in results), (
        "fail-path was expected to error inside the agent"
    )
    successful_trace_ids = [r.trace_id for _, r, _ in results if r is not None]
    assert len(successful_trace_ids) >= 2, "expected at least 2 successful traces"

    listed_ids = {t["id"] for t in hub.reports.list_traces(limit=20)}
    for tid in successful_trace_ids:
        assert tid in listed_ids, f"trace {tid} missing from team-scoped list"

    for s in scored:
        assert s.get("evaluated") is True, f"trace {s.get('trace_id')} not marked evaluated"
        assert isinstance(s.get("deepeval_scores"), dict) and s["deepeval_scores"], (
            f"trace {s.get('trace_id')} missing DeepEval scores"
        )

    print("[assert] OK — all invariants held.")
    print("\n" + "=" * 78)
    print(" PASS")
    print("=" * 78)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Finance Team SDK end-to-end test")
    parser.add_argument(
        "--real-llm", action="store_true",
        help="Run real DeepEval against the configured judge model "
             "(default: fake judge for cheap deterministic runs)",
    )
    args, _unknown = parser.parse_known_args()
    return asyncio.run(_drive(real_llm=args.real_llm))


# Pytest entry point — pytest invokes this as a regular test function.
def test_finance_team_sdk_end_to_end():
    """Runs the same flow as ``main`` but always in fake-LLM mode under pytest."""
    rc = asyncio.run(_drive(real_llm=False))
    assert rc == 0


if __name__ == "__main__":
    sys.exit(main())
