"""
Demo script — show the SDK retrieving the *same* team-scoped data that the
UI displays in Monitoring / Regression Performance Analysis / Evaluation.

Run from the repo root with the venv activated:

    python -m scripts.sdk_demo                 # uses team name "Finance Team"
    python -m scripts.sdk_demo --team "ABC"    # any seeded team

Each section below mirrors a specific page/section in the UI so you can
copy the SDK call into your own automation and trust it returns exactly
what the dashboard shows.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.sdk import HubClient


def _truncate(value: Any, n: int = 80) -> str:
    s = str(value)
    return s if len(s) <= n else s[: n - 1] + "…"


def _print_section(title: str) -> None:
    print()
    print("─" * 78)
    print(title)
    print("─" * 78)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="Finance Team", help="team name (default: Finance Team)")
    parser.add_argument("--days", type=int, default=7, help="look-back window for perf reports")
    args = parser.parse_args()

    print(f"Connecting SDK to team: {args.team!r}")
    hub = HubClient(team=args.team)
    print(f"Resolved team_id = {hub.team_id}")

    # ── 1. Stats — mirrors the small KPI tiles at the top of /monitoring ──
    _print_section("1. hub.reports.stats() — mirrors /monitoring KPI tiles")
    stats = hub.reports.stats()
    print(json.dumps(stats, indent=2, default=str))

    # ── 2. List traces — mirrors the "All Requests" list on /evaluation
    #    and the recent-traces panel on /monitoring.
    _print_section("2. hub.reports.list_traces(limit=5) — mirrors /evaluation 'All Requests'")
    traces = hub.reports.list_traces(limit=5)
    print(f"Returned {len(traces)} trace(s):")
    for t in traces:
        scores = (t.get("eval_scores") or {}).get("deepeval_scores") or {}
        # DeepEval mixes numeric scores (e.g. task_completion=0.85) with
        # string reasoning fields (e.g. task_completion_reason="…") under
        # the same dict — only the numeric ones are meaningful here.
        numeric = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
        de_summary = ", ".join(f"{k}={v:.2f}" for k, v in list(numeric.items())[:3]) or "—"
        print(
            f"  {t['id'][:8]}  "
            f"agent={t.get('agent_used') or '—':<10}  "
            f"latency={t['total_latency_ms']:.0f}ms  "
            f"tok={t['total_tokens']:<5}  "
            f"$={t['total_cost']:.4f}  "
            f"status={t['status']}  "
            f"prompt={_truncate(t.get('user_prompt'), 40)}  "
            f"deepeval=[{de_summary}]"
        )

    # ── 3. Single trace + spans — mirrors the expanded row on /evaluation
    if traces:
        first_id = traces[0]["id"]
        _print_section(f"3. hub.reports.trace({first_id[:8]}…) — full trace + spans")
        full = hub.reports.trace(first_id)
        if full:
            print(f"  user_prompt : {_truncate(full.get('user_prompt'), 200)}")
            print(f"  agent_used  : {full.get('agent_used')}")
            print(f"  total_cost  : ${full.get('total_cost'):.4f}")
            print(f"  spans       : {len(full.get('spans') or [])}")
            for sp in (full.get("spans") or [])[:5]:
                print(
                    f"    - {sp['span_type']:<14} {sp['name']:<40} "
                    f"{(sp.get('end_time') or '') or '—'}"
                )

    # ── 4. Performance report — mirrors /regression > Performance Analysis
    _print_section(
        f"4. hub.reports.performance_report(days={args.days}) — "
        "mirrors /regression Performance Analysis"
    )
    report = hub.reports.performance_report(days=args.days)
    print("Top-level keys:", sorted(report.keys()))
    al = report.get("agent_latency_percentiles") or {}
    if al:
        print("\nAgent latency percentiles (p50 / p95 / p99 ms):")
        for agent, pct in al.items():
            print(
                f"  {agent:<14} "
                f"p50={pct.get('p50', 0):>7.0f}  "
                f"p95={pct.get('p95', 0):>7.0f}  "
                f"p99={pct.get('p99', 0):>7.0f}  "
                f"count={pct.get('count', 0)}"
            )
    cb = (report.get("cost_breakdown") or {}).get("summary") or {}
    if cb:
        print(f"\nCost summary: total=${cb.get('total_cost', 0):.4f} "
              f"runs={cb.get('total_runs', 0)} "
              f"avg=${cb.get('avg_cost_per_run', 0):.4f}")

    # ── 5. Anomalies — mirrors /regression > Performance Analysis > Anomalies
    _print_section(f"5. hub.reports.anomalies(days={args.days}, z_threshold=2.5)")
    anoms = hub.reports.anomalies(days=args.days, z_threshold=2.5)
    print(f"Returned {len(anoms)} anomaly(ies):")
    for a in anoms[:5]:
        print(
            f"  [{a.get('severity', '?'):<6}] "
            f"agent={a.get('agent_used', '—')} "
            f"latency={a.get('latency_ms', 0):.0f}ms "
            f"z={a.get('z_score', 0):.2f}  "
            f"reason={_truncate(a.get('reason'), 60)}"
        )

    # ── 6. UI URL — open this in a browser to see the SAME data filtered
    #    by team in the dashboard.
    _print_section("6. UI URL — open in a browser to see the same data in the dashboard")
    print(hub.reports.ui_url())
    print(
        "\nTip: switch the team selector in /monitoring, /regression, /evaluation\n"
        "to this team to see the SDK-generated traces alongside the chat-generated ones."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
