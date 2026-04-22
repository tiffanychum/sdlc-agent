"""
Cross-team comparison: golden_001, golden_022, golden_030 on default vs sdlc_2_0.

Within a team, cases run concurrently (max_parallel=3) to keep wall-clock low.
Teams run sequentially to avoid noisy-neighbor effects on the local machine.
Output: human-readable table + JSON at /tmp/xteam_012230_report.json.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, ".env"))

from src.evaluation.regression import RegressionRunner  # noqa: E402
from src.evaluation.golden import sync_golden_to_db  # noqa: E402


CASE_IDS = ["golden_001", "golden_022", "golden_030"]
TEAMS = ["default", "sdlc_2_0"]


def _hdr(s: str, width: int = 82) -> None:
    print(f"\n{'=' * width}")
    print(f"  {s}")
    print("=" * width, flush=True)


def _one_liner_result(r: dict) -> dict:
    ta = r.get("trace_assertions") or {}
    safety = ta.get("safety_no_local_git_write") or {}
    qs = r.get("quality_scores") or {}
    ds = r.get("deepeval_scores") or {}
    return {
        "case_id": r.get("golden_case_id"),
        "passed": bool(r.get("overall_pass")),
        "skipped": bool(r.get("skipped")),
        "verdict": r.get("verdict") or ("pass" if r.get("overall_pass") else "fail"),
        "latency_ms": int(r.get("actual_latency_ms") or 0),
        "llm_calls": r.get("actual_llm_calls") or 0,
        "tool_calls": r.get("actual_tool_calls") or 0,
        "tokens_in": r.get("actual_tokens_in") or 0,
        "tokens_out": r.get("actual_tokens_out") or 0,
        "tokens": (r.get("actual_tokens_in") or 0) + (r.get("actual_tokens_out") or 0),
        "cost": float(r.get("actual_cost") or 0.0),
        "similarity": float(r.get("semantic_similarity") or 0.0),
        "delegation": " → ".join(r.get("actual_delegation_pattern") or []),
        "tools": r.get("actual_tools") or [],
        "step_efficiency": float(qs.get("step_efficiency") or 0.0),
        "tool_usage":      float(qs.get("tool_usage") or 0.0),
        "completeness":    float(qs.get("completeness") or 0.0),
        "coherence":       float(qs.get("coherence") or 0.0),
        "correctness":     float(qs.get("correctness") or 0.0),
        "answer_relevancy":float(ds.get("answer_relevancy") or 0.0),
        "faithfulness":    float(ds.get("faithfulness") or 0.0),
        "safety_passed": bool(safety.get("passed", True)) if safety else None,
        "safety_reason": safety.get("reason", "") if safety else "",
        "trace_fails": [name for name, data in ta.items() if not data.get("passed")],
        "error": r.get("error") or "",
    }


async def run_team(team_id: str) -> tuple[list[dict], str, float]:
    print(f"\n>>> team='{team_id}' — running {CASE_IDS} (parallel=3) ...", flush=True)
    started = time.time()
    runner = RegressionRunner(team_id=team_id)
    out = await runner.run(case_ids=CASE_IDS, max_parallel=3)
    elapsed = time.time() - started
    if "error" in out:
        print(f"    RUNNER ERROR on team={team_id}: {out['error']}")
        return [], "", elapsed
    results = out.get("results") or []
    run_id = out.get("run_id") or ""
    print(f"    done in {elapsed:.1f}s ({len(results)} results, run_id={run_id})")
    return results, run_id, elapsed


def _print_side_by_side(by_team: dict[str, list[dict]]) -> bool:
    all_pass = True
    fields = [
        ("verdict",          "str"),
        ("latency_ms",       "int"),
        ("llm_calls",        "int"),
        ("tool_calls",       "int"),
        ("tokens",           "int"),
        ("cost",             "cost"),
        ("similarity",       "f3"),
        ("step_efficiency",  "f3"),
        ("tool_usage",       "f3"),
        ("completeness",     "f3"),
        ("correctness",      "f3"),
        ("answer_relevancy", "f3"),
        ("faithfulness",     "f3"),
        ("delegation",       "str"),
        ("safety_passed",    "str"),
    ]
    for cid in CASE_IDS:
        _hdr(f"{cid} — cross-team comparison")
        print(f"  {'field':<20}  {'default':<28}  {'sdlc_2_0':<28}  {'delta':<12}")
        print(f"  {'-'*20}  {'-'*28}  {'-'*28}  {'-'*12}")
        rows = {t: next((_one_liner_result(r) for r in by_team[t] if r.get("golden_case_id") == cid), None)
                for t in TEAMS}
        for field, kind in fields:
            l = rows["default"].get(field) if rows["default"] else None
            r = rows["sdlc_2_0"].get(field) if rows["sdlc_2_0"] else None
            if kind == "int":
                ls = f"{int(l):,}" if l is not None else "—"
                rs = f"{int(r):,}" if r is not None else "—"
                delta = f"{int(r) - int(l):+,}" if (l is not None and r is not None) else ""
            elif kind == "cost":
                ls = f"${float(l):.4f}" if l is not None else "—"
                rs = f"${float(r):.4f}" if r is not None else "—"
                delta = f"${float(r) - float(l):+.4f}" if (l is not None and r is not None) else ""
            elif kind == "f3":
                ls = f"{float(l):.3f}" if l is not None else "—"
                rs = f"{float(r):.3f}" if r is not None else "—"
                delta = f"{float(r) - float(l):+.3f}" if (l is not None and r is not None) else ""
            else:
                ls = str(l)[:28] if l is not None else "—"
                rs = str(r)[:28] if r is not None else "—"
                delta = ""
            print(f"  {field:<20}  {ls:<28}  {rs:<28}  {delta:<12}")
        for team in TEAMS:
            row = rows[team]
            if row and row["trace_fails"]:
                print(f"    [{team}] trace failures: {row['trace_fails']}")
            if row and row["safety_reason"] and not row["safety_passed"]:
                print(f"    [{team}] SAFETY: {row['safety_reason'][:180]}")
            if row and row["error"]:
                print(f"    [{team}] ERROR: {row['error'][:180]}")
            if row and not row["passed"]:
                all_pass = False
    return all_pass


async def main() -> int:
    sync_golden_to_db()

    _hdr(f"Cross-team comparison — {datetime.utcnow().isoformat()}Z")
    print(f"  cases: {CASE_IDS}")
    print(f"  teams: {TEAMS}")

    by_team: dict[str, list[dict]] = {}
    run_ids: dict[str, str] = {}
    wallclock: dict[str, float] = {}
    for team in TEAMS:
        results, rid, elapsed = await run_team(team)
        by_team[team] = results
        run_ids[team] = rid
        wallclock[team] = elapsed

    all_pass = _print_side_by_side(by_team)

    _hdr("TEAM ROLL-UP")
    for team in TEAMS:
        rs = [_one_liner_result(r) for r in by_team[team]]
        n = len(rs)
        passed = sum(1 for r in rs if r["passed"])
        total_tokens = sum(r["tokens"] for r in rs)
        total_cost = sum(r["cost"] for r in rs)
        total_latency = sum(r["latency_ms"] for r in rs)
        avg_step_eff = sum(r["step_efficiency"] for r in rs) / max(n, 1)
        avg_tool_use = sum(r["tool_usage"] for r in rs) / max(n, 1)
        print(f"  [{team}] run_id={run_ids[team]} pass={passed}/{n} "
              f"tokens={total_tokens:,} cost=${total_cost:.4f} "
              f"sum_latency={total_latency/1000:.1f}s wallclock={wallclock[team]:.1f}s "
              f"step_eff={avg_step_eff:.3f} tool_use={avg_tool_use:.3f}")

    report_path = "/tmp/xteam_012230_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "when": datetime.utcnow().isoformat() + "Z",
            "cases": CASE_IDS,
            "teams": TEAMS,
            "run_ids": run_ids,
            "wallclock_s": wallclock,
            "by_team": {t: [_one_liner_result(r) for r in rs] for t, rs in by_team.items()},
            "all_pass": all_pass,
        }, f, indent=2)

    _hdr("OVERALL")
    print(f"  Report: {report_path}")
    print(f"  run_ids: {run_ids}")
    if all_pass:
        print("  ✅  ALL PASS across both teams.")
        return 0
    print("  ❌  Some cases failed. See details above.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
