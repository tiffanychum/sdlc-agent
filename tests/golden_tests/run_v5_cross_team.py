"""
v5 dataset smoke: run 3 representative goldens against BOTH teams and
print a side-by-side comparison so we can confirm:
  - team-skip filter behaviour (none of these should skip — all are cross-team)
  - scratch-dir discipline (no sdlc-agent repo writes)
  - safety_no_local_git_write passes on tagged cases (022 + 025)

Cases:
  golden_001 — read-smoke (quick, cross-team sanity)
  golden_022 — scratch-scoped code + local git inside /tmp/validate-util/
  golden_025 — full SDLC via GitHub REST only (no local git writes)

Output: human-readable table to stdout, plus JSON summary at
  /tmp/v5_xteam_report.json for later reference.
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


CASE_IDS = ["golden_001", "golden_022", "golden_025"]
TEAMS = ["default", "sdlc_2_0"]


def _hdr(s: str, width: int = 78) -> None:
    print(f"\n{'=' * width}")
    print(f"  {s}")
    print("=" * width, flush=True)


def _one_liner_result(r: dict) -> dict:
    """Boil a regression result down to the fields we compare across teams."""
    ta = r.get("trace_assertions") or {}
    safety = ta.get("safety_no_local_git_write") or {}
    return {
        "case_id": r.get("golden_case_id"),
        "passed": bool(r.get("overall_pass")),
        "skipped": bool(r.get("skipped")),
        "verdict": r.get("verdict") or ("pass" if r.get("overall_pass") else "fail"),
        "latency_ms": int(r.get("actual_latency_ms") or 0),
        "llm_calls": r.get("actual_llm_calls") or 0,
        "tool_calls": r.get("actual_tool_calls") or 0,
        "tokens": (r.get("actual_tokens_in") or 0) + (r.get("actual_tokens_out") or 0),
        "cost": float(r.get("actual_cost") or 0.0),
        "similarity": float(r.get("semantic_similarity") or 0.0),
        "delegation": " → ".join(r.get("actual_delegation_pattern") or []),
        "tools": r.get("actual_tools") or [],
        "safety_passed": bool(safety.get("passed", True)) if safety else None,
        "safety_reason": safety.get("reason", "") if safety else "",
        "trace_fails": [
            name for name, data in ta.items() if not data.get("passed")
        ],
        "error": r.get("error") or "",
    }


async def run_team(team_id: str) -> list[dict]:
    print(f"\n>>> team='{team_id}' — running {CASE_IDS} ...", flush=True)
    started = time.time()
    runner = RegressionRunner(team_id=team_id)
    out = await runner.run(case_ids=CASE_IDS)
    elapsed = time.time() - started
    if "error" in out:
        print(f"    RUNNER ERROR on team={team_id}: {out['error']}")
        return []
    results = out.get("results") or []
    print(f"    done in {elapsed:.1f}s ({len(results)} results, run_id={out.get('run_id')})")
    return results


def _print_side_by_side(by_team: dict[str, list[dict]]) -> bool:
    all_pass = True
    for cid in CASE_IDS:
        _hdr(f"{cid} — cross-team comparison")
        print(f"  {'field':<20}  {'default':<35}  {'sdlc_2_0':<35}")
        print(f"  {'-'*20}  {'-'*35}  {'-'*35}")
        rows = {t: next((_one_liner_result(r) for r in by_team[t] if r["case_id"] == cid), None)
                for t in TEAMS}
        for field in ["verdict", "latency_ms", "llm_calls", "tool_calls",
                      "tokens", "cost", "similarity", "delegation",
                      "safety_passed"]:
            l = str(rows["default"].get(field) if rows["default"] else "—")
            r = str(rows["sdlc_2_0"].get(field) if rows["sdlc_2_0"] else "—")
            if field in ("latency_ms", "tokens"):
                try:
                    l = f"{int(l):,}"
                except Exception:
                    pass
                try:
                    r = f"{int(r):,}"
                except Exception:
                    pass
            if field == "cost":
                try:
                    l = f"${float(l):.4f}"
                except Exception:
                    pass
                try:
                    r = f"${float(r):.4f}"
                except Exception:
                    pass
            if field == "similarity":
                try:
                    l = f"{float(l):.3f}"
                except Exception:
                    pass
                try:
                    r = f"{float(r):.3f}"
                except Exception:
                    pass
            print(f"  {field:<20}  {l[:35]:<35}  {r[:35]:<35}")
        for team in TEAMS:
            row = rows[team]
            if row and row["trace_fails"]:
                print(f"    [{team}] trace failures: {row['trace_fails']}")
            if row and row["safety_reason"] and not row["safety_passed"]:
                print(f"    [{team}] SAFETY: {row['safety_reason'][:200]}")
            if row and row["error"]:
                print(f"    [{team}] ERROR: {row['error'][:200]}")
            if row and not row["passed"]:
                all_pass = False
    return all_pass


async def main() -> int:
    sync_golden_to_db()

    _hdr(f"v5 golden-dataset cross-team smoke — {datetime.utcnow().isoformat()}Z")
    print(f"  cases: {CASE_IDS}")
    print(f"  teams: {TEAMS}")

    by_team: dict[str, list[dict]] = {}
    for team in TEAMS:
        by_team[team] = await run_team(team)

    all_pass = _print_side_by_side(by_team)

    report_path = "/tmp/v5_xteam_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "when": datetime.utcnow().isoformat() + "Z",
            "cases": CASE_IDS,
            "teams": TEAMS,
            "by_team": {t: [_one_liner_result(r) for r in rs] for t, rs in by_team.items()},
            "all_pass": all_pass,
        }, f, indent=2)

    _hdr("OVERALL")
    print(f"  Full JSON summary: {report_path}")
    if all_pass:
        print("  ✅  ALL PASS across both teams.")
        return 0
    print("  ❌  Some cases failed. See details above and the JSON report.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
