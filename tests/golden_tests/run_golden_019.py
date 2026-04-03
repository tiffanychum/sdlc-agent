"""
Standalone runner for golden_019: Jira create Workstream + Task + assignment.
Run: python tests/golden_tests/run_golden_019.py
"""

import asyncio
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, ".env"))

from src.evaluation.regression import RegressionRunner
from src.evaluation.golden import sync_golden_to_db


def _print_section(title: str, width: int = 60) -> None:
    print(f"\n{'='*width}")
    print(f"  {title}")
    print("=" * width)


async def main():
    # Sync JSON → DB so any threshold changes in golden_dataset.json take effect
    sync_golden_to_db()

    print("\n🚀 Running golden_019: Jira Workstream + Task + Assignment")
    print("  Project : SDLC (sdlc_agent at makszelai.atlassian.net)")
    print("  Epic    : Feature: User Authentication (Workstream)")
    print("  Task    : Implement JWT login endpoint → Tiffany Chum")

    runner = RegressionRunner()
    result = await runner.run(case_ids=["golden_019"])

    if "error" in result:
        print(f"\n❌ Runner error: {result['error']}")
        sys.exit(1)

    results = result.get("results", [])
    if not results:
        print("\n❌ No results returned.")
        sys.exit(1)

    r = results[0]

    _print_section("VERDICT")
    status = "✅  PASS" if r.get("overall_pass") else "❌  FAIL"
    print(f"  Status         : {status}")
    print(f"  Routed to      : {r.get('actual_agent', '?')}")
    print(f"  Expected agent : {r.get('expected_agent', '?')}")
    print(f"  Latency        : {r.get('actual_latency_ms', 0):,.0f} ms")
    print(f"  LLM calls      : {r.get('actual_llm_calls', '?')}")
    print(f"  Tool calls     : {r.get('actual_tool_calls', '?')}")
    print(f"  Tokens total   : {r.get('actual_tokens_in', 0) + r.get('actual_tokens_out', 0):,}")
    print(f"  Similarity     : {r.get('semantic_similarity', 0):.3f}")

    _print_section("TOOLS USED")
    for t in r.get("actual_tools", []):
        print(f"  ✔ {t}")

    _print_section("QUALITY SCORES")
    for k, v in (r.get("quality_scores") or {}).items():
        mark = "✅" if isinstance(v, (int, float)) and v >= 0.6 else "⚠️ "
        print(f"  {mark} {k:20s}: {v}")

    _print_section("TRACE ASSERTIONS")
    for assertion, data in (r.get("trace_assertions") or {}).items():
        mark = "✅" if data.get("passed") else "❌"
        print(f"  {mark} {assertion:30s}: {data.get('reason', '')}")

    regressions = []
    if r.get("cost_regression"):
        regressions.append("cost")
    if r.get("latency_regression"):
        regressions.append("latency")
    if r.get("quality_regression"):
        regressions.append("quality")
    if r.get("trace_regression"):
        regressions.append("trace")

    if regressions:
        _print_section("REGRESSION DETAILS")
        print(f"  Failed checks: {', '.join(regressions)}")

    _print_section("ACTUAL OUTPUT (truncated to 1200 chars)")
    output = r.get("actual_output", "")
    print(output[:1200] + ("..." if len(output) > 1200 else ""))

    _print_section("SUMMARY")
    if r.get("overall_pass"):
        print("  ✅  golden_019 PASSED — Jira integration test complete.")
        print("  Created tickets are visible at: https://makszelai.atlassian.net/browse/SDLC")
    else:
        print("  ❌  golden_019 FAILED — see regressions above.")

    return 0 if r.get("overall_pass") else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
