"""
Standalone runner for golden_021:
  End-to-end SDLC — fetch Jira SDLC-11 → implement → run tests → push → PR.

Run: python tests/golden_tests/run_golden_021.py
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


def _section(title: str, width: int = 62) -> None:
    print(f"\n{'='*width}")
    print(f"  {title}")
    print("=" * width)


async def main():
    sync_golden_to_db()

    print("\n🚀 golden_021 — End-to-end SDLC workflow")
    print("  Jira task : SDLC-11 (discount_utils.py + pytest)")
    print("  Repo      : tiffanychum/sdlc-agent-integration-test")
    print("  Branch    : feature/discount-util → main")

    runner = RegressionRunner()
    result = await runner.run(case_ids=["golden_021"])

    if "error" in result:
        print(f"\n❌ Runner error: {result['error']}")
        sys.exit(1)

    results = result.get("results", [])
    if not results:
        print("\n❌ No results returned.")
        sys.exit(1)

    r = results[0]

    _section("VERDICT")
    status = "✅  PASS" if r.get("overall_pass") else "❌  FAIL"
    print(f"  Status         : {status}")
    print(f"  Routed to      : {r.get('actual_agent', '?')}")
    print(f"  Expected agent : {r.get('expected_agent', '?')}")
    print(f"  Latency        : {r.get('actual_latency_ms', 0):,.0f} ms")
    print(f"  LLM calls      : {r.get('actual_llm_calls', '?')}")
    print(f"  Tool calls     : {r.get('actual_tool_calls', '?')}")
    print(f"  Tokens total   : {r.get('actual_tokens_in', 0) + r.get('actual_tokens_out', 0):,}")
    print(f"  Similarity     : {r.get('semantic_similarity', 0):.3f}")

    _section("TOOLS USED")
    for t in r.get("actual_tools", []):
        print(f"  ✔ {t}")

    _section("QUALITY SCORES")
    for k, v in (r.get("quality_scores") or {}).items():
        mark = "✅" if isinstance(v, (int, float)) and v >= 0.6 else "⚠️ "
        print(f"  {mark} {k:20s}: {v}")

    _section("TRACE ASSERTIONS")
    for assertion, data in (r.get("trace_assertions") or {}).items():
        mark = "✅" if data.get("passed") else "❌"
        print(f"  {mark} {assertion:30s}: {data.get('reason', '')}")

    regressions = [k for k in ("cost", "latency", "quality", "trace")
                   if r.get(f"{k}_regression")]
    if regressions:
        _section("REGRESSION DETAILS")
        print(f"  Failed checks : {', '.join(regressions)}")

    _section("ACTUAL OUTPUT (first 1800 chars)")
    output = r.get("actual_output", "")
    print(output[:1800] + ("..." if len(output) > 1800 else ""))

    _section("SUMMARY")
    if r.get("overall_pass"):
        print("  ✅  golden_021 PASSED — full SDLC workflow verified.")
        print("  PR: https://github.com/tiffanychum/sdlc-agent-integration-test/pulls")
        print("  Jira: https://makszelai.atlassian.net/browse/SDLC-11")
    else:
        print("  ❌  golden_021 FAILED — see details above.")

    return 0 if r.get("overall_pass") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
