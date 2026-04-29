"""
Smoke-test every registered team end-to-end.

For each team this script will:
  1. send ONE short chat through the SDK (HTTP transport, hits the live
     backend, writes a trace) and print the agent that handled it; and
  2. run ONE regression case via the SDK and print pass/fail with the
     headline metric scores.

Usage:
    PYTHONPATH=. python -m scripts.smoke_all_teams

The script picks a sensible prompt + a single golden case per team based
on its ``decision_strategy`` and its ``config_json["dataset_groups"]``
subscription.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Optional

from src.db.database import get_session
from src.db.models import Team, GoldenTestCase
from src.sdk.client import HubClient


# ── Team-specific prompts (one-line, low-cost smoke payloads) ──────────────

TEAM_PROMPTS: dict[str, str] = {
    "default": "List the files in the current directory and tell me what this repo is about.",
    "sdlc_2_0": "Read README.md and summarise what this project does in two sentences.",
    "finance_team": "Pull the last 30 days of AAPL price and tell me whether the trend is bullish, bearish, or sideways.",
}

DEFAULT_PROMPT = "Hello — what can you do?"


def _pick_golden(session, team: Team) -> Optional[GoldenTestCase]:
    """Pick the first active golden case in the team's dataset_groups."""
    cj = team.config_json or {}
    groups = cj.get("dataset_groups") or ["sdlc_v2"]
    return (
        session.query(GoldenTestCase)
        .filter(GoldenTestCase.is_active.is_(True))
        .filter(GoldenTestCase.dataset_group.in_(groups))
        .order_by(GoldenTestCase.id)
        .first()
    )


def _short(text: str, n: int = 220) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


async def main() -> int:
    backend = os.getenv("SDLC_HUB_BACKEND", "http://localhost:8000")
    print(f"\nUsing backend: {backend}\n")

    session = get_session()
    try:
        teams = session.query(Team).order_by(Team.id).all()
    finally:
        session.close()

    failures = 0

    for team in teams:
        print("=" * 88)
        print(f"TEAM  id={team.id!r}  name={team.name!r}  strategy={team.decision_strategy!r}")
        print("=" * 88)

        hub = HubClient(team_id=team.id, backend_url=backend)

        # ── 1. Chat smoke ─────────────────────────────────────────────
        prompt = TEAM_PROMPTS.get(team.id, DEFAULT_PROMPT)
        print(f"\n[chat] prompt: {prompt}")
        chat_started = time.time()
        try:
            result = hub.chat.send(prompt, transport="http", timeout=420.0)
            agent_used = result.get("agent_used") or result.get("selected_agent")
            response = result.get("response") or ""
            trace_id = (result.get("trace") or {}).get("trace_id")
            tool_calls = result.get("tool_calls") or []
            elapsed = time.time() - chat_started
            print(f"[chat] OK in {elapsed:.1f}s")
            print(f"  agent_used  : {agent_used}")
            print(f"  tool_calls  : {len(tool_calls)}")
            print(f"  trace_id    : {trace_id}")
            print(f"  response    : {_short(response)}")
            if not response:
                print("  WARN: empty response — check trace in Monitoring")
                failures += 1
        except Exception as e:
            print(f"[chat] FAIL: {type(e).__name__}: {e}")
            failures += 1

        # ── 2. Regression smoke (1 golden case) ───────────────────────
        s2 = get_session()
        try:
            case = _pick_golden(s2, team)
        finally:
            s2.close()

        if not case:
            print("\n[regression] SKIP — no active golden case in this team's dataset_groups")
            continue

        print(f"\n[regression] case={case.id!r} ({case.name})  group={case.dataset_group}")
        reg_started = time.time()
        try:
            run = await hub.regression.arun(
                case_ids=[case.id],
                model=None,  # let backend pick the team-default model
                max_parallel=1,
            )
            elapsed = time.time() - reg_started
            run_id = run.get("run_id") or run.get("id")
            results = run.get("results") or []
            r0 = results[0] if results else {}
            qs = r0.get("quality_scores") or {}
            ds = r0.get("deepeval_scores") or {}
            passed = bool(r0.get("overall_pass"))
            print(f"[regression] {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s  run_id={run_id}")
            if qs:
                print("  quality:   " + ", ".join(f"{k}={v:.2f}" for k, v in qs.items() if isinstance(v, (int, float))))
            if ds:
                print("  deepeval:  " + ", ".join(f"{k}={v:.2f}" for k, v in ds.items() if isinstance(v, (int, float))))
            if not passed:
                failures += 1
        except Exception as e:
            print(f"[regression] FAIL: {type(e).__name__}: {e}")
            failures += 1

        print()

    print("=" * 88)
    print(f"SMOKE TEST COMPLETE — failures: {failures}")
    print("=" * 88)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
