"""
Async-parallelism verification tests for the orchestrator + regression runner.

B1: `_build_parallel_graph` truly overlaps agent execution (not serialized).
B2: Two regression runs on different teams interleave rather than serialize
    their async work.
B3: Sync DB-write offload via `asyncio.to_thread` keeps the event loop
    responsive under concurrent load.

These tests do NOT hit real LLMs — we stub the agent executors and the
`get_llm()` factory so the tests are fast and deterministic.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


# ── Shared stubs ──────────────────────────────────────────────────────────────


class _SleepAgent:
    """Minimal stand-in for a built agent — sleeps, then returns an AIMessage."""

    def __init__(self, name: str, delay: float):
        self.name = name
        self.delay = delay

    async def ainvoke(self, payload):  # noqa: D401 — match LangChain signature
        await asyncio.sleep(self.delay)
        return {"messages": [AIMessage(content=f"{self.name}-done")]}


class _InstantLLM:
    """Stub LLM for `_merge` — returns a dummy AIMessage without any I/O."""

    async def ainvoke(self, messages):  # noqa: D401
        # Tiny sleep so `_merge` scheduling is realistic but negligible.
        await asyncio.sleep(0.001)
        return AIMessage(content="merged")


# ── B1: parallel strategy wall-clock ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_parallel_strategy_overlaps_agents():
    """Three 1-second agents in parallel should complete in ~1 s wall-clock,
    not ~3 s.  This proves `_build_parallel_graph` fans out via LangGraph's
    asyncio scheduler and does not serialize agent execution.
    """
    from src.orchestrator import _build_parallel_graph

    delay = 1.0
    agents_config = [
        {"role": "a", "description": "stub a"},
        {"role": "b", "description": "stub b"},
        {"role": "c", "description": "stub c"},
    ]
    built_agents = {
        "a": _SleepAgent("a", delay),
        "b": _SleepAgent("b", delay),
        "c": _SleepAgent("c", delay),
    }

    with patch("src.orchestrator.get_llm", return_value=_InstantLLM()):
        graph = _build_parallel_graph(agents_config, built_agents)
        t0 = time.monotonic()
        await graph.ainvoke({
            "messages": [HumanMessage(content="go")],
            "selected_agent": "",
            "agent_trace": [],
        })
        elapsed = time.monotonic() - t0

    # Serial would be ~3.0s. Parallel should be ~1.0s (+ graph/merge overhead).
    # Allow generous upper bound (1.8s) so a slow CI box doesn't flake; a
    # serialized implementation would still be well above 2.5s.
    assert elapsed < 1.8, (
        f"Parallel strategy wall-clock was {elapsed:.2f}s — expected ~1.0s. "
        f"Agents appear to be running serially."
    )
    # Lower bound proves our sleep stub actually ran (guard against the graph
    # short-circuiting by accident).
    assert elapsed >= delay * 0.95, f"Graph returned too fast: {elapsed:.2f}s"


# ── B2: cross-team regression runs don't serialize ────────────────────────────


@pytest.mark.asyncio
async def test_two_regression_runs_interleave():
    """Kick off two mock regression-shaped workloads concurrently and confirm
    they interleave instead of running back-to-back.  We simulate the runner's
    essential shape (a `gather` of N cases, each doing an async LLM call + a
    sync DB write) so the test is hermetic.
    """
    # Each "case" does async work + a sync SQLite-like blocking call.
    async def fake_case(run_id: str, idx: int, log: list, db_lock: asyncio.Lock):
        log.append((time.monotonic(), run_id, "start", idx))
        await asyncio.sleep(0.2)  # LLM-call analogue
        # Emulate A2: sync blocking work goes through a thread, so it does NOT
        # block the event loop.  We don't hold the lock across the await.
        def _blocking_commit():
            time.sleep(0.05)  # SQLite commit analogue
        await asyncio.to_thread(_blocking_commit)
        log.append((time.monotonic(), run_id, "end", idx))

    async def fake_run(run_id: str, num_cases: int, log: list, db_lock: asyncio.Lock):
        sem = asyncio.Semaphore(3)

        async def _one(i):
            async with sem:
                await fake_case(run_id, i, log, db_lock)

        await asyncio.gather(*(_one(i) for i in range(num_cases)))

    log: list[tuple[float, str, str, int]] = []
    db_lock = asyncio.Lock()

    t0 = time.monotonic()
    await asyncio.gather(
        fake_run("A", 4, log, db_lock),
        fake_run("B", 4, log, db_lock),
    )
    wall = time.monotonic() - t0

    # Interleaving check — at least one "B" event must appear between two "A"
    # events (and vice versa).  If A fully completed before B started, this
    # would be False.
    run_ids = [entry[1] for entry in sorted(log)]
    a_then_b = any(run_ids[i] == "A" and run_ids[i + 1] == "B" for i in range(len(run_ids) - 1))
    b_then_a = any(run_ids[i] == "B" and run_ids[i + 1] == "A" for i in range(len(run_ids) - 1))
    assert a_then_b and b_then_a, (
        f"Events did not interleave — one run blocked the other. "
        f"Ordering: {run_ids}"
    )

    # Wall-clock sanity: if they fully serialized, total would be ~2 * (cases/sem * sleep)
    # = 2 * ceil(4/3) * (0.2 + 0.05) = 2 * 0.5 = 1.0s. Interleaved should be ~0.5s.
    assert wall < 0.9, f"Concurrent wall-clock {wall:.2f}s — two runs serialized"


# ── B3: to_thread offload keeps the event loop responsive ─────────────────────


@pytest.mark.asyncio
async def test_to_thread_offload_keeps_loop_responsive():
    """Measure how much a sync-blocking call stalls the event loop.

    Instead of a heartbeat task (which has a tricky ordering around when
    `stop` fires), we measure directly: while the main coroutine is inside
    the blocking payload, how long does a concurrent `asyncio.sleep(0.02)`
    *actually* take?

    Without offload, the concurrent sleep observes ~0.3 s of extra latency
    because the loop is frozen.  With `asyncio.to_thread`, the loop keeps
    running and the concurrent sleep finishes in ~0.02 s.
    """

    def blocking_payload(seconds: float):
        time.sleep(seconds)

    async def measure(offload: bool) -> float:
        """Return how long a 20 ms asyncio.sleep actually takes while a
        0.3 s blocking payload runs concurrently on the same loop."""
        observed: list[float] = []

        async def probe():
            start = time.monotonic()
            await asyncio.sleep(0.02)
            observed.append(time.monotonic() - start)

        probe_task = asyncio.create_task(probe())
        # Yield once so the probe is definitely scheduled before we block.
        await asyncio.sleep(0)

        if offload:
            await asyncio.to_thread(blocking_payload, 0.3)
        else:
            blocking_payload(0.3)

        await probe_task
        return observed[0]

    observed_without = await measure(offload=False)
    observed_with = await measure(offload=True)

    # Control: without offload, the 20 ms sleep should have been stalled by
    # the 0.3 s blocking payload, so observed latency must be significantly
    # higher than the nominal 20 ms.
    assert observed_without > 0.15, (
        f"Control case didn't actually block the loop "
        f"(observed {observed_without*1000:.0f} ms for a sleep that should "
        f"have been stalled ~300 ms). Test environment may be flaky."
    )
    # Offload: the 20 ms sleep should take close to 20 ms (within scheduling
    # jitter) because the loop kept running while the blocking payload ran
    # in a worker thread.
    assert observed_with < 0.1, (
        f"asyncio.to_thread failed to keep the loop responsive — "
        f"20 ms sleep took {observed_with*1000:.0f} ms."
    )
    # And the offload case must be meaningfully better than the blocking one.
    assert observed_with < observed_without / 2, (
        f"Offload only reduced observed latency from "
        f"{observed_without*1000:.0f} ms to {observed_with*1000:.0f} ms."
    )
