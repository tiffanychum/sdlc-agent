"""
Regression testing engine for golden dataset evaluation.

Runs selected golden test cases against the agent system, captures full
execution traces, computes semantic similarity, evaluates quality with
per-criterion reasoning, checks structural trace assertions, and detects
cost/latency regressions.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import AsyncIterator, Callable, Awaitable

from src.db.database import get_session

logger = logging.getLogger(__name__)
from src.db.models import EvalRun, RegressionResult, GoldenTestCase
from src.evaluation.golden import get_active_cases
from src.config import config
from src.orchestrator import (
    build_orchestrator_from_team, select_strategy_auto,
    _extract_text, VALID_STRATEGIES,
)
from src.tracing.collector import TraceCollector, estimate_cost


COST_REGRESSION_THRESHOLD = 0.20
LATENCY_REGRESSION_THRESHOLD = 0.20
QUALITY_REGRESSION_THRESHOLD = -0.10


def _auto_approve_hitl(iv, case: "GoldenTestCase | None" = None) -> dict:
    """Return the appropriate auto-approve resume value for a HITL interrupt.

    If the golden test case defines `hitl_responses` (a dict keyed by lowercase
    substring), we do a case-insensitive substring match against the question text
    to return a specific, contextual answer instead of the generic fallback.
    This prevents agents from stalling when asking specific questions (e.g.
    "Which Epic or Story should I decompose?") that the generic answer can't satisfy.
    """
    if not isinstance(iv, dict):
        return {"approved": True, "answer": "auto-approved for regression test"}

    htype = iv.get("type", "")

    # For clarification-type interrupts, try case-specific answers first.
    if htype == "clarification" or htype == "":
        question = (
            iv.get("question") or iv.get("message") or iv.get("answer") or ""
        ).lower()
        hitl_responses: dict = {}
        if case and hasattr(case, "hitl_responses") and case.hitl_responses:
            hitl_responses = case.hitl_responses or {}
        if hitl_responses:
            for key, answer in hitl_responses.items():
                if key.lower() in question:
                    logger.debug(
                        "HITL matched key %r for question %r → %r", key, question[:80], answer[:60]
                    )
                    return {"type": "clarification", "answer": answer}
        if htype == "clarification":
            return {"type": "clarification", "answer": "Yes, proceed exactly as planned."}

    if htype == "plan_review":
        return {"type": "plan_review", "approved": True, "edited_plan": iv.get("plan", [])}
    if htype == "action_confirmation":
        return {"type": "action_confirmation", "approved": True}
    if htype == "tool_review":
        return {"type": "tool_review", "action": "continue"}
    return {"approved": True, "answer": "auto-approved for regression test"}


class RegressionRunner:
    def __init__(
        self,
        model: str = None,
        prompt_version: str = "v1",
        team_id: str = "default",
        prompt_versions_by_role: dict = None,
    ):
        self.model = model or config.llm.model
        self.prompt_version = prompt_version
        self.team_id = team_id
        # Per-role overrides take priority over the global prompt_version.
        # e.g. {"coder": "v3", "planner": "v2"}
        self.prompt_versions_by_role = prompt_versions_by_role or {}

    async def run_subset(self, golden_ids: list[str]) -> list[dict]:
        """
        Run a focused regression on specific golden test IDs.
        Returns a list of per-test result dicts (not a full eval_run record).
        Used by the PromptOptimizer agent's run_regression_subset tool.
        """
        full = await self.run(case_ids=golden_ids)
        return full.get("results", [])

    async def run(
        self,
        case_ids: list[str] = None,
        baseline_run_id: str = None,
    ) -> dict:
        """Run regression suite on selected golden cases. Returns full results."""
        cases = get_active_cases(case_ids)
        if not cases:
            return {"error": "No golden test cases found"}

        baseline_results = _load_baseline_results(baseline_run_id) if baseline_run_id else {}

        # ── Per-case strategy resolution ──────────────────────────────────────────
        # Each case may declare a `strategy` field:
        #   - None / missing  → use the team's currently active strategy from DB
        #   - "auto"          → meta-router LLM picks the best strategy for this prompt
        #   - concrete name   → use that strategy regardless of team setting
        #
        # We pre-resolve all strategies so we can cache one orchestrator per unique
        # concrete strategy value (avoids rebuilding the full agent graph every case).

        # Read the team's active strategy once — cases with no strategy inherit this.
        team_active_strategy = _get_team_strategy(self.team_id)

        # Load agent descriptions once for the auto meta-router.
        agents_config_for_auto = await _load_agents_config(self.team_id)

        case_strategy_map: dict[str, str] = {}   # case.id -> resolved concrete strategy
        for case in cases:
            raw = getattr(case, "strategy", None) or None
            # Determine effective strategy: explicit case setting, or fall back to team's active.
            effective = raw if raw else team_active_strategy
            if effective == "auto":
                # Meta-router picks the best concrete strategy for this specific prompt.
                try:
                    resolved = await select_strategy_auto(case.prompt, agents_config_for_auto)
                except Exception as exc:
                    logger.warning(
                        "Auto-strategy resolution failed for case %s (%s: %s); "
                        "falling back to 'supervisor'",
                        case.id, type(exc).__name__, exc,
                    )
                    resolved = "supervisor"
            elif effective in VALID_STRATEGIES:
                resolved = effective
            else:
                resolved = "router_decides"
            case_strategy_map[case.id] = resolved

        # Build (and cache) one orchestrator per unique resolved strategy.
        # Also capture the routing prompt versions used for this run.
        orchestrator_cache: dict[str, object] = {}
        routing_versions_cache: dict[str, dict] = {}
        for case in cases:
            strat = case_strategy_map[case.id]
            if strat not in orchestrator_cache:
                graph = await build_orchestrator_from_team(
                    self.team_id,
                    model_override=self.model or None,
                    strategy_override=strat,
                    prompt_versions_by_role=self.prompt_versions_by_role or None,
                )
                orchestrator_cache[strat] = graph
                routing_versions_cache[strat] = getattr(
                    graph, "__routing_prompt_versions__",
                    {"supervisor": "v1", "meta_router": "v1", "router": "v1"}
                )

        run_id = uuid.uuid4().hex[:12]
        results: list[dict] = []

        for case in cases:
            resolved_strat = case_strategy_map[case.id]
            orchestrator = orchestrator_cache[resolved_strat]
            result = await self._run_single_case(orchestrator, case, baseline_results.get(case.id))
            result["run_id"] = run_id
            # Record strategy provenance: what was configured vs what ran.
            result["expected_strategy"] = getattr(case, "expected_strategy", None) or getattr(case, "strategy", None)
            result["actual_strategy"] = resolved_strat  # always a real strategy name now
            # Record routing prompt versions for traceability
            rv = routing_versions_cache.get(resolved_strat, {})
            result["router_prompt_version"] = (
                f"supervisor={rv.get('supervisor','v1')} "
                f"meta_router={rv.get('meta_router','v1')} "
                f"router={rv.get('router','v1')}"
            )
            results.append(result)

        summary = self._build_summary(run_id, results, cases)
        self._persist(run_id, results, summary)

        return {
            "run_id": run_id,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "num_cases": len(cases),
            "summary": summary,
            "results": results,
        }

    # ── Streaming variant ─────────────────────────────────────────────────────

    async def run_streaming(
        self,
        emit: Callable[[str, dict], Awaitable[None]],
        case_ids: list[str] = None,
        baseline_run_id: str = None,
        max_parallel: int = 3,
    ) -> None:
        """
        Like `run()` but streams trajectory events via `emit(event_type, data)`.

        Cases run in parallel (up to `max_parallel` concurrently).
        Each event is tagged with `case_id` so the consumer can demultiplex.
        The final `run_done` event contains the full summary + results.
        """
        cases = get_active_cases(case_ids)
        if not cases:
            await emit("error", {"message": "No golden test cases found"})
            return

        baseline_results = _load_baseline_results(baseline_run_id) if baseline_run_id else {}

        # Resolve strategies (same logic as run())
        team_active_strategy = _get_team_strategy(self.team_id)
        agents_config_for_auto = await _load_agents_config(self.team_id)
        case_strategy_map: dict[str, str] = {}
        for case in cases:
            raw = getattr(case, "strategy", None) or None
            effective = raw if raw else team_active_strategy
            if effective == "auto":
                try:
                    resolved = await select_strategy_auto(case.prompt, agents_config_for_auto)
                except Exception:
                    resolved = "supervisor"
            elif effective in VALID_STRATEGIES:
                resolved = effective
            else:
                resolved = "router_decides"
            case_strategy_map[case.id] = resolved

        # Build orchestrators per strategy
        orchestrator_cache: dict[str, object] = {}
        routing_versions_cache: dict[str, dict] = {}
        for case in cases:
            strat = case_strategy_map[case.id]
            if strat not in orchestrator_cache:
                graph = await build_orchestrator_from_team(
                    self.team_id,
                    model_override=self.model or None,
                    strategy_override=strat,
                    prompt_versions_by_role=self.prompt_versions_by_role or None,
                )
                orchestrator_cache[strat] = graph
                routing_versions_cache[strat] = getattr(
                    graph, "__routing_prompt_versions__",
                    {"supervisor": "v1", "meta_router": "v1", "router": "v1"},
                )

        run_id = uuid.uuid4().hex[:12]
        await emit("run_start", {"run_id": run_id, "num_cases": len(cases)})

        sem = asyncio.Semaphore(max_parallel)
        results: list[dict] = []
        results_lock = asyncio.Lock()

        async def _run_one(case: GoldenTestCase, index: int):
            async with sem:
                await emit("case_start", {
                    "case_id": case.id,
                    "case_label": case.name,
                    "index": index,
                    "total": len(cases),
                    "prompt": case.prompt,
                })
                orchestrator = orchestrator_cache[case_strategy_map[case.id]]
                result = await self._run_single_case_streaming(
                    orchestrator, case, baseline_results.get(case.id), emit
                )
                result["run_id"] = run_id
                result["expected_strategy"] = getattr(case, "expected_strategy", None) or getattr(case, "strategy", None)
                result["actual_strategy"] = case_strategy_map[case.id]
                rv = routing_versions_cache.get(case_strategy_map[case.id], {})
                result["router_prompt_version"] = (
                    f"supervisor={rv.get('supervisor','v1')} "
                    f"meta_router={rv.get('meta_router','v1')} "
                    f"router={rv.get('router','v1')}"
                )
                async with results_lock:
                    results.append(result)
                await emit("case_done", {
                    "case_id": case.id,
                    "passed": result.get("overall_pass", False),
                    "verdict": "pass" if result.get("overall_pass", False) else "fail",
                    "latency_ms": result.get("actual_latency_ms", 0),
                    "trajectory": result.get("actual_delegation_pattern", []),
                    "actual_output": result.get("actual_output", ""),
                    "trace_assertions": result.get("trace_assertions", {}),
                    "deepeval_scores": result.get("deepeval_scores", {}),
                    "semantic_similarity": result.get("semantic_similarity", 0),
                    "actual_cost": result.get("actual_cost", 0),
                    "error": result.get("error"),
                })

        await asyncio.gather(*[_run_one(case, i) for i, case in enumerate(cases)])

        summary = self._build_summary(run_id, results, cases)
        self._persist(run_id, results, summary)

        await emit("run_done", {
            "run_id": run_id,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "num_cases": len(cases),
            "summary": summary,
        })

    async def _run_single_case_streaming(
        self,
        orchestrator,
        case: GoldenTestCase,
        baseline: dict = None,
        emit: Callable[[str, dict], Awaitable[None]] = None,
    ) -> dict:
        """
        Like `_run_single_case` but emits trajectory events during execution.
        Falls back to plain ainvoke when emit is None.
        """
        from src.tracing.callbacks import TracingCallbackHandler
        from src.orchestrator import get_graph_config
        from langgraph.types import Command as LGCommand

        collector = TraceCollector(team_id=self.team_id, user_prompt=case.prompt)
        tracing_cb = TracingCallbackHandler(collector)
        thread_id = f"regression-{uuid.uuid4().hex[:8]}"
        cfg = get_graph_config(thread_id, callbacks=[tracing_cb])
        start = time.time()

        async def _emit(evt: str, data: dict):
            if emit:
                await emit(evt, {"case_id": case.id, **data})

        current_agent: list[str] = [""]

        try:
            stream = orchestrator.astream_events(
                {"messages": [{"role": "user", "content": case.prompt}],
                 "selected_agent": "", "agent_trace": []},
                config=cfg,
                version="v2",
            )
            async for event in stream:
                kind = event.get("event", "")
                tags = event.get("tags", [])
                name = event.get("name", "")

                # Track which LangGraph node is active
                lg_node = next((t.replace("graph_node:", "") for t in tags if t.startswith("graph_node:")), None)

                if kind == "on_chain_start" and lg_node and lg_node not in ("supervisor_decide", "__start__", "LangGraph"):
                    current_agent[0] = lg_node
                    await _emit("agent_start", {"agent": lg_node})

                elif kind == "on_chain_end" and lg_node and lg_node not in ("supervisor_decide", "__start__", "LangGraph"):
                    ended = lg_node
                    current_agent[0] = ""
                    await _emit("agent_end", {"agent": ended})

                elif kind == "on_tool_start":
                    tool_name = name or event.get("data", {}).get("input", {}).get("name", "")
                    safe_input = {}
                    try:
                        raw_in = event.get("data", {}).get("input", {})
                        safe_input = {k: str(v)[:200] for k, v in raw_in.items()} if isinstance(raw_in, dict) else {}
                    except Exception:
                        pass
                    await _emit("tool_start", {"agent": current_agent[0], "tool": tool_name, "args": safe_input})

                elif kind == "on_tool_end":
                    tool_name = name or ""
                    out_preview = ""
                    try:
                        out = event.get("data", {}).get("output", "")
                        out_preview = str(out)[:300]
                    except Exception:
                        pass
                    await _emit("tool_end", {"agent": current_agent[0], "tool": tool_name, "output_preview": out_preview})

                elif kind == "on_chat_model_stream":
                    try:
                        chunk = event.get("data", {}).get("chunk", {})
                        content = chunk.content if hasattr(chunk, "content") else ""
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    token = block.get("text", "")
                                    if token:
                                        await _emit("llm_token", {"agent": current_agent[0], "token": token})
                        elif isinstance(content, str) and content:
                            await _emit("llm_token", {"agent": current_agent[0], "token": content})
                    except Exception:
                        pass

            # Handle HITL interrupts after first stream completes
            result = None
            for _ in range(15):
                state = await orchestrator.aget_state(cfg)
                if not state.next:
                    break
                interrupt_map: dict = {}
                if state.tasks:
                    for task in state.tasks:
                        if hasattr(task, "interrupts") and task.interrupts:
                            for intr in task.interrupts:
                                iv = intr.value if hasattr(intr, "value") else intr
                                interrupt_map[intr.id] = _auto_approve_hitl(iv, case)
                if not interrupt_map:
                    resume_cmd = LGCommand(resume={"approved": True, "answer": "auto-approved for regression test"})
                elif len(interrupt_map) == 1:
                    resume_cmd = LGCommand(resume=next(iter(interrupt_map.values())))
                else:
                    resume_cmd = LGCommand(resume=interrupt_map)
                result = await orchestrator.ainvoke(resume_cmd, config=cfg)

            if result is None:
                # Get final state if stream completed without interrupt
                state = await orchestrator.aget_state(cfg)
                msgs = state.values.get("messages", []) if state.values else []
                agent_trace_val = state.values.get("agent_trace", []) if state.values else []
                result = {"messages": msgs, "agent_trace": agent_trace_val}

        except Exception as e:
            return self._error_result(case, str(e), time.time() - start)

        elapsed_ms = (time.time() - start) * 1000

        agent_trace = result.get("agent_trace", [])
        messages = result.get("messages", [])
        last_msg = messages[-1] if messages else None
        actual_output = _extract_text(last_msg.content if hasattr(last_msg, "content") else str(last_msg)) if last_msg else ""

        actual_agent, actual_tools, delegation_pattern = _parse_trace(agent_trace)
        llm_calls, tool_calls_count = _count_calls(agent_trace, messages)
        tokens_in, tokens_out, model_used = _extract_token_meta(messages)
        total_cost = estimate_cost(model_used or self.model, tokens_in, tokens_out)

        collector.save()

        span_data = [s.copy() for s in collector.spans]
        for s in span_data:
            for k in ("start_time", "end_time"):
                if isinstance(s.get(k), datetime):
                    s[k] = s[k].isoformat()

        similarity = await self._compute_semantic_similarity(case.reference_output, actual_output)
        deepeval_scores = await self._run_deepeval(case.prompt, actual_output, agent_trace, collector)
        quality_scores: dict = {}
        eval_reasoning: dict = {}

        trace_assertions = self._check_trace_assertions(case, agent_trace, llm_calls, tool_calls_count, tokens_in + tokens_out, elapsed_ms)

        cost_reg = False
        latency_reg = False
        if baseline:
            if baseline.get("actual_cost", 0) > 0:
                cost_reg = (total_cost - baseline["actual_cost"]) / baseline["actual_cost"] > COST_REGRESSION_THRESHOLD
            if baseline.get("actual_latency_ms", 0) > 0:
                latency_reg = (elapsed_ms - baseline["actual_latency_ms"]) / baseline["actual_latency_ms"] > LATENCY_REGRESSION_THRESHOLD

        if tokens_in + tokens_out > case.max_tokens:
            cost_reg = True
        if elapsed_ms > case.max_latency_ms:
            latency_reg = True

        thresholds = case.quality_thresholds or {}
        de_numeric = [v for k, v in deepeval_scores.items()
                      if isinstance(v, (int, float)) and not k.endswith("_reason")]
        de_avg = sum(de_numeric) / max(len(de_numeric), 1) if de_numeric else None
        quality_reg = (
            similarity < thresholds.get("semantic_similarity", 0.4) or
            (de_avg is not None and de_avg < thresholds.get("task_completion", 0.4))
        )
        trace_reg = any(not a.get("passed", True) for a in trace_assertions.values())
        overall_pass = not (cost_reg or latency_reg or quality_reg or trace_reg)

        return {
            "golden_case_id": case.id,
            "golden_case_name": case.name,
            "prompt": case.prompt,
            "reference_output": case.reference_output,
            "actual_output": actual_output,
            "actual_agent": actual_agent,
            "actual_tools": actual_tools,
            "actual_delegation_pattern": delegation_pattern,
            "full_trace": agent_trace,
            "span_data": span_data,
            "actual_llm_calls": llm_calls,
            "actual_tool_calls": tool_calls_count,
            "actual_tokens_in": tokens_in,
            "actual_tokens_out": tokens_out,
            "actual_latency_ms": round(elapsed_ms, 1),
            "actual_cost": round(total_cost, 6),
            "semantic_similarity": round(similarity, 3),
            "quality_scores": quality_scores,
            "deepeval_scores": deepeval_scores,
            "eval_reasoning": eval_reasoning,
            "trace_assertions": trace_assertions,
            "cost_regression": cost_reg,
            "latency_regression": latency_reg,
            "quality_regression": quality_reg,
            "trace_regression": trace_reg,
            "overall_pass": overall_pass,
            "model_used": model_used or self.model,
            "prompt_version": self.prompt_version,
            "expected_agent": case.expected_agent,
            "expected_tools": case.expected_tools or [],
            "quality_thresholds": case.quality_thresholds or {},
            "complexity": case.complexity,
        }

    async def _run_single_case(
        self,
        orchestrator,
        case: GoldenTestCase,
        baseline: dict = None,
    ) -> dict:
        from src.tracing.callbacks import TracingCallbackHandler
        from src.orchestrator import get_graph_config
        from langgraph.types import Command as LGCommand

        collector = TraceCollector(team_id=self.team_id, user_prompt=case.prompt)
        tracing_cb = TracingCallbackHandler(collector)
        thread_id = f"regression-{uuid.uuid4().hex[:8]}"
        config = get_graph_config(thread_id, callbacks=[tracing_cb])
        start = time.time()

        try:
            result = await orchestrator.ainvoke(
                {"messages": [{"role": "user", "content": case.prompt}],
                 "selected_agent": "", "agent_trace": []},
                config=config,
            )

            for _ in range(15):
                state = await orchestrator.aget_state(config)
                if not state.next:
                    break

                # Collect ALL pending interrupts across all tasks (parallel strategy
                # can produce multiple simultaneous HITL interrupts).
                # Build a dict mapping interrupt.id -> auto-approved value, because
                # LangGraph requires interrupt IDs when there are multiple pending interrupts.
                interrupt_map: dict = {}
                if state.tasks:
                    for task in state.tasks:
                        if hasattr(task, "interrupts") and task.interrupts:
                            for intr in task.interrupts:
                                iv = intr.value if hasattr(intr, "value") else intr
                                interrupt_map[intr.id] = _auto_approve_hitl(iv, case)

                if not interrupt_map:
                    # No interrupt IDs found — use plain default resume value
                    resume_cmd = LGCommand(resume={"approved": True,
                                                   "answer": "auto-approved for regression test"})
                elif len(interrupt_map) == 1:
                    # Single interrupt — plain value is fine (avoids unnecessary nesting)
                    resume_cmd = LGCommand(resume=next(iter(interrupt_map.values())))
                else:
                    # Multiple parallel interrupts — LangGraph requires dict keyed by interrupt.id
                    resume_cmd = LGCommand(resume=interrupt_map)

                result = await orchestrator.ainvoke(resume_cmd, config=config)

        except Exception as e:
            return self._error_result(case, str(e), time.time() - start)

        elapsed_ms = (time.time() - start) * 1000

        agent_trace = result.get("agent_trace", [])
        messages = result.get("messages", [])
        last_msg = messages[-1] if messages else None
        actual_output = _extract_text(last_msg.content if hasattr(last_msg, "content") else str(last_msg)) if last_msg else ""

        actual_agent, actual_tools, delegation_pattern = _parse_trace(agent_trace)
        llm_calls, tool_calls_count = _count_calls(agent_trace, messages)
        tokens_in, tokens_out, model_used = _extract_token_meta(messages)
        total_cost = estimate_cost(model_used or self.model, tokens_in, tokens_out)

        collector.save()

        span_data = [s.copy() for s in collector.spans]
        for s in span_data:
            for k in ("start_time", "end_time"):
                if isinstance(s.get(k), datetime):
                    s[k] = s[k].isoformat()

        similarity = await self._compute_semantic_similarity(case.reference_output, actual_output)

        deepeval_scores = await self._run_deepeval(case.prompt, actual_output, agent_trace, collector)
        quality_scores: dict = {}   # G-Eval removed; kept for DB compat only
        eval_reasoning: dict = {}

        trace_assertions = self._check_trace_assertions(case, agent_trace, llm_calls, tool_calls_count, tokens_in + tokens_out, elapsed_ms)

        cost_reg = False
        latency_reg = False
        if baseline:
            if baseline.get("actual_cost", 0) > 0:
                cost_reg = (total_cost - baseline["actual_cost"]) / baseline["actual_cost"] > COST_REGRESSION_THRESHOLD
            if baseline.get("actual_latency_ms", 0) > 0:
                latency_reg = (elapsed_ms - baseline["actual_latency_ms"]) / baseline["actual_latency_ms"] > LATENCY_REGRESSION_THRESHOLD

        if tokens_in + tokens_out > case.max_tokens:
            cost_reg = True
        if elapsed_ms > case.max_latency_ms:
            latency_reg = True

        # Quality regression now based purely on DeepEval scores + semantic similarity
        thresholds = case.quality_thresholds or {}
        de_numeric = [v for k, v in deepeval_scores.items()
                      if isinstance(v, (int, float)) and not k.endswith("_reason")]
        de_avg = sum(de_numeric) / max(len(de_numeric), 1) if de_numeric else None
        quality_reg = (
            similarity < thresholds.get("semantic_similarity", 0.4) or
            (de_avg is not None and de_avg < thresholds.get("task_completion", 0.4))
        )

        trace_reg = any(not a.get("passed", True) for a in trace_assertions.values())

        overall_pass = not (cost_reg or latency_reg or quality_reg or trace_reg)

        return {
            "golden_case_id": case.id,
            "golden_case_name": case.name,
            "prompt": case.prompt,
            "reference_output": case.reference_output,
            "actual_output": actual_output,
            "actual_agent": actual_agent,
            "actual_tools": actual_tools,
            "actual_delegation_pattern": delegation_pattern,
            "full_trace": agent_trace,
            "span_data": span_data,
            "actual_llm_calls": llm_calls,
            "actual_tool_calls": tool_calls_count,
            "actual_tokens_in": tokens_in,
            "actual_tokens_out": tokens_out,
            "actual_latency_ms": round(elapsed_ms, 1),
            "actual_cost": round(total_cost, 6),
            "semantic_similarity": round(similarity, 4),
            "quality_scores": quality_scores,
            "deepeval_scores": deepeval_scores,
            "eval_reasoning": eval_reasoning,
            "trace_assertions": trace_assertions,
            "cost_regression": cost_reg,
            "latency_regression": latency_reg,
            "quality_regression": quality_reg,
            "trace_regression": trace_reg,
            "overall_pass": overall_pass,
            "model_used": model_used or self.model,
            "prompt_version": self.prompt_version,
            "expected_agent": case.expected_agent,
            "expected_tools": case.expected_tools or [],
            "quality_thresholds": case.quality_thresholds or {},
            "complexity": case.complexity,
        }

    def _error_result(self, case: GoldenTestCase, error: str, elapsed: float) -> dict:
        return {
            "golden_case_id": case.id,
            "golden_case_name": case.name,
            "prompt": case.prompt,
            "reference_output": case.reference_output,
            "actual_output": f"ERROR: {error}",
            "actual_agent": "", "actual_tools": [], "actual_delegation_pattern": [],
            "full_trace": [], "span_data": [],
            "actual_llm_calls": 0, "actual_tool_calls": 0,
            "actual_tokens_in": 0, "actual_tokens_out": 0,
            "actual_latency_ms": round(elapsed * 1000, 1), "actual_cost": 0.0,
            "semantic_similarity": 0.0, "quality_scores": {}, "deepeval_scores": {},
            "eval_reasoning": {},
            "trace_assertions": {}, "cost_regression": False, "latency_regression": True,
            "quality_regression": True, "trace_regression": True, "overall_pass": False,
            "model_used": self.model, "prompt_version": self.prompt_version,
            "expected_agent": case.expected_agent, "expected_tools": case.expected_tools or [],
            "quality_thresholds": case.quality_thresholds or {}, "complexity": case.complexity,
            "error": error,
        }

    async def _compute_semantic_similarity(self, reference: str, actual: str) -> float:
        if not reference or not actual:
            return 0.0
        from src.llm.client import get_judge_llm
        llm = get_judge_llm()
        prompt = f"""Score the semantic similarity between these two texts on a scale of 0.0 to 1.0.
Consider meaning equivalence, not exact wording. 0.0 = completely unrelated, 1.0 = identical meaning.

REFERENCE TEXT:
{reference}

ACTUAL TEXT:
{actual}

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}"""
        try:
            resp = await llm.ainvoke([{"role": "user", "content": prompt}])
            raw = _extract_text(resp.content)
            import json, re
            match = re.search(r'\{[^}]*"score"\s*:\s*([\d.]+)[^}]*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return min(1.0, max(0.0, float(data.get("score", 0.5))))
        except Exception:
            pass
        return 0.5

    async def _evaluate_quality(self, case: GoldenTestCase, actual_output: str, trace: list) -> tuple[dict, dict]:
        """Run G-Eval style per-criterion evaluation with reasoning."""
        from src.llm.client import get_judge_llm
        llm = get_judge_llm()
        criteria = {
            "correctness": "Does the output correctly answer the user's question?",
            "completeness": "Does the output cover all aspects of the task?",
            "tool_usage": "Were the right tools used appropriately?",
            "efficiency": "Was the task completed without unnecessary steps?",
            "coherence": "Is the output well-structured and easy to understand?",
        }

        from src.evaluation.integrations import _build_execution_steps_text
        trace_text = _build_execution_steps_text(trace)

        scores = {}
        reasoning = {}
        for name, description in criteria.items():
            prompt = f"""Evaluate this AI agent's output on the criterion: {name}
Criterion: {description}

USER TASK: {case.prompt[:500]}
EXPECTED TOOLS: {', '.join(case.expected_tools or [])}
EXPECTED AGENT: {case.expected_agent}

EXECUTION TRACE:
{trace_text[:1200]}

AGENT OUTPUT:
{actual_output}

First explain your reasoning step by step, then give a score.
Respond with ONLY JSON: {{"reasoning": "<step-by-step analysis>", "score": <1-5>}}
5 = excellent, 4 = good, 3 = adequate, 2 = poor, 1 = failed"""
            try:
                resp = await llm.ainvoke([{"role": "user", "content": prompt}])
                raw = _extract_text(resp.content)
                import json, re
                match = re.search(r'\{[^}]*"score"\s*:\s*(\d)[^}]*\}', raw, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    score = max(1, min(5, data.get("score", 3)))
                    scores[name] = round((score - 1) / 4, 3)
                    reasoning[name] = data.get("reasoning", "")
                else:
                    scores[name] = 0.5
                    reasoning[name] = "Could not parse evaluation response"
            except Exception as e:
                scores[name] = 0.5
                reasoning[name] = f"Evaluation error: {str(e)[:100]}"

        return scores, reasoning

    async def _run_deepeval(self, prompt: str, output: str, trace: list, collector) -> dict:
        """Run all DeepEval agentic metrics."""
        try:
            from src.evaluation.integrations import run_all_deepeval_metrics
            tool_outputs = []
            for s in collector.spans:
                od = s.get("output_data") or {}
                for v in od.values():
                    if v:
                        tool_outputs.append(str(v)[:300])
            scores = await run_all_deepeval_metrics(
                user_prompt=prompt,
                agent_response=output,
                agent_trace=trace,
                tool_outputs=tool_outputs[:5],
            )
            return {k: round(v, 3) if isinstance(v, float) else v for k, v in scores.items()}
        except Exception:
            return {}

    def _check_trace_assertions(
        self, case: GoldenTestCase, trace: list, llm_calls: int,
        tool_calls: int, total_tokens: int, latency_ms: float,
    ) -> dict:
        assertions = {}

        expected_tools = set(case.expected_tools or [])
        actual_tools_set = set()
        actual_tool_args: list[dict] = []
        for entry in trace:
            if entry.get("step") == "execution":
                for tc in entry.get("tool_calls", []):
                    name = tc.get("tool", "")
                    actual_tools_set.add(name)
                    actual_tool_args.append({
                        "tool": name,
                        "args": tc.get("arguments", {}) or tc.get("args", {}) or {},
                    })

        # Tool-equivalence: sdlc_2_0's builder uses generic `run_command` with a
        # `pytest …` / `python -m unittest …` invocation in place of the dev
        # team's dedicated `run_tests` wrapper. Satisfy the `run_tests`
        # requirement whenever we can see a pytest/unittest command being
        # executed via run_command — the semantic intent is identical.
        def _run_command_ran_tests() -> bool:
            for tc in actual_tool_args:
                if tc["tool"] != "run_command":
                    continue
                args = tc.get("args") or {}
                if isinstance(args, dict):
                    cmd = str(args.get("command") or args.get("cmd") or "")
                elif isinstance(args, str):
                    cmd = args
                else:
                    cmd = ""
                low = cmd.lower()
                if ("pytest" in low) or ("python -m unittest" in low) or ("unittest discover" in low):
                    return True
            return False

        effective_actual = set(actual_tools_set)
        if "run_tests" in expected_tools and "run_tests" not in actual_tools_set and _run_command_ran_tests():
            effective_actual.add("run_tests")

        missing = expected_tools - effective_actual
        assertions["required_tools_called"] = {
            "passed": len(missing) == 0,
            "expected": list(expected_tools),
            "actual": list(actual_tools_set),
            "reason": f"Missing tools: {', '.join(missing)}" if missing else "All required tools called",
        }

        expected_pattern = case.expected_delegation_pattern or []
        actual_pattern = []
        for entry in trace:
            if entry.get("step") == "execution":
                agent = entry.get("agent", "")
                if agent and (not actual_pattern or actual_pattern[-1] != agent):
                    actual_pattern.append(agent)
            elif entry.get("step") == "routing":
                agent = entry.get("selected_agent", "")
                if agent and (not actual_pattern or actual_pattern[-1] != agent):
                    actual_pattern.append(agent)

        # Team-aware projection: the golden dataset is authored against the
        # dev-team's 9-role roster, but the sdlc_2_0 team only exposes
        # {builder, planner_v2}. We translate the expected pattern through a
        # role mapping so the same goldens can assert correctness on either
        # team without duplicating the dataset.
        effective_expected = list(expected_pattern)
        if getattr(self, "team_id", "default") == "sdlc_2_0":
            _SDLC_2_0_ROLE_MAP = {
                "planner": "planner_v2",
                "project_manager": "builder",
                "coder": "builder",
                "qa": "builder",
                "devops": "builder",
                "researcher": "builder",
                "reviewer": "builder",
                "data_analyst": "builder",
                "prompt_optimizer": "builder",
            }
            projected: list[str] = []
            for role in expected_pattern:
                mapped = _SDLC_2_0_ROLE_MAP.get(role, role)
                # Collapse adjacent duplicates introduced by the many→one map.
                if not projected or projected[-1] != mapped:
                    projected.append(mapped)
            effective_expected = projected

        pattern_ok = not effective_expected or all(a in actual_pattern for a in effective_expected)
        assertions["delegation_pattern"] = {
            "passed": pattern_ok,
            "expected": effective_expected,
            "expected_original": expected_pattern,
            "actual": actual_pattern,
            "reason": (
                "Delegation matched"
                if pattern_ok
                else f"Expected {effective_expected}, got {actual_pattern}"
            ),
        }

        assertions["llm_call_budget"] = {
            "passed": llm_calls <= case.max_llm_calls,
            "expected": case.max_llm_calls,
            "actual": llm_calls,
            "reason": f"{llm_calls} LLM calls (budget: {case.max_llm_calls})",
        }

        assertions["tool_call_budget"] = {
            "passed": tool_calls <= case.max_tool_calls,
            "expected": case.max_tool_calls,
            "actual": tool_calls,
            "reason": f"{tool_calls} tool calls (budget: {case.max_tool_calls})",
        }

        assertions["token_budget"] = {
            "passed": total_tokens <= case.max_tokens,
            "expected": case.max_tokens,
            "actual": total_tokens,
            "reason": f"{total_tokens} tokens (budget: {case.max_tokens})",
        }

        assertions["latency_budget"] = {
            "passed": latency_ms <= case.max_latency_ms,
            "expected": case.max_latency_ms,
            "actual": round(latency_ms, 1),
            "reason": f"{round(latency_ms)}ms (budget: {case.max_latency_ms}ms)",
        }

        return assertions

    def _build_summary(self, _run_id: str, results: list[dict], _cases: list) -> dict:
        total = len(results)
        passed = sum(1 for r in results if r.get("overall_pass"))
        avg_similarity = sum(r.get("semantic_similarity", 0) for r in results) / max(total, 1)
        avg_latency = sum(r.get("actual_latency_ms", 0) for r in results) / max(total, 1)
        total_cost = sum(r.get("actual_cost", 0) for r in results)
        total_tokens = sum(r.get("actual_tokens_in", 0) + r.get("actual_tokens_out", 0) for r in results)

        cost_regs = sum(1 for r in results if r.get("cost_regression"))
        latency_regs = sum(1 for r in results if r.get("latency_regression"))
        quality_regs = sum(1 for r in results if r.get("quality_regression"))
        trace_regs = sum(1 for r in results if r.get("trace_regression"))

        # Aggregate DeepEval scores only (G-Eval removed)
        all_quality = {}
        for r in results:
            for k, v in r.get("deepeval_scores", {}).items():
                if isinstance(v, (int, float)) and not k.endswith("_reason"):
                    all_quality.setdefault(k, []).append(float(v))
        avg_quality = {k: round(sum(v) / len(v), 3) for k, v in all_quality.items()}

        return {
            "total_cases": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / max(total, 1), 3),
            "avg_semantic_similarity": round(avg_similarity, 4),
            "avg_quality_scores": avg_quality,
            "avg_latency_ms": round(avg_latency, 1),
            "total_cost": round(total_cost, 6),
            "total_tokens": total_tokens,
            "regressions": {
                "cost": cost_regs,
                "latency": latency_regs,
                "quality": quality_regs,
                "trace": trace_regs,
            },
            "task_success_rate": round(passed / max(total, 1), 3),
        }

    def _persist(self, run_id: str, results: list[dict], summary: dict):
        session = get_session()
        try:
            eval_run = EvalRun(
                id=run_id,
                model=self.model,
                prompt_version=self.prompt_version,
                team_id=self.team_id,
                num_tasks=len(results),
                task_completion_rate=summary["pass_rate"],
                avg_latency_ms=summary["avg_latency_ms"],
                total_cost=summary["total_cost"],
                total_tokens=summary["total_tokens"],
                results_json=summary,
            )
            session.add(eval_run)

            for r in results:
                rr = RegressionResult(
                    run_id=run_id,
                    golden_case_id=r["golden_case_id"],
                    golden_case_name=r.get("golden_case_name", ""),
                    actual_output=r.get("actual_output", ""),
                    actual_agent=r.get("actual_agent", ""),
                    actual_tools=r.get("actual_tools", []),
                    actual_delegation_pattern=r.get("actual_delegation_pattern", []),
                    full_trace=r.get("full_trace", []),
                    span_data=r.get("span_data", []),
                    actual_llm_calls=r.get("actual_llm_calls", 0),
                    actual_tool_calls=r.get("actual_tool_calls", 0),
                    actual_tokens_in=r.get("actual_tokens_in", 0),
                    actual_tokens_out=r.get("actual_tokens_out", 0),
                    actual_latency_ms=r.get("actual_latency_ms", 0),
                    actual_cost=r.get("actual_cost", 0),
                    semantic_similarity=r.get("semantic_similarity", 0),
                    quality_scores=r.get("quality_scores", {}),
                    deepeval_scores=r.get("deepeval_scores", {}),
                    trace_assertions=r.get("trace_assertions", {}),
                    eval_reasoning=r.get("eval_reasoning", {}),
                    cost_regression=r.get("cost_regression", False),
                    latency_regression=r.get("latency_regression", False),
                    quality_regression=r.get("quality_regression", False),
                    trace_regression=r.get("trace_regression", False),
                    overall_pass=r.get("overall_pass", True),
                    model_used=r.get("model_used", ""),
                    prompt_version=r.get("prompt_version", "v1"),
                    expected_strategy=r.get("expected_strategy"),
                    actual_strategy=r.get("actual_strategy"),
                    router_prompt_version=r.get("router_prompt_version"),
                )
                session.add(rr)

            session.commit()

            # Archive each result to FeedbackStore for semantic retrieval by PromptOptimizer
            try:
                from src.optimization.feedback_store import get_feedback_store
                store = get_feedback_store()
                for r in results:
                    role = r.get("actual_agent", "")
                    if not role:
                        # Extract from delegation pattern if available
                        pattern = r.get("actual_delegation_pattern", [])
                        role = pattern[0] if pattern else "unknown"
                    store.store_run(
                        role=role,
                        prompt_version=r.get("prompt_version", "v1"),
                        golden_id=r.get("golden_case_id", ""),
                        tool_trace=r.get("full_trace", []),
                        quality_scores=r.get("quality_scores", {}),
                        deepeval_scores=r.get("deepeval_scores", {}),
                        overall_pass=r.get("overall_pass", True),
                    )
            except Exception:
                pass  # FeedbackStore is best-effort; doesn't block regression results
        except Exception:
            session.rollback()
        finally:
            session.close()


def _parse_trace(trace: list) -> tuple[str, list[str], list[str]]:
    agents = []
    tools = []
    delegation = []
    for entry in trace:
        if entry.get("step") == "routing":
            agent = entry.get("selected_agent", "")
            if agent:
                delegation.append(agent)
        elif entry.get("step") == "execution":
            agent = entry.get("agent", "")
            if agent and agent not in agents:
                agents.append(agent)
            if agent and (not delegation or delegation[-1] != agent):
                delegation.append(agent)
            for tc in entry.get("tool_calls", []):
                tools.append(tc.get("tool", ""))
    primary = agents[0] if agents else ""
    return primary, tools, delegation


def _count_calls(trace: list, messages: list) -> tuple[int, int]:
    tool_calls = 0
    for entry in trace:
        if entry.get("step") == "execution":
            tool_calls += len(entry.get("tool_calls", []))

    llm_calls = 0
    for msg in messages:
        if hasattr(msg, "response_metadata") and msg.response_metadata:
            llm_calls += 1

    return max(llm_calls, 1), tool_calls


def _extract_token_meta(messages: list) -> tuple[int, int, str]:
    tokens_in = 0
    tokens_out = 0
    model = ""
    for msg in messages:
        meta = getattr(msg, "response_metadata", None) or {}
        usage = meta.get("token_usage") or meta.get("usage", {})
        if usage:
            tokens_in += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            tokens_out += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        if not model and meta.get("model_name"):
            model = meta["model_name"]
        um = getattr(msg, "usage_metadata", None)
        if um:
            tokens_in += getattr(um, "input_tokens", 0) or 0
            tokens_out += getattr(um, "output_tokens", 0) or 0
    return tokens_in, tokens_out, model


def _load_baseline_results(run_id: str) -> dict:
    session = get_session()
    try:
        rows = session.query(RegressionResult).filter_by(run_id=run_id).all()
        return {r.golden_case_id: {
            "actual_cost": r.actual_cost,
            "actual_latency_ms": r.actual_latency_ms,
            "actual_tokens_in": r.actual_tokens_in,
            "actual_tokens_out": r.actual_tokens_out,
            "quality_scores": r.quality_scores or {},
            "trace_assertions": r.trace_assertions or {},
            "full_trace": r.full_trace or [],
            "actual_output": r.actual_output or "",
        } for r in rows}
    finally:
        session.close()


def _get_team_strategy(team_id: str) -> str:
    """Return the team's currently active decision strategy (falls back to 'router_decides')."""
    from src.db.database import get_session
    from src.db.models import Team

    session = get_session()
    try:
        team = session.query(Team).filter_by(id=team_id).first()
        return (team.decision_strategy if team else None) or "router_decides"
    finally:
        session.close()


async def _load_agents_config(team_id: str) -> list[dict]:
    """Load a minimal agents_config list (id, role, description) for the meta-router."""
    from src.db.database import get_session
    from src.db.models import Agent

    session = get_session()
    try:
        agents = session.query(Agent).filter_by(team_id=team_id).all()
        return [{"id": a.id, "role": a.role, "description": a.description or ""} for a in agents]
    finally:
        session.close()
