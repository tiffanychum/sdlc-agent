"""Global Prompt Optimizer — deterministic, non-agentic optimisation loop.

This module implements prompt optimisation as a pure Python state machine. It
is intentionally *not* an LLM agent: the only LLM call per cycle is the
drafter (see ``drafter.py``). Everything else — baseline collection, in-memory
validation via ``RegressionRunner(prompt_text_overrides=…, persist=False)``,
classification, winner selection, and the commit decision — is deterministic.

Why a function, not an agent:
  * No supervisor / meta-router to misroute the task.
  * One LLM call per cycle instead of ~10 tool-calling supervisor turns.
  * Deterministic control flow ⇒ unit-testable; mocks only the drafter.
  * "Run Optimization Loop" in the UI gets crisper events (phase, cycle, report)
    instead of a blind token stream.

Commit policy (Aggressive tie-break, confirmed by user):
  * A cycle that crosses THRESHOLD ⇒ adopt winner (always).
  * A cycle that "IMPROVES" significantly (ΔPassRate ≥ 5pp OR ΔMetric ≥ 0.05)
    without crossing THRESHOLD ⇒ still commits the winner as a new version
    with a "below threshold but improved" rationale.
  * MARGINAL / PLATEAU ⇒ commit ONLY if caller passes ``commit_on_plateau=True``.
  * All cycles REGRESSED (ΔPassRate ≤ -5pp OR ΔMetric ≤ -0.05 on the winner)
    ⇒ never commit, regardless of ``commit_on_plateau`` (hard rule).

Winner tie-break key (descending):
  (crossed?, metric_avg, pass_rate, -diff_lines, -cycle_number)

The function emits progress events via an optional ``progress_cb(event, payload)``
awaitable; the HTTP endpoint wraps these into SSE events.
"""

from __future__ import annotations

import difflib
import logging
import uuid
from statistics import fmean
from typing import Any, Awaitable, Callable, Literal, Optional

from pydantic import BaseModel, Field

from src.db.database import get_session
from src.db.models import RegressionResult
from src.evaluation.regression import RegressionRunner
from src.optimization.drafter import DraftResult, draft_new_prompt
from src.optimization.feedback_store import get_feedback_store
from src.prompts.registry import get_registry


logger = logging.getLogger(__name__)


# ── Thresholds (in one place so the decision tree stays self-documenting) ─────

SIG_PASS_PP: float = 0.05          # ≥5pp pass-rate gain = significant
SIG_METRIC: float = 0.05           # ≥0.05 metric gain = significant
NOISE_PASS_PP: float = 0.05        # |delta| under this = noise (plateau)
NOISE_METRIC: float = 0.03         # |delta| under this = noise (plateau)
REGRESS_PP: float = 0.05           # ≥5pp pass-rate drop = regression
REGRESS_METRIC: float = 0.05       # ≥0.05 metric drop = regression
DEFAULT_MAX_CYCLES: int = 3
DEFAULT_BOOTSTRAP_IDS: list[str] = [
    "golden_001", "golden_004", "golden_005", "golden_006", "golden_021",
]


Classification = Literal["crossed", "improved", "marginal", "plateau", "regressed"]
LoopDecision = Literal[
    "adopt",
    "early_exit_crossed",
    "iterate_recover",
    "regress_max_cycles",
    "iterate_plateau",
    "plateau_stop",
    "plateau_max_cycles",
    "iterate_improve",
    "improved_max_cycles",
    "iterate_marginal",
    "marginal_max_cycles",
]
CommitStatus = Literal[
    "adopted",
    "improved_below_threshold",
    "forced_marginal",
    "forced_plateau",
    "plateau_no_commit",
    "all_regressed",
    "no_cycles",
    "dry_run",
]


# ── Public output types ───────────────────────────────────────────────────────


class CycleResult(BaseModel):
    cycle: int
    classification: Classification
    pass_rate: float
    metric_avg: float
    delta_pass_pp: float
    delta_metric: float
    loop_decision: LoopDecision
    prompt_text: str
    rationale: str
    change_type: str
    diff_lines: int
    validate_run_id: str                 # in-memory run id (not persisted)
    raw_scores: list[dict] = Field(default_factory=list)


class OptimizationReport(BaseModel):
    role: str
    target_version: str
    metric: str
    threshold: float
    team_id: str
    model: str
    baseline_run_id: Optional[str]
    golden_ids: list[str]
    baseline_pass_rate: float
    baseline_metric_avg: float
    cycles: list[CycleResult]
    winner_cycle: Optional[int]
    commit_status: CommitStatus
    committed_version: Optional[str]
    recommendation: str
    early_exit: bool
    commit_on_plateau: bool
    dry_run: bool


# ── Event emitter protocol ────────────────────────────────────────────────────

ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


async def _noop_emit(_event: str, _payload: dict[str, Any]) -> None:
    return None


# ── Classification + decision helpers ─────────────────────────────────────────


def _classify(
    pass_rate: float,
    metric_avg: float,
    threshold: float,
    base_pass: float,
    base_metric: float,
) -> Classification:
    d_pass = pass_rate - base_pass
    d_met = metric_avg - base_metric

    crossed = metric_avg >= threshold and pass_rate >= base_pass
    if crossed:
        return "crossed"

    if d_pass <= -REGRESS_PP or d_met <= -REGRESS_METRIC:
        return "regressed"

    if d_pass >= SIG_PASS_PP or d_met >= SIG_METRIC:
        return "improved"

    if abs(d_pass) < NOISE_PASS_PP and abs(d_met) < NOISE_METRIC:
        return "plateau"

    # Positive but not significant → marginal; negative but not regressive → plateau-ish.
    # Treat the small-positive band as marginal, anything else as plateau.
    if d_pass > 0 or d_met > 0:
        return "marginal"
    return "plateau"


def _loop_decision(cls: Classification, cycle: int, max_cycles: int, early_exit: bool) -> LoopDecision:
    if cls == "crossed":
        return "early_exit_crossed" if early_exit else "adopt"
    if cls == "regressed":
        return "iterate_recover" if cycle < max_cycles else "regress_max_cycles"
    if cls == "plateau":
        if cycle == 1:
            return "iterate_plateau"
        return "plateau_stop" if cycle == 2 else "plateau_max_cycles"
    if cls == "improved":
        return "iterate_improve" if cycle < max_cycles else "improved_max_cycles"
    # marginal
    return "iterate_marginal" if cycle < max_cycles else "marginal_max_cycles"


def _should_break(decision: LoopDecision) -> bool:
    return decision in {
        "adopt",
        "early_exit_crossed",
        "plateau_stop",
        "plateau_max_cycles",
        "regress_max_cycles",
        "improved_max_cycles",
        "marginal_max_cycles",
    }


def _rank_key(cr: CycleResult) -> tuple:
    # (crossed? desc, metric desc, pass desc, -diff desc (smaller=safer), -cycle (earlier=cheaper))
    return (
        1 if cr.classification == "crossed" else 0,
        cr.metric_avg,
        cr.pass_rate,
        -cr.diff_lines,
        -cr.cycle,
    )


def _pick_winner(cycles: list[CycleResult]) -> Optional[CycleResult]:
    if not cycles:
        return None
    return max(cycles, key=_rank_key)


# ── DB helpers (all read-only — writes go through PromptRegistry.register) ────


def _fetch_rows(
    run_id: str, role: str, golden_ids: list[str]
) -> list[RegressionResult]:
    session = get_session()
    try:
        q = session.query(RegressionResult).filter(
            RegressionResult.run_id == run_id,
            RegressionResult.actual_agent.like(f"%{role}%"),
        )
        if golden_ids:
            q = q.filter(RegressionResult.golden_case_id.in_(golden_ids))
        return q.all()
    finally:
        session.close()


def _fetch_rows_inmemory_like(
    role: str, results: list[dict]
) -> list[dict]:
    """Return the in-memory run_subset results filtered to rows for ``role``."""
    return [r for r in results if role in (r.get("actual_agent") or "")]


def _extract_metric(row_scores: dict, metric: str) -> Optional[float]:
    """Pull a metric out of the merged quality+deepeval scores dict.

    We accept both the canonical key and a ``{metric}_de`` fallback (DeepEval
    stores some metrics under a suffixed name, e.g. ``step_efficiency_de``).
    """
    if not row_scores:
        return None
    v = row_scores.get(metric)
    if isinstance(v, (int, float)):
        return float(v)
    v = row_scores.get(f"{metric}_de")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _merge_scores(result_row: dict) -> dict:
    qs = result_row.get("quality_scores") or {}
    ds = result_row.get("deepeval_scores") or {}
    if isinstance(qs, str):
        import json
        try:
            qs = json.loads(qs)
        except Exception:
            qs = {}
    if isinstance(ds, str):
        import json
        try:
            ds = json.loads(ds)
        except Exception:
            ds = {}
    return {**qs, **ds}


def _pass_rate_and_metric(
    rows: list[dict], metric: str
) -> tuple[float, float, list[dict]]:
    """Compute (pass_rate, metric_avg, per-row detail) from a list of result dicts."""
    if not rows:
        return 0.0, 0.0, []
    details: list[dict] = []
    pass_count = 0
    metric_vals: list[float] = []
    for r in rows:
        merged = _merge_scores(r)
        mv = _extract_metric(merged, metric)
        passed = bool(r.get("overall_pass"))
        if passed:
            pass_count += 1
        if mv is not None:
            metric_vals.append(mv)
        details.append({
            "golden_id": r.get("golden_case_id"),
            "pass": passed,
            "metric": mv,
            "scores": merged,
        })
    pass_rate = pass_count / len(rows)
    metric_avg = fmean(metric_vals) if metric_vals else 0.0
    return pass_rate, metric_avg, details


def _format_rows_for_drafter(rows: list[dict], metric: str, threshold: float) -> str:
    if not rows:
        return "(no baseline rows)"
    lines = []
    for r in rows:
        merged = _merge_scores(r)
        mv = _extract_metric(merged, metric)
        score_str = f"{mv:.3f}" if mv is not None else "N/A"
        verdict = "PASS" if r.get("overall_pass") else "FAIL"
        lines.append(
            f"  [{verdict}] {r.get('golden_case_id')} | {metric}={score_str}"
        )
        bad = [
            f"{k}={v:.2f}" for k, v in merged.items()
            if isinstance(v, (int, float)) and v < threshold
        ]
        if bad:
            lines.append(f"    low sub-metrics (<{threshold}): {', '.join(bad)}")
    return "\n".join(lines)


def _count_diff_lines(old: str, new: str) -> int:
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), n=0))
    return sum(1 for ln in diff if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---")))


# ── Baseline resolver ─────────────────────────────────────────────────────────


def _fetch_baseline_from_db(
    role: str, _version: str, run_id: str, golden_ids: list[str]
) -> list[dict]:
    """Re-read a persisted regression run to reconstruct baseline metrics."""
    rows = _fetch_rows(run_id, role, golden_ids)
    out: list[dict] = []
    for r in rows:
        out.append({
            "golden_case_id": r.golden_case_id,
            "actual_agent": r.actual_agent,
            "overall_pass": r.overall_pass,
            "quality_scores": r.quality_scores,
            "deepeval_scores": r.deepeval_scores,
            "actual_tool_calls": r.actual_tool_calls,
        })
    return out


# ── Main entry point ──────────────────────────────────────────────────────────


async def run_optimization_loop(
    *,
    role: str,
    metric: str = "step_efficiency",
    threshold: float = 0.7,
    version: Optional[str] = None,       # None => latest
    team_id: str = "default",
    baseline_run_id: Optional[str] = None,
    golden_ids: Optional[list[str]] = None,
    early_exit: bool = True,
    commit_on_plateau: bool = False,
    dry_run: bool = False,
    model: str = "claude-sonnet-4.6",
    max_cycles: int = DEFAULT_MAX_CYCLES,
    progress_cb: Optional[ProgressCallback] = None,
) -> OptimizationReport:
    """Run the full optimization loop for one agent role.

    Parameters
    ----------
    role, metric, threshold, version
        Target agent role, metric name, pass threshold, prompt version to
        improve FROM. ``version=None`` resolves to the registry's latest.
    team_id
        Team whose goldens will be used for validation (target team). The
        function does NOT build a team orchestrator graph — it only uses the
        team to scope regression goldens and the validate-runner.
    baseline_run_id, golden_ids
        Pin the baseline to a specific past regression run + specific golden
        IDs. If either is missing, the function bootstraps a fresh baseline
        regression over ``DEFAULT_BOOTSTRAP_IDS``.
    early_exit
        Stop as soon as a cycle crosses threshold.
    commit_on_plateau
        Persist best attempt even when only MARGINAL / PLATEAU. Never persists
        a REGRESSED attempt.
    dry_run
        Skip the final commit step; report ``commit_status="dry_run"`` and
        ``committed_version=None`` but include the full cycle trace.
    progress_cb
        Awaitable invoked as ``await progress_cb(event_name, payload)`` at
        every phase / tool / cycle boundary. SSE endpoints wrap this.
    """

    emit = progress_cb or _noop_emit
    registry = get_registry()

    # ── Resolve version ──────────────────────────────────────────────────────
    resolved_version = version or registry.latest_version(role)

    await emit("phase_start", {
        "phase": "Setup",
        "role": role,
        "metric": metric,
        "threshold": threshold,
        "version": resolved_version,
        "team_id": team_id,
        "baseline_run_id": baseline_run_id,
        "golden_ids": list(golden_ids or []),
        "early_exit": early_exit,
        "commit_on_plateau": commit_on_plateau,
        "dry_run": dry_run,
        "model": model,
        "max_cycles": max_cycles,
    })

    # ── Phase 1 — Baseline ───────────────────────────────────────────────────
    await emit("phase_start", {"phase": "Baseline", "role": role})

    baseline_rows: list[dict] = []
    effective_golden_ids = list(golden_ids or [])

    if baseline_run_id and effective_golden_ids:
        await emit("tool_start", {
            "tool": "fetch_baseline_from_db",
            "args": {"run_id": baseline_run_id, "goldens": effective_golden_ids},
        })
        baseline_rows = _fetch_baseline_from_db(
            role, resolved_version, baseline_run_id, effective_golden_ids
        )
        await emit("tool_end", {
            "tool": "fetch_baseline_from_db",
            "rows_found": len(baseline_rows),
        })

    if not baseline_rows:
        # Bootstrap: run a real regression over the user's golden_ids (or the
        # default bootstrap set) and use that as the pinned baseline.
        bootstrap_ids = effective_golden_ids or list(DEFAULT_BOOTSTRAP_IDS)
        await emit("tool_start", {
            "tool": "bootstrap_regression",
            "args": {"role": role, "version": resolved_version, "goldens": bootstrap_ids},
        })
        runner = RegressionRunner(
            model=model,
            prompt_version=resolved_version,
            team_id=team_id,
            prompt_versions_by_role={role: resolved_version},
            persist=True,      # persist the bootstrap so users can re-pin it
        )
        bootstrap_results = await runner.run_subset(bootstrap_ids, max_parallel=3)
        baseline_run_id = (bootstrap_results[0].get("run_id")
                           if bootstrap_results else None)
        baseline_rows = _fetch_rows_inmemory_like(role, bootstrap_results)
        effective_golden_ids = bootstrap_ids
        await emit("tool_end", {
            "tool": "bootstrap_regression",
            "run_id": baseline_run_id,
            "rows_found": len(baseline_rows),
        })

    base_pass_rate, base_metric_avg, _base_details = _pass_rate_and_metric(
        baseline_rows, metric
    )

    await emit("baseline_computed", {
        "role": role,
        "pass_rate": base_pass_rate,
        "metric_avg": base_metric_avg,
        "n_rows": len(baseline_rows),
    })

    # Retrieve few-shot failures (best effort — non-fatal on failure)
    similar_text = "(no similar past failures)"
    try:
        fb = get_feedback_store()
        similar = fb.retrieve_similar_failures(
            role, f"{metric} below {threshold}", limit=5
        )
        if similar:
            similar_text = "\n".join(
                f"  [{s['golden_id']}] v={s['version']} "
                f"tools={s.get('tool_trace_summary', '')} "
                f"failures={s.get('failure_types', '')}"
                for s in similar
            )
    except Exception as e:
        logger.debug("feedback_store lookup failed (non-fatal): %s", e)

    current_prompt = registry.get_prompt(role, resolved_version) or ""
    if not current_prompt:
        raise ValueError(
            f"No prompt text found for role={role} version={resolved_version}. "
            "Cannot optimize without a baseline prompt."
        )

    baseline_rows_text = _format_rows_for_drafter(baseline_rows, metric, threshold)

    # ── Phase 2 — Loop ───────────────────────────────────────────────────────
    await emit("phase_start", {"phase": "Loop", "role": role})

    cycles: list[CycleResult] = []
    prior_summary_lines: list[str] = []

    for cycle_n in range(1, max_cycles + 1):
        await emit("cycle_start", {
            "role": role,
            "cycle": cycle_n,
            "from_version": resolved_version,
        })

        # L1 — Draft (one LLM call)
        await emit("llm_drafting", {"role": role, "cycle": cycle_n})
        try:
            draft: DraftResult = await draft_new_prompt(
                role=role,
                metric=metric,
                threshold=threshold,
                version=resolved_version,
                current_prompt=current_prompt,
                baseline_failures=baseline_rows_text,
                similar_failures=similar_text,
                cycle=cycle_n,
                max_cycles=max_cycles,
                prior_summary="\n".join(prior_summary_lines),
                model=model,
            )
        except Exception as e:
            await emit("error", {
                "role": role, "cycle": cycle_n,
                "phase": "draft", "message": str(e),
            })
            logger.exception("Drafter failed on cycle %d: %s", cycle_n, e)
            break

        diff_lines = _count_diff_lines(current_prompt, draft.new_prompt)

        # L5 — Validate via in-memory override (NO DB writes)
        await emit("tool_start", {
            "role": role, "cycle": cycle_n,
            "tool": "validate_regression",
            "args": {"goldens": effective_golden_ids, "persist": False},
        })
        validate_run_id = uuid.uuid4().hex[:12]
        try:
            val_runner = RegressionRunner(
                model=model,
                prompt_version=resolved_version,
                team_id=team_id,
                prompt_versions_by_role={role: resolved_version},
                prompt_text_overrides={role: draft.new_prompt},
                persist=False,
            )
            val_results = await val_runner.run_subset(
                effective_golden_ids, max_parallel=3
            )
        except Exception as e:
            await emit("error", {
                "role": role, "cycle": cycle_n,
                "phase": "validate", "message": str(e),
            })
            logger.exception("Validation regression failed on cycle %d: %s", cycle_n, e)
            break

        val_rows = _fetch_rows_inmemory_like(role, val_results)
        pass_rate, metric_avg, details = _pass_rate_and_metric(val_rows, metric)
        d_pass = pass_rate - base_pass_rate
        d_met = metric_avg - base_metric_avg
        cls = _classify(pass_rate, metric_avg, threshold, base_pass_rate, base_metric_avg)
        decision = _loop_decision(cls, cycle_n, max_cycles, early_exit)

        await emit("tool_end", {
            "role": role, "cycle": cycle_n,
            "tool": "validate_regression",
            "pass_rate": pass_rate, "metric_avg": metric_avg,
            "n_rows": len(val_rows),
        })

        cr = CycleResult(
            cycle=cycle_n,
            classification=cls,
            pass_rate=pass_rate,
            metric_avg=metric_avg,
            delta_pass_pp=d_pass,
            delta_metric=d_met,
            loop_decision=decision,
            prompt_text=draft.new_prompt,
            rationale=draft.rationale,
            change_type=draft.change_type,
            diff_lines=diff_lines,
            validate_run_id=validate_run_id,
            raw_scores=details,
        )
        cycles.append(cr)
        await emit("cycle_end", cr.model_dump())

        prior_summary_lines.append(
            f"cycle {cycle_n}: {cls} (ΔPass={d_pass:+.1%}, Δ{metric}={d_met:+.3f}) "
            f"change_type={draft.change_type}"
        )

        if _should_break(decision):
            break

    # ── Phase 3 — Commit decision ────────────────────────────────────────────
    await emit("phase_start", {"phase": "Report", "role": role})

    winner = _pick_winner(cycles)
    status, committed_ver, recommendation = _commit_decision(
        cycles=cycles,
        winner=winner,
        commit_on_plateau=commit_on_plateau,
        role=role,
        parent_version=resolved_version,
        metric=metric,
        base_pass=base_pass_rate,
        base_metric=base_metric_avg,
        registry=registry,
        dry_run=dry_run,
    )

    report = OptimizationReport(
        role=role,
        target_version=resolved_version,
        metric=metric,
        threshold=threshold,
        team_id=team_id,
        model=model,
        baseline_run_id=baseline_run_id,
        golden_ids=effective_golden_ids,
        baseline_pass_rate=base_pass_rate,
        baseline_metric_avg=base_metric_avg,
        cycles=cycles,
        winner_cycle=winner.cycle if winner else None,
        commit_status=status,
        committed_version=committed_ver,
        recommendation=recommendation,
        early_exit=early_exit,
        commit_on_plateau=commit_on_plateau,
        dry_run=dry_run,
    )

    await emit("report", report.model_dump())
    await emit("done", {
        "role": role,
        "status": status,
        "committed_version": committed_ver,
    })

    return report


# ── Commit decision (pure function — unit-testable) ──────────────────────────


def _commit_decision(
    *,
    cycles: list[CycleResult],
    winner: Optional[CycleResult],
    commit_on_plateau: bool,
    role: str,
    parent_version: str,
    metric: str,
    base_pass: float,
    base_metric: float,
    registry,
    dry_run: bool,
) -> tuple[CommitStatus, Optional[str], str]:
    if dry_run:
        return (
            "dry_run",
            None,
            "Dry run — no prompt version registered. "
            f"Winner cycle: {winner.cycle if winner else 'none'}.",
        )

    if not cycles or winner is None:
        return "no_cycles", None, "No cycles completed — nothing to commit."

    has_crossed = any(c.classification == "crossed" for c in cycles)
    crossed_note = (
        f"Adopted cycle-{winner.cycle}: {metric} crossed threshold "
        f"(pass {base_pass:.0%} → {winner.pass_rate:.0%}, "
        f"{metric} {base_metric:.3f} → {winner.metric_avg:.3f}). "
        f"Drafter rationale: {winner.rationale}"
    )
    improved_note = (
        f"Cycle-{winner.cycle}: below threshold but improved "
        f"(ΔPass {winner.delta_pass_pp:+.1%}, Δ{metric} {winner.delta_metric:+.3f}). "
        f"Human review recommended. Drafter rationale: {winner.rationale}"
    )

    def _commit(rationale: str) -> str:
        new_ver = registry.register(
            role=role,
            prompt_text=winner.prompt_text,
            rationale=rationale,
            parent_version=parent_version,
            created_by="optimizer_fn",
        )
        try:
            registry.update_metric_scores(role, new_ver, {
                metric: winner.metric_avg,
                "pass_rate": winner.pass_rate,
                "delta_pass_pp": winner.delta_pass_pp,
                "delta_metric": winner.delta_metric,
                "classification": winner.classification,
                "winner_cycle": winner.cycle,
            })
        except Exception as e:
            logger.debug("update_metric_scores failed (non-fatal): %s", e)
        return new_ver

    if has_crossed:
        new_ver = _commit(crossed_note)
        return "adopted", new_ver, (
            f"Adopted cycle-{winner.cycle} as {new_ver} "
            f"(crossed threshold; ΔPass {winner.delta_pass_pp:+.1%})."
        )

    if winner.classification == "improved":
        new_ver = _commit(improved_note)
        return "improved_below_threshold", new_ver, (
            f"Committed below-threshold winner cycle-{winner.cycle} as {new_ver} "
            f"(ΔPass {winner.delta_pass_pp:+.1%}, Δ{metric} {winner.delta_metric:+.3f}). "
            "Threshold not crossed — recommend review."
        )

    if winner.classification in ("marginal", "plateau"):
        if commit_on_plateau:
            note = (
                f"Forced commit on {winner.classification}: "
                f"cycle-{winner.cycle} (ΔPass {winner.delta_pass_pp:+.1%}, "
                f"Δ{metric} {winner.delta_metric:+.3f}). Drafter rationale: {winner.rationale}"
            )
            new_ver = _commit(note)
            status: CommitStatus = (
                "forced_marginal" if winner.classification == "marginal" else "forced_plateau"
            )
            return status, new_ver, (
                f"Forced-committed {winner.classification} winner cycle-{winner.cycle} as {new_ver}."
            )
        return "plateau_no_commit", None, (
            f"All cycles {winner.classification} (best cycle-{winner.cycle}, "
            f"ΔPass {winner.delta_pass_pp:+.1%}, Δ{metric} {winner.delta_metric:+.3f}). "
            "Nothing committed. Set commit_on_plateau=true to force."
        )

    # REGRESSED — hard rule, never commit
    return "all_regressed", None, (
        f"All cycles regressed vs baseline (best cycle-{winner.cycle}, "
        f"ΔPass {winner.delta_pass_pp:+.1%}, Δ{metric} {winner.delta_metric:+.3f}). "
        "Nothing committed — recommend human review of the drafter heuristics."
    )
