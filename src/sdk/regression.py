"""
RegressionClient — programmatic access to the regression suite.

Provides three operations that map 1:1 onto the same code paths the
``/api/regression`` HTTP routes use:

    * ``run``           — execute golden cases against a (model, prompt) pair
    * ``list_cases``    — golden cases visible to this team (filterable)
    * ``recent_runs``   — paginate ``EvalRun`` rows for the team
    * ``result``        — fetch a single run + its per-case payloads
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable, Optional

from src.db.database import get_session
from src.db.models import EvalRun, RegressionResult, Team
from src.evaluation.regression import RegressionRunner


class RegressionClient:
    def __init__(self, team_id: str) -> None:
        self.team_id = team_id

    # ── Helpers ──────────────────────────────────────────────────────

    def _team_dataset_groups(self) -> list[str] | None:
        """Resolve the team's ``config_json["dataset_groups"]`` if set."""
        session = get_session()
        try:
            t = session.query(Team).filter_by(id=self.team_id).one_or_none()
            if t is None:
                return None
            cfg = dict(t.config_json or {})
            groups = cfg.get("dataset_groups")
            if isinstance(groups, list) and groups:
                return [str(g) for g in groups]
            return None
        finally:
            session.close()

    # ── Reads ────────────────────────────────────────────────────────

    def list_cases(
        self,
        *,
        dataset_groups: Optional[Iterable[str]] = None,
        case_ids: Optional[Iterable[str]] = None,
    ) -> list[dict[str, Any]]:
        """Return golden cases visible to this team.

        ``dataset_groups`` overrides the team default; otherwise the team's
        ``config_json["dataset_groups"]`` filter is applied (or no filter at
        all when neither is set).
        """
        from src.evaluation.golden import get_active_cases
        groups = list(dataset_groups) if dataset_groups else self._team_dataset_groups()
        cases = get_active_cases(
            list(case_ids) if case_ids else None,
            dataset_groups=groups,
        )
        return [
            {
                "id": c.id,
                "name": c.name,
                "prompt": c.prompt,
                "complexity": c.complexity,
                "dataset_group": c.dataset_group,
                "expected_agent": c.expected_agent,
                "expected_tools": c.expected_tools,
                "max_llm_calls": c.max_llm_calls,
                "max_tool_calls": c.max_tool_calls,
                "max_tokens": c.max_tokens,
                "max_latency_ms": c.max_latency_ms,
            }
            for c in cases
        ]

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        session = get_session()
        try:
            q = (
                session.query(EvalRun)
                .filter(EvalRun.team_id == self.team_id)
                .order_by(EvalRun.created_at.desc())
                .limit(limit)
            )
            return [
                {
                    "run_id": r.id,
                    "model": r.model,
                    "prompt_version": r.prompt_version,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "summary": r.summary,
                }
                for r in q.all()
            ]
        finally:
            session.close()

    def result(self, run_id: str) -> dict[str, Any] | None:
        session = get_session()
        try:
            run = session.query(EvalRun).filter_by(id=run_id).one_or_none()
            if run is None:
                return None
            results = (
                session.query(RegressionResult)
                .filter_by(eval_run_id=run_id)
                .all()
            )
            return {
                "run_id": run.id,
                "team_id": run.team_id,
                "model": run.model,
                "prompt_version": run.prompt_version,
                "summary": run.summary,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "results": [
                    {
                        "id": r.id,
                        "case_id": r.case_id,
                        "passed": r.passed,
                        "regressed": r.regressed,
                        "summary": r.summary,
                        "trace_id": r.trace_id,
                    }
                    for r in results
                ],
            }
        finally:
            session.close()

    # ── Writes ───────────────────────────────────────────────────────

    def run(
        self,
        *,
        case_ids: Optional[Iterable[str]] = None,
        dataset_groups: Optional[Iterable[str]] = None,
        model: Optional[str] = None,
        prompt_version: str = "v1",
        prompt_versions_by_role: Optional[dict[str, str]] = None,
        baseline_run_id: Optional[str] = None,
        max_parallel: int = 1,
    ) -> dict[str, Any]:
        """Run regression cases for this team and return the summary payload.

        Mirrors the args of ``RegressionRunner`` but defaults the dataset
        filter to the team's configured groups (so callers don't have to
        repeat ``dataset_groups=["finance_v1"]`` in every call).
        """
        groups = list(dataset_groups) if dataset_groups else self._team_dataset_groups()
        runner = RegressionRunner(
            model=model,
            prompt_version=prompt_version,
            team_id=self.team_id,
            prompt_versions_by_role=prompt_versions_by_role or {},
            dataset_groups=groups,
        )
        return asyncio.run(runner.run(
            case_ids=list(case_ids) if case_ids else None,
            baseline_run_id=baseline_run_id,
            max_parallel=max_parallel,
        ))

    async def arun(
        self,
        *,
        case_ids: Optional[Iterable[str]] = None,
        dataset_groups: Optional[Iterable[str]] = None,
        model: Optional[str] = None,
        prompt_version: str = "v1",
        prompt_versions_by_role: Optional[dict[str, str]] = None,
        baseline_run_id: Optional[str] = None,
        max_parallel: int = 1,
    ) -> dict[str, Any]:
        """Async variant of :meth:`run` — useful from inside event loops."""
        groups = list(dataset_groups) if dataset_groups else self._team_dataset_groups()
        runner = RegressionRunner(
            model=model,
            prompt_version=prompt_version,
            team_id=self.team_id,
            prompt_versions_by_role=prompt_versions_by_role or {},
            dataset_groups=groups,
        )
        return await runner.run(
            case_ids=list(case_ids) if case_ids else None,
            baseline_run_id=baseline_run_id,
            max_parallel=max_parallel,
        )
