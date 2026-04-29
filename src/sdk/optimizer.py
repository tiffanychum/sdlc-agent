"""
OptimizerClient — thin wrapper around ``run_optimization_loop``.

Lets SDK callers iterate on a prompt the same way the Studio "Run
Optimization Loop" button does: baseline → drafter → in-memory validate →
classify → commit (or dry-run).
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from src.optimization.optimizer import OptimizationReport, run_optimization_loop


class OptimizerClient:
    def __init__(self, team_id: str) -> None:
        self.team_id = team_id

    def run(
        self,
        *,
        role: str,
        metric: str = "step_efficiency",
        threshold: float = 0.7,
        version: Optional[str] = None,
        baseline_run_id: Optional[str] = None,
        golden_ids: Optional[list[str]] = None,
        early_exit: bool = True,
        commit_on_plateau: bool = False,
        dry_run: bool = False,
        model: str = "claude-sonnet-4.6",
        max_cycles: int = 3,
    ) -> dict[str, Any]:
        """Synchronous front-end for the optimization loop.

        Returns the ``OptimizationReport`` as a dict so it round-trips through
        JSON for notebooks / demo scripts.
        """
        report: OptimizationReport = asyncio.run(run_optimization_loop(
            role=role,
            metric=metric,
            threshold=threshold,
            version=version,
            team_id=self.team_id,
            baseline_run_id=baseline_run_id,
            golden_ids=list(golden_ids) if golden_ids else None,
            early_exit=early_exit,
            commit_on_plateau=commit_on_plateau,
            dry_run=dry_run,
            model=model,
            max_cycles=max_cycles,
        ))
        return report.model_dump() if hasattr(report, "model_dump") else dict(report)

    async def arun(
        self,
        *,
        role: str,
        metric: str = "step_efficiency",
        threshold: float = 0.7,
        version: Optional[str] = None,
        baseline_run_id: Optional[str] = None,
        golden_ids: Optional[list[str]] = None,
        early_exit: bool = True,
        commit_on_plateau: bool = False,
        dry_run: bool = False,
        model: str = "claude-sonnet-4.6",
        max_cycles: int = 3,
    ) -> dict[str, Any]:
        report: OptimizationReport = await run_optimization_loop(
            role=role,
            metric=metric,
            threshold=threshold,
            version=version,
            team_id=self.team_id,
            baseline_run_id=baseline_run_id,
            golden_ids=list(golden_ids) if golden_ids else None,
            early_exit=early_exit,
            commit_on_plateau=commit_on_plateau,
            dry_run=dry_run,
            model=model,
            max_cycles=max_cycles,
        )
        return report.model_dump() if hasattr(report, "model_dump") else dict(report)
