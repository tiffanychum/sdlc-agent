"""
EvalClient — DeepEval scoring for traces produced by the SDK.

Calls ``run_all_deepeval_metrics`` (the same function the FastAPI server
uses in ``/api/traces/evaluate``) and persists the resulting scores back
to the ``traces.eval_scores`` JSON column. The legacy G-Eval path was
removed app-wide; only DeepEval scores and a defensive default fallback
remain.
"""

from __future__ import annotations

import copy
import json as _json
import logging
from typing import Any

from sqlalchemy.orm.attributes import flag_modified

from src.db.database import get_session
from src.db.models import Span, Trace as TraceModel

logger = logging.getLogger(__name__)


_DEFAULT_DEEPEVAL_FALLBACK = {
    "deepeval_relevancy": 0.5,
    "deepeval_faithfulness": 0.5,
    "tool_correctness": 0.5,
    "argument_correctness": 0.5,
    "task_completion": 0.5,
    "step_efficiency_de": 0.5,
    "plan_quality": 0.5,
    "plan_adherence": 0.5,
}


class EvalClient:
    """Scores traces using DeepEval. Always team-scoped via the SDK."""

    def __init__(self, team_id: str) -> None:
        self.team_id = team_id

    async def run(self, trace_id: str) -> dict[str, Any]:
        """Run DeepEval on a single trace and persist scores.

        Returns a dict with keys::

            {
                "trace_id": <id>,
                "deepeval_scores": {...},
                "evaluated": True | False,
                "error": <optional message>,
            }

        The function is idempotent: re-running on an already-evaluated
        trace overwrites the DeepEval scores but leaves any legacy keys
        (e.g. semantic_similarity) untouched.
        """
        from src.evaluation.integrations import run_all_deepeval_metrics

        session = get_session()
        try:
            tr = session.query(TraceModel).filter(TraceModel.id == trace_id).one_or_none()
            if tr is None:
                return {"trace_id": trace_id, "evaluated": False, "error": "trace not found"}
            if tr.team_id and tr.team_id != self.team_id:
                return {
                    "trace_id": trace_id,
                    "evaluated": False,
                    "error": (
                        f"trace belongs to team {tr.team_id!r}, "
                        f"refusing to evaluate from client scoped to {self.team_id!r}"
                    ),
                }

            existing = copy.deepcopy(tr.eval_scores) if tr.eval_scores else {}
            if isinstance(existing, str):
                existing = _json.loads(existing) if existing else {}

            # Collect span output_data — same shape the server uses.
            tool_outputs: list[str] = []
            for span_row in session.query(Span).filter_by(trace_id=tr.id).all():
                if span_row.output_data:
                    for v in span_row.output_data.values():
                        if v:
                            tool_outputs.append(str(v)[:300])

            agent_trace = tr.tool_calls_json if isinstance(tr.tool_calls_json, list) else []

            try:
                deepeval_scores = await run_all_deepeval_metrics(
                    user_prompt=tr.user_prompt or "",
                    agent_response=tr.agent_response or "",
                    agent_trace=agent_trace,
                    tool_outputs=tool_outputs[:5],
                )
                existing["deepeval_scores"] = deepeval_scores
                error = None
            except Exception as exc:
                logger.warning("DeepEval scoring failed for %s: %s", trace_id, exc)
                existing["deepeval_scores"] = dict(_DEFAULT_DEEPEVAL_FALLBACK)
                error = f"deepeval failure: {exc!s}"

            tr.eval_scores = existing
            tr.eval_status = "evaluated"
            flag_modified(tr, "eval_scores")
            session.commit()

            return {
                "trace_id": trace_id,
                "deepeval_scores": existing.get("deepeval_scores", {}),
                "evaluated": True,
                "error": error,
            }
        finally:
            session.close()

    async def run_batch(self, trace_ids: list[str]) -> list[dict[str, Any]]:
        """Sequentially evaluate a list of trace_ids — mirrors the
        single-trace contract per id."""
        results = []
        for tid in trace_ids:
            results.append(await self.run(tid))
        return results
