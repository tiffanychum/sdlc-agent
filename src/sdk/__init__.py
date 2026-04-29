"""
SDLC Hub SDK — in-process Python client for observability, evaluation,
and report retrieval.

The SDK is a thin wrapper over the same code paths used by the FastAPI
server, so traces produced via the SDK appear in the existing UI exactly
like traces produced via /api/teams/{id}/chat.

Quick start:

    from src.sdk import HubClient

    hub = HubClient(team="Finance Team")

    @hub.trace.trace_agent(name="finance_report", agent_used="planner")
    async def report(question: str) -> str:
        with hub.trace.span("fetch_data", span_type="tool_call") as s:
            ...
            s.set_tokens(in_=300, out_=80, model="gpt-4o-mini")
        return answer

    result = await report("Q4 burn rate")
    await hub.eval.run(result.trace_id)              # DeepEval scoring
    perf = hub.reports.performance_report(days=7)    # team-scoped
"""

from src.sdk.client import HubClient
from src.sdk.observability import TraceResult
from src.sdk.teams import create_team_static

__all__ = ["HubClient", "TraceResult", "create_team_static"]
