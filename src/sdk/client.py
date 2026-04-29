"""
HubClient — entry point for the SDLC Hub SDK.

Resolves a team (by name or id) once at construction time and exposes the
three sub-clients used in normal SDK flows:

- ``hub.trace``    — observability primitives (decorator + span ctx mgr)
- ``hub.eval``     — DeepEval-backed scoring of completed traces
- ``hub.reports``  — programmatic, team-scoped retrieval of traces,
                     performance reports, anomalies and stats
"""

from __future__ import annotations

from src.db.database import get_session, init_db
from src.db.models import Team

from src.sdk.chat import ChatClient
from src.sdk.evaluation import EvalClient
from src.sdk.observability import ObservabilityClient
from src.sdk.optimizer import OptimizerClient
from src.sdk.prompts import PromptsClient
from src.sdk.rag import RagClient
from src.sdk.regression import RegressionClient
from src.sdk.reports import ReportsClient
from src.sdk.teams import TeamsClient, create_team_static
from src.sdk.tools import ToolsClient


class HubClient:
    """Project/team-scoped, in-process client.

    Args:
        team:        Human-readable team name (case-insensitive). Resolved
                     to a ``team_id`` via the existing ``teams`` table.
        team_id:     Direct ``team_id`` if the caller already knows it.
                     Takes precedence over ``team`` when both are supplied.
        auto_init_db:  When True (default) ``init_db()`` is called once so
                       SDK calls work in any process — including tests
                       that boot independently of the FastAPI server.
        auto_create: When True, create the team if it doesn't yet exist.
                     Used by ``HubClient.auto_create(...)`` factory.
        backend_url: Base URL for the FastAPI backend (used by ``chat``).
                     Defaults to ``$SDLC_HUB_BACKEND`` or
                     ``http://localhost:8000``.
    """

    def __init__(
        self,
        team: str | None = None,
        team_id: str | None = None,
        *,
        auto_init_db: bool = True,
        backend_url: str | None = None,
    ) -> None:
        if not team and not team_id:
            raise ValueError("HubClient requires either `team` or `team_id`.")

        if auto_init_db:
            try:
                init_db()
            except Exception:
                # init_db is idempotent for the FastAPI lifespan; if it fails
                # here it will fail loudly on the first DB call below.
                pass

        if team_id:
            self.team_id = team_id
            self.team_name = self._lookup_name(team_id) or team or team_id
        else:
            self.team_id, self.team_name = self._resolve_by_name(team)

        self._backend_url = backend_url

        # Lazy sub-clients
        self._trace_client: ObservabilityClient | None = None
        self._eval_client: EvalClient | None = None
        self._reports_client: ReportsClient | None = None
        self._teams_client: TeamsClient | None = None
        self._tools_client: ToolsClient | None = None
        self._prompts_client: PromptsClient | None = None
        self._regression_client: RegressionClient | None = None
        self._optimizer_client: OptimizerClient | None = None
        self._rag_client: RagClient | None = None
        self._chat_client: ChatClient | None = None

    @staticmethod
    def _resolve_by_name(name: str) -> tuple[str, str]:
        """Look up a team by case-insensitive name match."""
        target = (name or "").strip().lower()
        if not target:
            raise ValueError("Team name must be a non-empty string.")
        session = get_session()
        try:
            for t in session.query(Team).all():
                if (t.name or "").strip().lower() == target:
                    return t.id, t.name
            available = ", ".join(sorted(repr(t.name) for t in session.query(Team).all()))
            raise LookupError(
                f"Team {name!r} not found in the hub. "
                f"Available teams: {available or '(none)'}"
            )
        finally:
            session.close()

    @staticmethod
    def _lookup_name(team_id: str) -> str | None:
        session = get_session()
        try:
            t = session.query(Team).filter(Team.id == team_id).one_or_none()
            return t.name if t else None
        finally:
            session.close()

    # ── Sub-clients (lazy) ────────────────────────────────────────────

    @property
    def trace(self) -> ObservabilityClient:
        if self._trace_client is None:
            self._trace_client = ObservabilityClient(self.team_id)
        return self._trace_client

    @property
    def eval(self) -> EvalClient:
        if self._eval_client is None:
            self._eval_client = EvalClient(self.team_id)
        return self._eval_client

    @property
    def reports(self) -> ReportsClient:
        if self._reports_client is None:
            self._reports_client = ReportsClient(self.team_id)
        return self._reports_client

    @property
    def teams(self) -> TeamsClient:
        if self._teams_client is None:
            self._teams_client = TeamsClient(self.team_id)
        return self._teams_client

    @property
    def tools(self) -> ToolsClient:
        if self._tools_client is None:
            self._tools_client = ToolsClient()
        return self._tools_client

    @property
    def prompts(self) -> PromptsClient:
        if self._prompts_client is None:
            self._prompts_client = PromptsClient(self.team_id)
        return self._prompts_client

    @property
    def regression(self) -> RegressionClient:
        if self._regression_client is None:
            self._regression_client = RegressionClient(self.team_id)
        return self._regression_client

    @property
    def optimizer(self) -> OptimizerClient:
        if self._optimizer_client is None:
            self._optimizer_client = OptimizerClient(self.team_id)
        return self._optimizer_client

    @property
    def rag(self) -> RagClient:
        if self._rag_client is None:
            self._rag_client = RagClient(self.team_id)
        return self._rag_client

    @property
    def chat(self) -> ChatClient:
        if self._chat_client is None:
            self._chat_client = ChatClient(self.team_id, backend_url=self._backend_url)
        return self._chat_client

    # ── Factories ────────────────────────────────────────────────────

    @classmethod
    def auto_create(
        cls,
        *,
        team_id: str,
        name: str,
        description: str = "",
        decision_strategy: str = "router_decides",
        config_json: dict | None = None,
        agents: list[dict] | None = None,
        backend_url: str | None = None,
    ) -> "HubClient":
        """Create or upsert a team + agents in one call, then return a bound client.

        Equivalent to::

            create_team_static(team_id=..., agents=[...])
            HubClient(team_id=...)

        — but in a single call. Used by ``sdk_finance_demo.py`` to stand
        up the Finance Team without an HTTP round-trip.
        """
        try:
            init_db()
        except Exception:
            pass
        create_team_static(
            team_id=team_id,
            name=name,
            description=description,
            decision_strategy=decision_strategy,
            config_json=config_json,
            agents=agents,
        )
        return cls(team_id=team_id, backend_url=backend_url)

    def __repr__(self) -> str:
        return f"HubClient(team={self.team_name!r}, team_id={self.team_id!r})"
