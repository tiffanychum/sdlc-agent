"""
PromptsClient — versioned per-role prompt store.

Wraps ``PromptRegistry`` so SDK callers can:
    * register a new prompt version
    * read the active prompt for any role
    * list version history
    * activate / deactivate specific versions

Versions are scoped to a team when ``team_id`` is supplied to ``register``;
omitting it stores the prompt globally (visible to every team).
"""

from __future__ import annotations

from typing import Any, Optional

from src.prompts.registry import get_registry


class PromptsClient:
    def __init__(self, team_id: str) -> None:
        self.team_id = team_id

    # ── Reads ─────────────────────────────────────────────────────────

    def get(
        self,
        role: str,
        version: str = "latest",
        *,
        team_scoped: bool = True,
    ) -> Optional[str]:
        """Return the prompt text for (role, version).

        With ``team_scoped=True`` (default) the lookup tries the team-scoped
        row first and falls back to the global row when none exists — same
        semantics the orchestrator uses at build time.
        """
        team_id = self.team_id if team_scoped else None
        return get_registry().get_prompt(role, version=version, team_id=team_id)

    def latest_version(self, role: str, *, team_scoped: bool = True) -> str:
        team_id = self.team_id if team_scoped else None
        return get_registry().latest_version(role, team_id=team_id)

    def list_versions(
        self,
        role: str,
        *,
        team_scoped: bool = True,
    ) -> list[dict[str, Any]]:
        team_id = self.team_id if team_scoped else None
        return get_registry().list_versions(role, team_id=team_id)

    def list_roles(self, *, team_scoped: bool = True) -> list[str]:
        team_id = self.team_id if team_scoped else None
        return sorted(get_registry().list_all_roles(team_id=team_id))

    # ── Writes ────────────────────────────────────────────────────────

    def register(
        self,
        role: str,
        prompt_text: str,
        *,
        rationale: str = "",
        parent_version: Optional[str] = None,
        cot_enhanced: bool = False,
        scope: str = "team",
        notes: str = "",
    ) -> str:
        """Register a new prompt version. ``scope`` is ``"team"`` or ``"global"``.

        Returns the auto-incremented version label (e.g. ``"v3"``).
        """
        if scope not in ("team", "global"):
            raise ValueError("scope must be 'team' or 'global'")
        team_id = self.team_id if scope == "team" else None
        return get_registry().register(
            role=role,
            prompt_text=prompt_text,
            rationale=rationale,
            parent_version=parent_version,
            created_by="sdk",
            cot_enhanced=cot_enhanced,
            notes=notes,
            team_id=team_id,
        )

    def deactivate_version(
        self,
        role: str,
        version: str,
        *,
        scope: str = "team",
    ) -> None:
        team_id = self.team_id if scope == "team" else None
        get_registry().deactivate_version(role=role, version=version, team_id=team_id)

    def update_metric_scores(
        self,
        role: str,
        version: str,
        scores: dict,
        *,
        scope: str = "team",
    ) -> None:
        team_id = self.team_id if scope == "team" else None
        get_registry().update_metric_scores(
            role=role, version=version, scores=scores, team_id=team_id,
        )

    # ── Convenience: assign a prompt to an agent ──────────────────────

    def set_agent_version(self, agent_id: str, version: str) -> None:
        """Set ``agents.prompt_version`` so the orchestrator picks up the new prompt."""
        from src.db.database import get_session
        from src.db.models import Agent
        session = get_session()
        try:
            agent = session.query(Agent).filter_by(id=agent_id).one_or_none()
            if agent is None:
                raise LookupError(f"Agent {agent_id!r} not found")
            agent.prompt_version = version
            session.commit()
        finally:
            session.close()
