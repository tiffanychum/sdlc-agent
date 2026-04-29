"""
TeamsClient — programmatic CRUD for the ``teams`` and ``agents`` tables.

Lets SDK callers stand up an entire agent team (Finance Team, etc.) the
same way they would in Studio: pick a strategy, declare agent roles,
attach tool groups and prompts, all without HTTP round-trips.

Methods:
    list                — every team in the hub
    get                 — full payload for the team this client is bound to
    update_strategy     — set ``decision_strategy``
    update_config       — merge or replace ``config_json``
    upsert_agent        — create or update an agent on this team
    list_agents         — agents currently attached to the team
    create_team_static  — module-level factory used during ``auto_create``
"""

from __future__ import annotations

from typing import Any, Iterable

from sqlalchemy.orm.attributes import flag_modified

from src.db.database import get_session
from src.db.models import Agent, AgentToolMapping, Team


# ── Module-level factory (used by HubClient.auto_create) ─────────────────────


def create_team_static(
    *,
    team_id: str,
    name: str,
    description: str = "",
    decision_strategy: str = "router_decides",
    config_json: dict | None = None,
    agents: Iterable[dict] | None = None,
) -> str:
    """Create or upsert a team row plus its agents in one call.

    ``agents`` is a list of dicts shaped like::

        {
            "id":   "market_analyst",        # required
            "name": "Market Analyst",        # required
            "role": "market_analyst",        # defaults to id when omitted
            "description": "…",
            "system_prompt": "…",
            "model": "claude-sonnet-4.6",
            "tool_groups": ["finance_market"],
            "decision_strategy": "react",     # ReAct loop is the safe default
            "prompt_version": "v1",
        }

    Returns the team id.
    """
    session = get_session()
    try:
        team = session.query(Team).filter_by(id=team_id).first()
        if team is None:
            team = Team(
                id=team_id,
                name=name,
                description=description,
                decision_strategy=decision_strategy,
                config_json=dict(config_json or {}),
            )
            session.add(team)
        else:
            team.name = name
            team.description = description or team.description
            team.decision_strategy = decision_strategy
            if config_json is not None:
                team.config_json = dict(config_json)
                flag_modified(team, "config_json")

        if agents:
            for ad in agents:
                _upsert_agent(session, team_id=team_id, agent_def=ad)

        session.commit()
        return team_id
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Internal helpers ─────────────────────────────────────────────────────────


def _upsert_agent(session, *, team_id: str, agent_def: dict) -> Agent:
    aid = agent_def["id"]
    role = agent_def.get("role") or aid
    name = agent_def["name"]
    desc = agent_def.get("description", "")
    prompt = agent_def.get("system_prompt", "")
    model = agent_def.get("model", "")
    decision_strategy = agent_def.get("decision_strategy", "react")
    prompt_version = agent_def.get("prompt_version", "v1")
    tool_groups = list(agent_def.get("tool_groups") or [])

    agent = session.query(Agent).filter_by(id=aid).one_or_none()
    if agent is None:
        agent = Agent(
            id=aid, team_id=team_id, name=name, role=role, description=desc,
            system_prompt=prompt, model=model, decision_strategy=decision_strategy,
            prompt_version=prompt_version,
        )
        session.add(agent)
    else:
        agent.team_id = team_id
        agent.name = name
        agent.role = role
        agent.description = desc
        agent.system_prompt = prompt
        agent.model = model
        agent.decision_strategy = decision_strategy
        agent.prompt_version = prompt_version

    # Reconcile tool group mappings (add new, drop missing).
    existing = session.query(AgentToolMapping).filter_by(agent_id=aid).all()
    existing_groups = {m.tool_group for m in existing}
    target = set(tool_groups)
    for g in target - existing_groups:
        session.add(AgentToolMapping(agent_id=aid, tool_group=g))
    for m in existing:
        if m.tool_group not in target:
            session.delete(m)
    return agent


# ── Per-hub team client ──────────────────────────────────────────────────────


class TeamsClient:
    """Operations scoped to ``hub.team_id`` plus a global ``list``."""

    def __init__(self, team_id: str) -> None:
        self.team_id = team_id

    # ── Reads ─────────────────────────────────────────────────────────

    def list(self) -> list[dict[str, Any]]:
        """Return all teams in the hub (for cross-team lookup)."""
        session = get_session()
        try:
            return [
                self._team_to_dict(t)
                for t in session.query(Team).order_by(Team.created_at.asc()).all()
            ]
        finally:
            session.close()

    def get(self) -> dict[str, Any] | None:
        session = get_session()
        try:
            t = session.query(Team).filter_by(id=self.team_id).one_or_none()
            if t is None:
                return None
            payload = self._team_to_dict(t)
            payload["agents"] = self._list_agents(session)
            return payload
        finally:
            session.close()

    def list_agents(self) -> list[dict[str, Any]]:
        session = get_session()
        try:
            return self._list_agents(session)
        finally:
            session.close()

    # ── Writes ────────────────────────────────────────────────────────

    def update_strategy(self, decision_strategy: str) -> None:
        from src.orchestrator import VALID_STRATEGIES
        if decision_strategy not in VALID_STRATEGIES and decision_strategy != "auto":
            raise ValueError(
                f"Unknown decision_strategy {decision_strategy!r}. "
                f"Valid: {sorted(VALID_STRATEGIES) + ['auto']}"
            )
        session = get_session()
        try:
            t = session.query(Team).filter_by(id=self.team_id).one_or_none()
            if t is None:
                raise LookupError(f"Team {self.team_id!r} not found")
            t.decision_strategy = decision_strategy
            session.commit()
        finally:
            session.close()

    def update_config(self, patch: dict, *, replace: bool = False) -> dict:
        """Merge ``patch`` into ``team.config_json`` (or replace entirely).

        Returns the resulting config dict.
        """
        session = get_session()
        try:
            t = session.query(Team).filter_by(id=self.team_id).one_or_none()
            if t is None:
                raise LookupError(f"Team {self.team_id!r} not found")
            current = dict(t.config_json or {})
            if replace:
                merged = dict(patch)
            else:
                merged = current
                merged.update(patch)
            t.config_json = merged
            flag_modified(t, "config_json")
            session.commit()
            return merged
        finally:
            session.close()

    def upsert_agent(self, agent_def: dict) -> dict[str, Any]:
        """Create or update an agent on this team. Returns the agent dict."""
        session = get_session()
        try:
            agent = _upsert_agent(session, team_id=self.team_id, agent_def=agent_def)
            session.commit()
            session.refresh(agent)
            return self._agent_to_dict(agent, session)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_agents(self, agent_defs: Iterable[dict]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for d in agent_defs:
            out.append(self.upsert_agent(d))
        return out

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _team_to_dict(t: Team) -> dict[str, Any]:
        return {
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "decision_strategy": t.decision_strategy,
            "config_json": dict(t.config_json or {}),
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }

    def _list_agents(self, session) -> list[dict[str, Any]]:
        agents = (
            session.query(Agent)
            .filter_by(team_id=self.team_id)
            .order_by(Agent.id.asc())
            .all()
        )
        return [self._agent_to_dict(a, session) for a in agents]

    @staticmethod
    def _agent_to_dict(a: Agent, session) -> dict[str, Any]:
        groups = [
            m.tool_group
            for m in session.query(AgentToolMapping).filter_by(agent_id=a.id).all()
        ]
        return {
            "id": a.id,
            "name": a.name,
            "role": a.role,
            "description": a.description,
            "system_prompt": a.system_prompt,
            "model": a.model,
            "decision_strategy": a.decision_strategy,
            "prompt_version": a.prompt_version,
            "tool_groups": groups,
        }
