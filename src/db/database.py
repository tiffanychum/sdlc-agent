"""
Database connection and session management.
Uses SQLite for zero-setup portability.

Agent prompts, descriptions, and tool lists live in src/agents/prompts.py.
To update an agent's prompt: edit that file, then call update_agent_data() (or restart
the server — it calls update_agent_data() automatically on startup).
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.db.models import Base

DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "data", "sdlc_agent.db"))


def get_engine():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    return engine


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    _migrate(engine)
    return engine


def _migrate(engine):
    """Add columns that may be missing in older DBs."""
    from sqlalchemy import text, inspect
    insp = inspect(engine)
    if "agents" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("agents")}
        with engine.connect() as conn:
            if "model" not in cols:
                conn.execute(text("ALTER TABLE agents ADD COLUMN model TEXT DEFAULT ''"))
            if "decision_strategy" not in cols:
                conn.execute(text("ALTER TABLE agents ADD COLUMN decision_strategy TEXT DEFAULT 'react'"))
            conn.commit()

    if "traces" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("traces")}
        with engine.connect() as conn:
            for col, default in [("agent_used", "''"), ("agent_response", "''"),
                                 ("tool_calls_json", "'[]'"), ("eval_scores", "'{}'"),
                                 ("eval_status", "'pending'")]:
                if col not in cols:
                    conn.execute(text(f"ALTER TABLE traces ADD COLUMN {col} TEXT DEFAULT {default}"))
            conn.commit()

    if "regression_results" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("regression_results")}
        with engine.connect() as conn:
            for col, default in [
                ("deepeval_scores", "'{}'"),
                ("eval_reasoning", "'{}'"),
                ("model_used", "''"),
                ("prompt_version", "'v1'"),
                ("expected_strategy", "NULL"),
                ("actual_strategy", "NULL"),
            ]:
                if col not in cols:
                    conn.execute(text(f"ALTER TABLE regression_results ADD COLUMN {col} TEXT DEFAULT {default}"))
            conn.commit()

    if "golden_test_cases" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("golden_test_cases")}
        with engine.connect() as conn:
            for col in ("strategy", "expected_strategy"):
                if col not in cols:
                    conn.execute(text(f"ALTER TABLE golden_test_cases ADD COLUMN {col} TEXT"))
            conn.commit()

    # RAG tables — created by Base.metadata.create_all but guarded here for safety
    tables = set(insp.get_table_names())
    if "rag_configs" in tables:
        cols = {c["name"] for c in insp.get_columns("rag_configs")}
        with engine.connect() as conn:
            for col, default in [
                ("mmr_lambda", "0.5"),
                ("multi_query_n", "3"),
                ("system_prompt", "NULL"),
                ("reranker", "'none'"),
            ]:
                if col not in cols:
                    conn.execute(text(f"ALTER TABLE rag_configs ADD COLUMN {col} TEXT DEFAULT {default}"))
            conn.commit()

    if "rag_queries" in tables:
        cols = {c["name"] for c in insp.get_columns("rag_queries")}
        with engine.connect() as conn:
            for col, default in [
                ("eval_scores", "NULL"),
                ("eval_status", "'pending'"),
                ("eval_error", "NULL"),
                ("trace_id", "NULL"),
            ]:
                if col not in cols:
                    conn.execute(text(f"ALTER TABLE rag_queries ADD COLUMN {col} TEXT DEFAULT {default}"))
            conn.commit()

    if "agents" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("agents")}
        with engine.connect() as conn:
            if "prompt_version" not in cols:
                conn.execute(text("ALTER TABLE agents ADD COLUMN prompt_version TEXT DEFAULT 'v1'"))
            conn.commit()

    # Workflow tables — created by Base.metadata.create_all.  Migration
    # block kept for future column additions (matches the RAG pattern above).
    # No-op on fresh DBs; safe to leave in place.
    if "workflow_definitions" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("workflow_definitions")}
        with engine.connect() as conn:
            if "is_active" not in cols:
                conn.execute(text("ALTER TABLE workflow_definitions ADD COLUMN is_active BOOLEAN DEFAULT 1"))
            conn.commit()

    # Patch 5: team-scope PromptVersionEntry.  Existing rows keep team_id = NULL
    # (= "global"), which is the correct default — the agent-role prompts were
    # never team-specific anyway, and the first orchestrator build after this
    # migration will seed per-team supervisor/router rows on demand.
    if "prompt_version_entries" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("prompt_version_entries")}
        with engine.connect() as conn:
            if "team_id" not in cols:
                conn.execute(text("ALTER TABLE prompt_version_entries ADD COLUMN team_id TEXT"))
                # Backfill existing rows explicitly as NULL (global) — no-op in
                # SQLite but keeps the intent obvious to future readers.
                conn.execute(text("UPDATE prompt_version_entries SET team_id = NULL WHERE team_id IS NULL"))
            conn.commit()


def get_session() -> Session:
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def seed_defaults():
    """Create default team, agents, skills, and assignments if the DB is empty."""
    from src.db.models import Team, Agent, AgentToolMapping, Skill, AgentSkillMapping
    from src.agents.prompts import AGENT_DEFINITIONS, SKILL_DEFINITIONS, SKILL_ASSIGNMENTS

    session = get_session()
    try:
        if session.query(Team).count() > 0:
            return

        team = Team(
            id="default",
            name="Dev Team",
            description=(
                "Full-stack development team with specialized agents for coding, testing, "
                "research, planning, and DevOps. Uses supervisor strategy so complex tasks "
                "are automatically delegated across multiple agents."
            ),
            decision_strategy="supervisor",
        )
        session.add(team)

        for ad in AGENT_DEFINITIONS:
            agent = Agent(
                id=ad["id"],
                team_id="default",
                name=ad["name"],
                role=ad["role"],
                description=ad["description"],
                system_prompt=ad["prompt"],
                decision_strategy=ad["decision_strategy"],
                model=ad.get("model", ""),
            )
            session.add(agent)
            for tg in ad["tools"]:
                session.add(AgentToolMapping(agent_id=ad["id"], tool_group=tg))

        for sd in SKILL_DEFINITIONS:
            session.add(Skill(
                id=sd["id"],
                name=sd["name"],
                description=sd["description"],
                instructions=sd["instructions"],
                trigger_pattern=sd["trigger_pattern"],
            ))
        session.flush()

        for agent_id, skill_ids in SKILL_ASSIGNMENTS:
            for sid in skill_ids:
                session.add(AgentSkillMapping(agent_id=agent_id, skill_id=sid))

        session.commit()
    finally:
        session.close()


def update_agent_data():
    """
    Synchronise the DB with the canonical definitions in src/agents/prompts.py.

    For every agent in AGENT_DEFINITIONS:
      - Insert the agent row if it does not exist yet.
      - Update system_prompt, description, model, and decision_strategy to the latest values.
      - Reconcile tool groups: add missing ones AND remove groups no longer in the definition.

    Deprecated agents (e.g. runner) are soft-deleted (team_id set to None) so existing
    trace records remain intact.

    Also runs _migrate_tool_mappings() to fix any legacy tool-group names.
    This function is idempotent and safe to call on every startup.
    """
    from src.db.models import Agent, AgentToolMapping
    from src.agents.prompts import AGENT_DEFINITIONS

    session = get_session()
    try:
        canonical_ids = {ad["id"] for ad in AGENT_DEFINITIONS}

        # Hard-delete deprecated agents that are no longer in AGENT_DEFINITIONS.
        # AgentToolMapping and AgentSkillMapping cascade via the relationship.
        # traces.agent_used is a plain string column (not a FK), so no orphan risk.
        for agent in session.query(Agent).filter(Agent.team_id == "default").all():
            if agent.id not in canonical_ids:
                session.delete(agent)

        for ad in AGENT_DEFINITIONS:
            agent = session.query(Agent).filter_by(id=ad["id"]).first()
            if agent is None:
                agent = Agent(
                    id=ad["id"],
                    team_id="default",
                    name=ad["name"],
                    role=ad["role"],
                    description=ad["description"],
                    system_prompt=ad["prompt"],
                    decision_strategy=ad["decision_strategy"],
                    model=ad.get("model", ""),
                )
                session.add(agent)
            else:
                # Re-attach if previously soft-deleted
                agent.team_id = "default"
                agent.name = ad["name"]
                agent.role = ad["role"]
                agent.description = ad["description"]
                agent.system_prompt = ad["prompt"]
                agent.decision_strategy = ad["decision_strategy"]
                agent.model = ad.get("model", "")

            canonical_tools = set(ad["tools"])
            existing_mappings = session.query(AgentToolMapping).filter_by(agent_id=ad["id"]).all()
            existing_groups = {m.tool_group for m in existing_mappings}

            for tg in canonical_tools - existing_groups:
                session.add(AgentToolMapping(agent_id=ad["id"], tool_group=tg))
            for m in existing_mappings:
                if m.tool_group not in canonical_tools:
                    session.delete(m)

        _migrate_tool_mappings(session)
        _seed_skill_assignments(session)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # Seed PromptRegistry with v1 and v2 prompts (idempotent)
    try:
        from src.prompts.registry import get_registry
        from src.agents.prompts import AGENT_DEFINITIONS, SDLC_2_0_AGENT_DEFINITIONS
        reg = get_registry()
        inserted_v1 = reg.seed_from_definitions(AGENT_DEFINITIONS)
        cot_roles = ["coder", "planner", "qa", "researcher"]
        inserted_v2 = reg.seed_cot_v2(cot_roles)
        # Drift detection: register a new version for any role whose code prompt
        # has changed since the last registered version (e.g. after tester removal,
        # the Coder/DevOps prompts changed but the DB still held the stale v1).
        synced = reg.sync_from_definitions(AGENT_DEFINITIONS)
        # sdlc_2_0 team roles also get versioning so they appear in Studio + A/B.
        inserted_v1_v2 = reg.seed_from_definitions(SDLC_2_0_AGENT_DEFINITIONS)
        synced_v2 = reg.sync_from_definitions(SDLC_2_0_AGENT_DEFINITIONS)
        if inserted_v1 or inserted_v2 or synced or inserted_v1_v2 or synced_v2:
            pass  # silently seeded / synced
    except Exception:
        pass  # best-effort; doesn't block startup

    # Seed the sdlc_2_0 team alongside the default dev team.
    try:
        seed_sdlc_2_0_team()
    except Exception:
        pass  # best-effort; doesn't block startup


def seed_sdlc_2_0_team():
    """
    Idempotently create the `sdlc_2_0` team (Cursor/OpenCode-style simplified roster).

    The team is a separate row in the `teams` table with its own Agents (builder,
    planner_v2). It reuses the existing skill library and golden dataset — the only
    thing that differs is the agent roster and the supervisor routing logic (handled
    in orchestrator._derive_required_steps based on team_id).

    Safe to call on every startup: it upserts rows rather than wiping existing data,
    so user edits in Studio (per-agent prompt_version) survive restarts.
    """
    from src.db.models import Team, Agent, AgentToolMapping, AgentSkillMapping, Skill
    from src.agents.prompts import (
        SDLC_2_0_AGENT_DEFINITIONS,
        SDLC_2_0_SKILL_ASSIGNMENTS,
        SKILL_DEFINITIONS,
    )

    team_id = "sdlc_2_0"
    session = get_session()
    try:
        team = session.query(Team).filter_by(id=team_id).first()
        if team is None:
            team = Team(
                id=team_id,
                name="SDLC 2.0",
                description=(
                    "Simplified Cursor / OpenCode-style team with TWO agents: "
                    "a Builder that owns end-to-end execution (code, tests, git, jira, "
                    "research) and a Planner-2 invoked only for complex multi-concern "
                    "tasks. Designed for direct A/B comparison against the Dev Team "
                    "on the same golden dataset."
                ),
                decision_strategy="supervisor",
            )
            session.add(team)

        canonical_ids = {ad["id"] for ad in SDLC_2_0_AGENT_DEFINITIONS}

        # Detach any sdlc_2_0 agents that are no longer in the canonical list
        # (defensive — keeps the team roster in sync if we rename roles later).
        for agent in session.query(Agent).filter(Agent.team_id == team_id).all():
            if agent.id not in canonical_ids:
                agent.team_id = None  # soft-delete

        for ad in SDLC_2_0_AGENT_DEFINITIONS:
            agent = session.query(Agent).filter_by(id=ad["id"]).first()
            if agent is None:
                agent = Agent(
                    id=ad["id"],
                    team_id=team_id,
                    name=ad["name"],
                    role=ad["role"],
                    description=ad["description"],
                    system_prompt=ad["prompt"],
                    decision_strategy=ad["decision_strategy"],
                    model=ad.get("model", ""),
                )
                session.add(agent)
            else:
                # Reconcile — user-edited prompt_version / model survives because
                # those columns are NOT overwritten here; everything else tracks
                # the canonical definition.
                agent.team_id = team_id
                agent.name = ad["name"]
                agent.role = ad["role"]
                agent.description = ad["description"]
                agent.system_prompt = ad["prompt"]
                agent.decision_strategy = ad["decision_strategy"]
                # Only set model if not yet set (preserves user overrides in Studio).
                if not (getattr(agent, "model", None) or "").strip():
                    agent.model = ad.get("model", "")

            canonical_tools = set(ad["tools"])
            existing = session.query(AgentToolMapping).filter_by(agent_id=ad["id"]).all()
            existing_groups = {m.tool_group for m in existing}
            for tg in canonical_tools - existing_groups:
                session.add(AgentToolMapping(agent_id=ad["id"], tool_group=tg))
            for m in existing:
                if m.tool_group not in canonical_tools:
                    session.delete(m)

        # Upsert any SKILL_DEFINITIONS rows that the initial seed missed
        # (e.g. skills added to code after the DB was first seeded). Without
        # this, SDLC_2_0_SKILL_ASSIGNMENTS referencing newer skills silently
        # drops them.
        existing_skill_ids = {s.id for s in session.query(Skill).all()}
        for sd in SKILL_DEFINITIONS:
            if sd["id"] not in existing_skill_ids:
                session.add(Skill(
                    id=sd["id"],
                    name=sd["name"],
                    description=sd["description"],
                    instructions=sd["instructions"],
                    trigger_pattern=sd["trigger_pattern"],
                ))

        session.flush()

        for agent_id, skill_ids in SDLC_2_0_SKILL_ASSIGNMENTS:
            if session.query(Agent).filter_by(id=agent_id).first() is None:
                continue
            existing = {
                m.skill_id
                for m in session.query(AgentSkillMapping).filter_by(agent_id=agent_id).all()
            }
            for sid in skill_ids:
                if session.query(Skill).filter_by(id=sid).first() and sid not in existing:
                    session.add(AgentSkillMapping(agent_id=agent_id, skill_id=sid))

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _seed_skill_assignments(session):
    """Idempotently ensure each agent in SKILL_ASSIGNMENTS has the expected skills.

    Unlike seed_defaults() which only runs once, this runs on every startup so new
    agents (e.g. qa) get their skills even when the DB was initialized before they existed.
    """
    from src.db.models import AgentSkillMapping, Agent, Skill
    from src.agents.prompts import SKILL_ASSIGNMENTS

    for agent_id, skill_ids in SKILL_ASSIGNMENTS:
        agent = session.query(Agent).filter_by(id=agent_id).first()
        if agent is None:
            continue
        existing = {m.skill_id for m in session.query(AgentSkillMapping).filter_by(agent_id=agent_id).all()}
        for sid in skill_ids:
            skill = session.query(Skill).filter_by(id=sid).first()
            if skill and sid not in existing:
                session.add(AgentSkillMapping(agent_id=agent_id, skill_id=sid))


def _migrate_tool_mappings(session):
    """Fix legacy tool-group names and enforce current canonical tool group assignments."""
    from src.db.models import AgentToolMapping

    # PM: rename legacy 'planner' tool group → 'jira'
    for agent_id in ("project_manager",):
        mappings = session.query(AgentToolMapping).filter_by(agent_id=agent_id).all()
        existing_groups = {m.tool_group for m in mappings}
        if "planner" in existing_groups and "jira" not in existing_groups:
            for m in mappings:
                if m.tool_group == "planner":
                    session.delete(m)
            session.add(AgentToolMapping(agent_id=agent_id, tool_group="jira"))

    # Planner: downgrade 'filesystem' → 'filesystem_read' (read-only enforcement)
    planner_mappings = session.query(AgentToolMapping).filter_by(agent_id="planner").all()
    planner_groups = {m.tool_group for m in planner_mappings}
    if "filesystem" in planner_groups and "filesystem_read" not in planner_groups:
        for m in planner_mappings:
            if m.tool_group == "filesystem":
                session.delete(m)
        session.add(AgentToolMapping(agent_id="planner", tool_group="filesystem_read"))
