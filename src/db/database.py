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
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _migrate_tool_mappings(session):
    """Fix legacy tool-group names (e.g. PM/BA agents that had 'planner' instead of 'jira')."""
    from src.db.models import AgentToolMapping

    for agent_id in ("project_manager", "business_analyst"):
        mappings = session.query(AgentToolMapping).filter_by(agent_id=agent_id).all()
        existing_groups = {m.tool_group for m in mappings}
        if "planner" in existing_groups and "jira" not in existing_groups:
            for m in mappings:
                if m.tool_group == "planner":
                    session.delete(m)
            session.add(AgentToolMapping(agent_id=agent_id, tool_group="jira"))
