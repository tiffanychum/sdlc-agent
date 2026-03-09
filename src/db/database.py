"""
Database connection and session management.
Uses SQLite for zero-setup portability.
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
        if "model" not in cols:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE agents ADD COLUMN model TEXT DEFAULT ''"))
                conn.commit()


def get_session() -> Session:
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def seed_defaults():
    """Create default team, agents, and skills if DB is empty."""
    session = get_session()
    from src.db.models import Team, Agent, AgentToolMapping, Skill

    if session.query(Team).count() > 0:
        session.close()
        return

    team = Team(id="default", name="Default Team", description="General-purpose coding assistant", decision_strategy="router_decides")
    session.add(team)

    agents_data = [
        {
            "id": "coder", "name": "Coder Agent", "role": "coder",
            "description": "Reads, writes, and manages code. Handles file operations and git.",
            "system_prompt": _default_coder_prompt(),
            "tools": ["filesystem", "git"],
        },
        {
            "id": "runner", "name": "Runner Agent", "role": "runner",
            "description": "Executes commands, runs tests, builds projects.",
            "system_prompt": _default_runner_prompt(),
            "tools": ["shell"],
        },
        {
            "id": "researcher", "name": "Researcher Agent", "role": "researcher",
            "description": "Searches the web, fetches documentation, researches solutions.",
            "system_prompt": _default_researcher_prompt(),
            "tools": ["web"],
        },
    ]

    for ad in agents_data:
        agent = Agent(
            id=ad["id"], team_id="default", name=ad["name"], role=ad["role"],
            description=ad["description"], system_prompt=ad["system_prompt"],
        )
        session.add(agent)
        for tg in ad["tools"]:
            session.add(AgentToolMapping(agent_id=ad["id"], tool_group=tg))

    default_skills = [
        Skill(id="code-review", name="Code Review Standards",
              description="Enforce code review best practices",
              instructions="When reviewing code, always check for: 1) Security vulnerabilities 2) Performance issues 3) Test coverage 4) Code style consistency. Provide specific line references.",
              trigger_pattern="review"),
        Skill(id="git-conventions", name="Git Commit Conventions",
              description="Enforce conventional commits format",
              instructions="When making git commits, always use conventional commits format: feat:, fix:, docs:, refactor:, test:, chore:. Include a brief description of WHY the change was made.",
              trigger_pattern="commit"),
        Skill(id="error-recovery", name="Error Recovery Policy",
              description="Handle tool failures gracefully",
              instructions="When a tool call fails: 1) Analyze the error message 2) Attempt one retry with adjusted parameters 3) If retry fails, explain the failure clearly and suggest manual alternatives.",
              trigger_pattern=""),
    ]
    for s in default_skills:
        session.add(s)

    session.commit()
    session.close()


def _default_coder_prompt() -> str:
    from src.agents.prompts import CODER_AGENT_PROMPT
    return CODER_AGENT_PROMPT

def _default_runner_prompt() -> str:
    from src.agents.prompts import RUNNER_AGENT_PROMPT
    return RUNNER_AGENT_PROMPT

def _default_researcher_prompt() -> str:
    from src.agents.prompts import RESEARCHER_AGENT_PROMPT
    return RESEARCHER_AGENT_PROMPT
