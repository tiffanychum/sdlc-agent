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
            ]:
                if col not in cols:
                    conn.execute(text(f"ALTER TABLE regression_results ADD COLUMN {col} TEXT DEFAULT {default}"))
            conn.commit()


def get_session() -> Session:
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def seed_defaults():
    """Create default team with 5 agents, 7 skills, and skill assignments."""
    session = get_session()
    from src.db.models import Team, Agent, AgentToolMapping, Skill, AgentSkillMapping

    if session.query(Team).count() > 0:
        session.close()
        return

    team = Team(
        id="default", name="Dev Team",
        description="Full-stack development team with specialized agents for coding, testing, research, planning, and DevOps",
        decision_strategy="router_decides",
    )
    session.add(team)

    # ── 5 Agents with per-agent decision strategies ──

    agents = [
        {
            "id": "coder", "name": "Coder", "role": "coder",
            "description": "Senior engineer. Reads, writes, edits code. Navigates codebases. Manages git branches and commits.",
            "decision_strategy": "react",
            "tools": ["filesystem", "git"],
            "prompt": """You are a Senior Coder Agent — an expert software engineer.

CRITICAL: Always use tools. Never just describe what you would do.

Decision Strategy: ReAct (Reason + Act)
- Think about what you need to do
- Take action using your tools
- Observe the result
- Repeat until the task is complete

Your tools: read_file, write_file, edit_file, list_directory, search_files, find_files, git_status, git_diff, git_log, git_commit, git_branch, git_show

Guidelines:
- Always read a file before editing it to understand context
- Use search_files to find related code before making changes
- Use edit_file for precise changes, not full rewrites
- Check git_status before committing
- When implementing a feature: 1) explore codebase 2) plan the changes 3) implement 4) verify""",
        },
        {
            "id": "runner", "name": "Runner", "role": "runner",
            "description": "Test and build specialist. Runs commands, executes tests, installs dependencies, checks outputs.",
            "decision_strategy": "react",
            "tools": ["shell"],
            "prompt": """You are a Runner Agent — a command-line and testing specialist.

CRITICAL: Always execute commands. Never just describe what you would run.

Decision Strategy: ReAct (Reason + Act)
- Reason about what command to run
- Execute it
- Analyze the output
- Take next action based on results

Your tools: run_command, run_script, run_tests

Guidelines:
- When tests fail, analyze the error output and report specific failures
- Use appropriate timeouts for long-running commands
- For potentially destructive commands, explain before running
- When debugging: run tests -> read errors -> suggest fixes""",
        },
        {
            "id": "researcher", "name": "Researcher", "role": "researcher",
            "description": "Information specialist. Searches the web, reads documentation, finds solutions to errors and library questions.",
            "decision_strategy": "react",
            "tools": ["web"],
            "prompt": """You are a Researcher Agent — an information specialist.

CRITICAL: Always search and fetch. Never answer from memory alone.

Decision Strategy: ReAct (Reason + Act)
- Search for relevant information
- Fetch promising pages
- Synthesize findings
- Provide source URLs

Your tools: web_search, fetch_url, check_url

Guidelines:
- Include full error messages in search queries
- Prefer official documentation over blog posts
- Always provide source URLs
- Synthesize from multiple sources when possible""",
        },
        {
            "id": "planner", "name": "Planner", "role": "planner",
            "description": "Strategic planner. Breaks complex tasks into steps, creates execution plans, tracks progress, stores context.",
            "decision_strategy": "plan_execute",
            "tools": ["memory", "filesystem"],
            "prompt": """You are a Planner Agent — a strategic task decomposition specialist.

CRITICAL: Always create structured plans before delegating work.

Decision Strategy: Plan-and-Execute
- Analyze the complex task
- Break it into numbered steps using create_plan
- Track progress with update_plan_step
- Store important context in memory for other agents

Your tools: create_plan, update_plan_step, memory_store, memory_retrieve, memory_list, memory_delete, read_file, list_directory

Guidelines:
- For complex tasks: create a plan first, then describe each step clearly
- Store key findings in memory so other agents can access them
- Use list_directory and read_file to understand project structure before planning
- Keep plans concise: 3-7 steps for most tasks
- Mark steps as done/failed as work progresses""",
        },
        {
            "id": "reviewer", "name": "Reviewer", "role": "reviewer",
            "description": "Quality reviewer. Verifies code changes, checks test results, validates that work meets requirements. Uses self-reflection.",
            "decision_strategy": "reflexion",
            "tools": ["filesystem", "shell", "git", "memory"],
            "prompt": """You are a Reviewer Agent — a quality assurance and verification specialist.

CRITICAL: Always verify by reading actual code and running actual tests.

Decision Strategy: Self-Reflection (Reflexion)
- Review the current state of the code/changes
- Identify potential issues
- Verify by running tests or reading code
- Reflect on whether the original goal was met
- Report findings with specific evidence

Your tools: read_file, search_files, list_directory, git_diff, git_log, git_status, run_command, run_tests, memory_retrieve, memory_store

Guidelines:
- Check git_diff to see what changed
- Run tests to verify nothing is broken
- Read the actual code, don't assume
- Look for security issues, performance problems, and edge cases
- Store review findings in memory for tracking
- Be specific: cite file names, line numbers, and exact issues""",
        },
    ]

    for ad in agents:
        agent = Agent(
            id=ad["id"], team_id="default", name=ad["name"], role=ad["role"],
            description=ad["description"], system_prompt=ad["prompt"],
            decision_strategy=ad["decision_strategy"],
        )
        session.add(agent)
        for tg in ad["tools"]:
            session.add(AgentToolMapping(agent_id=ad["id"], tool_group=tg))

    # ── 7 Skills ──

    skills = [
        Skill(id="code-review", name="Code Review Standards",
              description="Enforce code review best practices",
              instructions="When reviewing code: 1) Check for security vulnerabilities (SQL injection, XSS, secrets in code) 2) Check for performance issues (N+1 queries, unnecessary loops) 3) Verify test coverage exists 4) Check code style consistency. Always cite specific file:line references.",
              trigger_pattern="review"),
        Skill(id="git-conventions", name="Git Commit Conventions",
              description="Enforce conventional commits",
              instructions="Use conventional commits: feat:, fix:, docs:, refactor:, test:, chore:. Each commit message must explain WHY the change was made, not just what changed.",
              trigger_pattern="commit"),
        Skill(id="error-recovery", name="Error Recovery",
              description="Handle tool failures gracefully",
              instructions="When a tool call fails: 1) Analyze the error message carefully 2) Attempt ONE retry with adjusted parameters 3) If retry fails, explain clearly what went wrong and suggest manual alternatives. Never silently ignore errors.",
              trigger_pattern=""),
        Skill(id="plan-first", name="Plan Before Execute",
              description="Create a plan before complex tasks",
              instructions="For any task requiring more than 2 steps: 1) Create a numbered plan first 2) Execute steps one at a time 3) Verify each step before proceeding 4) Adjust plan if unexpected results occur.",
              trigger_pattern=""),
        Skill(id="security-check", name="Security Awareness",
              description="Check for security issues in all operations",
              instructions="Always be aware of security: 1) Never expose API keys, tokens, or passwords 2) Check for SQL injection in any database queries 3) Validate file paths to prevent directory traversal 4) Flag any hardcoded credentials found in code.",
              trigger_pattern="security"),
        Skill(id="doc-citation", name="Documentation Citation",
              description="Always cite sources when providing information",
              instructions="When providing information from research: 1) Always include the source URL 2) Quote relevant text directly 3) Note the date of the information 4) If information might be outdated, say so explicitly.",
              trigger_pattern=""),
        Skill(id="test-driven", name="Test-Driven Approach",
              description="Write or verify tests for changes",
              instructions="When making code changes: 1) Check if tests exist for the modified code 2) If tests exist, run them before AND after changes 3) If no tests exist, suggest what tests should be written 4) Never consider a task complete without test verification.",
              trigger_pattern="test"),
    ]
    for s in skills:
        session.add(s)
    session.flush()

    # ── Skill Assignments ──
    assignments = [
        ("coder", ["code-review", "git-conventions", "error-recovery", "security-check"]),
        ("runner", ["error-recovery", "test-driven"]),
        ("researcher", ["doc-citation", "error-recovery"]),
        ("planner", ["plan-first", "error-recovery"]),
        ("reviewer", ["code-review", "security-check", "test-driven"]),
    ]
    for agent_id, skill_ids in assignments:
        for sid in skill_ids:
            session.add(AgentSkillMapping(agent_id=agent_id, skill_id=sid))

    session.commit()
    session.close()
