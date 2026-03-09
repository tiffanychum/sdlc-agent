"""
Skill Engine — Injects behavioral instructions into agent prompts at runtime.

Following the Claude Code principle: "MCP gives power. Skills control power."
- MCP Tools define WHAT the agent can do (tools, APIs, data)
- Skills define HOW the agent should do it (process, policy, validation)

Skills are model-invoked, instruction-only, lightweight, and safe.
They never add tools — only behavioral guidance injected into the system prompt.
"""

import re
from src.db.database import get_session
from src.db.models import Agent, AgentSkillMapping, Skill


def get_agent_skills(agent_id: str) -> list[Skill]:
    """Fetch all skills assigned to an agent."""
    session = get_session()
    try:
        mappings = session.query(AgentSkillMapping).filter_by(agent_id=agent_id).all()
        skill_ids = [m.skill_id for m in mappings]
        if not skill_ids:
            return []
        return session.query(Skill).filter(Skill.id.in_(skill_ids)).all()
    finally:
        session.close()


def inject_skills(base_prompt: str, skills: list[Skill]) -> str:
    """
    Inject skill instructions into an agent's system prompt.
    Skills are appended as a structured section at the end.
    """
    if not skills:
        return base_prompt

    skill_blocks = []
    for skill in skills:
        block = f"### {skill.name}\n{skill.instructions}"
        skill_blocks.append(block)

    skills_section = "\n\n## Active Skills\n\n" + "\n\n".join(skill_blocks)
    return base_prompt + skills_section


def match_skills_for_prompt(user_prompt: str, available_skills: list[Skill]) -> list[Skill]:
    """
    Auto-select skills whose trigger_pattern matches the user prompt.
    Skills with empty trigger_pattern are always included.
    """
    matched = []
    prompt_lower = user_prompt.lower()

    for skill in available_skills:
        if not skill.trigger_pattern:
            matched.append(skill)
        elif skill.trigger_pattern.lower() in prompt_lower:
            matched.append(skill)

    return matched


def build_agent_prompt(agent_id: str, base_prompt: str, user_prompt: str = "") -> str:
    """
    Build the final system prompt for an agent by:
    1. Fetching assigned skills
    2. Filtering by trigger pattern (if user_prompt provided)
    3. Injecting matching skill instructions
    """
    all_skills = get_agent_skills(agent_id)

    if user_prompt:
        active_skills = match_skills_for_prompt(user_prompt, all_skills)
    else:
        active_skills = all_skills

    return inject_skills(base_prompt, active_skills)
