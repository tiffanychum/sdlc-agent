"""
Agent definitions and factory functions.

Each agent is a LangGraph ReAct agent configured with specific MCP tools
and a system prompt tailored to its role in the development workflow.
"""

from dataclasses import dataclass
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

from src.llm.client import get_llm
from src.agents.prompts import (
    CODER_AGENT_PROMPT,
    RUNNER_AGENT_PROMPT,
    RESEARCHER_AGENT_PROMPT,
)


@dataclass
class AgentConfig:
    name: str
    role: str
    description: str
    system_prompt: str
    tool_groups: list[str]


AGENT_CONFIGS: dict[str, AgentConfig] = {
    "coder": AgentConfig(
        name="Coder Agent",
        role="coder",
        description="Reads, writes, and manages code. Handles file operations and git.",
        system_prompt=CODER_AGENT_PROMPT,
        tool_groups=["filesystem", "git"],
    ),
    "runner": AgentConfig(
        name="Runner Agent",
        role="runner",
        description="Executes commands, runs tests, builds projects.",
        system_prompt=RUNNER_AGENT_PROMPT,
        tool_groups=["shell"],
    ),
    "researcher": AgentConfig(
        name="Researcher Agent",
        role="researcher",
        description="Searches the web, fetches documentation, researches solutions.",
        system_prompt=RESEARCHER_AGENT_PROMPT,
        tool_groups=["web"],
    ),
}


def build_agent(agent_id: str, tools: list[StructuredTool]):
    """Build a LangGraph ReAct agent from config and tools."""
    cfg = AGENT_CONFIGS[agent_id]
    llm = get_llm()
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=cfg.system_prompt,
    )
