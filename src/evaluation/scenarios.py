"""
Evaluation scenarios for testing agent performance.

Each scenario includes expected_min_steps for step efficiency scoring.
"""

from dataclasses import dataclass, field


@dataclass
class EvalScenario:
    name: str
    prompt: str
    expected_agent: str
    expected_tools: list[str]
    success_keywords: list[str]
    expected_min_steps: int = 1
    description: str = ""


SCENARIOS: list[EvalScenario] = [
    EvalScenario(
        name="route_to_coder_read_file",
        prompt="Read the main.py file and explain what it does",
        expected_agent="coder",
        expected_tools=["read_file"],
        success_keywords=["main", "chat", "eval"],
        expected_min_steps=1,
    ),
    EvalScenario(
        name="route_to_runner_run_tests",
        prompt="Run the test suite and tell me if anything is failing",
        expected_agent="runner",
        expected_tools=["run_tests"],
        success_keywords=["test", "pass", "fail"],
        expected_min_steps=1,
    ),
    EvalScenario(
        name="route_to_researcher_lookup",
        prompt="What's the latest version of LangGraph and how do I install it?",
        expected_agent="researcher",
        expected_tools=["web_search", "fetch_url"],
        success_keywords=["langgraph", "install", "pip"],
        expected_min_steps=1,
    ),
    EvalScenario(
        name="route_to_coder_git_operation",
        prompt="Show me what files have changed since the last commit",
        expected_agent="coder",
        expected_tools=["git_status", "git_diff"],
        success_keywords=["modified", "change", "diff"],
        expected_min_steps=1,
    ),
    EvalScenario(
        name="explore_before_edit",
        prompt="Find where the router prompt is defined and add a new agent type called 'devops'",
        expected_agent="coder",
        expected_tools=["search_files", "read_file", "edit_file"],
        success_keywords=["prompts.py", "devops"],
        expected_min_steps=3,
    ),
    EvalScenario(
        name="multi_tool_research",
        prompt="Search for how to use FastMCP in Python and show me an example",
        expected_agent="researcher",
        expected_tools=["web_search", "fetch_url"],
        success_keywords=["FastMCP", "example", "python"],
        expected_min_steps=2,
    ),
    EvalScenario(
        name="write_and_execute",
        prompt="Write a Python script that prints the fibonacci sequence up to 10 terms, then run it",
        expected_agent="coder",
        expected_tools=["write_file"],
        success_keywords=["fibonacci", "print"],
        expected_min_steps=1,
    ),
    EvalScenario(
        name="handle_missing_file",
        prompt="Read the file src/nonexistent_module.py",
        expected_agent="coder",
        expected_tools=["read_file"],
        success_keywords=["not found", "error", "doesn't exist", "does not exist"],
        expected_min_steps=1,
    ),
    EvalScenario(
        name="handle_failing_command",
        prompt="Run 'python nonexistent_script.py'",
        expected_agent="runner",
        expected_tools=["run_command"],
        success_keywords=["error", "no such file", "not found"],
        expected_min_steps=1,
    ),
    EvalScenario(
        name="understand_project_structure",
        prompt="Give me an overview of this project — what are the main components and how do they connect?",
        expected_agent="coder",
        expected_tools=["list_directory", "read_file"],
        success_keywords=["agent", "mcp", "tool"],
        expected_min_steps=2,
    ),
    EvalScenario(
        name="debug_workflow",
        prompt="Run the tests and if any fail, read the failing test file to understand why",
        expected_agent="runner",
        expected_tools=["run_tests"],
        success_keywords=["test"],
        expected_min_steps=1,
    ),
]
