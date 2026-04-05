"""
Evaluation scenarios — from simple single-step to complex multi-step trajectories.

Organized by complexity:
- Quick: single tool call, tests routing accuracy
- Medium: 2-3 tool calls, tests tool selection
- Complex: 4+ steps, tests planning, multi-tool workflows, and recovery
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
    complexity: str = "quick"
    description: str = ""


# ── Quick Scenarios (1 step, routing accuracy) ──

QUICK_SCENARIOS = [
    EvalScenario(
        name="read_single_file",
        prompt="Read the main.py file and tell me what it does",
        expected_agent="coder", expected_tools=["read_file"],
        success_keywords=["main", "chat", "eval"],
        expected_min_steps=1, complexity="quick",
    ),
    EvalScenario(
        name="run_tests",
        prompt="Run the test suite and tell me the results",
        expected_agent="tester", expected_tools=["run_tests"],
        success_keywords=["test", "pass", "fail"],
        expected_min_steps=1, complexity="quick",
    ),
    EvalScenario(
        name="check_git_status",
        prompt="What is the current git status? Show me changed files.",
        expected_agent="devops", expected_tools=["git_status"],
        success_keywords=["modified", "clean", "branch"],
        expected_min_steps=1, complexity="quick",
    ),
    EvalScenario(
        name="list_project_files",
        prompt="List all Python files in the src/mcp_servers directory",
        expected_agent="coder", expected_tools=["list_directory", "find_files"],
        success_keywords=["filesystem_server", "shell_server", ".py"],
        expected_min_steps=1, complexity="quick",
    ),
    EvalScenario(
        name="handle_missing_file",
        prompt="Read the file src/nonexistent_module.py",
        expected_agent="coder", expected_tools=["read_file"],
        success_keywords=["not found", "error", "doesn't exist", "does not exist"],
        expected_min_steps=1, complexity="quick",
    ),
]

# ── Medium Scenarios (2-3 steps, tool accuracy) ──

MEDIUM_SCENARIOS = [
    EvalScenario(
        name="search_and_read",
        prompt="Find where AGENT_DEFINITIONS is defined, then read that file and explain what each entry contains",
        expected_agent="coder", expected_tools=["search_files", "read_file"],
        success_keywords=["AGENT_DEFINITIONS", "prompt", "tools", "role"],
        expected_min_steps=2, complexity="medium",
    ),
    EvalScenario(
        name="explore_then_summarize",
        prompt="List the contents of the src/ directory, then read the orchestrator.py and explain how routing works",
        expected_agent="coder", expected_tools=["list_directory", "read_file"],
        success_keywords=["router", "agent", "route"],
        expected_min_steps=2, complexity="medium",
    ),
    EvalScenario(
        name="run_specific_test",
        prompt="Run only the test_evaluation.py tests and report how many passed",
        expected_agent="tester", expected_tools=["run_tests", "run_command"],
        success_keywords=["passed", "test"],
        expected_min_steps=1, complexity="medium",
    ),
    EvalScenario(
        name="search_web_and_summarize",
        prompt="Search for how to use Python's dataclasses module and give me a quick example",
        expected_agent="researcher", expected_tools=["web_search"],
        success_keywords=["dataclass", "python"],
        expected_min_steps=1, complexity="medium",
    ),
    EvalScenario(
        name="git_history_analysis",
        prompt="Show me the last 5 commits and explain what changes were made recently",
        expected_agent="devops", expected_tools=["git_log"],
        success_keywords=["commit"],
        expected_min_steps=1, complexity="medium",
    ),
]

# ── Complex Scenarios (4+ steps, long trajectory) ──

COMPLEX_SCENARIOS = [
    EvalScenario(
        name="full_codebase_audit",
        prompt="Explore this project's directory structure, then read the README and 2 key source files. Give me a comprehensive architecture overview.",
        expected_agent="coder",
        expected_tools=["list_directory", "read_file"],
        success_keywords=["agent", "mcp", "orchestrator", "tool"],
        expected_min_steps=4, complexity="complex",
    ),
    EvalScenario(
        name="find_and_fix_pattern",
        prompt="Search for all files that import 'get_llm', read each one, and tell me which ones could benefit from caching the LLM instance",
        expected_agent="coder",
        expected_tools=["search_files", "read_file"],
        success_keywords=["get_llm", "import"],
        expected_min_steps=3, complexity="complex",
    ),
    EvalScenario(
        name="test_debug_cycle",
        prompt="Run all the tests. If any fail, find the test file and read the failing test to understand what it expects.",
        expected_agent="tester",
        expected_tools=["run_tests", "run_command"],
        success_keywords=["test", "pass"],
        expected_min_steps=2, complexity="complex",
    ),
    EvalScenario(
        name="write_test_and_verify",
        prompt="Create a simple Python file called 'test_hello.py' that has a test function which asserts 1+1==2, then run it with pytest",
        expected_agent="coder",
        expected_tools=["write_file"],
        success_keywords=["test", "assert", "hello"],
        expected_min_steps=2, complexity="complex",
    ),
    EvalScenario(
        name="multi_file_understanding",
        prompt="Read the config.py file AND the llm/client.py file, then explain how the LLM is configured and what environment variables are needed",
        expected_agent="coder",
        expected_tools=["read_file"],
        success_keywords=["POE_API_KEY", "LLM", "config", "model"],
        expected_min_steps=2, complexity="complex",
    ),
    EvalScenario(
        name="git_diff_review",
        prompt="Check git status, show me the diff of any changed files, and review the changes for potential issues",
        expected_agent="coder",
        expected_tools=["git_status", "git_diff"],
        success_keywords=["diff", "change", "modified"],
        expected_min_steps=2, complexity="complex",
    ),
]

ALL_SCENARIOS = QUICK_SCENARIOS + MEDIUM_SCENARIOS + COMPLEX_SCENARIOS
SCENARIOS = ALL_SCENARIOS  # default: all scenarios
FAST_SCENARIOS = QUICK_SCENARIOS  # for quick API eval


# ── Example Test Prompts for Manual Testing ──
# Use these in the Chat UI to test long trajectories:

EXAMPLE_PROMPTS = [
    # Simple (should route to single agent, 1-2 tool calls)
    "Read main.py and explain the architecture",
    "Run the tests and tell me the results",
    "What files are in the src/evaluation directory?",

    # Medium (2-3 tool calls, some reasoning)
    "Find where decision strategies are defined and list all supported strategies",
    "Check git status and show me the diff of any changed files",
    "Search for all TODO comments across the codebase",

    # Complex (4+ steps, multi-tool workflows)
    "Explore the full project structure, read the 3 most important files, and write a summary",
    "Run tests, check if any fail, and if they do read the test file to diagnose the issue",
    "Create a new Python file that implements a Fibonacci function, write a test for it, then run the test",
    "Search for all files that use the 'config' module, read each one, and list all configuration keys used",
    "Check git log for recent changes, read the most recently changed file, and review it for code quality",
]
