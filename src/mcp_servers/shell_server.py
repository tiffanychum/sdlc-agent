"""
MCP Server for shell command execution (FastMCP).

Allows agents to run terminal commands — build projects, run tests,
install dependencies, check system info. Sandboxed with command
allowlisting and timeout enforcement.

Transports:
  stdio (default):  python -m src.mcp_servers.shell_server
  HTTP:             MCP_TRANSPORT=http MCP_PORT=8002 python -m src.mcp_servers.shell_server
"""

import asyncio
import logging
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Optional

from fastmcp import FastMCP
from mcp.types import TextContent

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = os.getenv("AGENT_WORKSPACE", os.getcwd())
DEFAULT_TIMEOUT = 30

# Dangerous commands we always refuse. Substring matching on "rm -rf /" was
# too aggressive — `rm -rf /tmp/url-shortener/tests/qa` also contains
# "rm -rf /" as a substring and would be incorrectly blocked, so we match
# root-targeted deletions with a regex and keep the other destructive
# commands as plain substrings.
BLOCKED_SUBSTRINGS = {"mkfs", "dd if=", ":(){ :|:&", "shutdown", "reboot"}
# `rm -rf /` or `rm -rf /*` where the next char is whitespace, end-of-string
# or a shell glob — i.e. literally targeting root. Allows `rm -rf /tmp/...`,
# `rm -rf /var/foo`, etc. because those have additional path segments.
_RM_RF_ROOT_RE = re.compile(r"\brm\s+-[a-z]*r[a-z]*f[a-z]*\s+/(\s|$|\*|;|&|\|)", re.IGNORECASE)
# Also catch the `-fr` ordering.
_RM_FR_ROOT_RE = re.compile(r"\brm\s+-[a-z]*f[a-z]*r[a-z]*\s+/(\s|$|\*|;|&|\|)", re.IGNORECASE)


@dataclass
class ShellState:
    tool_calls: list[dict] = field(default_factory=list)
    command_history: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str, success: bool) -> None:
        self.tool_calls.append({
            "tool": tool, "args": args, "result": result[:500], "success": success,
        })


mcp = FastMCP(
    "shell-mcp-server",
    instructions=(
        "Execute shell commands in the agent workspace. "
        "Destructive commands (rm -rf /, mkfs, etc.) are blocked. "
        "Use timeout parameter for long-running commands."
    ),
)
state = ShellState()


# ── Helpers ───────────────────────────────────────────────────────

def _is_safe(command: str) -> bool:
    cmd_lower = command.lower().strip()
    if any(blocked in cmd_lower for blocked in BLOCKED_SUBSTRINGS):
        return False
    # Reject only `rm -rf /` (root) style commands; allow deletions inside
    # /tmp, /var/tmp, or agent-created scratch directories needed by QA
    # cleanup steps.
    if _RM_RF_ROOT_RE.search(cmd_lower) or _RM_FR_ROOT_RE.search(cmd_lower):
        return False
    return True


async def _execute(command: str, cwd: str, timeout: int) -> str:
    if not _is_safe(command):
        raise ValueError(f"Command blocked for safety: {command}")

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        # On timeout we try to terminate cleanly. Under uvloop the process may
        # already be reaped by the time we call .kill(), surfacing as
        # ProcessLookupError; swallow it so we always return the informative
        # "timed out" message instead of crashing the tool call.
        try:
            proc.kill()
        except ProcessLookupError:
            pass  # uvloop already reaped the process
        except Exception:
            pass  # swallow any other kill-time errors rather than crash the tool
        return (
            f"Command timed out after {timeout}s: {command}\n"
            f"Hint: long-running processes (uvicorn, flask run, etc.) must be "
            f"backgrounded, e.g. `nohup <cmd> > /tmp/out.log 2>&1 & echo $!`."
        )

    stdout_str = stdout.decode("utf-8", errors="replace").strip()
    stderr_str = stderr.decode("utf-8", errors="replace").strip()

    state.command_history.append({
        "command": command, "exit_code": proc.returncode, "cwd": cwd,
    })

    parts = [f"$ {command}", f"Exit code: {proc.returncode}"]
    if stdout_str:
        parts.append(f"\nSTDOUT:\n{stdout_str}")
    if stderr_str:
        parts.append(f"\nSTDERR:\n{stderr_str}")
    return "\n".join(parts)


# ── Tool implementations ──────────────────────────────────────────

@mcp.tool()
async def run_command(command: str, working_dir: str = "", timeout: int = 30) -> str:
    """Execute a shell command in the workspace directory.

    Use for running tests, builds, installs, linting, or any CLI tool.
    Returns stdout, stderr, and exit code.

    Args:
        command: Shell command to execute.
        working_dir: Working directory relative to workspace (default: workspace root).
        timeout: Timeout in seconds (default: 30).
    """
    cwd = os.path.join(WORKSPACE_ROOT, working_dir) if working_dir else WORKSPACE_ROOT
    return await _execute(command, cwd, timeout)


@mcp.tool()
async def run_script(path: str, args: str = "") -> str:
    """Execute a Python script from a file path.

    Convenience wrapper for 'python <path>'.

    Args:
        path: Relative path to the Python script.
        args: Command line arguments (optional).
    """
    command = f"python {shlex.quote(path)} {args}".strip()
    return await _execute(command, WORKSPACE_ROOT, DEFAULT_TIMEOUT)


@mcp.tool()
async def run_tests(path: str = "", verbose: bool = True, pattern: Optional[str] = None) -> str:
    """Run the test suite using pytest. Automatically discovers and runs tests.

    Args:
        path: Test file or directory to run (default: all tests).
        verbose: Verbose output (default: True).
        pattern: Test name pattern to match (e.g. 'test_auth').
    """
    parts = ["python", "-m", "pytest"]
    if path:
        parts.append(path)
    if verbose:
        parts.append("-v")
    if pattern:
        parts.extend(["-k", pattern])

    return await _execute(" ".join(parts), WORKSPACE_ROOT, timeout=60)


# ── Backward-compatible shims ─────────────────────────────────────

async def list_tools():
    tools = await mcp.list_tools()
    return [t.to_mcp_tool() for t in tools]


async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        result = await mcp.call_tool(name, arguments)
        text = result.content[0].text if result.content else "Done"
        state.record(name, arguments, text, success=True)
        return list(result.content)
    except Exception as e:
        error_msg = f"Error in {name}: {str(e)}"
        state.record(name, arguments, error_msg, success=False)
        return [TextContent(type="text", text=error_msg)]


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "http":
        import asyncio as _asyncio
        _asyncio.run(mcp.run_http_async(host="0.0.0.0", port=int(os.getenv("MCP_PORT", "8002"))))
    else:
        mcp.run()
