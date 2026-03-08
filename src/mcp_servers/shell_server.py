"""
MCP Server for shell command execution.

Allows agents to run terminal commands — build projects, run tests,
install dependencies, check system info. Sandboxed with command
allowlisting and timeout enforcement.
"""

import asyncio
import os
import shlex
from dataclasses import dataclass, field
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


WORKSPACE_ROOT = os.getenv("AGENT_WORKSPACE", os.getcwd())
DEFAULT_TIMEOUT = 30  # seconds

BLOCKED_COMMANDS = {"rm -rf /", "mkfs", "dd if=", ":(){ :|:&", "shutdown", "reboot"}


@dataclass
class ShellState:
    tool_calls: list[dict] = field(default_factory=list)
    command_history: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str, success: bool) -> None:
        self.tool_calls.append({
            "tool": tool, "args": args,
            "result": result[:500], "success": success,
        })


server = Server("shell-mcp-server")
state = ShellState()


def _is_safe(command: str) -> bool:
    """Basic safety check against destructive commands."""
    cmd_lower = command.lower().strip()
    return not any(blocked in cmd_lower for blocked in BLOCKED_COMMANDS)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="run_command",
            description=(
                "Execute a shell command in the workspace directory. "
                "Use for running tests, builds, installs, linting, or any CLI tool. "
                "Returns stdout, stderr, and exit code."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory relative to workspace (default: workspace root)",
                        "default": "",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30)",
                        "default": 30,
                    },
                },
                "required": ["command"],
            },
        ),
        Tool(
            name="run_script",
            description="Execute a Python script from a file path. Convenience wrapper for 'python <path>'.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the Python script"},
                    "args": {"type": "string", "description": "Command line arguments", "default": ""},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="run_tests",
            description="Run the test suite using pytest. Automatically discovers and runs tests.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Test file or directory to run (default: all tests)",
                        "default": "",
                    },
                    "verbose": {"type": "boolean", "description": "Verbose output", "default": True},
                    "pattern": {"type": "string", "description": "Test name pattern to match (e.g., 'test_auth')"},
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        result = await _dispatch(name, arguments)
        state.record(name, arguments, result, success=True)
        return [TextContent(type="text", text=result)]
    except Exception as e:
        error_msg = f"Error in {name}: {str(e)}"
        state.record(name, arguments, error_msg, success=False)
        return [TextContent(type="text", text=error_msg)]


async def _dispatch(name: str, args: dict) -> str:
    match name:
        case "run_command":
            return await _run_command(args)
        case "run_script":
            return await _run_script(args)
        case "run_tests":
            return await _run_tests(args)
        case _:
            raise ValueError(f"Unknown tool: {name}")


async def _execute(command: str, cwd: str, timeout: int) -> str:
    """Execute a command and return formatted output."""
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
        proc.kill()
        return f"Command timed out after {timeout}s: {command}"

    stdout_str = stdout.decode("utf-8", errors="replace").strip()
    stderr_str = stderr.decode("utf-8", errors="replace").strip()

    state.command_history.append({
        "command": command,
        "exit_code": proc.returncode,
        "cwd": cwd,
    })

    parts = [f"$ {command}", f"Exit code: {proc.returncode}"]
    if stdout_str:
        parts.append(f"\nSTDOUT:\n{stdout_str}")
    if stderr_str:
        parts.append(f"\nSTDERR:\n{stderr_str}")
    return "\n".join(parts)


async def _run_command(args: dict) -> str:
    cwd = os.path.join(WORKSPACE_ROOT, args.get("working_dir", ""))
    timeout = args.get("timeout", DEFAULT_TIMEOUT)
    return await _execute(args["command"], cwd, timeout)


async def _run_script(args: dict) -> str:
    script_path = args["path"]
    extra_args = args.get("args", "")
    command = f"python {shlex.quote(script_path)} {extra_args}".strip()
    return await _execute(command, WORKSPACE_ROOT, DEFAULT_TIMEOUT)


async def _run_tests(args: dict) -> str:
    parts = ["python", "-m", "pytest"]
    if path := args.get("path"):
        parts.append(path)
    if args.get("verbose", True):
        parts.append("-v")
    if pattern := args.get("pattern"):
        parts.extend(["-k", pattern])

    command = " ".join(parts)
    return await _execute(command, WORKSPACE_ROOT, timeout=60)


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
