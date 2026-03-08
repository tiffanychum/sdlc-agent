"""
MCP Server for Git operations.

Enables agents to understand and manipulate version control — check status,
view diffs, create commits, manage branches. Essential for any coding agent
that modifies files.
"""

import asyncio
import os
from dataclasses import dataclass, field
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


WORKSPACE_ROOT = os.getenv("AGENT_WORKSPACE", os.getcwd())


@dataclass
class GitState:
    tool_calls: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str, success: bool) -> None:
        self.tool_calls.append({
            "tool": tool, "args": args,
            "result": result[:500], "success": success,
        })


server = Server("git-mcp-server")
state = GitState()


async def _git(command: str, cwd: str = WORKSPACE_ROOT) -> str:
    """Execute a git command and return output."""
    proc = await asyncio.create_subprocess_shell(
        f"git {command}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
    stdout_str = stdout.decode("utf-8", errors="replace").strip()
    stderr_str = stderr.decode("utf-8", errors="replace").strip()

    if proc.returncode != 0:
        raise RuntimeError(f"git {command} failed (exit {proc.returncode}): {stderr_str or stdout_str}")
    return stdout_str


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="git_status",
            description="Show the working tree status — modified, staged, and untracked files.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="git_diff",
            description="Show changes between working tree and index, or between commits.",
            inputSchema={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "What to diff against (e.g., 'HEAD', 'main', a commit SHA). Default: unstaged changes.",
                        "default": "",
                    },
                    "file": {"type": "string", "description": "Optional: diff a specific file"},
                    "staged": {"type": "boolean", "description": "Show staged changes (--cached)", "default": False},
                },
            },
        ),
        Tool(
            name="git_log",
            description="Show commit history.",
            inputSchema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of commits to show", "default": 10},
                    "oneline": {"type": "boolean", "description": "One-line format", "default": True},
                    "file": {"type": "string", "description": "Show history for a specific file"},
                },
            },
        ),
        Tool(
            name="git_commit",
            description="Stage all changes and create a commit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Commit message"},
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific files to stage (default: all changes)",
                    },
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="git_branch",
            description="List, create, or switch branches.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "create", "switch"],
                        "description": "Branch operation",
                        "default": "list",
                    },
                    "name": {"type": "string", "description": "Branch name (for create/switch)"},
                },
            },
        ),
        Tool(
            name="git_show",
            description="Show the contents of a specific commit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "Commit hash or ref (default: HEAD)", "default": "HEAD"},
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
        case "git_status":
            return await _git("status --short")
        case "git_diff":
            return await _git_diff(args)
        case "git_log":
            return await _git_log(args)
        case "git_commit":
            return await _git_commit(args)
        case "git_branch":
            return await _git_branch(args)
        case "git_show":
            ref = args.get("ref", "HEAD")
            return await _git(f"show --stat {ref}")
        case _:
            raise ValueError(f"Unknown tool: {name}")


async def _git_diff(args: dict) -> str:
    cmd = "diff"
    if args.get("staged"):
        cmd += " --cached"
    if target := args.get("target"):
        cmd += f" {target}"
    if file := args.get("file"):
        cmd += f" -- {file}"
    result = await _git(cmd)
    return result or "No changes."


async def _git_log(args: dict) -> str:
    count = args.get("count", 10)
    fmt = "--oneline" if args.get("oneline", True) else '--format="%h %an %s (%ar)"'
    cmd = f"log -{count} {fmt}"
    if file := args.get("file"):
        cmd += f" -- {file}"
    return await _git(cmd)


async def _git_commit(args: dict) -> str:
    files = args.get("files")
    if files:
        for f in files:
            await _git(f"add {f}")
    else:
        await _git("add -A")

    message = args["message"].replace('"', '\\"')
    return await _git(f'commit -m "{message}"')


async def _git_branch(args: dict) -> str:
    action = args.get("action", "list")
    name = args.get("name", "")

    match action:
        case "list":
            return await _git("branch -a")
        case "create":
            if not name:
                raise ValueError("Branch name required for create")
            return await _git(f"checkout -b {name}")
        case "switch":
            if not name:
                raise ValueError("Branch name required for switch")
            return await _git(f"checkout {name}")
        case _:
            raise ValueError(f"Unknown branch action: {action}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
