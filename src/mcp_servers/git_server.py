"""
MCP Server for Git operations (FastMCP).

Enables agents to understand and manipulate version control — check status,
view diffs, create commits, manage branches. Essential for any coding agent
that modifies files.

Transports:
  stdio (default):  python -m src.mcp_servers.git_server
  HTTP:             MCP_TRANSPORT=http MCP_PORT=8003 python -m src.mcp_servers.git_server
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from fastmcp import FastMCP
from mcp.types import TextContent

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = os.getenv("AGENT_WORKSPACE", os.getcwd())


@dataclass
class GitState:
    tool_calls: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str, success: bool) -> None:
        self.tool_calls.append({
            "tool": tool, "args": args, "result": result[:500], "success": success,
        })


mcp = FastMCP(
    "git-mcp-server",
    instructions="Git version control operations in the agent workspace.",
)
state = GitState()


# ── Git helper ────────────────────────────────────────────────────

async def _git(command: str, cwd: str = WORKSPACE_ROOT) -> str:
    proc = await asyncio.create_subprocess_shell(
        f"git {command}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
    stdout_str = stdout.decode("utf-8", errors="replace").strip()
    stderr_str = stderr.decode("utf-8", errors="replace").strip()
    combined = stderr_str or stdout_str

    if proc.returncode != 0:
        # "nothing to commit" is idempotent — treat as success so the agent can continue.
        if "nothing to commit" in combined or "nothing added to commit" in combined:
            return f"Nothing to commit — working tree already clean. {combined}"
        # Paths outside the workspace git repo (e.g. /tmp/calc-app/) are a common
        # pattern for standalone scratch projects built by Coder. Return a soft
        # informational message instead of crashing the supervisor flow — the
        # agent can acknowledge that git is not applicable and move on.
        if "is outside repository" in combined or "outside of repository" in combined:
            return (
                "NOTE: target path is outside the main repository — git operations "
                "do not apply to standalone scratch projects. Skip git and continue. "
                f"(raw: {combined[:200]})"
            )
        if "not a git repository" in combined.lower():
            return (
                "NOTE: working directory is not a git repository. Skip git and "
                f"continue without committing. (raw: {combined[:200]})"
            )
        raise RuntimeError(f"git {command} failed (exit {proc.returncode}): {combined}")
    return stdout_str


# ── Tool implementations ──────────────────────────────────────────

@mcp.tool()
async def git_status() -> str:
    """Show the working tree status — modified, staged, and untracked files."""
    return await _git("status --short")


@mcp.tool()
async def git_diff(target: str = "", file: Optional[str] = None, staged: bool = False) -> str:
    """Show changes between working tree and index, or between commits.

    Args:
        target: What to diff against (e.g. 'HEAD', 'main', a commit SHA). Default: unstaged changes.
        file: Diff a specific file (optional).
        staged: Show staged changes (--cached).
    """
    cmd = "diff"
    if staged:
        cmd += " --cached"
    if target:
        cmd += f" {target}"
    if file:
        cmd += f" -- {file}"
    result = await _git(cmd)
    return result or "No changes."


@mcp.tool()
async def git_log(count: int = 10, oneline: bool = True, file: Optional[str] = None) -> str:
    """Show commit history.

    Args:
        count: Number of commits to show (default: 10).
        oneline: One-line format (default: True).
        file: Show history for a specific file (optional).
    """
    fmt = "--oneline" if oneline else '--format="%h %an %s (%ar)"'
    cmd = f"log -{count} {fmt}"
    if file:
        cmd += f" -- {file}"
    return await _git(cmd)


@mcp.tool()
async def git_commit(message: str, files: Optional[list] = None) -> str:
    """Stage all changes and create a commit.

    Args:
        message: Commit message.
        files: Specific files to stage (default: all changes).
    """
    if files:
        for f in files:
            await _git(f"add {f}")
    else:
        await _git("add -A")
    message_escaped = message.replace('"', '\\"')
    return await _git(f'commit -m "{message_escaped}"')


@mcp.tool()
async def git_branch(action: str = "list", name: str = "") -> str:
    """List, create, or switch branches.

    Args:
        action: Branch operation: 'list', 'create', or 'switch' (default: 'list').
        name: Branch name (required for create/switch).
    """
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
            raise ValueError(f"Unknown branch action: {action}. Use 'list', 'create', or 'switch'.")


@mcp.tool()
async def git_show(ref: str = "HEAD") -> str:
    """Show the contents of a specific commit.

    Args:
        ref: Commit hash or ref (default: HEAD).
    """
    return await _git(f"show --stat {ref}")


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
        _asyncio.run(mcp.run_http_async(host="0.0.0.0", port=int(os.getenv("MCP_PORT", "8003"))))
    else:
        mcp.run()
