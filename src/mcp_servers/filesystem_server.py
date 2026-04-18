"""
MCP Server for filesystem operations (FastMCP).

The most fundamental tool for any coding agent — read, write, search,
and navigate codebases. Sandboxed to a configurable workspace root
for safety.

Transports:
  stdio (default):  python -m src.mcp_servers.filesystem_server
  HTTP:             MCP_TRANSPORT=http MCP_PORT=8001 python -m src.mcp_servers.filesystem_server
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP
from mcp.types import TextContent

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = os.getenv("AGENT_WORKSPACE", os.getcwd())
MAX_FILE_SIZE = 100_000  # 100KB — files above this are returned as truncated excerpts
MAX_LINES_TRUNCATED = 200  # Lines returned when a file exceeds MAX_FILE_SIZE
MAX_RESULTS = 50


def _allow_absolute_paths() -> bool:
    return os.getenv("AGENT_ALLOW_ABSOLUTE_PATHS", "").lower() in ("1", "true", "yes")


@dataclass
class FilesystemState:
    tool_calls: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str, success: bool) -> None:
        self.tool_calls.append({
            "tool": tool, "args": args, "result": result[:500], "success": success,
        })


mcp = FastMCP(
    "filesystem-mcp-server",
    instructions=(
        "Filesystem operations sandboxed to workspace root. "
        "All paths are relative to the workspace unless AGENT_ALLOW_ABSOLUTE_PATHS=1."
    ),
)
state = FilesystemState()


# ── Path safety ───────────────────────────────────────────────────

def _safe_path(path_arg: str) -> Path:
    """Resolve a path: relative paths stay under WORKSPACE_ROOT; absolute paths allowed if env enables it."""
    raw = (path_arg or "").strip()
    if not raw:
        return Path(WORKSPACE_ROOT).resolve()

    candidate = Path(raw)
    if candidate.is_absolute():
        if not _allow_absolute_paths():
            raise ValueError(
                "Absolute paths are disabled. Set AGENT_ALLOW_ABSOLUTE_PATHS=1 "
                "to allow read/write outside the workspace."
            )
        return candidate.resolve()

    resolved = Path(WORKSPACE_ROOT, raw).resolve()
    workspace = Path(WORKSPACE_ROOT).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise ValueError(f"Path escapes workspace: {path_arg}") from None
    return resolved


# ── Tool implementations ──────────────────────────────────────────

@mcp.tool()
async def read_file(
    path: str,
    start_line: int = 1,
    end_line: int = 0,
    query: str = "",
) -> str:
    """Read the contents of a file with line numbers.

    For LARGE files (>100 KB) always provide a `query` describing what you are
    looking for.  The tool will use semantic search to return the most relevant
    sections rather than a truncated head, saving you many follow-up calls.

    Args:
        path: Path to the file (relative to workspace root, or absolute if
            AGENT_ALLOW_ABSOLUTE_PATHS=1).
        start_line: Start reading from this line (1-indexed, default 1).
            Ignored when `query` is provided.
        end_line: Stop reading at this line (0 = end of file, default 0).
            Ignored when `query` is provided.
        query: Natural-language description of what you are looking for
            (e.g. "regression run endpoint" or "build_orchestrator_from_team").
            When provided for a large file, semantic search returns the most
            relevant chunks instead of a truncated head.
    """
    filepath = _safe_path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {path}")
    # Graceful degradation: if a directory is passed, return a listing instead
    # of erroring — the agent can then pick the specific file it needs.
    if filepath.is_dir():
        entries = sorted(filepath.iterdir())
        lines_out = [f"NOTE: '{path}' is a directory. Listing its contents instead of reading:"]
        for e in entries[:MAX_RESULTS]:
            kind = "dir/" if e.is_dir() else f"{e.stat().st_size:,}B"
            lines_out.append(f"  {e.name}  ({kind})")
        lines_out.append("Use read_file with a specific file path, or list_directory for recursive exploration.")
        return "\n".join(lines_out)

    file_size = filepath.stat().st_size
    lines = filepath.read_text(encoding="utf-8").splitlines()
    total_lines = len(lines)

    # ── Semantic retrieval path ────────────────────────────────────────────
    # Activated when: file is large AND a query is provided (with no explicit range).
    if file_size > MAX_FILE_SIZE and query and start_line == 1 and end_line == 0:
        from src.mcp_servers.file_index import semantic_read
        return await semantic_read(str(filepath), query)

    # ── Truncation fallback (large file, no query, no explicit range) ──────
    # Guide the agent to use query= for semantic search on the next call.
    if file_size > MAX_FILE_SIZE and start_line == 1 and end_line == 0:
        selected = lines[:MAX_LINES_TRUNCATED]
        numbered = "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(selected))
        return (
            f"File: {path} ({total_lines} lines, {file_size} bytes)\n"
            f"⚠️  FILE TRUNCATED — showing lines 1–{len(selected)} of {total_lines}.\n"
            f"💡 For better results, call read_file with query=\"<what you are looking for>\" "
            f"to use semantic search and jump directly to relevant sections.\n"
            f"   OR use start_line/end_line for manual pagination.\n\n"
            + numbered
        )

    # ── Normal linear read ─────────────────────────────────────────────────
    resolved_end = end_line if end_line and end_line > 0 else total_lines
    selected = lines[max(0, start_line - 1): resolved_end]
    numbered = "\n".join(f"{i + start_line:4d} | {line}" for i, line in enumerate(selected))
    return f"File: {path} ({total_lines} lines total)\n\n{numbered}"


@mcp.tool()
async def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates the file if it doesn't exist, overwrites if it does.

    Args:
        path: File path (workspace-relative or absolute if AGENT_ALLOW_ABSOLUTE_PATHS=1).
        content: Content to write.
    """
    filepath = _safe_path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    return f"Written {len(content)} bytes to {path}"


@mcp.tool()
async def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace a specific string in a file. Use for precise edits without rewriting the entire file.

    Args:
        path: File path (workspace-relative or absolute if AGENT_ALLOW_ABSOLUTE_PATHS=1).
        old_text: Exact text to find and replace (must appear exactly once).
        new_text: Replacement text.
    """
    filepath = _safe_path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {path}")

    content = filepath.read_text(encoding="utf-8")
    if old_text not in content:
        raise ValueError(f"Text not found in {path}. Ensure exact match including whitespace.")

    count = content.count(old_text)
    if count > 1:
        raise ValueError(f"Found {count} occurrences. Provide more context for a unique match.")

    filepath.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {path}: replaced 1 occurrence"


@mcp.tool()
async def list_directory(path: str = "", recursive: bool = False) -> str:
    """List files and directories at a given path. Shows file sizes and types.

    Args:
        path: Directory path (workspace-relative, empty = workspace root; or absolute if enabled).
        recursive: List recursively (default False).
    """
    dirpath = _safe_path(path)
    if not dirpath.exists():
        # Path doesn't exist yet — treat it as an empty new directory rather than crashing.
        return f"(empty — '{path}' does not exist yet; use write_file to create files there)"
    if not dirpath.is_dir():
        raise NotADirectoryError(f"Not a directory: {path or '.'}")

    anchor = dirpath.resolve()
    entries = []
    if recursive:
        for p in sorted(dirpath.rglob("*")):
            if any(part.startswith(".") for part in p.relative_to(anchor).parts):
                continue
            kind = "dir" if p.is_dir() else f"{p.stat().st_size:,}B"
            entries.append(f"  {p.relative_to(anchor)}  ({kind})")
    else:
        for p in sorted(dirpath.iterdir()):
            if p.name.startswith("."):
                continue
            kind = "dir/" if p.is_dir() else f"{p.stat().st_size:,}B"
            entries.append(f"  {p.relative_to(anchor)}  ({kind})")

    if not entries:
        return "Directory is empty."
    return f"Contents of {path or '.'}:\n" + "\n".join(entries[:MAX_RESULTS])


@mcp.tool()
async def search_files(pattern: str, path: str = "", file_pattern: str = "*") -> str:
    """Search for a text pattern across files in the workspace. Like grep/ripgrep.

    Args:
        pattern: Text or regex pattern to search for.
        path: Directory to search in (relative or absolute if enabled, default: workspace root).
        file_pattern: Glob pattern to filter files (e.g. '*.py', default: '*').
    """
    search_dir = _safe_path(path)
    anchor = search_dir.resolve()
    matches = []
    for filepath in search_dir.rglob(file_pattern):
        if not filepath.is_file() or filepath.stat().st_size > MAX_FILE_SIZE:
            continue
        if any(part.startswith(".") for part in filepath.relative_to(anchor).parts):
            continue
        try:
            content = filepath.read_text(encoding="utf-8")
            for i, line in enumerate(content.splitlines(), 1):
                if re.search(pattern, line):
                    matches.append(f"  {filepath.relative_to(anchor)}:{i}: {line.strip()}")
                    if len(matches) >= MAX_RESULTS:
                        return f"Found {len(matches)}+ matches:\n" + "\n".join(matches)
        except (UnicodeDecodeError, PermissionError):
            continue

    if not matches:
        return f"No matches found for '{pattern}'"
    return f"Found {len(matches)} match(es):\n" + "\n".join(matches)


@mcp.tool()
async def find_files(pattern: str, path: str = "") -> str:
    """Find files by name pattern. Like 'find' command with glob matching.

    Args:
        pattern: Glob pattern (e.g. '*.py', 'test_*.ts').
        path: Directory to search in (relative or absolute if enabled, default: workspace root).
    """
    search_dir = _safe_path(path)
    anchor = search_dir.resolve()
    results = []
    for filepath in sorted(search_dir.rglob(pattern)):
        if any(part.startswith(".") for part in filepath.relative_to(anchor).parts):
            continue
        kind = "dir/" if filepath.is_dir() else f"{filepath.stat().st_size:,}B"
        results.append(f"  {filepath.relative_to(anchor)}  ({kind})")
        if len(results) >= MAX_RESULTS:
            break

    if not results:
        return f"No files matching '{pattern}'"
    return f"Found {len(results)} file(s):\n" + "\n".join(results)


# ── Backward-compatible shims (used by registry.py and existing tests) ──

async def list_tools():
    """Return MCP-protocol Tool objects for all registered tools."""
    tools = await mcp.list_tools()
    return [t.to_mcp_tool() for t in tools]


async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Invoke a tool by name and return MCP-protocol TextContent results."""
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
        import asyncio
        asyncio.run(mcp.run_http_async(host="0.0.0.0", port=int(os.getenv("MCP_PORT", "8001"))))
    else:
        mcp.run()
