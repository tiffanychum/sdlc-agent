"""
MCP Server for filesystem operations.

The most fundamental tool for any coding agent — read, write, search,
and navigate codebases. Sandboxed to a configurable workspace root
for safety.
"""

import os
import fnmatch
from pathlib import Path
from dataclasses import dataclass, field
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


WORKSPACE_ROOT = os.getenv("AGENT_WORKSPACE", os.getcwd())
MAX_FILE_SIZE = 100_000  # 100KB read limit
MAX_RESULTS = 50


@dataclass
class FilesystemState:
    tool_calls: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str, success: bool) -> None:
        self.tool_calls.append({
            "tool": tool, "args": args,
            "result": result[:500], "success": success,
        })


server = Server("filesystem-mcp-server")
state = FilesystemState()


def _safe_path(relative_path: str) -> Path:
    """Resolve a path within the workspace, preventing directory traversal."""
    resolved = Path(WORKSPACE_ROOT, relative_path).resolve()
    workspace = Path(WORKSPACE_ROOT).resolve()
    if not str(resolved).startswith(str(workspace)):
        raise ValueError(f"Path escapes workspace: {relative_path}")
    return resolved


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="read_file",
            description="Read the contents of a file. Returns the file content as text with line numbers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file from workspace root"},
                    "start_line": {"type": "integer", "description": "Optional: start reading from this line (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Optional: stop reading at this line"},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="write_file",
            description="Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="edit_file",
            description="Replace a specific string in a file. Use for precise edits without rewriting the entire file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file"},
                    "old_text": {"type": "string", "description": "Exact text to find and replace"},
                    "new_text": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        ),
        Tool(
            name="list_directory",
            description="List files and directories at a given path. Shows file sizes and types.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to directory (empty string for root)", "default": ""},
                    "recursive": {"type": "boolean", "description": "List recursively", "default": False},
                },
            },
        ),
        Tool(
            name="search_files",
            description="Search for a text pattern across files in the workspace. Like grep/ripgrep.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text or regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search in (relative path)", "default": ""},
                    "file_pattern": {"type": "string", "description": "Glob pattern to filter files (e.g., '*.py')", "default": "*"},
                },
                "required": ["pattern"],
            },
        ),
        Tool(
            name="find_files",
            description="Find files by name pattern. Like 'find' command with glob matching.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g., '*.py', 'test_*.ts')"},
                    "path": {"type": "string", "description": "Directory to search in", "default": ""},
                },
                "required": ["pattern"],
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
        case "read_file":
            return _read_file(args)
        case "write_file":
            return _write_file(args)
        case "edit_file":
            return _edit_file(args)
        case "list_directory":
            return _list_directory(args)
        case "search_files":
            return _search_files(args)
        case "find_files":
            return _find_files(args)
        case _:
            raise ValueError(f"Unknown tool: {name}")


def _read_file(args: dict) -> str:
    filepath = _safe_path(args["path"])
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {args['path']}")
    if filepath.stat().st_size > MAX_FILE_SIZE:
        raise ValueError(f"File too large ({filepath.stat().st_size} bytes). Use start_line/end_line.")

    lines = filepath.read_text(encoding="utf-8").splitlines()
    start = args.get("start_line", 1) - 1
    end = args.get("end_line", len(lines))
    selected = lines[max(0, start):end]

    numbered = "\n".join(f"{i+start+1:4d} | {line}" for i, line in enumerate(selected))
    return f"File: {args['path']} ({len(lines)} lines total)\n\n{numbered}"


def _write_file(args: dict) -> str:
    filepath = _safe_path(args["path"])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(args["content"], encoding="utf-8")
    return f"Written {len(args['content'])} bytes to {args['path']}"


def _edit_file(args: dict) -> str:
    filepath = _safe_path(args["path"])
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {args['path']}")

    content = filepath.read_text(encoding="utf-8")
    old_text = args["old_text"]
    if old_text not in content:
        raise ValueError(f"Text not found in {args['path']}. Ensure exact match including whitespace.")

    count = content.count(old_text)
    if count > 1:
        raise ValueError(f"Found {count} occurrences. Provide more context for a unique match.")

    new_content = content.replace(old_text, args["new_text"], 1)
    filepath.write_text(new_content, encoding="utf-8")
    return f"Edited {args['path']}: replaced 1 occurrence"


def _list_directory(args: dict) -> str:
    dirpath = _safe_path(args.get("path", ""))
    if not dirpath.is_dir():
        raise NotADirectoryError(f"Not a directory: {args.get('path', '')}")

    entries = []
    if args.get("recursive"):
        for p in sorted(dirpath.rglob("*")):
            if any(part.startswith(".") for part in p.relative_to(dirpath).parts):
                continue
            rel = p.relative_to(Path(WORKSPACE_ROOT))
            kind = "dir" if p.is_dir() else f"{p.stat().st_size:,}B"
            entries.append(f"  {rel}  ({kind})")
    else:
        for p in sorted(dirpath.iterdir()):
            if p.name.startswith("."):
                continue
            rel = p.relative_to(Path(WORKSPACE_ROOT))
            kind = "dir/" if p.is_dir() else f"{p.stat().st_size:,}B"
            entries.append(f"  {rel}  ({kind})")

    if not entries:
        return "Directory is empty."
    return f"Contents of {args.get('path', '.')}:\n" + "\n".join(entries[:MAX_RESULTS])


def _search_files(args: dict) -> str:
    import re
    search_dir = _safe_path(args.get("path", ""))
    pattern = args["pattern"]
    file_glob = args.get("file_pattern", "*")

    matches = []
    for filepath in search_dir.rglob(file_glob):
        if not filepath.is_file() or filepath.stat().st_size > MAX_FILE_SIZE:
            continue
        if any(part.startswith(".") for part in filepath.relative_to(search_dir).parts):
            continue
        try:
            content = filepath.read_text(encoding="utf-8")
            for i, line in enumerate(content.splitlines(), 1):
                if re.search(pattern, line):
                    rel = filepath.relative_to(Path(WORKSPACE_ROOT))
                    matches.append(f"  {rel}:{i}: {line.strip()}")
                    if len(matches) >= MAX_RESULTS:
                        return f"Found {len(matches)}+ matches:\n" + "\n".join(matches)
        except (UnicodeDecodeError, PermissionError):
            continue

    if not matches:
        return f"No matches found for '{pattern}'"
    return f"Found {len(matches)} match(es):\n" + "\n".join(matches)


def _find_files(args: dict) -> str:
    search_dir = _safe_path(args.get("path", ""))
    pattern = args["pattern"]

    results = []
    for filepath in sorted(search_dir.rglob(pattern)):
        if any(part.startswith(".") for part in filepath.relative_to(search_dir).parts):
            continue
        rel = filepath.relative_to(Path(WORKSPACE_ROOT))
        kind = "dir/" if filepath.is_dir() else f"{filepath.stat().st_size:,}B"
        results.append(f"  {rel}  ({kind})")
        if len(results) >= MAX_RESULTS:
            break

    if not results:
        return f"No files matching '{pattern}'"
    return f"Found {len(results)} file(s):\n" + "\n".join(results)


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
