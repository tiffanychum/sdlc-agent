"""
MCP Server for agent memory and note-taking (FastMCP).

Provides persistent key-value storage for agents to remember context
across conversations, store plans, and track task progress.
Essential for Plan-and-Execute and self-reflection strategies.

Transports:
  stdio (default):  python -m src.mcp_servers.memory_server
  HTTP:             MCP_TRANSPORT=http MCP_PORT=8005 python -m src.mcp_servers.memory_server
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from fastmcp import FastMCP
from mcp.types import TextContent

logger = logging.getLogger(__name__)

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "agent_memory.json")


@dataclass
class MemoryState:
    tool_calls: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str, success: bool) -> None:
        self.tool_calls.append({"tool": tool, "args": args, "result": result[:500], "success": success})


mcp = FastMCP(
    "memory-mcp-server",
    instructions="Persistent key-value memory for agents. Stores notes, plans, and context across sessions.",
)
state = MemoryState()


# ── Storage helpers ───────────────────────────────────────────────

def _load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return {"notes": {}, "plans": {}, "context": {}}


def _save_memory(data: dict) -> None:
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Tool implementations ──────────────────────────────────────────

@mcp.tool()
async def memory_store(key: str, value: str, category: str = "notes") -> str:
    """Store a key-value pair in persistent memory.

    Use to save plans, notes, or context for later retrieval.

    Args:
        key: Memory key (e.g. 'current_plan', 'user_preference').
        value: Content to remember.
        category: Storage category: 'notes', 'plans', or 'context' (default: 'notes').
    """
    mem = _load_memory()
    mem.setdefault(category, {})[key] = {
        "value": value,
        "updated_at": datetime.now().isoformat(),
    }
    _save_memory(mem)
    return f"Stored '{key}' in {category}"


@mcp.tool()
async def memory_retrieve(key: str, category: str = "notes") -> str:
    """Retrieve a value from persistent memory by key.

    Args:
        key: Memory key to retrieve.
        category: Category to search in (default: 'notes').
    """
    mem = _load_memory()
    entry = mem.get(category, {}).get(key)
    if not entry:
        return f"Key '{key}' not found in {category}"
    # Tolerate legacy shapes: older rows stored either a bare string or a dict
    # with different field names (e.g. "content" instead of "value"). Falling
    # back instead of raising `KeyError` keeps the whole multi-agent run from
    # crashing on a stale memory file.
    if isinstance(entry, str):
        value, updated = entry, "?"
    elif isinstance(entry, dict):
        value = entry.get("value") or entry.get("content") or entry.get("data") or str(entry)
        updated = entry.get("updated_at", "?")
    else:
        value, updated = str(entry), "?"
    return f"[{category}/{key}] {value} (updated: {updated})"


@mcp.tool()
async def memory_list(category: str = "") -> str:
    """List all stored memory keys, optionally filtered by category.

    Args:
        category: Filter by category ('notes', 'plans', 'context'). Empty = all.
    """
    mem = _load_memory()
    results = []
    categories = [category] if category else list(mem.keys())
    for c in categories:
        for key, entry in mem.get(c, {}).items():
            val_preview = str(entry.get("value", ""))[:80]
            results.append(f"[{c}/{key}] {val_preview}")
    return "\n".join(results) if results else "Memory is empty"


@mcp.tool()
async def memory_delete(key: str, category: str = "notes") -> str:
    """Delete a key from memory.

    Args:
        key: Key to delete.
        category: Category (default: 'notes').
    """
    mem = _load_memory()
    if key in mem.get(category, {}):
        del mem[category][key]
        _save_memory(mem)
        return f"Deleted '{key}' from {category}"
    return f"Key '{key}' not found"


@mcp.tool()
async def create_plan(name: str, steps: list) -> str:
    """Create a structured task plan with numbered steps.

    Use for complex multi-step tasks before execution.

    Args:
        name: Plan name.
        steps: Ordered list of steps to execute.
    """
    mem = _load_memory()
    plan = {
        "steps": [{"description": s, "status": "pending"} for s in steps],
        "created_at": datetime.now().isoformat(),
    }
    mem.setdefault("plans", {})[name] = plan
    _save_memory(mem)
    formatted = "\n".join(
        f"  {i + 1}. [{s['status']}] {s['description']}"
        for i, s in enumerate(plan["steps"])
    )
    return f"Plan '{name}' created:\n{formatted}"


@mcp.tool()
async def update_plan_step(plan_name: str, step_index: int, status: str) -> str:
    """Mark a plan step as completed, in-progress, or failed.

    Args:
        plan_name: Name of the plan.
        step_index: Step index (0-based).
        status: New status: 'pending', 'in_progress', 'done', or 'failed'.
    """
    mem = _load_memory()
    plan = mem.get("plans", {}).get(plan_name)
    if not plan:
        return f"Plan '{plan_name}' not found"
    if step_index < 0 or step_index >= len(plan["steps"]):
        return f"Step index {step_index} out of range"
    plan["steps"][step_index]["status"] = status
    _save_memory(mem)
    formatted = "\n".join(
        f"  {i + 1}. [{s['status']}] {s['description']}"
        for i, s in enumerate(plan["steps"])
    )
    return f"Plan '{plan_name}' updated:\n{formatted}"


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
        import asyncio
        asyncio.run(mcp.run_http_async(host="0.0.0.0", port=int(os.getenv("MCP_PORT", "8005"))))
    else:
        mcp.run()
