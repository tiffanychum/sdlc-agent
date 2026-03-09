"""
MCP Server for agent memory and note-taking.

Provides persistent key-value storage for agents to remember context
across conversations, store plans, and track task progress.
Essential for Plan-and-Execute and self-reflection strategies.
"""

import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


MEMORY_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "agent_memory.json")


@dataclass
class MemoryState:
    tool_calls: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str, success: bool) -> None:
        self.tool_calls.append({"tool": tool, "args": args, "result": result[:500], "success": success})


server = Server("memory-mcp-server")
state = MemoryState()


def _load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return {"notes": {}, "plans": {}, "context": {}}


def _save_memory(data: dict):
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="memory_store",
            description="Store a key-value pair in persistent memory. Use to save plans, notes, or context for later retrieval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Memory key (e.g., 'current_plan', 'user_preference')"},
                    "value": {"type": "string", "description": "Content to remember"},
                    "category": {"type": "string", "description": "Category: notes, plans, or context", "default": "notes"},
                },
                "required": ["key", "value"],
            },
        ),
        Tool(
            name="memory_retrieve",
            description="Retrieve a value from persistent memory by key.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Memory key to retrieve"},
                    "category": {"type": "string", "description": "Category to search in", "default": "notes"},
                },
                "required": ["key"],
            },
        ),
        Tool(
            name="memory_list",
            description="List all stored memory keys, optionally filtered by category.",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Filter by category (notes, plans, context). Empty = all."},
                },
            },
        ),
        Tool(
            name="memory_delete",
            description="Delete a key from memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Key to delete"},
                    "category": {"type": "string", "description": "Category", "default": "notes"},
                },
                "required": ["key"],
            },
        ),
        Tool(
            name="create_plan",
            description="Create a structured task plan with numbered steps. Use for complex multi-step tasks before execution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Plan name"},
                    "steps": {"type": "array", "items": {"type": "string"}, "description": "Ordered list of steps to execute"},
                },
                "required": ["name", "steps"],
            },
        ),
        Tool(
            name="update_plan_step",
            description="Mark a plan step as completed, in-progress, or failed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "plan_name": {"type": "string", "description": "Name of the plan"},
                    "step_index": {"type": "integer", "description": "Step index (0-based)"},
                    "status": {"type": "string", "description": "New status: pending, in_progress, done, failed"},
                },
                "required": ["plan_name", "step_index", "status"],
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
        case "memory_store":
            return _store(args)
        case "memory_retrieve":
            return _retrieve(args)
        case "memory_list":
            return _list(args)
        case "memory_delete":
            return _delete(args)
        case "create_plan":
            return _create_plan(args)
        case "update_plan_step":
            return _update_plan_step(args)
        case _:
            raise ValueError(f"Unknown tool: {name}")


def _store(args: dict) -> str:
    mem = _load_memory()
    cat = args.get("category", "notes")
    mem.setdefault(cat, {})[args["key"]] = {
        "value": args["value"],
        "updated_at": datetime.now().isoformat(),
    }
    _save_memory(mem)
    return f"Stored '{args['key']}' in {cat}"


def _retrieve(args: dict) -> str:
    mem = _load_memory()
    cat = args.get("category", "notes")
    entry = mem.get(cat, {}).get(args["key"])
    if not entry:
        return f"Key '{args['key']}' not found in {cat}"
    return f"[{cat}/{args['key']}] {entry['value']} (updated: {entry.get('updated_at', '?')})"


def _list(args: dict) -> str:
    mem = _load_memory()
    cat = args.get("category", "")
    results = []
    categories = [cat] if cat else list(mem.keys())
    for c in categories:
        for key, entry in mem.get(c, {}).items():
            val_preview = str(entry.get("value", ""))[:80]
            results.append(f"[{c}/{key}] {val_preview}")
    return "\n".join(results) if results else "Memory is empty"


def _delete(args: dict) -> str:
    mem = _load_memory()
    cat = args.get("category", "notes")
    if args["key"] in mem.get(cat, {}):
        del mem[cat][args["key"]]
        _save_memory(mem)
        return f"Deleted '{args['key']}' from {cat}"
    return f"Key '{args['key']}' not found"


def _create_plan(args: dict) -> str:
    mem = _load_memory()
    plan = {
        "steps": [{"description": s, "status": "pending"} for s in args["steps"]],
        "created_at": datetime.now().isoformat(),
    }
    mem.setdefault("plans", {})[args["name"]] = plan
    _save_memory(mem)
    formatted = "\n".join(f"  {i+1}. [{s['status']}] {s['description']}" for i, s in enumerate(plan["steps"]))
    return f"Plan '{args['name']}' created:\n{formatted}"


def _update_plan_step(args: dict) -> str:
    mem = _load_memory()
    plan = mem.get("plans", {}).get(args["plan_name"])
    if not plan:
        return f"Plan '{args['plan_name']}' not found"
    idx = args["step_index"]
    if idx < 0 or idx >= len(plan["steps"]):
        return f"Step index {idx} out of range"
    plan["steps"][idx]["status"] = args["status"]
    _save_memory(mem)
    formatted = "\n".join(f"  {i+1}. [{s['status']}] {s['description']}" for i, s in enumerate(plan["steps"]))
    return f"Plan '{args['plan_name']}' updated:\n{formatted}"


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
