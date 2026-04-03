"""
Tool registry that wraps MCP server tools as LangChain-compatible tools.

This bridge layer allows LangGraph agents to invoke MCP tools through
a standardized interface. It converts MCP tool schemas into Pydantic
models so LangChain can properly validate and pass arguments.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field, create_model
from langchain_core.tools import StructuredTool

from src.mcp_servers.filesystem_server import (
    call_tool as fs_call, list_tools as fs_list,
)
from src.mcp_servers.shell_server import (
    call_tool as shell_call, list_tools as shell_list,
)
from src.mcp_servers.git_server import (
    call_tool as git_call, list_tools as git_list,
)
from src.mcp_servers.web_server import (
    call_tool as web_call, list_tools as web_list,
)
from src.mcp_servers.memory_server import (
    call_tool as mem_call, list_tools as mem_list,
)
from src.mcp_servers.planner_server import (
    call_tool as planner_call, list_tools as planner_list,
)
from src.mcp_servers.github_server import (
    call_tool as github_call, list_tools as github_list,
)
from src.mcp_servers.jira_server import (
    call_tool as jira_call, list_tools as jira_list,
)


JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_to_pydantic(tool_name: str, schema: dict) -> type[BaseModel]:
    """Convert a JSON Schema from an MCP tool into a Pydantic model."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields = {}

    for prop_name, prop_schema in properties.items():
        py_type = JSON_TYPE_MAP.get(prop_schema.get("type", "string"), str)
        description = prop_schema.get("description", "")
        default = prop_schema.get("default")

        if prop_name in required:
            fields[prop_name] = (py_type, Field(description=description))
        elif default is not None:
            fields[prop_name] = (Optional[py_type], Field(default=default, description=description))
        else:
            fields[prop_name] = (Optional[py_type], Field(default=None, description=description))

    model_name = "".join(part.capitalize() for part in tool_name.split("_")) + "Args"
    return create_model(model_name, **fields)


def _make_tool(name: str, description: str, schema: dict, call_fn) -> StructuredTool:
    """Create a LangChain StructuredTool from an MCP tool definition."""
    args_model = _schema_to_pydantic(name, schema)

    async def _invoke(**kwargs) -> str:
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        results = await call_fn(name, cleaned)
        return results[0].text if results else "No result"

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=name,
        description=description,
        args_schema=args_model,
        func=lambda **kw: None,
    )


async def get_filesystem_tools() -> list[StructuredTool]:
    mcp_tools = await fs_list()
    return [_make_tool(t.name, t.description, t.inputSchema, fs_call) for t in mcp_tools]


async def get_shell_tools() -> list[StructuredTool]:
    mcp_tools = await shell_list()
    return [_make_tool(t.name, t.description, t.inputSchema, shell_call) for t in mcp_tools]


async def get_git_tools() -> list[StructuredTool]:
    mcp_tools = await git_list()
    return [_make_tool(t.name, t.description, t.inputSchema, git_call) for t in mcp_tools]


async def get_web_tools() -> list[StructuredTool]:
    mcp_tools = await web_list()
    return [_make_tool(t.name, t.description, t.inputSchema, web_call) for t in mcp_tools]


async def get_memory_tools() -> list[StructuredTool]:
    mcp_tools = await mem_list()
    return [_make_tool(t.name, t.description, t.inputSchema, mem_call) for t in mcp_tools]


async def get_planner_tools() -> list[StructuredTool]:
    mcp_tools = await planner_list()
    return [_make_tool(t.name, t.description, t.inputSchema, planner_call) for t in mcp_tools]


async def get_github_tools() -> list[StructuredTool]:
    mcp_tools = await github_list()
    return [_make_tool(t.name, t.description, t.inputSchema, github_call) for t in mcp_tools]


async def get_jira_tools() -> list[StructuredTool]:
    mcp_tools = await jira_list()
    return [_make_tool(t.name, t.description, t.inputSchema, jira_call) for t in mcp_tools]


async def get_all_tools() -> dict[str, list[StructuredTool]]:
    """Returns all tools grouped by MCP server origin."""
    return {
        "filesystem": await get_filesystem_tools(),
        "shell": await get_shell_tools(),
        "git": await get_git_tools(),
        "web": await get_web_tools(),
        "memory": await get_memory_tools(),
        "planner": await get_planner_tools(),
        "github": await get_github_tools(),
        "jira": await get_jira_tools(),
    }
