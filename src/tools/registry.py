"""
Tool registry that wraps MCP server tools as LangChain-compatible tools.

This bridge layer allows LangGraph agents to invoke MCP tools through
a standardized interface, handling serialization and error recovery.
"""

from langchain_core.tools import StructuredTool

from src.mcp_servers.filesystem_server import (
    server as fs_server, call_tool as fs_call, list_tools as fs_list,
)
from src.mcp_servers.shell_server import (
    server as shell_server, call_tool as shell_call, list_tools as shell_list,
)
from src.mcp_servers.git_server import (
    server as git_server, call_tool as git_call, list_tools as git_list,
)
from src.mcp_servers.web_server import (
    server as web_server, call_tool as web_call, list_tools as web_list,
)


def _make_tool(name: str, description: str, call_fn) -> StructuredTool:
    """Create a LangChain StructuredTool from an MCP tool definition."""

    async def _invoke(**kwargs) -> str:
        results = await call_fn(name, kwargs)
        return results[0].text if results else "No result"

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=name,
        description=description,
        func=lambda **kw: None,
    )


async def get_filesystem_tools() -> list[StructuredTool]:
    mcp_tools = await fs_list()
    return [_make_tool(t.name, t.description, fs_call) for t in mcp_tools]


async def get_shell_tools() -> list[StructuredTool]:
    mcp_tools = await shell_list()
    return [_make_tool(t.name, t.description, shell_call) for t in mcp_tools]


async def get_git_tools() -> list[StructuredTool]:
    mcp_tools = await git_list()
    return [_make_tool(t.name, t.description, git_call) for t in mcp_tools]


async def get_web_tools() -> list[StructuredTool]:
    mcp_tools = await web_list()
    return [_make_tool(t.name, t.description, web_call) for t in mcp_tools]


async def get_all_tools() -> dict[str, list[StructuredTool]]:
    """Returns all tools grouped by MCP server origin."""
    return {
        "filesystem": await get_filesystem_tools(),
        "shell": await get_shell_tools(),
        "git": await get_git_tools(),
        "web": await get_web_tools(),
    }
