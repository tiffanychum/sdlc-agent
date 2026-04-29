"""
ToolsClient — read-only inspection of registered tool groups.

Lets SDK callers see exactly which MCP-backed tool groups are available
and which underlying tool names each group ships, so an agent definition
can be built programmatically without poking at the registry source.
"""

from __future__ import annotations

import asyncio
from typing import Any


class ToolsClient:
    """Discoverability for the tool registry."""

    def list_groups(self) -> dict[str, list[str]]:
        """Return ``{group_name: [tool_name, ...]}``.

        Discovery is async upstream (FastMCP's ``list_tools``); we drive it
        through ``asyncio.run`` so SDK callers get a synchronous, simple API
        that's easy to use from notebooks and demo scripts.
        """
        from src.tools.registry import get_all_tools

        try:
            tool_map = asyncio.run(get_all_tools())
        except RuntimeError:
            # Already inside an event loop — caller should use ``alist_groups`` instead.
            raise RuntimeError(
                "ToolsClient.list_groups() cannot be called from within a "
                "running event loop. Use `await client.alist_groups()` instead."
            )
        return {group: [t.name for t in tools] for group, tools in tool_map.items()}

    async def alist_groups(self) -> dict[str, list[str]]:
        """Async variant of ``list_groups``."""
        from src.tools.registry import get_all_tools
        tool_map = await get_all_tools()
        return {group: [t.name for t in tools] for group, tools in tool_map.items()}

    def describe(self, tool_name: str) -> dict[str, Any] | None:
        """Return the full descriptor for one tool (group + description + arg schema)."""
        groups = self._all_tools_sync()
        for group, tools in groups.items():
            for t in tools:
                if t.name == tool_name:
                    return {
                        "group": group,
                        "name": t.name,
                        "description": t.description,
                        "args_schema": t.args_schema.model_json_schema()
                        if t.args_schema else {},
                    }
        return None

    @staticmethod
    def _all_tools_sync():
        from src.tools.registry import get_all_tools
        return asyncio.run(get_all_tools())
