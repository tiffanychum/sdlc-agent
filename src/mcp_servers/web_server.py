"""
MCP Server for web operations.

Enables agents to fetch documentation, search the web, and read
online resources. Essential for researching APIs, finding solutions,
and staying up-to-date with libraries.
"""

import re
from dataclasses import dataclass, field
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import httpx


@dataclass
class WebState:
    tool_calls: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str, success: bool) -> None:
        self.tool_calls.append({
            "tool": tool, "args": args,
            "result": result[:500], "success": success,
        })


server = Server("web-mcp-server")
state = WebState()


def _html_to_text(html: str) -> str:
    """Basic HTML to text conversion for readability."""
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:10000]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="fetch_url",
            description=(
                "Fetch the content of a URL and return it as readable text. "
                "Use for reading documentation, API references, blog posts, or any web page."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to fetch"},
                    "raw": {
                        "type": "boolean",
                        "description": "Return raw HTML instead of extracted text",
                        "default": False,
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="web_search",
            description=(
                "Search the web for information. Returns titles, URLs, and snippets. "
                "Use for finding documentation, solutions to errors, or library references."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="check_url",
            description="Check if a URL is reachable and return its HTTP status code and headers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to check"},
                },
                "required": ["url"],
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
        case "fetch_url":
            return await _fetch_url(args)
        case "web_search":
            return await _web_search(args)
        case "check_url":
            return await _check_url(args)
        case _:
            raise ValueError(f"Unknown tool: {name}")


async def _fetch_url(args: dict) -> str:
    url = args["url"]
    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
        resp = await client.get(url, headers={"User-Agent": "SDLC-Agent/0.1"})
        resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    if "json" in content_type:
        return f"URL: {url}\nContent-Type: {content_type}\n\n{resp.text[:10000]}"

    if args.get("raw"):
        return f"URL: {url}\n\n{resp.text[:10000]}"

    text = _html_to_text(resp.text)
    return f"URL: {url}\nContent-Type: {content_type}\n\n{text}"


async def _web_search(args: dict) -> str:
    """
    Web search implementation. Uses a simple approach for demo.
    In production, integrate with Google Custom Search API, Brave Search API,
    or SerpAPI for real results.
    """
    query = args["query"]
    num = args.get("num_results", 5)

    # For demo: use DuckDuckGo Lite (no API key needed)
    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        resp = await client.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers={"User-Agent": "SDLC-Agent/0.1"},
        )

    text = _html_to_text(resp.text)
    lines = [line.strip() for line in text.split(".") if len(line.strip()) > 20][:num * 2]

    if not lines:
        return f"Search results for: {query}\n\nNo results found. Try a different query."

    return f"Search results for: {query}\n\n" + "\n".join(f"- {line.strip()}" for line in lines[:num])


async def _check_url(args: dict) -> str:
    url = args["url"]
    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        resp = await client.head(url, headers={"User-Agent": "SDLC-Agent/0.1"})

    headers_str = "\n".join(f"  {k}: {v}" for k, v in list(resp.headers.items())[:10])
    return f"URL: {url}\nStatus: {resp.status_code}\nHeaders:\n{headers_str}"


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
