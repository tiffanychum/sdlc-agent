"""
MCP Server for web operations (FastMCP).

Enables agents to fetch documentation, search the web, and read
online resources. Essential for researching APIs, finding solutions,
and staying up-to-date with libraries.

Transports:
  stdio (default):  python -m src.mcp_servers.web_server
  HTTP:             MCP_TRANSPORT=http MCP_PORT=8004 python -m src.mcp_servers.web_server
"""

import logging
import os
import re
from dataclasses import dataclass, field

import httpx
from fastmcp import FastMCP
from mcp.types import TextContent

logger = logging.getLogger(__name__)


@dataclass
class WebState:
    tool_calls: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str, success: bool) -> None:
        self.tool_calls.append({
            "tool": tool, "args": args, "result": result[:500], "success": success,
        })


mcp = FastMCP(
    "web-mcp-server",
    instructions="Fetch URLs, search the web, and check URL reachability.",
)
state = WebState()


# ── Helpers ───────────────────────────────────────────────────────

def _html_to_text(html: str) -> str:
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:10000]


# ── Tool implementations ──────────────────────────────────────────

@mcp.tool()
async def fetch_url(url: str, raw: bool = False) -> str:
    """Fetch the content of a URL and return it as readable text.

    Use for reading documentation, API references, blog posts, or any web page.

    Args:
        url: Full URL to fetch.
        raw: Return raw HTML instead of extracted text (default: False).
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
        try:
            resp = await client.get(url, headers={"User-Agent": "SDLC-Agent/0.1"})
        except httpx.TimeoutException:
            return (
                f"URL: {url}\n"
                "⚠️  FETCH FAILED — request timed out after 15 s.\n"
                "Recovery: use web_search to find similar content by topic instead."
            )
        except httpx.RequestError as exc:
            return (
                f"URL: {url}\n"
                f"⚠️  FETCH FAILED — network error: {exc}.\n"
                "Recovery: use web_search to find alternative sources."
            )

    if resp.status_code in (301, 302, 307, 308):
        location = resp.headers.get("location", "")
        return f"URL: {url}\nRedirect → {location}\nFetch the redirect target directly."

    if resp.status_code == 403:
        return (
            f"URL: {url}\n"
            "⚠️  FETCH FAILED — 403 Forbidden (geo-block or auth required).\n"
            "Recovery: use web_search to find this information from accessible sources."
        )

    if resp.status_code == 404:
        return (
            f"URL: {url}\n"
            "⚠️  FETCH FAILED — 404 Not Found. The page no longer exists.\n"
            "Recovery: use web_search to find updated or alternative sources on the same topic."
        )

    if resp.status_code >= 400:
        return (
            f"URL: {url}\n"
            f"⚠️  FETCH FAILED — HTTP {resp.status_code}.\n"
            "Recovery: use web_search to find this information from another source."
        )

    content_type = resp.headers.get("content-type", "")
    if "json" in content_type:
        return f"URL: {url}\nContent-Type: {content_type}\n\n{resp.text[:10000]}"
    if raw:
        return f"URL: {url}\n\n{resp.text[:10000]}"

    text = _html_to_text(resp.text)
    return f"URL: {url}\nContent-Type: {content_type}\n\n{text}"


@mcp.tool()
async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web for information. Returns titles, URLs, and snippets.

    Use for finding documentation, solutions to errors, or library references.

    Args:
        query: Search query.
        num_results: Number of results to return (default: 5).
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        resp = await client.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers={"User-Agent": "SDLC-Agent/0.1"},
        )

    text = _html_to_text(resp.text)
    lines = [line.strip() for line in text.split(".") if len(line.strip()) > 20][:num_results * 2]

    if not lines:
        return f"Search results for: {query}\n\nNo results found. Try a different query."
    return f"Search results for: {query}\n\n" + "\n".join(f"- {line}" for line in lines[:num_results])


@mcp.tool()
async def check_url(url: str) -> str:
    """Check if a URL is reachable and return its HTTP status code and headers.

    Args:
        url: URL to check.
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        resp = await client.head(url, headers={"User-Agent": "SDLC-Agent/0.1"})

    headers_str = "\n".join(f"  {k}: {v}" for k, v in list(resp.headers.items())[:10])
    return f"URL: {url}\nStatus: {resp.status_code}\nHeaders:\n{headers_str}"


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
        asyncio.run(mcp.run_http_async(host="0.0.0.0", port=int(os.getenv("MCP_PORT", "8004"))))
    else:
        mcp.run()
