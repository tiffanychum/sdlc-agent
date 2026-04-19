"""
MCP Server for Jira via Atlassian REST API v3 (FastMCP).

Allows PM and BA agents to manage Jira projects, Epics, Stories, Tasks,
user assignments, and issue transitions.

Auth: Basic auth with email + API token (Jira Cloud).
Required: JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN in .env

Transports:
  stdio (default):  python -m src.mcp_servers.jira_server
  HTTP:             MCP_TRANSPORT=http MCP_PORT=8008 python -m src.mcp_servers.jira_server
"""

import asyncio
import base64
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx
from fastmcp import FastMCP
from mcp.types import TextContent

logger = logging.getLogger(__name__)

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "").rstrip("/")
JIRA_EMAIL = os.getenv("JIRA_EMAIL", "")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")


@dataclass
class JiraState:
    tool_calls: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str, success: bool) -> None:
        self.tool_calls.append({
            "tool": tool, "args": args, "result": result[:500], "success": success,
        })


mcp = FastMCP(
    "jira-mcp-server",
    instructions=(
        "Jira project and issue management via REST API v3. "
        "Requires JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN in environment."
    ),
)
state = JiraState()


# ── Auth & HTTP helpers ───────────────────────────────────────────

def _auth_header() -> str:
    if not all([JIRA_EMAIL, JIRA_API_TOKEN]):
        raise RuntimeError(
            "Jira credentials not configured. "
            "Set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN in .env"
        )
    token = base64.b64encode(f"{JIRA_EMAIL}:{JIRA_API_TOKEN}".encode()).decode()
    return f"Basic {token}"


async def _jira(method: str, path: str, json_body: dict | None = None,
                params: dict | None = None) -> dict | list:
    url = f"{JIRA_BASE_URL}/rest/api/3{path}"
    headers = {
        "Authorization": _auth_header(),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.request(method, url, headers=headers, json=json_body, params=params)
        if resp.status_code == 204:
            return {}
        if not resp.is_success:
            try:
                err = resp.json()
                messages = err.get("errorMessages", []) + list(err.get("errors", {}).values())
                detail = "; ".join(messages) if messages else resp.text[:400]
            except Exception:
                detail = resp.text[:400]
            raise RuntimeError(f"Jira API {method} {path} failed ({resp.status_code}): {detail}")
        if not resp.content:
            return {}
        return resp.json()


def _adf(text: str) -> dict:
    """Convert plain text to Atlassian Document Format (required for Jira v3 descriptions)."""
    if not text:
        return {"type": "doc", "version": 1, "content": []}
    paragraphs = []
    for para in text.split("\n\n"):
        lines = para.strip().splitlines()
        if lines:
            content_nodes = []
            for i, line in enumerate(lines):
                if i > 0:
                    content_nodes.append({"type": "hardBreak"})
                content_nodes.append({"type": "text", "text": line})
            paragraphs.append({"type": "paragraph", "content": content_nodes})
    return {"type": "doc", "version": 1, "content": paragraphs or [
        {"type": "paragraph", "content": [{"type": "text", "text": text}]}
    ]}


def _fmt_issue(issue: dict) -> str:
    fields = issue.get("fields", {})
    assignee = fields.get("assignee") or {}
    parent = fields.get("parent") or {}
    return (
        f"  [{issue['key']}] {fields.get('summary', '?')}\n"
        f"    Type: {fields.get('issuetype', {}).get('name', '?')} "
        f"| Status: {fields.get('status', {}).get('name', '?')} "
        f"| Assignee: {assignee.get('displayName', 'Unassigned')}\n"
        f"    Parent: {parent.get('key', 'none')} "
        f"| Priority: {(fields.get('priority') or {}).get('name', 'None')}"
    )


# ── Tool implementations ──────────────────────────────────────────

@mcp.tool()
async def jira_list_projects() -> str:
    """List all Jira projects in the workspace.

    Use this to let the user choose an existing project or decide to create a new one.
    """
    data = await _jira("GET", "/project?expand=description")
    if not data:
        return "No Jira projects found."
    lines = ["Jira Projects:"]
    for p in data:
        lines.append(
            f"  Key: {p['key']} | Name: {p.get('name', '?')} "
            f"| Type: {p.get('projectTypeKey', '?')} "
            f"| URL: {JIRA_BASE_URL}/browse/{p['key']}"
        )
    return "\n".join(lines)


@mcp.tool()
async def jira_get_project(project_key: str) -> str:
    """Get full details of a Jira project including recent issues, Epics, and Stories.

    Args:
        project_key: Jira project key (e.g. SDLC, PROJ).
    """
    key = project_key.upper()
    project = await _jira("GET", f"/project/{key}")
    search = await _jira("GET", "/search/jql", params={
        "jql": f"project={key} ORDER BY created DESC",
        "maxResults": 15,
        "fields": "summary,issuetype,status,assignee,parent,priority",
    })
    issues = search.get("issues", [])
    lines = [
        f"Project: {project.get('name', key)} (Key: {key})",
        f"Type: {project.get('projectTypeKey', '?')}",
        f"URL: {JIRA_BASE_URL}/browse/{key}",
        f"Total issues: {search.get('total', 0)}",
        "",
        "Recent issues:",
    ]
    for issue in issues:
        lines.append(_fmt_issue(issue))
    return "\n".join(lines)


@mcp.tool()
async def jira_list_issue_types(project_key: str) -> str:
    """List available issue types for a project (Epic, Story, Task, Bug, Subtask).

    Call this before creating issues to confirm which types are available.

    Args:
        project_key: Jira project key.
    """
    key = project_key.upper()
    meta = await _jira("GET", f"/issue/createmeta?projectKeys={key}&expand=projects.issuetypes")
    projects = meta.get("projects", [])
    if not projects:
        return f"No issue types found for project {key}."
    types = projects[0].get("issuetypes", [])
    lines = [f"Available issue types for {key}:"]
    for t in types:
        subtask_marker = " (subtask)" if t.get("subtask") else ""
        lines.append(f"  - {t['name']}{subtask_marker} (id: {t['id']})")
    return "\n".join(lines)


@mcp.tool()
async def jira_list_issues(
    project_key: str,
    issue_type: str = "",
    status: str = "",
    assignee_account_id: str = "",
    max_results: int = 20,
) -> str:
    """List issues in a Jira project using JQL.

    Filter by type, status, assignee, or parent Epic.

    Args:
        project_key: Jira project key.
        issue_type: Filter by type: Epic, Story, Task, Bug (optional).
        status: Filter by status: 'To Do', 'In Progress', 'Done' (optional).
        assignee_account_id: Filter by assignee account ID (optional).
        max_results: Max issues to return (default: 20).
    """
    key = project_key.upper()
    jql_parts = [f"project={key}"]
    if issue_type:
        jql_parts.append(f'issuetype="{issue_type}"')
    if status:
        jql_parts.append(f'status="{status}"')
    if assignee_account_id:
        jql_parts.append(f"assignee={assignee_account_id}")
    jql = " AND ".join(jql_parts) + " ORDER BY created DESC"

    data = await _jira("GET", "/search/jql", params={
        "jql": jql, "maxResults": max_results,
        "fields": "summary,issuetype,status,assignee,parent,priority",
    })
    issues = data.get("issues", [])
    total = data.get("total", 0)
    if not issues:
        return f"No issues found matching the filter in project {key}."
    lines = [f"Issues in {key} ({total} total, showing {len(issues)}):"]
    for issue in issues:
        lines.append(_fmt_issue(issue))
    return "\n".join(lines)


@mcp.tool()
async def jira_get_issue(issue_key: str) -> str:
    """Get full details of a specific Jira issue by key.

    Args:
        issue_key: Jira issue key (e.g. SDLC-1, PROJ-42).
    """
    key = issue_key.upper()
    issue = await _jira(
        "GET",
        f"/issue/{key}?fields=summary,description,issuetype,status,assignee,parent,priority,labels,subtasks,comment",
    )
    fields = issue.get("fields", {})
    assignee = fields.get("assignee") or {}
    parent = fields.get("parent") or {}
    subtasks = fields.get("subtasks", [])

    desc_raw = fields.get("description") or {}
    desc_text = ""
    for block in desc_raw.get("content", []):
        for node in block.get("content", []):
            if node.get("type") == "text":
                desc_text += node.get("text", "")
        desc_text += "\n"

    lines = [
        f"Issue: {key} — {fields.get('summary', '?')}",
        f"  Type: {fields.get('issuetype', {}).get('name', '?')}",
        f"  Status: {fields.get('status', {}).get('name', '?')}",
        f"  Priority: {(fields.get('priority') or {}).get('name', 'None')}",
        f"  Assignee: {assignee.get('displayName', 'Unassigned')} ({assignee.get('emailAddress', '')})",
        f"  Parent: {parent.get('key', 'none')} — {parent.get('fields', {}).get('summary', '')}",
        f"  Labels: {', '.join(fields.get('labels', [])) or 'none'}",
        f"  URL: {JIRA_BASE_URL}/browse/{key}",
    ]
    if desc_text.strip():
        lines.append(f"  Description:\n    {desc_text.strip()[:500]}")
    if subtasks:
        lines.append(f"  Subtasks ({len(subtasks)}):")
        for st in subtasks:
            lines.append(f"    [{st['key']}] {st.get('fields', {}).get('summary', '?')}")
    return "\n".join(lines)


@mcp.tool()
async def jira_create_issue(
    project_key: str,
    issue_type: str,
    summary: str,
    description: str = "",
    assignee_account_id: str = "",
    parent_key: str = "",
    priority: str = "Medium",
    labels: Optional[list] = None,
) -> str:
    """Create a Jira issue (Epic, Story, Task, Bug, or Subtask).

    DANGEROUS — creates a real ticket.
    ALWAYS show full issue details to user and get confirmation before calling.
    Call jira_list_issue_types first to verify available types.

    Args:
        project_key: Jira project key (e.g. SDLC).
        issue_type: Issue type: Epic, Story, Task, Bug, or Subtask.
        summary: Issue title / summary.
        description: Issue description or acceptance criteria (optional).
        assignee_account_id: Account ID of the user to assign (from jira_list_assignable_users).
        parent_key: Parent issue key (e.g. link a Story to an Epic, or a Subtask to a Story).
        priority: Priority: Highest, High, Medium, Low, Lowest (default: Medium).
        labels: Optional labels/tags.
    """
    key = project_key.upper()
    fields: dict = {
        "project": {"key": key},
        "issuetype": {"name": issue_type},
        "summary": summary,
        "priority": {"name": priority},
    }
    if description:
        fields["description"] = _adf(description)
    if assignee_account_id:
        fields["assignee"] = {"accountId": assignee_account_id}
    if parent_key:
        fields["parent"] = {"key": parent_key.upper()}
    if labels:
        fields["labels"] = labels
    if issue_type.lower() == "epic":
        fields["customfield_10011"] = summary

    result = await _jira("POST", "/issue", json_body={"fields": fields})
    issue_key = result.get("key", "?")
    return (
        f"Issue created successfully.\n"
        f"  Key: {issue_key}\n"
        f"  Type: {issue_type}\n"
        f"  Summary: {summary}\n"
        f"  Project: {key}\n"
        f"  Parent: {parent_key or 'none'}\n"
        f"  Assignee account: {assignee_account_id or 'unassigned'}\n"
        f"  Priority: {priority}\n"
        f"  URL: {JIRA_BASE_URL}/browse/{issue_key}"
    )


@mcp.tool()
async def jira_update_issue(
    issue_key: str,
    summary: str = "",
    description: str = "",
    priority: str = "",
) -> str:
    """Update an existing Jira issue (summary, description, priority).

    DANGEROUS — modifies an existing ticket.
    Always show planned changes to user before calling.

    Args:
        issue_key: Jira issue key (e.g. SDLC-5).
        summary: New summary/title (optional).
        description: New description (optional).
        priority: New priority: Highest, High, Medium, Low, Lowest (optional).
    """
    key = issue_key.upper()
    fields: dict = {}
    updates = []
    if summary:
        fields["summary"] = summary
        updates.append(f"summary='{summary}'")
    if description:
        fields["description"] = _adf(description)
        updates.append("description updated")
    if priority:
        fields["priority"] = {"name": priority}
        updates.append(f"priority={priority}")
    if not fields:
        return "No updates provided."
    await _jira("PUT", f"/issue/{key}", json_body={"fields": fields})
    return f"Issue {key} updated: {', '.join(updates)}\n  URL: {JIRA_BASE_URL}/browse/{key}"


@mcp.tool()
async def jira_assign_issue(issue_key: str, account_id: str) -> str:
    """Assign a Jira issue to a developer.

    DANGEROUS — sends notification to the assignee.
    Call jira_list_assignable_users first to get account IDs.

    Args:
        issue_key: Jira issue key.
        account_id: Jira user account ID (from jira_list_assignable_users).
    """
    key = issue_key.upper()
    await _jira("PUT", f"/issue/{key}/assignee", json_body={"accountId": account_id})
    return f"Issue {key} assigned to account {account_id}.\n  URL: {JIRA_BASE_URL}/browse/{key}"


@mcp.tool()
async def jira_transition_issue(issue_key: str, status_name: str) -> str:
    """Change the status of a Jira issue (e.g. To Do → In Progress → Done).

    DANGEROUS — changes ticket status. Always confirm with user first.

    Args:
        issue_key: Jira issue key.
        status_name: Target status name (e.g. 'In Progress', 'Done', 'To Do').
    """
    key = issue_key.upper()
    transitions_data = await _jira("GET", f"/issue/{key}/transitions")
    transitions = transitions_data.get("transitions", [])

    match = next((t for t in transitions if t["name"].lower() == status_name.lower()), None)
    if not match:
        available = [t["name"] for t in transitions]
        return (
            f"Status '{status_name}' not available for {key}. "
            f"Available transitions: {', '.join(available)}"
        )
    await _jira("POST", f"/issue/{key}/transitions", json_body={"transition": {"id": match["id"]}})
    return f"Issue {key} transitioned to '{match['name']}'.\n  URL: {JIRA_BASE_URL}/browse/{key}"


@mcp.tool()
async def jira_list_assignable_users(project_key: str) -> str:
    """List users who can be assigned to issues in a Jira project.

    Returns display name, email, and account ID needed for assignment.

    Args:
        project_key: Jira project key.
    """
    key = project_key.upper()
    data = await _jira("GET", "/user/assignable/search", params={"project": key, "maxResults": 50})
    if not data:
        return f"No assignable users found for project {key}."
    lines = [f"Assignable users for project {key}:"]
    for u in data:
        lines.append(
            f"  Account ID: {u['accountId']} "
            f"| Name: {u.get('displayName', '?')} "
            f"| Email: {u.get('emailAddress', 'N/A')}"
        )
    return "\n".join(lines)


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
        asyncio.run(mcp.run_http_async(host="0.0.0.0", port=int(os.getenv("MCP_PORT", "8008"))))
    else:
        mcp.run()
