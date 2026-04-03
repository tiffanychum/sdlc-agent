"""
MCP Server for Jira via Atlassian REST API v3.

Allows PM and BA agents to manage Jira projects, Epics, Stories, Tasks,
user assignments, and issue transitions.

Auth: Basic auth with email + API token (Jira Cloud).
Required: JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN in .env
"""

import asyncio
import base64
import json
import os
from dataclasses import dataclass, field
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import httpx


JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "").rstrip("/")
JIRA_EMAIL = os.getenv("JIRA_EMAIL", "")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")


@dataclass
class JiraState:
    tool_calls: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str, success: bool) -> None:
        self.tool_calls.append({
            "tool": tool, "args": args,
            "result": result[:500], "success": success,
        })


server = Server("jira-mcp-server")
state = JiraState()


# ── Auth & HTTP ───────────────────────────────────────────────────

def _auth_header() -> str:
    """Build Basic auth header from email + API token."""
    if not all([JIRA_EMAIL, JIRA_API_TOKEN]):
        raise RuntimeError(
            "Jira credentials not configured. "
            "Set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN in .env"
        )
    token = base64.b64encode(f"{JIRA_EMAIL}:{JIRA_API_TOKEN}".encode()).decode()
    return f"Basic {token}"


async def _jira(method: str, path: str, json_body: dict | None = None,
                params: dict | None = None) -> dict | list:
    """Make an authenticated Jira REST API v3 request."""
    url = f"{JIRA_BASE_URL}/rest/api/3{path}"
    headers = {
        "Authorization": _auth_header(),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.request(
            method, url,
            headers=headers,
            json=json_body,
            params=params,
        )
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
    """Format a Jira issue dict into a readable string."""
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


# ── Tool Definitions ──────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="jira_list_projects",
            description=(
                "List all Jira projects in the workspace. "
                "Use this to let the user choose an existing project or decide to create a new one."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="jira_get_project",
            description=(
                "Get full details of a Jira project including recent issues, Epics, and Stories. "
                "Use before showing content to user for confirmation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_key": {
                        "type": "string",
                        "description": "Jira project key (e.g. SDLC, PROJ)",
                    }
                },
                "required": ["project_key"],
            },
        ),
        Tool(
            name="jira_list_issue_types",
            description=(
                "List available issue types for a project (Epic, Story, Task, Bug, Subtask). "
                "Call this before creating issues to confirm which types are available."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_key": {
                        "type": "string",
                        "description": "Jira project key",
                    }
                },
                "required": ["project_key"],
            },
        ),
        Tool(
            name="jira_list_issues",
            description=(
                "List issues in a Jira project using JQL. "
                "Filter by type, status, assignee, or parent Epic."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_key": {
                        "type": "string",
                        "description": "Jira project key",
                    },
                    "issue_type": {
                        "type": "string",
                        "description": "Filter by type: Epic, Story, Task, Bug (optional)",
                        "default": "",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by status: 'To Do', 'In Progress', 'Done' (optional)",
                        "default": "",
                    },
                    "assignee_account_id": {
                        "type": "string",
                        "description": "Filter by assignee account ID (optional)",
                        "default": "",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max issues to return (default: 20)",
                        "default": 20,
                    },
                },
                "required": ["project_key"],
            },
        ),
        Tool(
            name="jira_get_issue",
            description="Get full details of a specific Jira issue by key.",
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "Jira issue key (e.g. SDLC-1, PROJ-42)",
                    }
                },
                "required": ["issue_key"],
            },
        ),
        Tool(
            name="jira_create_issue",
            description=(
                "Create a Jira issue (Epic, Story, Task, Bug, or Subtask). "
                "DANGEROUS — creates a real ticket. "
                "ALWAYS show full issue details to user and get confirmation before calling. "
                "Call jira_list_issue_types first to verify available types."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_key": {
                        "type": "string",
                        "description": "Jira project key (e.g. SDLC)",
                    },
                    "issue_type": {
                        "type": "string",
                        "description": "Issue type: Epic, Story, Task, Bug, or Subtask",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Issue title / summary",
                    },
                    "description": {
                        "type": "string",
                        "description": "Issue description or acceptance criteria",
                        "default": "",
                    },
                    "assignee_account_id": {
                        "type": "string",
                        "description": "Account ID of the user to assign (from jira_list_assignable_users)",
                        "default": "",
                    },
                    "parent_key": {
                        "type": "string",
                        "description": "Parent issue key (e.g. link a Story to an Epic, or a Subtask to a Story)",
                        "default": "",
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority: Highest, High, Medium, Low, Lowest",
                        "default": "Medium",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional labels/tags",
                        "default": [],
                    },
                },
                "required": ["project_key", "issue_type", "summary"],
            },
        ),
        Tool(
            name="jira_update_issue",
            description=(
                "Update an existing Jira issue (summary, description, priority). "
                "DANGEROUS — modifies an existing ticket. "
                "Always show planned changes to user before calling."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "Jira issue key (e.g. SDLC-5)",
                    },
                    "summary": {
                        "type": "string",
                        "description": "New summary/title (optional)",
                        "default": "",
                    },
                    "description": {
                        "type": "string",
                        "description": "New description (optional)",
                        "default": "",
                    },
                    "priority": {
                        "type": "string",
                        "description": "New priority: Highest, High, Medium, Low, Lowest (optional)",
                        "default": "",
                    },
                },
                "required": ["issue_key"],
            },
        ),
        Tool(
            name="jira_assign_issue",
            description=(
                "Assign a Jira issue to a developer. "
                "DANGEROUS — sends notification to the assignee. "
                "Call jira_list_assignable_users first to get account IDs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "Jira issue key",
                    },
                    "account_id": {
                        "type": "string",
                        "description": "Jira user account ID (from jira_list_assignable_users)",
                    },
                },
                "required": ["issue_key", "account_id"],
            },
        ),
        Tool(
            name="jira_transition_issue",
            description=(
                "Change the status of a Jira issue (e.g. To Do → In Progress → Done). "
                "DANGEROUS — changes ticket status. Always confirm with user first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "Jira issue key",
                    },
                    "status_name": {
                        "type": "string",
                        "description": "Target status name (e.g. 'In Progress', 'Done', 'To Do')",
                    },
                },
                "required": ["issue_key", "status_name"],
            },
        ),
        Tool(
            name="jira_list_assignable_users",
            description=(
                "List users who can be assigned to issues in a Jira project. "
                "Returns display name, email, and account ID needed for assignment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_key": {
                        "type": "string",
                        "description": "Jira project key",
                    }
                },
                "required": ["project_key"],
            },
        ),
    ]


# ── Tool Call Handler ─────────────────────────────────────────────

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
        case "jira_list_projects":
            return await _list_projects()
        case "jira_get_project":
            return await _get_project(args)
        case "jira_list_issue_types":
            return await _list_issue_types(args)
        case "jira_list_issues":
            return await _list_issues(args)
        case "jira_get_issue":
            return await _get_issue(args)
        case "jira_create_issue":
            return await _create_issue(args)
        case "jira_update_issue":
            return await _update_issue(args)
        case "jira_assign_issue":
            return await _assign_issue(args)
        case "jira_transition_issue":
            return await _transition_issue(args)
        case "jira_list_assignable_users":
            return await _list_assignable_users(args)
        case _:
            raise ValueError(f"Unknown tool: {name}")


# ── Tool Implementations ──────────────────────────────────────────

async def _list_projects() -> str:
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


async def _get_project(args: dict) -> str:
    key = args["project_key"].upper()
    project = await _jira("GET", f"/project/{key}")
    # Get recent issues
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


async def _list_issue_types(args: dict) -> str:
    key = args["project_key"].upper()
    data = await _jira("GET", f"/project/{key}/statuses")
    # Get issue types from createmeta
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


async def _list_issues(args: dict) -> str:
    key = args["project_key"].upper()
    jql_parts = [f"project={key}"]
    if issue_type := args.get("issue_type"):
        jql_parts.append(f'issuetype="{issue_type}"')
    if status := args.get("status"):
        jql_parts.append(f'status="{status}"')
    if assignee_id := args.get("assignee_account_id"):
        jql_parts.append(f"assignee={assignee_id}")
    jql = " AND ".join(jql_parts) + " ORDER BY created DESC"
    max_results = args.get("max_results", 20)

    data = await _jira("GET", "/search/jql", params={
        "jql": jql,
        "maxResults": max_results,
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


async def _get_issue(args: dict) -> str:
    key = args["issue_key"].upper()
    issue = await _jira("GET", f"/issue/{key}?fields=summary,description,issuetype,status,assignee,parent,priority,labels,subtasks,comment")
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


async def _create_issue(args: dict) -> str:
    project_key = args["project_key"].upper()
    issue_type = args["issue_type"]
    summary = args["summary"]
    description = args.get("description", "")
    assignee_id = args.get("assignee_account_id", "")
    parent_key = args.get("parent_key", "")
    priority = args.get("priority", "Medium")
    labels = args.get("labels", [])

    fields: dict = {
        "project": {"key": project_key},
        "issuetype": {"name": issue_type},
        "summary": summary,
        "priority": {"name": priority},
    }
    if description:
        fields["description"] = _adf(description)
    if assignee_id:
        fields["assignee"] = {"accountId": assignee_id}
    if parent_key:
        fields["parent"] = {"key": parent_key.upper()}
    if labels:
        fields["labels"] = labels

    # Epic Name is required for Epics in some Jira configurations
    if issue_type.lower() == "epic":
        fields["customfield_10011"] = summary

    result = await _jira("POST", "/issue", json_body={"fields": fields})
    issue_key = result.get("key", "?")
    return (
        f"Issue created successfully.\n"
        f"  Key: {issue_key}\n"
        f"  Type: {issue_type}\n"
        f"  Summary: {summary}\n"
        f"  Project: {project_key}\n"
        f"  Parent: {parent_key or 'none'}\n"
        f"  Assignee account: {assignee_id or 'unassigned'}\n"
        f"  Priority: {priority}\n"
        f"  URL: {JIRA_BASE_URL}/browse/{issue_key}"
    )


async def _update_issue(args: dict) -> str:
    key = args["issue_key"].upper()
    fields: dict = {}
    updates = []
    if summary := args.get("summary"):
        fields["summary"] = summary
        updates.append(f"summary='{summary}'")
    if description := args.get("description"):
        fields["description"] = _adf(description)
        updates.append("description updated")
    if priority := args.get("priority"):
        fields["priority"] = {"name": priority}
        updates.append(f"priority={priority}")
    if not fields:
        return "No updates provided."
    await _jira("PUT", f"/issue/{key}", json_body={"fields": fields})
    return f"Issue {key} updated: {', '.join(updates)}\n  URL: {JIRA_BASE_URL}/browse/{key}"


async def _assign_issue(args: dict) -> str:
    key = args["issue_key"].upper()
    account_id = args["account_id"]
    await _jira("PUT", f"/issue/{key}/assignee", json_body={"accountId": account_id})
    return f"Issue {key} assigned to account {account_id}.\n  URL: {JIRA_BASE_URL}/browse/{key}"


async def _transition_issue(args: dict) -> str:
    key = args["issue_key"].upper()
    target_status = args["status_name"].lower()

    transitions_data = await _jira("GET", f"/issue/{key}/transitions")
    transitions = transitions_data.get("transitions", [])

    match = next(
        (t for t in transitions if t["name"].lower() == target_status), None
    )
    if not match:
        available = [t["name"] for t in transitions]
        return (
            f"Status '{args['status_name']}' not available for {key}. "
            f"Available transitions: {', '.join(available)}"
        )
    await _jira("POST", f"/issue/{key}/transitions",
                json_body={"transition": {"id": match["id"]}})
    return f"Issue {key} transitioned to '{match['name']}'.\n  URL: {JIRA_BASE_URL}/browse/{key}"


async def _list_assignable_users(args: dict) -> str:
    key = args["project_key"].upper()
    data = await _jira("GET", "/user/assignable/search", params={
        "project": key,
        "maxResults": 50,
    })
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


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
