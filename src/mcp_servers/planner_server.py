"""
MCP Server for Microsoft Planner via Microsoft Graph API (FastMCP).

Allows PM and BA agents to manage Planner plans, buckets (Epics),
tasks (Stories/Tasks), and user assignments through the MS Graph REST API.

Authentication: OAuth2 client credentials (app-only).
Required permissions: Tasks.ReadWrite, Group.ReadWrite.All, User.Read.All

Transports:
  stdio (default):  python -m src.mcp_servers.planner_server
  HTTP:             MCP_TRANSPORT=http MCP_PORT=8006 python -m src.mcp_servers.planner_server
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
from fastmcp import FastMCP
from mcp.types import TextContent

logger = logging.getLogger(__name__)

TENANT_ID = os.getenv("MS_TENANT_ID", "")
CLIENT_ID = os.getenv("MS_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET", "")
GRAPH_BASE = "https://graph.microsoft.com/v1.0"

_token_cache: dict = {"token": None, "expires_at": 0.0}


@dataclass
class PlannerState:
    tool_calls: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str, success: bool) -> None:
        self.tool_calls.append({
            "tool": tool, "args": args, "result": result[:500], "success": success,
        })


mcp = FastMCP(
    "planner-mcp-server",
    instructions=(
        "Microsoft Planner task management via MS Graph API. "
        "Requires MS_TENANT_ID, MS_CLIENT_ID, MS_CLIENT_SECRET in environment."
    ),
)
state = PlannerState()


# ── Auth ──────────────────────────────────────────────────────────

async def _get_token() -> str:
    if _token_cache["token"] and time.time() < _token_cache["expires_at"] - 60:
        return _token_cache["token"]

    if not all([TENANT_ID, CLIENT_ID, CLIENT_SECRET]):
        raise RuntimeError(
            "Microsoft Graph credentials not configured. "
            "Set MS_TENANT_ID, MS_CLIENT_ID, MS_CLIENT_SECRET in .env"
        )

    url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "https://graph.microsoft.com/.default",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, data=data)
        resp.raise_for_status()
        result = resp.json()
        _token_cache["token"] = result["access_token"]
        _token_cache["expires_at"] = time.time() + result.get("expires_in", 3600)
        return _token_cache["token"]


async def _graph(method: str, path: str, json_body: dict | None = None,
                 extra_headers: dict | None = None) -> dict:
    token = await _get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.request(method, f"{GRAPH_BASE}{path}", headers=headers, json=json_body)
        if resp.status_code == 204:
            return {}
        if not resp.is_success:
            raise RuntimeError(f"Graph API {method} {path} failed ({resp.status_code}): {resp.text[:500]}")
        return resp.json()


def _fmt(data: dict | list) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


# ── Tool implementations ──────────────────────────────────────────

@mcp.tool()
async def planner_list_groups(search: str = "") -> str:
    """List Microsoft 365 groups available in the tenant.

    Use this to find which group a Planner plan should belong to.
    Returns id and displayName for each group.

    Args:
        search: Optional filter groups by name (partial match).
    """
    path = "/groups?$select=id,displayName,mail&$top=50"
    if search:
        path += f'&$search="displayName:{search}"'
        data = await _graph("GET", path, extra_headers={"ConsistencyLevel": "eventual"})
    else:
        data = await _graph("GET", path)
    groups = data.get("value", [])
    if not groups:
        return "No M365 groups found. Create a group in Microsoft Teams or the M365 admin portal first."
    lines = ["Available M365 Groups:"]
    for g in groups:
        lines.append(f"  ID: {g['id']} | Name: {g.get('displayName', '?')} | Mail: {g.get('mail', '')}")
    return "\n".join(lines)


@mcp.tool()
async def planner_list_plans(group_id: str) -> str:
    """List all Planner plans in a Microsoft 365 group.

    Call planner_list_groups first to find the group_id.

    Args:
        group_id: The M365 group ID that owns the plans.
    """
    data = await _graph("GET", f"/groups/{group_id}/planner/plans")
    plans = data.get("value", [])
    if not plans:
        return f"No Planner plans found in group {group_id}. Use planner_create_plan to create one."
    lines = ["Existing Planner Plans:"]
    for p in plans:
        lines.append(f"  ID: {p['id']} | Title: {p.get('title', '?')}")
    return "\n".join(lines)


@mcp.tool()
async def planner_get_plan(plan_id: str) -> str:
    """Get full details of a Planner plan including all buckets (Epics) and tasks (Stories/Tasks).

    Args:
        plan_id: The Planner plan ID.
    """
    plan = await _graph("GET", f"/planner/plans/{plan_id}")
    buckets_data = await _graph("GET", f"/planner/plans/{plan_id}/buckets")
    buckets = buckets_data.get("value", [])

    lines = [
        f"Plan: {plan.get('title', '?')} (ID: {plan_id})",
        f"Buckets (Epics): {len(buckets)}",
    ]
    for b in buckets:
        lines.append(f"\n  Epic: {b.get('name', '?')} (bucket ID: {b['id']})")
        tasks_data = await _graph("GET", f"/planner/buckets/{b['id']}/tasks")
        for t in tasks_data.get("value", []):
            assignees = list(t.get("assignments", {}).keys())
            pct = t.get("percentComplete", 0)
            lines.append(
                f"    - [{pct}%] {t.get('title', '?')} (task ID: {t['id']}) "
                f"| Assigned: {', '.join(assignees) or 'unassigned'}"
            )
    return "\n".join(lines)


@mcp.tool()
async def planner_create_plan(group_id: str, title: str) -> str:
    """Create a new Planner plan inside a Microsoft 365 group.

    ALWAYS show the plan details to user and get confirmation before calling this.

    Args:
        group_id: The M365 group ID that will own the plan.
        title: Name of the plan.
    """
    result = await _graph("POST", "/planner/plans", json_body={"owner": group_id, "title": title})
    return (
        f"Plan created successfully.\n"
        f"  Plan ID: {result['id']}\n"
        f"  Title: {result.get('title', title)}\n"
        f"  Group: {group_id}"
    )


@mcp.tool()
async def planner_create_bucket(plan_id: str, name: str) -> str:
    """Create a bucket (Epic) in a Planner plan.

    Buckets represent Epics or feature areas.

    Args:
        plan_id: The Planner plan ID.
        name: Name of the Epic / bucket.
    """
    result = await _graph("POST", "/planner/buckets",
                          json_body={"name": name, "planId": plan_id, "orderHint": " !"})
    return (
        f"Bucket (Epic) created successfully.\n"
        f"  Bucket ID: {result['id']}\n"
        f"  Name: {result.get('name', name)}\n"
        f"  Plan ID: {plan_id}"
    )


@mcp.tool()
async def planner_create_task(
    plan_id: str,
    bucket_id: str,
    title: str,
    description: str = "",
    assigned_user_id: str = "",
    due_date: str = "",
) -> str:
    """Create a task (Story or Task) in a Planner bucket.

    ALWAYS show full task details to user for confirmation before creating.

    Args:
        plan_id: The Planner plan ID.
        bucket_id: The bucket (Epic) ID to create the task in.
        title: Task title.
        description: Task description / acceptance criteria (optional).
        assigned_user_id: Microsoft user ID to assign the task to (optional).
        due_date: Due date in ISO 8601 format e.g. 2026-04-15T00:00:00Z (optional).
    """
    body: dict = {"planId": plan_id, "bucketId": bucket_id, "title": title}
    if assigned_user_id:
        body["assignments"] = {
            assigned_user_id: {"@odata.type": "microsoft.graph.plannerAssignment", "orderHint": " !"}
        }
    if due_date:
        body["dueDateTime"] = due_date

    result = await _graph("POST", "/planner/tasks", json_body=body)
    task_id = result["id"]

    if description:
        try:
            detail = await _graph("GET", f"/planner/tasks/{task_id}/details")
            await _graph(
                "PATCH", f"/planner/tasks/{task_id}/details",
                json_body={"description": description},
                extra_headers={"If-Match": detail.get("@odata.etag", "")},
            )
        except Exception:
            pass

    return (
        f"Task created successfully.\n"
        f"  Task ID: {task_id}\n"
        f"  Title: {title}\n"
        f"  Bucket ID: {bucket_id}\n"
        f"  Assigned to: {assigned_user_id or 'unassigned'}\n"
        f"  Due: {due_date or 'not set'}"
    )


@mcp.tool()
async def planner_assign_task(task_id: str, user_id: str) -> str:
    """Assign an existing Planner task to a developer (Microsoft user).

    Call planner_list_members first to get user IDs.

    Args:
        task_id: The Planner task ID.
        user_id: Microsoft user ID of the developer to assign.
    """
    task = await _graph("GET", f"/planner/tasks/{task_id}")
    body = {
        "assignments": {
            user_id: {"@odata.type": "microsoft.graph.plannerAssignment", "orderHint": " !"}
        }
    }
    await _graph("PATCH", f"/planner/tasks/{task_id}", json_body=body,
                 extra_headers={"If-Match": task.get("@odata.etag", "")})
    return f"Task {task_id} successfully assigned to user {user_id}."


@mcp.tool()
async def planner_update_task(
    task_id: str,
    title: str = "",
    percent_complete: int = -1,
    due_date: str = "",
) -> str:
    """Update an existing Planner task (title, percent complete, due date).

    ALWAYS show changes to user for confirmation before calling.

    Args:
        task_id: The Planner task ID.
        title: New title (optional).
        percent_complete: Completion percentage: 0, 25, 50, 75, or 100 (optional, -1 = no change).
        due_date: New due date in ISO 8601 format (optional).
    """
    task = await _graph("GET", f"/planner/tasks/{task_id}")
    body: dict = {}
    if title:
        body["title"] = title
    if percent_complete is not None and percent_complete >= 0:
        body["percentComplete"] = percent_complete
    if due_date:
        body["dueDateTime"] = due_date

    if not body:
        return "No updates provided."

    await _graph("PATCH", f"/planner/tasks/{task_id}", json_body=body,
                 extra_headers={"If-Match": task.get("@odata.etag", "")})
    updated = ", ".join(f"{k}={v}" for k, v in body.items())
    return f"Task {task_id} updated: {updated}"


@mcp.tool()
async def planner_list_tasks(plan_id: str = "", bucket_id: str = "") -> str:
    """List all tasks in a Planner plan or a specific bucket.

    Args:
        plan_id: Plan ID to list all tasks in the plan.
        bucket_id: Bucket ID to list tasks in a specific bucket only.
    """
    if bucket_id:
        data = await _graph("GET", f"/planner/buckets/{bucket_id}/tasks")
    elif plan_id:
        data = await _graph("GET", f"/planner/plans/{plan_id}/tasks")
    else:
        return "Provide either plan_id or bucket_id."

    tasks = data.get("value", [])
    if not tasks:
        return "No tasks found."

    lines = [f"Tasks ({len(tasks)} found):"]
    for t in tasks:
        assignees = list(t.get("assignments", {}).keys())
        pct = t.get("percentComplete", 0)
        lines.append(
            f"  [{pct}%] {t.get('title', '?')} (ID: {t['id']}) "
            f"| Assigned: {', '.join(assignees) or 'unassigned'}"
        )
    return "\n".join(lines)


@mcp.tool()
async def planner_list_members(group_id: str) -> str:
    """List users in a Microsoft 365 group — developers available for task assignment.

    Returns user id, displayName, mail.

    Args:
        group_id: The M365 group ID to list members of.
    """
    data = await _graph("GET", f"/groups/{group_id}/members?$select=id,displayName,mail")
    members = data.get("value", [])
    if not members:
        return f"No members found in group {group_id}."
    lines = ["Group Members (available for task assignment):"]
    for m in members:
        lines.append(f"  User ID: {m['id']} | Name: {m.get('displayName', '?')} | Email: {m.get('mail', '')}")
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
        asyncio.run(mcp.run_http_async(host="0.0.0.0", port=int(os.getenv("MCP_PORT", "8006"))))
    else:
        mcp.run()
