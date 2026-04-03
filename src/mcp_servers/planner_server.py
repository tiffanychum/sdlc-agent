"""
MCP Server for Microsoft Planner via Microsoft Graph API.

Allows PM and BA agents to manage Planner plans, buckets (Epics),
tasks (Stories/Tasks), and user assignments through the MS Graph REST API.

Authentication: OAuth2 client credentials (app-only).
Required permissions: Tasks.ReadWrite, Group.ReadWrite.All, User.Read.All
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import httpx


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
            "tool": tool, "args": args,
            "result": result[:500], "success": success,
        })


server = Server("planner-mcp-server")
state = PlannerState()


# ── Auth ──────────────────────────────────────────────────────────

async def _get_token() -> str:
    """Get or refresh the MS Graph access token (client credentials flow)."""
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
    """Make an authenticated Microsoft Graph API request."""
    token = await _get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.request(
            method,
            f"{GRAPH_BASE}{path}",
            headers=headers,
            json=json_body,
        )
        if resp.status_code == 204:
            return {}
        if not resp.is_success:
            error_body = resp.text[:500]
            raise RuntimeError(f"Graph API {method} {path} failed ({resp.status_code}): {error_body}")
        return resp.json()


def _fmt(data: dict | list) -> str:
    """Format API response as readable JSON."""
    return json.dumps(data, indent=2, ensure_ascii=False)


# ── Tool Definitions ──────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="planner_list_groups",
            description=(
                "List Microsoft 365 groups available in the tenant. "
                "Use this to find which group a Planner plan should belong to. "
                "Returns id, displayName for each group."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "Optional: filter groups by name (partial match)",
                        "default": "",
                    }
                },
            },
        ),
        Tool(
            name="planner_list_plans",
            description=(
                "List all Planner plans in a Microsoft 365 group. "
                "Call planner_list_groups first to find the group_id."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "group_id": {
                        "type": "string",
                        "description": "The M365 group ID that owns the plans",
                    }
                },
                "required": ["group_id"],
            },
        ),
        Tool(
            name="planner_get_plan",
            description=(
                "Get full details of a Planner plan including all buckets (Epics) "
                "and tasks (Stories/Tasks). Use before showing content to user for confirmation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "The Planner plan ID",
                    }
                },
                "required": ["plan_id"],
            },
        ),
        Tool(
            name="planner_create_plan",
            description=(
                "Create a new Planner plan inside a Microsoft 365 group. "
                "ALWAYS show the plan details to user and get confirmation before calling this."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "group_id": {
                        "type": "string",
                        "description": "The M365 group ID that will own the plan",
                    },
                    "title": {
                        "type": "string",
                        "description": "Name of the plan",
                    },
                },
                "required": ["group_id", "title"],
            },
        ),
        Tool(
            name="planner_create_bucket",
            description=(
                "Create a bucket (Epic) in a Planner plan. "
                "Buckets represent Epics or feature areas."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "The Planner plan ID",
                    },
                    "name": {
                        "type": "string",
                        "description": "Name of the Epic / bucket",
                    },
                },
                "required": ["plan_id", "name"],
            },
        ),
        Tool(
            name="planner_create_task",
            description=(
                "Create a task (Story or Task) in a Planner bucket. "
                "Optionally assign to a user. "
                "ALWAYS show full task details to user for confirmation before creating."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "The Planner plan ID",
                    },
                    "bucket_id": {
                        "type": "string",
                        "description": "The bucket (Epic) ID to create the task in",
                    },
                    "title": {
                        "type": "string",
                        "description": "Task title",
                    },
                    "description": {
                        "type": "string",
                        "description": "Task description / acceptance criteria",
                        "default": "",
                    },
                    "assigned_user_id": {
                        "type": "string",
                        "description": "Microsoft user ID to assign the task to (optional)",
                        "default": "",
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Due date in ISO 8601 format (e.g. 2026-04-15T00:00:00Z). Optional.",
                        "default": "",
                    },
                },
                "required": ["plan_id", "bucket_id", "title"],
            },
        ),
        Tool(
            name="planner_assign_task",
            description=(
                "Assign an existing Planner task to a developer (Microsoft user). "
                "Call planner_list_members first to get user IDs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The Planner task ID",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Microsoft user ID of the developer to assign",
                    },
                },
                "required": ["task_id", "user_id"],
            },
        ),
        Tool(
            name="planner_update_task",
            description=(
                "Update an existing Planner task (title, description, percent complete, due date). "
                "ALWAYS show changes to user for confirmation before calling."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The Planner task ID",
                    },
                    "title": {
                        "type": "string",
                        "description": "New title (optional)",
                        "default": "",
                    },
                    "percent_complete": {
                        "type": "integer",
                        "description": "Completion percentage: 0, 25, 50, 75, or 100",
                        "default": -1,
                    },
                    "due_date": {
                        "type": "string",
                        "description": "New due date in ISO 8601 format (optional)",
                        "default": "",
                    },
                },
                "required": ["task_id"],
            },
        ),
        Tool(
            name="planner_list_tasks",
            description="List all tasks in a Planner plan or a specific bucket.",
            inputSchema={
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "Plan ID to list all tasks in the plan",
                        "default": "",
                    },
                    "bucket_id": {
                        "type": "string",
                        "description": "Bucket ID to list tasks in a specific bucket only",
                        "default": "",
                    },
                },
            },
        ),
        Tool(
            name="planner_list_members",
            description=(
                "List users in a Microsoft 365 group — these are the developers "
                "available for task assignment. Returns user id, displayName, mail."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "group_id": {
                        "type": "string",
                        "description": "The M365 group ID to list members of",
                    }
                },
                "required": ["group_id"],
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
        case "planner_list_groups":
            return await _list_groups(args)
        case "planner_list_plans":
            return await _list_plans(args)
        case "planner_get_plan":
            return await _get_plan(args)
        case "planner_create_plan":
            return await _create_plan(args)
        case "planner_create_bucket":
            return await _create_bucket(args)
        case "planner_create_task":
            return await _create_task(args)
        case "planner_assign_task":
            return await _assign_task(args)
        case "planner_update_task":
            return await _update_task(args)
        case "planner_list_tasks":
            return await _list_tasks(args)
        case "planner_list_members":
            return await _list_members(args)
        case _:
            raise ValueError(f"Unknown tool: {name}")


# ── Tool Implementations ──────────────────────────────────────────

async def _list_groups(args: dict) -> str:
    search = args.get("search", "")
    path = "/groups?$select=id,displayName,mail&$top=50"
    if search:
        path += f"&$search=\"displayName:{search}\""
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


async def _list_plans(args: dict) -> str:
    group_id = args["group_id"]
    data = await _graph("GET", f"/groups/{group_id}/planner/plans")
    plans = data.get("value", [])
    if not plans:
        return f"No Planner plans found in group {group_id}. Use planner_create_plan to create one."
    lines = ["Existing Planner Plans:"]
    for p in plans:
        lines.append(f"  ID: {p['id']} | Title: {p.get('title', '?')}")
    return "\n".join(lines)


async def _get_plan(args: dict) -> str:
    plan_id = args["plan_id"]
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
        tasks = tasks_data.get("value", [])
        for t in tasks:
            assignees = list(t.get("assignments", {}).keys())
            pct = t.get("percentComplete", 0)
            lines.append(
                f"    - [{pct}%] {t.get('title', '?')} (task ID: {t['id']}) "
                f"| Assigned: {', '.join(assignees) or 'unassigned'}"
            )
    return "\n".join(lines)


async def _create_plan(args: dict) -> str:
    group_id = args["group_id"]
    title = args["title"]
    body = {"owner": group_id, "title": title}
    result = await _graph("POST", "/planner/plans", json_body=body)
    return (
        f"Plan created successfully.\n"
        f"  Plan ID: {result['id']}\n"
        f"  Title: {result.get('title', title)}\n"
        f"  Group: {group_id}"
    )


async def _create_bucket(args: dict) -> str:
    plan_id = args["plan_id"]
    name = args["name"]
    body = {"name": name, "planId": plan_id, "orderHint": " !"}
    result = await _graph("POST", "/planner/buckets", json_body=body)
    return (
        f"Bucket (Epic) created successfully.\n"
        f"  Bucket ID: {result['id']}\n"
        f"  Name: {result.get('name', name)}\n"
        f"  Plan ID: {plan_id}"
    )


async def _create_task(args: dict) -> str:
    plan_id = args["plan_id"]
    bucket_id = args["bucket_id"]
    title = args["title"]
    description = args.get("description", "")
    assigned_user_id = args.get("assigned_user_id", "")
    due_date = args.get("due_date", "")

    body: dict = {
        "planId": plan_id,
        "bucketId": bucket_id,
        "title": title,
    }
    if assigned_user_id:
        body["assignments"] = {
            assigned_user_id: {
                "@odata.type": "microsoft.graph.plannerAssignment",
                "orderHint": " !",
            }
        }
    if due_date:
        body["dueDateTime"] = due_date

    result = await _graph("POST", "/planner/tasks", json_body=body)
    task_id = result["id"]

    # Add description via task details (separate endpoint)
    if description:
        try:
            detail = await _graph("GET", f"/planner/tasks/{task_id}/details")
            etag = detail.get("@odata.etag", "")
            await _graph(
                "PATCH",
                f"/planner/tasks/{task_id}/details",
                json_body={"description": description},
                extra_headers={"If-Match": etag},
            )
        except Exception:
            pass  # Description update is best-effort

    return (
        f"Task created successfully.\n"
        f"  Task ID: {task_id}\n"
        f"  Title: {title}\n"
        f"  Bucket ID: {bucket_id}\n"
        f"  Assigned to: {assigned_user_id or 'unassigned'}\n"
        f"  Due: {due_date or 'not set'}"
    )


async def _assign_task(args: dict) -> str:
    task_id = args["task_id"]
    user_id = args["user_id"]

    task = await _graph("GET", f"/planner/tasks/{task_id}")
    etag = task.get("@odata.etag", "")
    body = {
        "assignments": {
            user_id: {
                "@odata.type": "microsoft.graph.plannerAssignment",
                "orderHint": " !",
            }
        }
    }
    await _graph(
        "PATCH",
        f"/planner/tasks/{task_id}",
        json_body=body,
        extra_headers={"If-Match": etag},
    )
    return f"Task {task_id} successfully assigned to user {user_id}."


async def _update_task(args: dict) -> str:
    task_id = args["task_id"]
    task = await _graph("GET", f"/planner/tasks/{task_id}")
    etag = task.get("@odata.etag", "")

    body: dict = {}
    if title := args.get("title"):
        body["title"] = title
    pct = args.get("percent_complete", -1)
    if pct is not None and pct >= 0:
        body["percentComplete"] = pct
    if due_date := args.get("due_date"):
        body["dueDateTime"] = due_date

    if not body:
        return "No updates provided."

    await _graph(
        "PATCH",
        f"/planner/tasks/{task_id}",
        json_body=body,
        extra_headers={"If-Match": etag},
    )
    updated = ", ".join(f"{k}={v}" for k, v in body.items())
    return f"Task {task_id} updated: {updated}"


async def _list_tasks(args: dict) -> str:
    plan_id = args.get("plan_id", "")
    bucket_id = args.get("bucket_id", "")

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
            f"  [{pct}%] {t.get('title', '?')} "
            f"(ID: {t['id']}) | Assigned: {', '.join(assignees) or 'unassigned'}"
        )
    return "\n".join(lines)


async def _list_members(args: dict) -> str:
    group_id = args["group_id"]
    data = await _graph("GET", f"/groups/{group_id}/members?$select=id,displayName,mail")
    members = data.get("value", [])
    if not members:
        return f"No members found in group {group_id}."
    lines = ["Group Members (available for task assignment):"]
    for m in members:
        lines.append(
            f"  User ID: {m['id']} | Name: {m.get('displayName', '?')} | Email: {m.get('mail', '')}"
        )
    return "\n".join(lines)


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
