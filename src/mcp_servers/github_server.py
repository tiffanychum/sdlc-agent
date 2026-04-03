"""
MCP Server for GitHub repository management via GitHub REST API + git CLI.

Handles: repo creation, push, remote add, PR creation, and listing.

All write operations are in DANGEROUS_TOOLS and require HITL confirmation.
Read operations (list repos, list PRs, get repo info) are safe.

Auth: GITHUB_TOKEN env var (requires 'repo' scope for all write operations).
"""

import asyncio
import os
from dataclasses import dataclass, field
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import httpx


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_BASE = "https://api.github.com"
WORKSPACE_ROOT = os.getenv("AGENT_WORKSPACE", os.getcwd())


@dataclass
class GitHubState:
    tool_calls: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str, success: bool) -> None:
        self.tool_calls.append({
            "tool": tool, "args": args,
            "result": result[:500], "success": success,
        })


server = Server("github-mcp-server")
state = GitHubState()


# ── GitHub REST API Helper ────────────────────────────────────────

async def _gh(method: str, path: str, json_body: dict | None = None) -> dict | list:
    """Make an authenticated GitHub REST API request."""
    if not GITHUB_TOKEN:
        raise RuntimeError(
            "GITHUB_TOKEN not set in .env. "
            "Generate a token at github.com/settings/tokens with 'repo' scope."
        )
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.request(
            method,
            f"{GITHUB_BASE}{path}",
            headers=headers,
            json=json_body,
        )
        if resp.status_code in (204, 201) and not resp.content:
            return {}
        if not resp.is_success:
            error_body = resp.text[:500]
            raise RuntimeError(
                f"GitHub API {method} {path} failed ({resp.status_code}): {error_body}"
            )
        return resp.json()


async def _get_authenticated_user() -> str:
    """Get the login of the authenticated GitHub user."""
    data = await _gh("GET", "/user")
    return data.get("login", "")


async def _git(command: str, cwd: str = WORKSPACE_ROOT) -> str:
    """Execute a git CLI command and return output."""
    proc = await asyncio.create_subprocess_shell(
        f"git {command}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    stdout_str = stdout.decode("utf-8", errors="replace").strip()
    stderr_str = stderr.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {command} failed (exit {proc.returncode}): {stderr_str or stdout_str}"
        )
    return stdout_str or stderr_str


# ── Tool Definitions ──────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="github_get_whoami",
            description=(
                "Get the GitHub username of the authenticated user. "
                "Call this first to know the owner for new repos."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="github_list_repos",
            description=(
                "List GitHub repositories for the authenticated user or an org. "
                "Returns repo names, visibility, and clone URLs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "org": {
                        "type": "string",
                        "description": "Organization name (optional, defaults to authenticated user)",
                        "default": "",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max repos to return (default: 20)",
                        "default": 20,
                    },
                },
            },
        ),
        Tool(
            name="github_get_repo",
            description="Get details about a specific GitHub repository.",
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repo owner (user or org)"},
                    "repo": {"type": "string", "description": "Repository name"},
                },
                "required": ["owner", "repo"],
            },
        ),
        Tool(
            name="github_create_repo",
            description=(
                "Create a new GitHub repository. "
                "DANGEROUS — requires user confirmation before executing. "
                "Always ask the user if they want to use an existing repo first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Repository name (no spaces)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Repository description",
                        "default": "",
                    },
                    "private": {
                        "type": "boolean",
                        "description": "True for private repo, False for public",
                        "default": True,
                    },
                    "auto_init": {
                        "type": "boolean",
                        "description": "Initialize with README.md",
                        "default": True,
                    },
                    "org": {
                        "type": "string",
                        "description": "Create under this org (optional, defaults to authenticated user)",
                        "default": "",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="github_remote_add",
            description=(
                "Add a GitHub remote to the local git repository (git remote add). "
                "DANGEROUS — modifies local git config. Requires user confirmation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "remote_name": {
                        "type": "string",
                        "description": "Remote name (usually 'origin')",
                        "default": "origin",
                    },
                    "remote_url": {
                        "type": "string",
                        "description": "GitHub HTTPS or SSH URL (e.g. https://github.com/owner/repo.git)",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Local repo path (defaults to workspace root)",
                        "default": "",
                    },
                },
                "required": ["remote_url"],
            },
        ),
        Tool(
            name="github_push",
            description=(
                "Push local commits to a GitHub remote (git push). "
                "DANGEROUS — uploads code to GitHub. Requires user confirmation. "
                "The local repo must have commits and a remote configured."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "remote": {
                        "type": "string",
                        "description": "Remote name (default: origin)",
                        "default": "origin",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Branch to push (default: current branch)",
                        "default": "",
                    },
                    "set_upstream": {
                        "type": "boolean",
                        "description": "Set upstream tracking (-u flag, needed for first push)",
                        "default": True,
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Local repo path (defaults to workspace root)",
                        "default": "",
                    },
                },
            },
        ),
        Tool(
            name="github_create_pr",
            description=(
                "Create a GitHub pull request. "
                "DANGEROUS — creates a PR on GitHub. Requires user confirmation. "
                "The branch must already be pushed to the remote."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (user or org)",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name",
                    },
                    "title": {
                        "type": "string",
                        "description": "PR title",
                    },
                    "body": {
                        "type": "string",
                        "description": "PR description / body",
                        "default": "",
                    },
                    "head": {
                        "type": "string",
                        "description": "Source branch name (the branch with your changes)",
                    },
                    "base": {
                        "type": "string",
                        "description": "Target branch to merge into (default: main)",
                        "default": "main",
                    },
                    "draft": {
                        "type": "boolean",
                        "description": "Create as draft PR",
                        "default": False,
                    },
                },
                "required": ["owner", "repo", "title", "head"],
            },
        ),
        Tool(
            name="github_create_branch",
            description=(
                "Create a new branch in a GitHub repository from an existing branch or commit SHA. "
                "DANGEROUS — modifies remote branch state. Always confirm with user first. "
                "Use before github_create_file to ensure the target branch exists."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repo owner"},
                    "repo": {"type": "string", "description": "Repo name"},
                    "branch": {"type": "string", "description": "Name of the new branch to create"},
                    "from_branch": {
                        "type": "string",
                        "description": "Source branch to branch from (default: main)",
                        "default": "main",
                    },
                },
                "required": ["owner", "repo", "branch"],
            },
        ),
        Tool(
            name="github_create_file",
            description=(
                "Create or update a file directly in a GitHub repository branch via the Contents API. "
                "No local clone required. "
                "DANGEROUS — commits a file to the remote repo. Requires user confirmation. "
                "Call github_create_branch first if the branch does not exist yet."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repo owner"},
                    "repo": {"type": "string", "description": "Repo name"},
                    "path": {
                        "type": "string",
                        "description": "File path in the repo (e.g. test.py, src/utils/helper.py)",
                    },
                    "content": {
                        "type": "string",
                        "description": "Plain text content to write to the file",
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Commit message for the file creation/update",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Branch to commit to (default: main)",
                        "default": "main",
                    },
                },
                "required": ["owner", "repo", "path", "content", "commit_message"],
            },
        ),
        Tool(
            name="github_list_prs",
            description="List pull requests in a GitHub repository.",
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repo owner"},
                    "repo": {"type": "string", "description": "Repo name"},
                    "state": {
                        "type": "string",
                        "description": "PR state: open, closed, or all",
                        "default": "open",
                    },
                },
                "required": ["owner", "repo"],
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
        case "github_get_whoami":
            login = await _get_authenticated_user()
            return f"Authenticated as GitHub user: {login}"
        case "github_list_repos":
            return await _list_repos(args)
        case "github_get_repo":
            return await _get_repo(args)
        case "github_create_repo":
            return await _create_repo(args)
        case "github_remote_add":
            return await _remote_add(args)
        case "github_push":
            return await _push(args)
        case "github_create_branch":
            return await _create_branch(args)
        case "github_create_file":
            return await _create_file(args)
        case "github_create_pr":
            return await _create_pr(args)
        case "github_list_prs":
            return await _list_prs(args)
        case _:
            raise ValueError(f"Unknown tool: {name}")


# ── Tool Implementations ──────────────────────────────────────────

async def _list_repos(args: dict) -> str:
    org = args.get("org", "")
    limit = args.get("limit", 20)
    if org:
        path = f"/orgs/{org}/repos?per_page={limit}&sort=updated"
    else:
        path = f"/user/repos?per_page={limit}&sort=updated&affiliation=owner"
    repos = await _gh("GET", path)
    if not repos:
        return "No repositories found."
    lines = [f"Repositories ({len(repos)} found):"]
    for r in repos:
        visibility = "private" if r.get("private") else "public"
        lines.append(
            f"  {r['full_name']} [{visibility}] — {r.get('clone_url', '')} "
            f"| default branch: {r.get('default_branch', 'main')}"
        )
    return "\n".join(lines)


async def _get_repo(args: dict) -> str:
    owner = args["owner"]
    repo = args["repo"]
    data = await _gh("GET", f"/repos/{owner}/{repo}")
    return (
        f"Repository: {data['full_name']}\n"
        f"  URL: {data.get('html_url', '')}\n"
        f"  Clone: {data.get('clone_url', '')}\n"
        f"  Default branch: {data.get('default_branch', 'main')}\n"
        f"  Private: {data.get('private', False)}\n"
        f"  Description: {data.get('description', '')}\n"
        f"  Open PRs: {data.get('open_issues_count', 0)}"
    )


async def _create_repo(args: dict) -> str:
    name = args["name"]
    description = args.get("description", "")
    private = args.get("private", True)
    auto_init = args.get("auto_init", True)
    org = args.get("org", "")

    body = {
        "name": name,
        "description": description,
        "private": private,
        "auto_init": auto_init,
    }
    if org:
        path = f"/orgs/{org}/repos"
    else:
        path = "/user/repos"

    data = await _gh("POST", path, json_body=body)
    return (
        f"Repository created successfully.\n"
        f"  Name: {data.get('full_name', name)}\n"
        f"  URL: {data.get('html_url', '')}\n"
        f"  Clone URL: {data.get('clone_url', '')}\n"
        f"  SSH URL: {data.get('ssh_url', '')}\n"
        f"  Private: {data.get('private', private)}\n"
        f"  Default branch: {data.get('default_branch', 'main')}"
    )


async def _remote_add(args: dict) -> str:
    remote_name = args.get("remote_name", "origin")
    remote_url = args["remote_url"]
    working_dir = args.get("working_dir", "")
    cwd = os.path.join(WORKSPACE_ROOT, working_dir) if working_dir else WORKSPACE_ROOT

    # Check if remote already exists
    try:
        existing = await _git(f"remote get-url {remote_name}", cwd=cwd)
        if existing:
            return (
                f"Remote '{remote_name}' already exists pointing to: {existing}\n"
                f"Use 'git remote set-url {remote_name} {remote_url}' to update it."
            )
    except RuntimeError:
        pass  # Remote doesn't exist yet, proceed

    result = await _git(f"remote add {remote_name} {remote_url}", cwd=cwd)
    return (
        f"Remote '{remote_name}' added successfully.\n"
        f"  URL: {remote_url}\n"
        f"  Working dir: {cwd}\n"
        f"{result or ''}"
    ).strip()


async def _push(args: dict) -> str:
    remote = args.get("remote", "origin")
    branch = args.get("branch", "")
    set_upstream = args.get("set_upstream", True)
    working_dir = args.get("working_dir", "")
    cwd = os.path.join(WORKSPACE_ROOT, working_dir) if working_dir else WORKSPACE_ROOT

    if not branch:
        branch = await _git("rev-parse --abbrev-ref HEAD", cwd=cwd)

    cmd = f"push {remote} {branch}"
    if set_upstream:
        cmd = f"push -u {remote} {branch}"

    result = await _git(cmd, cwd=cwd)
    return (
        f"Push successful.\n"
        f"  Remote: {remote}\n"
        f"  Branch: {branch}\n"
        f"  Output: {result or 'Branch up to date'}"
    )


async def _create_pr(args: dict) -> str:
    owner = args["owner"]
    repo = args["repo"]
    title = args["title"]
    body = args.get("body", "")
    head = args["head"]
    base = args.get("base", "main")
    draft = args.get("draft", False)

    pr_body: dict = {
        "title": title,
        "body": body,
        "head": head,
        "base": base,
        "draft": draft,
    }
    try:
        data = await _gh("POST", f"/repos/{owner}/{repo}/pulls", json_body=pr_body)
    except RuntimeError as exc:
        # GitHub returns 422 when a PR already exists for this branch → base combo.
        if "422" in str(exc) or "already exists" in str(exc).lower() or "pull request" in str(exc).lower():
            existing = await _gh(
                "GET",
                f"/repos/{owner}/{repo}/pulls?state=open&head={owner}:{head}&base={base}",
            )
            if existing and isinstance(existing, list):
                pr = existing[0]
                return (
                    f"Pull Request already exists (returned existing).\n"
                    f"  PR #: {pr.get('number', '?')}\n"
                    f"  Title: {pr.get('title', title)}\n"
                    f"  URL: {pr.get('html_url', '')}\n"
                    f"  {head} → {base}\n"
                    f"  State: {pr.get('state', 'open')}"
                )
        raise
    return (
        f"Pull Request created successfully.\n"
        f"  PR #: {data.get('number', '?')}\n"
        f"  Title: {data.get('title', title)}\n"
        f"  URL: {data.get('html_url', '')}\n"
        f"  {head} → {base}\n"
        f"  State: {data.get('state', 'open')}\n"
        f"  Draft: {draft}"
    )


async def _create_branch(args: dict) -> str:
    import base64 as _b64  # noqa: PLC0415
    owner = args["owner"]
    repo = args["repo"]
    branch = args["branch"]
    from_branch = args.get("from_branch", "main")

    # Get the SHA of the source branch HEAD
    ref_data = await _gh("GET", f"/repos/{owner}/{repo}/git/refs/heads/{from_branch}")
    if isinstance(ref_data, list):
        ref_data = ref_data[0] if ref_data else {}
    sha = ref_data.get("object", {}).get("sha", "")
    if not sha:
        raise RuntimeError(f"Could not resolve SHA for branch '{from_branch}' in {owner}/{repo}.")

    # Create the new branch ref (idempotent — return success if it already exists)
    try:
        await _gh("POST", f"/repos/{owner}/{repo}/git/refs", json_body={
            "ref": f"refs/heads/{branch}",
            "sha": sha,
        })
        action = "created"
    except RuntimeError as exc:
        if "already exists" in str(exc).lower() or "422" in str(exc):
            action = "already exists"
        else:
            raise
    return (
        f"Branch '{branch}' {action} in {owner}/{repo}.\n"
        f"  Based on: {from_branch} ({sha[:8]})\n"
        f"  URL: https://github.com/{owner}/{repo}/tree/{branch}"
    )


async def _create_file(args: dict) -> str:
    import base64 as _b64  # noqa: PLC0415
    owner = args["owner"]
    repo = args["repo"]
    path = args["path"].lstrip("/")
    content = args["content"]
    commit_message = args["commit_message"]
    branch = args.get("branch", "main")

    encoded = _b64.b64encode(content.encode("utf-8")).decode("utf-8")

    # Check if the file already exists (need its SHA to update)
    sha = None
    try:
        existing = await _gh("GET", f"/repos/{owner}/{repo}/contents/{path}?ref={branch}")
        if isinstance(existing, dict):
            sha = existing.get("sha")
    except RuntimeError:
        pass  # File doesn't exist yet — create it

    body: dict = {
        "message": commit_message,
        "content": encoded,
        "branch": branch,
    }
    if sha:
        body["sha"] = sha  # Required for updates

    data = await _gh("PUT", f"/repos/{owner}/{repo}/contents/{path}", json_body=body)
    commit_info = data.get("commit", {})
    action = "updated" if sha else "created"
    return (
        f"File {action} successfully in {owner}/{repo}.\n"
        f"  Path: {path}\n"
        f"  Branch: {branch}\n"
        f"  Commit: {commit_info.get('sha', '?')[:8]} — {commit_message}\n"
        f"  URL: https://github.com/{owner}/{repo}/blob/{branch}/{path}"
    )


async def _list_prs(args: dict) -> str:
    owner = args["owner"]
    repo = args["repo"]
    pr_state = args.get("state", "open")
    prs = await _gh("GET", f"/repos/{owner}/{repo}/pulls?state={pr_state}&per_page=20")
    if not prs:
        return f"No {pr_state} pull requests found in {owner}/{repo}."
    lines = [f"Pull Requests ({pr_state}) in {owner}/{repo}:"]
    for pr in prs:
        lines.append(
            f"  PR #{pr['number']}: {pr['title']}\n"
            f"    {pr['head']['ref']} → {pr['base']['ref']} | {pr.get('html_url', '')}"
        )
    return "\n".join(lines)


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
