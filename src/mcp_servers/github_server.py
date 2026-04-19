"""
MCP Server for GitHub repository management via GitHub REST API + git CLI (FastMCP).

Handles: repo creation, push, remote add, PR creation, and listing.

All write operations are in DANGEROUS_TOOLS and require HITL confirmation.
Read operations (list repos, list PRs, get repo info) are safe.

Auth: GITHUB_TOKEN env var (requires 'repo' scope for all write operations).

Transports:
  stdio (default):  python -m src.mcp_servers.github_server
  HTTP:             MCP_TRANSPORT=http MCP_PORT=8007 python -m src.mcp_servers.github_server
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx
from fastmcp import FastMCP
from mcp.types import TextContent

logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_BASE = "https://api.github.com"
WORKSPACE_ROOT = os.getenv("AGENT_WORKSPACE", os.getcwd())


@dataclass
class GitHubState:
    tool_calls: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str, success: bool) -> None:
        self.tool_calls.append({
            "tool": tool, "args": args, "result": result[:500], "success": success,
        })


mcp = FastMCP(
    "github-mcp-server",
    instructions=(
        "GitHub repository management via REST API and git CLI. "
        "Requires GITHUB_TOKEN in environment with 'repo' scope for write operations."
    ),
)
state = GitHubState()


# ── GitHub REST API helpers ───────────────────────────────────────

async def _gh(method: str, path: str, json_body: dict | None = None) -> dict | list:
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
        resp = await client.request(method, f"{GITHUB_BASE}{path}", headers=headers, json=json_body)
        if resp.status_code in (204, 201) and not resp.content:
            return {}
        if not resp.is_success:
            raise RuntimeError(f"GitHub API {method} {path} failed ({resp.status_code}): {resp.text[:500]}")
        return resp.json()


async def _get_authenticated_user() -> str:
    data = await _gh("GET", "/user")
    return data.get("login", "")


async def _git(command: str, cwd: str = WORKSPACE_ROOT) -> str:
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
        raise RuntimeError(f"git {command} failed (exit {proc.returncode}): {stderr_str or stdout_str}")
    return stdout_str or stderr_str


# ── Tool implementations ──────────────────────────────────────────

@mcp.tool()
async def github_get_whoami() -> str:
    """Get the GitHub username of the authenticated user.

    Call this first to know the owner for new repos.
    """
    login = await _get_authenticated_user()
    return f"Authenticated as GitHub user: {login}"


@mcp.tool()
async def github_list_repos(org: str = "", limit: int = 20) -> str:
    """List GitHub repositories for the authenticated user or an org.

    Returns repo names, visibility, and clone URLs.

    Args:
        org: Organization name (optional, defaults to authenticated user).
        limit: Max repos to return (default: 20).
    """
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


@mcp.tool()
async def github_get_repo(owner: str, repo: str) -> str:
    """Get details about a specific GitHub repository.

    Args:
        owner: Repo owner (user or org).
        repo: Repository name.
    """
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


@mcp.tool()
async def github_create_repo(
    name: str,
    description: str = "",
    private: bool = True,
    auto_init: bool = True,
    org: str = "",
) -> str:
    """Create a new GitHub repository.

    DANGEROUS — requires user confirmation before executing.
    Always ask the user if they want to use an existing repo first.

    Args:
        name: Repository name (no spaces).
        description: Repository description (optional).
        private: True for private repo, False for public (default: True).
        auto_init: Initialize with README.md (default: True).
        org: Create under this org (optional, defaults to authenticated user).
    """
    body = {"name": name, "description": description, "private": private, "auto_init": auto_init}
    path = f"/orgs/{org}/repos" if org else "/user/repos"
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


@mcp.tool()
async def github_remote_add(
    remote_url: str,
    remote_name: str = "origin",
    working_dir: str = "",
) -> str:
    """Add a GitHub remote to the local git repository (git remote add).

    DANGEROUS — modifies local git config. Requires user confirmation.

    Args:
        remote_url: GitHub HTTPS or SSH URL (e.g. https://github.com/owner/repo.git).
        remote_name: Remote name (default: 'origin').
        working_dir: Local repo path (defaults to workspace root).
    """
    cwd = os.path.join(WORKSPACE_ROOT, working_dir) if working_dir else WORKSPACE_ROOT
    try:
        existing = await _git(f"remote get-url {remote_name}", cwd=cwd)
        if existing:
            return (
                f"Remote '{remote_name}' already exists pointing to: {existing}\n"
                f"Use 'git remote set-url {remote_name} {remote_url}' to update it."
            )
    except RuntimeError:
        pass

    result = await _git(f"remote add {remote_name} {remote_url}", cwd=cwd)
    return (
        f"Remote '{remote_name}' added successfully.\n"
        f"  URL: {remote_url}\n"
        f"  Working dir: {cwd}\n"
        f"{result or ''}"
    ).strip()


@mcp.tool()
async def github_push(
    remote: str = "origin",
    branch: str = "",
    set_upstream: bool = True,
    working_dir: str = "",
) -> str:
    """Push local commits to a GitHub remote (git push).

    DANGEROUS — uploads code to GitHub. Requires user confirmation.
    The local repo must have commits and a remote configured.

    Args:
        remote: Remote name (default: origin).
        branch: Branch to push (default: current branch).
        set_upstream: Set upstream tracking (-u flag, needed for first push).
        working_dir: Local repo path (defaults to workspace root).
    """
    cwd = os.path.join(WORKSPACE_ROOT, working_dir) if working_dir else WORKSPACE_ROOT
    if not branch:
        branch = await _git("rev-parse --abbrev-ref HEAD", cwd=cwd)

    cmd = f"push -u {remote} {branch}" if set_upstream else f"push {remote} {branch}"
    result = await _git(cmd, cwd=cwd)
    return (
        f"Push successful.\n"
        f"  Remote: {remote}\n"
        f"  Branch: {branch}\n"
        f"  Output: {result or 'Branch up to date'}"
    )


@mcp.tool()
async def github_create_pr(
    owner: str,
    repo: str,
    title: str,
    head: str,
    body: str = "",
    base: str = "main",
    draft: bool = False,
) -> str:
    """Create a GitHub pull request.

    DANGEROUS — creates a PR on GitHub. Requires user confirmation.
    The branch must already be pushed to the remote.

    Args:
        owner: Repository owner (user or org).
        repo: Repository name.
        title: PR title.
        head: Source branch name (the branch with your changes).
        body: PR description / body (optional).
        base: Target branch to merge into (default: main).
        draft: Create as draft PR (default: False).
    """
    pr_body: dict = {"title": title, "body": body, "head": head, "base": base, "draft": draft}
    try:
        data = await _gh("POST", f"/repos/{owner}/{repo}/pulls", json_body=pr_body)
    except RuntimeError as exc:
        if "422" in str(exc) or "already exists" in str(exc).lower() or "pull request" in str(exc).lower():
            existing = await _gh("GET", f"/repos/{owner}/{repo}/pulls?state=open&head={owner}:{head}&base={base}")
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


@mcp.tool()
async def github_create_branch(owner: str, repo: str, branch: str, from_branch: str = "main") -> str:
    """Create a new branch in a GitHub repository from an existing branch or commit SHA.

    DANGEROUS — modifies remote branch state. Always confirm with user first.
    Use before github_create_file to ensure the target branch exists.

    Args:
        owner: Repo owner.
        repo: Repo name.
        branch: Name of the new branch to create.
        from_branch: Source branch to branch from (default: main).
    """
    ref_data = await _gh("GET", f"/repos/{owner}/{repo}/git/refs/heads/{from_branch}")
    if isinstance(ref_data, list):
        ref_data = ref_data[0] if ref_data else {}
    sha = ref_data.get("object", {}).get("sha", "")
    if not sha:
        raise RuntimeError(f"Could not resolve SHA for branch '{from_branch}' in {owner}/{repo}.")

    try:
        await _gh("POST", f"/repos/{owner}/{repo}/git/refs",
                  json_body={"ref": f"refs/heads/{branch}", "sha": sha})
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


@mcp.tool()
async def github_create_file(
    owner: str,
    repo: str,
    path: str,
    content: str,
    commit_message: str,
    branch: str = "main",
) -> str:
    """Create or update a file directly in a GitHub repository branch via the Contents API.

    No local clone required.
    DANGEROUS — commits a file to the remote repo. Requires user confirmation.
    Call github_create_branch first if the branch does not exist yet.

    Args:
        owner: Repo owner.
        repo: Repo name.
        path: File path in the repo (e.g. test.py, src/utils/helper.py).
        content: Plain text content to write to the file.
        commit_message: Commit message for the file creation/update.
        branch: Branch to commit to (default: main).
    """
    import base64 as _b64
    clean_path = path.lstrip("/")
    encoded = _b64.b64encode(content.encode("utf-8")).decode("utf-8")

    sha = None
    try:
        existing = await _gh("GET", f"/repos/{owner}/{repo}/contents/{clean_path}?ref={branch}")
        if isinstance(existing, dict):
            sha = existing.get("sha")
    except RuntimeError:
        pass

    body: dict = {"message": commit_message, "content": encoded, "branch": branch}
    if sha:
        body["sha"] = sha

    data = await _gh("PUT", f"/repos/{owner}/{repo}/contents/{clean_path}", json_body=body)
    commit_info = data.get("commit", {})
    action = "updated" if sha else "created"
    return (
        f"File {action} successfully in {owner}/{repo}.\n"
        f"  Path: {clean_path}\n"
        f"  Branch: {branch}\n"
        f"  Commit: {commit_info.get('sha', '?')[:8]} — {commit_message}\n"
        f"  URL: https://github.com/{owner}/{repo}/blob/{branch}/{clean_path}"
    )


@mcp.tool()
async def github_list_prs(owner: str, repo: str, state: str = "open") -> str:
    """List pull requests in a GitHub repository.

    Args:
        owner: Repo owner.
        repo: Repo name.
        state: PR state: 'open', 'closed', or 'all' (default: 'open').
    """
    prs = await _gh("GET", f"/repos/{owner}/{repo}/pulls?state={state}&per_page=20")
    if not prs:
        return f"No {state} pull requests found in {owner}/{repo}."
    lines = [f"Pull Requests ({state}) in {owner}/{repo}:"]
    for pr in prs:
        lines.append(
            f"  PR #{pr['number']}: {pr['title']}\n"
            f"    {pr['head']['ref']} → {pr['base']['ref']} | {pr.get('html_url', '')}"
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
        asyncio.run(mcp.run_http_async(host="0.0.0.0", port=int(os.getenv("MCP_PORT", "8007"))))
    else:
        mcp.run()
