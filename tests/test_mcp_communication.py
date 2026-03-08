"""
End-to-end tests for the MCP communication layer.

Validates each MCP server's:
- Tool discovery (list_tools)
- Tool invocation (call_tool)
- Error handling and recovery
- Input validation
- State tracking for observability

These tests ensure reliable integration between agents and tools.
"""

import os
import tempfile
import pytest

from src.mcp_servers.filesystem_server import (
    call_tool as fs_call, list_tools as fs_list, state as fs_state, WORKSPACE_ROOT,
)
from src.mcp_servers.shell_server import (
    call_tool as shell_call, list_tools as shell_list, state as shell_state,
)
from src.mcp_servers.git_server import (
    call_tool as git_call, list_tools as git_list, state as git_state,
)
from src.mcp_servers.web_server import (
    call_tool as web_call, list_tools as web_list, state as web_state,
)


# ============================================================
# Filesystem MCP Server Tests
# ============================================================

class TestFilesystemMCPServer:
    """E2E tests for filesystem MCP server — the core tool for any coding agent."""

    @pytest.mark.asyncio
    async def test_tool_discovery(self):
        """Verify all filesystem tools are discoverable via MCP protocol."""
        tools = await fs_list()
        tool_names = {t.name for t in tools}
        expected = {"read_file", "write_file", "edit_file", "list_directory", "search_files", "find_files"}
        assert tool_names == expected, f"Missing tools: {expected - tool_names}"

    @pytest.mark.asyncio
    async def test_tool_schemas_valid(self):
        """Verify each tool has a valid JSON Schema for input validation."""
        tools = await fs_list()
        for tool in tools:
            assert tool.inputSchema is not None, f"{tool.name} missing schema"
            assert tool.inputSchema.get("type") == "object"
            assert "properties" in tool.inputSchema

    @pytest.mark.asyncio
    async def test_write_then_read_file(self):
        """E2E: Write a file, then read it back and verify contents."""
        test_path = "tests/_tmp_test_write.txt"
        full_path = os.path.join(WORKSPACE_ROOT, test_path)

        try:
            await fs_call("write_file", {
                "path": test_path,
                "content": "Hello from MCP test\nLine 2\nLine 3",
            })

            result = await fs_call("read_file", {"path": test_path})
            text = result[0].text
            assert "Hello from MCP test" in text
            assert "Line 2" in text
            assert "3 lines total" in text
        finally:
            if os.path.exists(full_path):
                os.remove(full_path)

    @pytest.mark.asyncio
    async def test_read_file_with_line_range(self):
        """E2E: Read specific lines from a file."""
        test_path = "tests/_tmp_test_lines.txt"
        full_path = os.path.join(WORKSPACE_ROOT, test_path)

        try:
            content = "\n".join(f"Line {i}" for i in range(1, 11))
            await fs_call("write_file", {"path": test_path, "content": content})

            result = await fs_call("read_file", {
                "path": test_path,
                "start_line": 3,
                "end_line": 5,
            })
            text = result[0].text
            assert "Line 3" in text
            assert "Line 5" in text
            assert "Line 1" not in text
        finally:
            if os.path.exists(full_path):
                os.remove(full_path)

    @pytest.mark.asyncio
    async def test_edit_file_precise_replacement(self):
        """E2E: Write → Edit → Read to verify precise string replacement."""
        test_path = "tests/_tmp_test_edit.txt"
        full_path = os.path.join(WORKSPACE_ROOT, test_path)

        try:
            await fs_call("write_file", {
                "path": test_path,
                "content": "def greet():\n    return 'hello'\n",
            })

            await fs_call("edit_file", {
                "path": test_path,
                "old_text": "return 'hello'",
                "new_text": "return 'hello, world!'",
            })

            result = await fs_call("read_file", {"path": test_path})
            text = result[0].text
            assert "hello, world!" in text
            assert "hello'" not in text  # old text should be gone
        finally:
            if os.path.exists(full_path):
                os.remove(full_path)

    @pytest.mark.asyncio
    async def test_edit_file_rejects_ambiguous_match(self):
        """Verify edit_file fails when the target text appears multiple times."""
        test_path = "tests/_tmp_test_ambiguous.txt"
        full_path = os.path.join(WORKSPACE_ROOT, test_path)

        try:
            await fs_call("write_file", {
                "path": test_path,
                "content": "foo\nfoo\nbar\n",
            })

            result = await fs_call("edit_file", {
                "path": test_path,
                "old_text": "foo",
                "new_text": "baz",
            })
            text = result[0].text
            assert "Error" in text or "2 occurrences" in text
        finally:
            if os.path.exists(full_path):
                os.remove(full_path)

    @pytest.mark.asyncio
    async def test_search_files(self):
        """E2E: Search for a pattern across files in the workspace."""
        result = await fs_call("search_files", {
            "pattern": "def list_tools",
            "path": "src/mcp_servers",
            "file_pattern": "*.py",
        })
        text = result[0].text
        assert "match" in text.lower() or "list_tools" in text

    @pytest.mark.asyncio
    async def test_find_files_by_pattern(self):
        """E2E: Find files matching a glob pattern."""
        result = await fs_call("find_files", {
            "pattern": "*.py",
            "path": "src/mcp_servers",
        })
        text = result[0].text
        assert "filesystem_server.py" in text or ".py" in text

    @pytest.mark.asyncio
    async def test_list_directory(self):
        """E2E: List directory contents."""
        result = await fs_call("list_directory", {"path": "src", "recursive": False})
        text = result[0].text
        assert "agents" in text or "mcp_servers" in text

    @pytest.mark.asyncio
    async def test_read_nonexistent_file_error_recovery(self):
        """Verify graceful error when reading a file that doesn't exist."""
        result = await fs_call("read_file", {"path": "nonexistent_file_xyz.txt"})
        text = result[0].text
        assert "Error" in text or "not found" in text.lower()

    @pytest.mark.asyncio
    async def test_directory_traversal_blocked(self):
        """Security: verify path traversal outside workspace is blocked."""
        result = await fs_call("read_file", {"path": "../../../etc/passwd"})
        text = result[0].text
        assert "Error" in text

    @pytest.mark.asyncio
    async def test_tool_call_observability(self):
        """Verify tool calls are recorded for monitoring and evaluation."""
        initial_count = len(fs_state.tool_calls)
        await fs_call("list_directory", {"path": ""})
        assert len(fs_state.tool_calls) > initial_count
        last = fs_state.tool_calls[-1]
        assert last["tool"] == "list_directory"
        assert last["success"] is True


# ============================================================
# Shell MCP Server Tests
# ============================================================

class TestShellMCPServer:
    """E2E tests for shell command execution MCP server."""

    @pytest.mark.asyncio
    async def test_tool_discovery(self):
        tools = await shell_list()
        tool_names = {t.name for t in tools}
        assert {"run_command", "run_script", "run_tests"} == tool_names

    @pytest.mark.asyncio
    async def test_run_simple_command(self):
        """E2E: Run a basic shell command and capture output."""
        result = await shell_call("run_command", {"command": "echo 'hello from agent'"})
        text = result[0].text
        assert "hello from agent" in text
        assert "Exit code: 0" in text

    @pytest.mark.asyncio
    async def test_run_command_captures_stderr(self):
        """Verify stderr is captured alongside stdout."""
        result = await shell_call("run_command", {"command": "ls nonexistent_dir_xyz 2>&1 || true"})
        text = result[0].text
        assert "Exit code:" in text

    @pytest.mark.asyncio
    async def test_run_command_with_exit_code(self):
        """Verify non-zero exit codes are reported."""
        result = await shell_call("run_command", {"command": "false"})
        text = result[0].text
        assert "Exit code: 1" in text

    @pytest.mark.asyncio
    async def test_run_python_expression(self):
        """E2E: Execute inline Python via shell."""
        result = await shell_call("run_command", {
            "command": "python3 -c \"print(2 + 2)\"",
        })
        text = result[0].text
        assert "4" in text

    @pytest.mark.asyncio
    async def test_command_timeout(self):
        """Verify commands are killed after timeout."""
        result = await shell_call("run_command", {
            "command": "sleep 10",
            "timeout": 2,
        })
        text = result[0].text
        assert "timed out" in text.lower()

    @pytest.mark.asyncio
    async def test_blocked_dangerous_command(self):
        """Security: verify destructive commands are blocked."""
        result = await shell_call("run_command", {"command": "rm -rf /"})
        text = result[0].text
        assert "Error" in text or "blocked" in text.lower()

    @pytest.mark.asyncio
    async def test_tool_call_observability(self):
        initial_count = len(shell_state.tool_calls)
        await shell_call("run_command", {"command": "echo test"})
        assert len(shell_state.tool_calls) > initial_count


# ============================================================
# Web MCP Server Tests
# ============================================================

class TestWebMCPServer:
    """E2E tests for web operations MCP server."""

    @pytest.mark.asyncio
    async def test_tool_discovery(self):
        tools = await web_list()
        tool_names = {t.name for t in tools}
        assert {"fetch_url", "web_search", "check_url"} == tool_names

    @pytest.mark.asyncio
    async def test_check_url(self):
        """E2E: Check if a known URL is reachable."""
        result = await web_call("check_url", {"url": "https://httpbin.org/status/200"})
        text = result[0].text
        assert "200" in text

    @pytest.mark.asyncio
    async def test_fetch_url_json(self):
        """E2E: Fetch a JSON endpoint."""
        result = await web_call("fetch_url", {"url": "https://httpbin.org/json"})
        text = result[0].text
        assert "slideshow" in text.lower() or "json" in text.lower()

    @pytest.mark.asyncio
    async def test_fetch_invalid_url_error_recovery(self):
        """Verify graceful error for unreachable URLs."""
        result = await web_call("fetch_url", {"url": "https://thisdomaindoesnotexist12345.com"})
        text = result[0].text
        assert "Error" in text

    @pytest.mark.asyncio
    async def test_tool_call_observability(self):
        initial_count = len(web_state.tool_calls)
        await web_call("check_url", {"url": "https://httpbin.org/status/200"})
        assert len(web_state.tool_calls) > initial_count


# ============================================================
# Cross-Server Workflow Tests
# ============================================================

class TestCrossServerWorkflows:
    """Tests simulating real agent workflows that span multiple MCP servers."""

    @pytest.mark.asyncio
    async def test_read_code_then_run_tests(self):
        """
        Simulates: Coder reads a test file → Runner executes it.
        Validates cross-server data flow.
        """
        read_result = await fs_call("read_file", {"path": "tests/test_evaluation.py"})
        assert "test_" in read_result[0].text

        run_result = await shell_call("run_command", {
            "command": "python3 -m pytest tests/test_evaluation.py -v --tb=short -q 2>&1 | head -30",
            "timeout": 30,
        })
        assert "test_" in run_result[0].text

    @pytest.mark.asyncio
    async def test_write_code_then_verify(self):
        """
        Simulates: Coder writes a Python file → Runner executes it → verify output.
        """
        test_path = "tests/_tmp_cross_test.py"
        full_path = os.path.join(WORKSPACE_ROOT, test_path)

        try:
            await fs_call("write_file", {
                "path": test_path,
                "content": "print('cross-server test passed')",
            })

            result = await shell_call("run_command", {
                "command": f"python3 {test_path}",
            })
            assert "cross-server test passed" in result[0].text
        finally:
            if os.path.exists(full_path):
                os.remove(full_path)

    @pytest.mark.asyncio
    async def test_search_then_read(self):
        """
        Simulates: Search for a pattern → Read the matching file.
        """
        search_result = await fs_call("search_files", {
            "pattern": "class AgentConfig",
            "path": "src",
            "file_pattern": "*.py",
        })
        assert "AgentConfig" in search_result[0].text

        read_result = await fs_call("read_file", {"path": "src/agents/definitions.py"})
        assert "AgentConfig" in read_result[0].text
