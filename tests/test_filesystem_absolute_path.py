"""Filesystem MCP: absolute paths when AGENT_ALLOW_ABSOLUTE_PATHS=1."""
import importlib
import os
import sys
from pathlib import Path

import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def abs_enabled(monkeypatch):
    monkeypatch.setenv("AGENT_ALLOW_ABSOLUTE_PATHS", "1")
    import src.mcp_servers.filesystem_server as fs
    importlib.reload(fs)
    yield fs
    monkeypatch.delenv("AGENT_ALLOW_ABSOLUTE_PATHS", raising=False)
    importlib.reload(fs)


@pytest.fixture
def abs_disabled(monkeypatch):
    monkeypatch.delenv("AGENT_ALLOW_ABSOLUTE_PATHS", raising=False)
    import src.mcp_servers.filesystem_server as fs
    importlib.reload(fs)
    yield fs


def test_absolute_path_disabled_raises(abs_disabled, tmp_path):
    fs = abs_disabled
    target = tmp_path / "x.txt"
    with pytest.raises(ValueError, match="Absolute paths are disabled"):
        fs._safe_path(str(target))


@pytest.mark.asyncio
async def test_write_read_absolute_path(abs_enabled, tmp_path):
    fs = abs_enabled
    target = tmp_path / "abs_write_test.txt"
    content = "hello absolute\n"

    out = await fs.call_tool("write_file", {"path": str(target), "content": content})
    assert "Written" in out[0].text
    assert target.read_text() == content

    out_read = await fs.call_tool("read_file", {"path": str(target)})
    assert "hello absolute" in out_read[0].text


@pytest.mark.asyncio
async def test_list_directory_absolute_root(abs_enabled, tmp_path):
    fs = abs_enabled
    (tmp_path / "a.txt").write_text("a")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("b")

    out = await fs.call_tool("list_directory", {"path": str(tmp_path), "recursive": False})
    text = out[0].text
    assert "a.txt" in text
    assert "sub" in text


@pytest.mark.asyncio
async def test_search_files_absolute(abs_enabled, tmp_path):
    fs = abs_enabled
    d = tmp_path / "search_here"
    d.mkdir()
    (d / "foo.py").write_text("unique_marker_xyz = 1\n")

    out = await fs.call_tool(
        "search_files",
        {"pattern": "unique_marker_xyz", "path": str(d), "file_pattern": "*.py"},
    )
    assert "unique_marker_xyz" in out[0].text


@pytest.mark.asyncio
async def test_edit_file_absolute(abs_enabled, tmp_path):
    fs = abs_enabled
    target = tmp_path / "edit_abs.txt"
    target.write_text("OLD_LINE\nkeep\n")

    out = await fs.call_tool(
        "edit_file",
        {"path": str(target), "old_text": "OLD_LINE", "new_text": "NEW_LINE"},
    )
    assert "Edited" in out[0].text
    assert "NEW_LINE" in target.read_text()


@pytest.mark.asyncio
async def test_find_files_absolute(abs_enabled, tmp_path):
    fs = abs_enabled
    (tmp_path / "globmatch_abs_123.tmp").write_text("")

    out = await fs.call_tool(
        "find_files",
        {"pattern": "globmatch_abs_*.tmp", "path": str(tmp_path)},
    )
    assert "globmatch_abs_123.tmp" in out[0].text
