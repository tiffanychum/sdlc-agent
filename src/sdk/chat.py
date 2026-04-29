"""
ChatClient — invoke a team's orchestrator from Python.

Two transports are supported:

* ``http`` (default) — POST to ``/api/teams/{id}/chat`` on a running FastAPI
  backend. Recommended because the server-side route writes the full trace
  to the DB exactly the same way the UI does, so chats placed via the SDK
  appear in the Monitoring page identically to chats placed via Studio.

* ``inprocess`` — build the team's LangGraph orchestrator in this Python
  process and invoke it directly. Useful for notebook demos that don't
  want the overhead of running uvicorn. Trace persistence still works
  because the orchestrator already writes spans/traces via OTel.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional


class ChatClient:
    def __init__(self, team_id: str, *, backend_url: Optional[str] = None) -> None:
        self.team_id = team_id
        self.backend_url = (
            backend_url
            or os.getenv("SDLC_HUB_BACKEND")
            or "http://localhost:8000"
        ).rstrip("/")

    # ── HTTP transport ────────────────────────────────────────────────

    def send(
        self,
        message: str,
        *,
        transport: str = "http",
        timeout: float = 600.0,
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send one message and return the full response payload.

        Result keys (HTTP route):
            ``response``, ``agent_used``, ``tool_calls``, ``agent_trace``,
            ``selected_agent``, ``trace`` (with ``trace_id``), ``thread_id``.
        """
        if transport == "http":
            return self._send_http(message, timeout=timeout, thread_id=thread_id)
        if transport == "inprocess":
            return asyncio.run(self._asend_inprocess(message))
        raise ValueError(f"Unknown transport {transport!r}")

    async def asend(
        self,
        message: str,
        *,
        transport: str = "http",
        timeout: float = 600.0,
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if transport == "http":
            return await self._asend_http(message, timeout=timeout, thread_id=thread_id)
        if transport == "inprocess":
            return await self._asend_inprocess(message)
        raise ValueError(f"Unknown transport {transport!r}")

    def _send_http(
        self,
        message: str,
        *,
        timeout: float,
        thread_id: Optional[str],
    ) -> dict[str, Any]:
        import httpx
        url = f"{self.backend_url}/api/teams/{self.team_id}/chat"
        payload: dict[str, Any] = {"message": message}
        if thread_id:
            payload["thread_id"] = thread_id
        resp = httpx.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    async def _asend_http(
        self,
        message: str,
        *,
        timeout: float,
        thread_id: Optional[str],
    ) -> dict[str, Any]:
        import httpx
        url = f"{self.backend_url}/api/teams/{self.team_id}/chat"
        payload: dict[str, Any] = {"message": message}
        if thread_id:
            payload["thread_id"] = thread_id
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    # ── In-process transport ─────────────────────────────────────────

    async def _asend_inprocess(self, message: str) -> dict[str, Any]:
        """Build the orchestrator locally and invoke it.

        Returns a payload shaped like the HTTP route's reply where possible —
        but ``trace_id`` may be absent because the in-process path doesn't
        run the FastAPI route's trace-persistence wrapper. Use the HTTP
        transport when you need DB-persisted traces.
        """
        from src.orchestrator import build_orchestrator_from_team, _extract_text

        graph = await build_orchestrator_from_team(self.team_id)
        result = await graph.ainvoke({
            "messages": [{"role": "user", "content": message}],
            "selected_agent": "",
            "agent_trace": [],
        })
        msgs = result.get("messages") or []
        last = msgs[-1] if msgs else None
        response = (
            _extract_text(getattr(last, "content", "")) if last is not None else ""
        )
        return {
            "response": response,
            "agent_used": result.get("selected_agent", ""),
            "selected_agent": result.get("selected_agent", ""),
            "agent_trace": result.get("agent_trace", []),
            "tool_calls": [],
            "trace": {},
        }
