"""
PromptRegistry — versioned per-role prompt store.

Provides a single source of truth for agent system prompts with full
version history. The optimizer agent writes new versions here; the
orchestrator reads the active version for each role at runtime.

Version numbering: v1 (seed), v2 (CoT-enhanced), v3+ (optimizer-generated).
"""

from __future__ import annotations

import re
from typing import Optional


# ── CoT header injected into v2 prompts ──────────────────────────────────────

COT_HEADER = """
## REASONING PROTOCOL (apply before every tool call)
Before taking action, think step by step:

**SITUATION** — What exactly is being asked? What context do I already have?
**PLAN** — List the exact steps I will take and which tool each step needs.
**EXECUTE** — Carry out step 1. Observe the result carefully.
**VERIFY** — Did it succeed? What does the result tell me? What is the next step?

Never skip PLAN. Never call a tool without first stating what you expect it to return.
"""


def _inject_cot(base_prompt: str) -> str:
    """Insert the CoT reasoning protocol after the first ## Role section."""
    # Insert after the first top-level section header (## Role / ## Scope / etc.)
    match = re.search(r"\n## ", base_prompt)
    if match:
        insert_at = match.start()
        return base_prompt[:insert_at] + COT_HEADER + base_prompt[insert_at:]
    # Fallback: prepend
    return COT_HEADER + "\n" + base_prompt


# ── PromptRegistry ────────────────────────────────────────────────────────────

class PromptRegistry:
    """
    Versioned per-role prompt store backed by the agent metrics SQLite DB.

    Usage:
        registry = PromptRegistry()
        registry.seed_from_definitions(AGENT_DEFINITIONS)   # idempotent v1 seed
        registry.seed_v2_cot(["coder", "planner", "tester", "researcher"])
        prompt = registry.get_prompt("coder")               # latest active version
        prompt = registry.get_prompt("coder", "v1")         # specific version
        new_ver = registry.register("coder", improved_prompt, rationale="…")
    """

    def __init__(self) -> None:
        self._ensure_tables()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _ensure_tables(self) -> None:
        from src.db.database import get_engine
        from src.db.models import Base
        Base.metadata.create_all(get_engine())

    def _session(self):
        from src.db.database import get_session
        return get_session()

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_prompt(self, role: str, version: str = "latest") -> Optional[str]:
        """Return the prompt text for (role, version). 'latest' = highest active version."""
        from src.db.models import PromptVersionEntry
        session = self._session()
        try:
            q = session.query(PromptVersionEntry).filter_by(role=role, is_active=True)
            if version == "latest":
                entry = q.order_by(PromptVersionEntry.created_at.desc()).first()
            else:
                entry = q.filter_by(version=version).first()
            return entry.prompt_text if entry else None
        finally:
            session.close()

    def latest_version(self, role: str) -> str:
        """Return the latest active version label for a role (e.g. 'v2')."""
        from src.db.models import PromptVersionEntry
        session = self._session()
        try:
            entry = (
                session.query(PromptVersionEntry)
                .filter_by(role=role, is_active=True)
                .order_by(PromptVersionEntry.created_at.desc())
                .first()
            )
            return entry.version if entry else "v1"
        finally:
            session.close()

    def list_versions(self, role: str) -> list[dict]:
        """Return all version entries for a role, newest first."""
        from src.db.models import PromptVersionEntry
        session = self._session()
        try:
            entries = (
                session.query(PromptVersionEntry)
                .filter_by(role=role)
                .order_by(PromptVersionEntry.created_at.desc())
                .all()
            )
            return [
                {
                    "role": e.role,
                    "version": e.version,
                    "cot_enhanced": e.cot_enhanced,
                    "parent_version": e.parent_version,
                    "rationale": e.rationale,
                    "metric_scores": e.metric_scores or {},
                    "created_by": e.created_by,
                    "is_active": e.is_active,
                    "notes": e.notes,
                    "created_at": e.created_at.isoformat() if e.created_at else "",
                }
                for e in entries
            ]
        finally:
            session.close()

    def list_all_roles(self) -> list[str]:
        """Return all roles that have at least one registered version."""
        from src.db.models import PromptVersionEntry
        from sqlalchemy import distinct
        session = self._session()
        try:
            rows = session.query(distinct(PromptVersionEntry.role)).all()
            return [r[0] for r in rows]
        finally:
            session.close()

    # ── Write ─────────────────────────────────────────────────────────────────

    def register(
        self,
        role: str,
        prompt_text: str,
        rationale: str = "",
        parent_version: Optional[str] = None,
        created_by: str = "optimizer",
        cot_enhanced: bool = False,
        notes: str = "",
    ) -> str:
        """Register a new prompt version for a role. Returns the new version label."""
        from src.db.models import PromptVersionEntry
        session = self._session()
        try:
            # Auto-increment version number
            existing = (
                session.query(PromptVersionEntry)
                .filter_by(role=role)
                .order_by(PromptVersionEntry.created_at.desc())
                .first()
            )
            if existing:
                m = re.match(r"v(\d+)", existing.version)
                next_n = int(m.group(1)) + 1 if m else 2
            else:
                next_n = 1
            new_version = f"v{next_n}"

            entry = PromptVersionEntry(
                role=role,
                version=new_version,
                prompt_text=prompt_text,
                cot_enhanced=cot_enhanced,
                parent_version=parent_version or (existing.version if existing else None),
                rationale=rationale,
                created_by=created_by,
                is_active=True,
                notes=notes,
            )
            session.add(entry)
            session.commit()
            return new_version
        finally:
            session.close()

    def update_metric_scores(self, role: str, version: str, scores: dict) -> None:
        """Store measured metric scores against a prompt version."""
        from src.db.models import PromptVersionEntry
        session = self._session()
        try:
            entry = (
                session.query(PromptVersionEntry)
                .filter_by(role=role, version=version)
                .first()
            )
            if entry:
                entry.metric_scores = scores
                session.commit()
        finally:
            session.close()

    def deactivate_version(self, role: str, version: str) -> None:
        """Mark a version inactive (does not delete it)."""
        from src.db.models import PromptVersionEntry
        session = self._session()
        try:
            entry = (
                session.query(PromptVersionEntry)
                .filter_by(role=role, version=version)
                .first()
            )
            if entry:
                entry.is_active = False
                session.commit()
        finally:
            session.close()

    # ── Seeding ───────────────────────────────────────────────────────────────

    def seed_from_definitions(self, agent_definitions: list[dict]) -> int:
        """Idempotently seed v1 prompts from AGENT_DEFINITIONS. Returns count inserted."""
        from src.db.models import PromptVersionEntry
        session = self._session()
        inserted = 0
        try:
            for defn in agent_definitions:
                role = defn.get("role") or defn.get("id")
                prompt = defn.get("prompt", "")
                if not role or not prompt:
                    continue
                exists = (
                    session.query(PromptVersionEntry)
                    .filter_by(role=role, version="v1")
                    .first()
                )
                if not exists:
                    session.add(PromptVersionEntry(
                        role=role,
                        version="v1",
                        prompt_text=prompt,
                        cot_enhanced=False,
                        rationale="Initial seed from AGENT_DEFINITIONS",
                        created_by="seed",
                        is_active=True,
                    ))
                    inserted += 1
            session.commit()
        finally:
            session.close()
        return inserted

    def seed_cot_v2(self, roles: list[str]) -> int:
        """
        Idempotently create v2 CoT-enhanced prompts for the given roles
        by injecting the COT_HEADER into each role's v1 prompt.
        Returns count inserted.
        """
        from src.db.models import PromptVersionEntry
        session = self._session()
        inserted = 0
        try:
            for role in roles:
                exists_v2 = (
                    session.query(PromptVersionEntry)
                    .filter_by(role=role, version="v2")
                    .first()
                )
                if exists_v2:
                    continue
                v1 = (
                    session.query(PromptVersionEntry)
                    .filter_by(role=role, version="v1")
                    .first()
                )
                if not v1:
                    continue
                cot_prompt = _inject_cot(v1.prompt_text)
                session.add(PromptVersionEntry(
                    role=role,
                    version="v2",
                    prompt_text=cot_prompt,
                    cot_enhanced=True,
                    parent_version="v1",
                    rationale=(
                        "CoT v2: injected explicit SITUATION/PLAN/EXECUTE/VERIFY reasoning "
                        "protocol to improve step_efficiency and tool_usage metrics."
                    ),
                    created_by="seed",
                    is_active=True,
                ))
                inserted += 1
            session.commit()
        finally:
            session.close()
        return inserted


    def seed_routing_prompts(self, supervisor_prompt: str, meta_router_prompt: str, router_prompt: str) -> int:
        """Idempotently seed v1 routing prompts for supervisor, meta_router, and router roles.

        These three roles control orchestration strategy selection and agent routing.
        Versioning them enables A/B testing routing quality and prompt optimization.
        """
        from src.db.models import PromptVersionEntry
        session = self._session()
        inserted = 0
        routing_prompts = {
            "supervisor": (supervisor_prompt, "Initial supervisor orchestration prompt — routes tasks between specialized agents using ReAct step tracking"),
            "meta_router": (meta_router_prompt, "Initial meta-router prompt — selects orchestration strategy (router_decides/sequential/parallel/supervisor)"),
            "router": (router_prompt, "Initial single-agent router prompt — routes a task to exactly one specialized agent"),
        }
        try:
            for role, (prompt_text, rationale) in routing_prompts.items():
                exists = (
                    session.query(PromptVersionEntry)
                    .filter_by(role=role, version="v1")
                    .first()
                )
                if not exists:
                    session.add(PromptVersionEntry(
                        role=role,
                        version="v1",
                        prompt_text=prompt_text,
                        cot_enhanced=False,
                        rationale=rationale,
                        created_by="seed",
                        is_active=True,
                        notes="routing_prompt",
                    ))
                    inserted += 1
            session.commit()
        finally:
            session.close()
        return inserted

    def get_routing_prompt_versions(self) -> dict[str, str]:
        """Return the latest active version label for each routing role."""
        return {
            role: self.latest_version(role)
            for role in ("supervisor", "meta_router", "router")
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_registry: Optional[PromptRegistry] = None


def get_registry() -> PromptRegistry:
    """Return the module-level PromptRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
