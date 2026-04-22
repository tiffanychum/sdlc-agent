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
    #
    # Scoping semantics (Patch 5):
    #   team_id=None  → read ONLY global rows (team_id IS NULL in DB).
    #   team_id="x"   → read team-scoped rows; fall back to global when the
    #                   team has no row for the requested (role[, version]).
    #
    # The fallback path is important because:
    #   - Agent-role prompts (coder, qa, …) are stored globally today.  Team A
    #     asking for "coder latest" must still get the global row.
    #   - A brand-new team created before its first orchestrator build has no
    #     routing rows yet; the UI should still be able to render the shared
    #     global template until the first build seeds per-team rows.

    def _role_query(self, session, role: str, team_id: Optional[str]):
        """Return a query scoped to (role, team_id).

        When team_id is a concrete string, uses equality.  When None, matches
        rows where team_id IS NULL (SQL equality with NULL would never match).
        """
        from src.db.models import PromptVersionEntry
        q = session.query(PromptVersionEntry).filter(PromptVersionEntry.role == role)
        if team_id is None:
            q = q.filter(PromptVersionEntry.team_id.is_(None))
        else:
            q = q.filter(PromptVersionEntry.team_id == team_id)
        return q

    def get_prompt(
        self,
        role: str,
        version: str = "latest",
        team_id: Optional[str] = None,
    ) -> Optional[str]:
        """Return the prompt text for (role, version[, team_id]).

        'latest' = highest active version.  When team_id is given, tries the
        team-scoped row first and falls back to the global row if the team has
        none.
        """
        from src.db.models import PromptVersionEntry
        session = self._session()
        try:
            def _lookup(scope_team_id: Optional[str]):
                q = self._role_query(session, role, scope_team_id).filter(
                    PromptVersionEntry.is_active.is_(True)
                )
                if version == "latest":
                    return q.order_by(PromptVersionEntry.created_at.desc()).first()
                return q.filter(PromptVersionEntry.version == version).first()

            entry = _lookup(team_id)
            if entry is None and team_id is not None:
                entry = _lookup(None)
            return entry.prompt_text if entry else None
        finally:
            session.close()

    def latest_version(self, role: str, team_id: Optional[str] = None) -> str:
        """Return the latest active version label for (role[, team_id])."""
        from src.db.models import PromptVersionEntry
        session = self._session()
        try:
            def _lookup(scope_team_id: Optional[str]):
                q = self._role_query(session, role, scope_team_id).filter(
                    PromptVersionEntry.is_active.is_(True)
                )
                return q.order_by(PromptVersionEntry.created_at.desc()).first()

            entry = _lookup(team_id)
            if entry is None and team_id is not None:
                entry = _lookup(None)
            return entry.version if entry else "v1"
        finally:
            session.close()

    def list_versions(self, role: str, team_id: Optional[str] = None) -> list[dict]:
        """Return all version entries for (role[, team_id]), newest first.

        When team_id is given and the team has no rows for this role, falls
        back to the global rows so the UI shows something meaningful.
        """
        from src.db.models import PromptVersionEntry
        session = self._session()
        try:
            entries = (
                self._role_query(session, role, team_id)
                .order_by(PromptVersionEntry.created_at.desc())
                .all()
            )
            if not entries and team_id is not None:
                entries = (
                    self._role_query(session, role, None)
                    .order_by(PromptVersionEntry.created_at.desc())
                    .all()
                )
            return [
                {
                    "role": e.role,
                    "team_id": e.team_id,
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

    def list_all_roles(self, team_id: Optional[str] = None) -> list[str]:
        """Return all roles that have at least one registered version.

        When team_id is given, returns the union of:
          - roles with a team-scoped row for that team, and
          - roles that exist globally.
        Callers typically want the union because team-scoped and global rows
        coexist (e.g. supervisor is team-scoped, coder is still global).
        """
        from src.db.models import PromptVersionEntry
        from sqlalchemy import distinct, or_
        session = self._session()
        try:
            q = session.query(distinct(PromptVersionEntry.role))
            if team_id is not None:
                q = q.filter(
                    or_(
                        PromptVersionEntry.team_id == team_id,
                        PromptVersionEntry.team_id.is_(None),
                    )
                )
            rows = q.all()
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
        team_id: Optional[str] = None,
    ) -> str:
        """Register a new prompt version for (role[, team_id]).

        Version auto-increment is per-scope: team A's router v2 and team B's
        router v2 are independent rows.  A global (team_id=None) row shares
        the version counter with nobody.
        """
        from src.db.models import PromptVersionEntry
        session = self._session()
        try:
            existing = (
                self._role_query(session, role, team_id)
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
                team_id=team_id,
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

    def update_metric_scores(
        self,
        role: str,
        version: str,
        scores: dict,
        team_id: Optional[str] = None,
    ) -> None:
        """Store measured metric scores against a prompt version."""
        from src.db.models import PromptVersionEntry
        session = self._session()
        try:
            entry = (
                self._role_query(session, role, team_id)
                .filter(PromptVersionEntry.version == version)
                .first()
            )
            if entry:
                entry.metric_scores = scores
                session.commit()
        finally:
            session.close()

    def deactivate_version(
        self,
        role: str,
        version: str,
        team_id: Optional[str] = None,
    ) -> None:
        """Mark a version inactive (does not delete it)."""
        from src.db.models import PromptVersionEntry
        session = self._session()
        try:
            entry = (
                self._role_query(session, role, team_id)
                .filter(PromptVersionEntry.version == version)
                .first()
            )
            if entry:
                entry.is_active = False
                session.commit()
        finally:
            session.close()

    # ── Seeding ───────────────────────────────────────────────────────────────

    def sync_from_definitions(self, agent_definitions: list[dict]) -> list[tuple[str, str]]:
        """
        For each role in AGENT_DEFINITIONS, compare the CURRENT code-level prompt to the
        latest ACTIVE version in the registry. If they differ, register a new version so
        `get_prompt(role, "latest")` always serves the current code.

        Returns a list of (role, new_version) for versions that were newly registered.
        """
        from src.db.models import PromptVersionEntry
        session = self._session()
        bumps: list[tuple[str, str]] = []
        try:
            for defn in agent_definitions:
                role = defn.get("role") or defn.get("id")
                current_prompt = (defn.get("prompt") or "").strip()
                if not role or not current_prompt:
                    continue
                latest = (
                    session.query(PromptVersionEntry)
                    .filter_by(role=role, is_active=True)
                    .order_by(PromptVersionEntry.created_at.desc())
                    .first()
                )
                if latest and (latest.prompt_text or "").strip() == current_prompt:
                    continue  # already in sync
                bumps.append((role, "pending"))
            session.close()
        except Exception:
            session.close()
            return bumps

        # Register new versions outside the read session
        for i, (role, _) in enumerate(bumps):
            current_prompt = next(
                (d.get("prompt", "") for d in agent_definitions if (d.get("role") or d.get("id")) == role),
                "",
            )
            new_ver = self.register(
                role=role,
                prompt_text=current_prompt,
                rationale="Auto-sync from AGENT_DEFINITIONS (code prompt drifted from DB).",
                created_by="sync",
                cot_enhanced=False,
            )
            bumps[i] = (role, new_ver)
        return bumps

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


    def seed_routing_prompts(
        self,
        supervisor_prompt: str,
        meta_router_prompt: str,
        router_prompt: str,
        team_id: Optional[str] = None,
    ) -> int:
        """Idempotently seed v1 routing prompts for supervisor, meta_router, and router.

        Patch 5 semantics:
          - team_id=None   → seed the *global* v1 row (one per role).  Used by
            the one-shot cleanup script / legacy flows.
          - team_id="x"    → seed *team x's* v1 row for supervisor and router.
            meta_router stays global and is skipped when team_id is set (the
            caller is expected to use the shared global template).

        Inserts are idempotent: if a row already exists at (role, team_id, v1),
        it is left alone.
        """
        from src.db.models import PromptVersionEntry
        session = self._session()
        inserted = 0
        if team_id is None:
            routing_prompts = {
                "supervisor": (supervisor_prompt, "Initial supervisor orchestration prompt — routes tasks between specialized agents using ReAct step tracking"),
                "meta_router": (meta_router_prompt, "Initial meta-router prompt — selects orchestration strategy (router_decides/sequential/parallel/supervisor)"),
                "router": (router_prompt, "Initial single-agent router prompt — routes a task to exactly one specialized agent"),
            }
        else:
            # Per-team seed: only supervisor + router are team-scoped.
            # meta_router remains a global shared row.
            routing_prompts = {
                "supervisor": (supervisor_prompt, f"Initial supervisor prompt for team {team_id}"),
                "router": (router_prompt, f"Initial router prompt for team {team_id}"),
            }
        try:
            for role, (prompt_text, rationale) in routing_prompts.items():
                q = session.query(PromptVersionEntry).filter(
                    PromptVersionEntry.role == role,
                    PromptVersionEntry.version == "v1",
                )
                if team_id is None:
                    q = q.filter(PromptVersionEntry.team_id.is_(None))
                else:
                    q = q.filter(PromptVersionEntry.team_id == team_id)
                exists = q.first()
                if not exists:
                    session.add(PromptVersionEntry(
                        role=role,
                        team_id=team_id,
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

    def sync_routing_prompts(
        self,
        supervisor_prompt: str,
        meta_router_prompt: str,
        router_prompt: str,
        team_id: Optional[str] = None,
    ) -> list[tuple[str, str]]:
        """Drift detection for the three orchestration roles.

        Compares the latest stored prompt to the provided text and registers a
        new version if they differ.  Respects team_id the same way as the read
        path (meta_router is always compared against the global row).

        NOTE: This is no longer called at orchestrator build time — it was the
        source of the router v2..v40 pollution (rendered-vs-template diff).
        Kept for admin tooling / cleanup scripts that want explicit drift
        detection.
        """
        pairs = {
            "supervisor": (supervisor_prompt, team_id),
            "meta_router": (meta_router_prompt, None),  # always global
            "router": (router_prompt, team_id),
        }
        bumps: list[tuple[str, str]] = []
        for role, (current_prompt, scope_tid) in pairs.items():
            current_prompt = (current_prompt or "").strip()
            if not current_prompt:
                continue
            latest_text = self.get_prompt(role, "latest", team_id=scope_tid) or ""
            if latest_text.strip() == current_prompt:
                continue
            new_ver = self.register(
                role=role,
                prompt_text=current_prompt,
                rationale="Auto-sync routing prompt: code drifted from DB.",
                created_by="sync",
                cot_enhanced=False,
                notes="routing_prompt",
                team_id=scope_tid,
            )
            bumps.append((role, new_ver))
        return bumps

    def get_routing_prompt_versions(self, team_id: Optional[str] = None) -> dict[str, str]:
        """Return the latest active version label for each routing role.

        supervisor + router resolve under team_id (with global fallback);
        meta_router always resolves globally because it's a single shared row.
        """
        return {
            "supervisor": self.latest_version("supervisor", team_id=team_id),
            "meta_router": self.latest_version("meta_router", team_id=None),
            "router": self.latest_version("router", team_id=team_id),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_registry: Optional[PromptRegistry] = None


def get_registry() -> PromptRegistry:
    """Return the module-level PromptRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
