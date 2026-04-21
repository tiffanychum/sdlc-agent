"""One-shot cleanup for the routing-prompt rows in PromptRegistry.

Run this once after pulling Patch 1 to:
  1. Hard-delete the router v2..v40 rows that accumulated from the previous
     runtime `sync_routing_prompts` drift bug.
  2. Delete the degenerate `supervisor` rows (the literal string "supervisor"
     and the mis-seeded meta-router template copy).
  3. Re-seed clean v1 rows for supervisor / meta_router / router using the
     canonical templates in src/orchestrator.py (with {agent_descs} /
     {agent_names} placeholders intact).

This is idempotent — running it twice is safe.  After Patch 5 lands, a
per-team seeding step will happen automatically on first orchestrator build;
this script only restores the global-row baseline.

Usage:
    python -m scripts.cleanup_routing_prompts
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.database import get_session  # noqa: E402
from src.db.models import PromptVersionEntry  # noqa: E402
from src.orchestrator import (  # noqa: E402
    _META_ROUTER_PROMPT,
    _ROUTER_PROMPT_TEMPLATE,
    _SUPERVISOR_PROMPT_DEV_TEMPLATE,
)
from src.prompts.registry import get_registry  # noqa: E402


def run() -> None:
    session = get_session()
    deleted_router = 0
    deleted_supervisor = 0

    try:
        # ── 1. Hard-delete all stale router rows ──────────────────────────────
        # The old code stored RENDERED router text (dev-team roles baked in)
        # rather than the template with {agent_descs} / {agent_names}
        # placeholders.  Any row without those placeholders is stale and must
        # be removed so the re-seed below can insert a clean template v1.
        router_rows = (
            session.query(PromptVersionEntry)
            .filter_by(role="router")
            .all()
        )
        for r in router_rows:
            txt = r.prompt_text or ""
            has_placeholders = "{agent_descs}" in txt and "{agent_names}" in txt
            # Delete:
            #   - any label other than v1 (team-switch artefacts v2..v40), OR
            #   - v1 if its text is the old rendered form (no placeholders)
            if r.version != "v1" or not has_placeholders:
                session.delete(r)
                deleted_router += 1

        # ── 2. Delete degenerate supervisor rows ──────────────────────────────
        sup_rows = (
            session.query(PromptVersionEntry)
            .filter_by(role="supervisor")
            .all()
        )
        for r in sup_rows:
            txt = (r.prompt_text or "").strip()
            is_degenerate = txt == "supervisor"
            is_mis_seeded_meta_router = txt.startswith(
                "You are an orchestration meta-router"
            )
            if is_degenerate or is_mis_seeded_meta_router:
                session.delete(r)
                deleted_supervisor += 1

        session.commit()
    finally:
        session.close()

    # ── 3. Re-seed clean v1 rows (idempotent) ─────────────────────────────────
    reg = get_registry()
    inserted = reg.seed_routing_prompts(
        supervisor_prompt=_SUPERVISOR_PROMPT_DEV_TEMPLATE,
        meta_router_prompt=_META_ROUTER_PROMPT,
        router_prompt=_ROUTER_PROMPT_TEMPLATE,
    )

    # ── 4. Report ─────────────────────────────────────────────────────────────
    print("cleanup_routing_prompts.py — done")
    print(f"  router rows deleted     : {deleted_router}")
    print(f"  supervisor rows deleted : {deleted_supervisor}")
    print(f"  v1 rows re-seeded       : {inserted}")

    # Verify end-state
    session = get_session()
    try:
        for role in ("supervisor", "meta_router", "router"):
            rows = (
                session.query(PromptVersionEntry)
                .filter_by(role=role)
                .order_by(PromptVersionEntry.created_at.desc())
                .all()
            )
            print(f"  {role:12s}: {len(rows)} row(s)  versions={[r.version for r in rows]}")
    finally:
        session.close()


if __name__ == "__main__":
    run()
