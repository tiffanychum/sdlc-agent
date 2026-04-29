"""
Golden dataset management.

Loads curated test cases from golden_dataset.json (git-tracked source of truth)
and syncs them to the GoldenTestCase DB table for UI access.

Multiple dataset groups (e.g. "sdlc_v2", "finance_v1") are supported via the
``GoldenTestCase.dataset_group`` column. Cases without an explicit group are
assigned to ``"default"`` for backward compatibility — the SDLC team continues
to see everything as before, while new teams can subscribe to specific groups
via ``Team.config_json["dataset_groups"]``.
"""

import json
import os

from src.db.database import get_session
from src.db.models import GoldenTestCase


GOLDEN_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
# Additional named JSON files merged into the same DB table, each tagged with
# their own dataset_group. Keeps the original file untouched while allowing
# brand-new groups (e.g. finance_v1) to live in their own JSON.
EXTRA_DATASET_PATHS: list[tuple[str, str]] = [
    (
        os.path.join(os.path.dirname(__file__), "golden_dataset_finance.json"),
        "finance_v1",
    ),
]


def load_golden_dataset() -> list[dict]:
    """Read the canonical SDLC golden dataset (default group)."""
    if not os.path.exists(GOLDEN_DATASET_PATH):
        return []
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_all_grouped_datasets() -> list[tuple[dict, str]]:
    """Yield ``(case_dict, dataset_group)`` for every JSON the system knows about.

    The canonical SDLC file lives in the ``"sdlc_v2"`` group unless a case
    overrides it. Files in ``EXTRA_DATASET_PATHS`` are forced to their
    declared group. Teams can subscribe to specific groups via
    ``Team.config_json["dataset_groups"]``; teams that don't configure any
    groups continue to see every case (legacy behaviour).
    """
    out: list[tuple[dict, str]] = []
    for case in load_golden_dataset():
        out.append((case, case.get("dataset_group") or "sdlc_v2"))
    for path, group_name in EXTRA_DATASET_PATHS:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                rows = json.load(f)
            for case in rows:
                out.append((case, case.get("dataset_group") or group_name))
        except Exception:
            # Best-effort load — broken extra files don't break the SDLC dataset
            continue
    return out


def sync_golden_to_db():
    """Upsert all known golden datasets into the DB.

    Cases removed from the JSON (e.g. via v5 consolidation) are deactivated —
    their DB rows remain for historical regression results to reference, but
    they won't show up in ``get_active_cases()`` / the Studio picker.
    """
    grouped = load_all_grouped_datasets()
    if not grouped:
        return

    json_ids = {c["id"] for c, _ in grouped}

    session = get_session()
    try:
        # Deactivate rows for ids no longer present in the JSON sources of truth.
        stale = (
            session.query(GoldenTestCase)
            .filter(GoldenTestCase.is_active.is_(True))
            .filter(~GoldenTestCase.id.in_(json_ids))
            .all()
        )
        for row in stale:
            row.is_active = False
        for c, dataset_group in grouped:
            existing = session.query(GoldenTestCase).filter_by(id=c["id"]).first()
            if existing:
                existing.name = c["name"]
                existing.prompt = c["prompt"]
                existing.expected_agent = c.get("expected_agent", "")
                existing.expected_tools = c.get("expected_tools", [])
                existing.expected_output_keywords = c.get("expected_output_keywords", [])
                existing.expected_delegation_pattern = c.get("expected_delegation_pattern", [])
                existing.quality_thresholds = c.get("quality_thresholds", {})
                existing.max_llm_calls = c.get("max_llm_calls", 15)
                existing.max_tool_calls = c.get("max_tool_calls", 10)
                existing.max_tokens = c.get("max_tokens", 8000)
                existing.max_latency_ms = c.get("max_latency_ms", 120000)
                existing.complexity = c.get("complexity", "quick")
                existing.version = c.get("version", "1.0")
                existing.reference_output = c.get("reference_output", "")
                existing.strategy = c.get("strategy", None)
                existing.expected_strategy = c.get("expected_strategy", None)
                existing.dataset_group = dataset_group
                existing.is_active = True
            else:
                session.add(GoldenTestCase(
                    id=c["id"],
                    name=c["name"],
                    prompt=c["prompt"],
                    expected_agent=c.get("expected_agent", ""),
                    expected_tools=c.get("expected_tools", []),
                    expected_output_keywords=c.get("expected_output_keywords", []),
                    expected_delegation_pattern=c.get("expected_delegation_pattern", []),
                    quality_thresholds=c.get("quality_thresholds", {}),
                    max_llm_calls=c.get("max_llm_calls", 15),
                    max_tool_calls=c.get("max_tool_calls", 10),
                    max_tokens=c.get("max_tokens", 8000),
                    max_latency_ms=c.get("max_latency_ms", 120000),
                    complexity=c.get("complexity", "quick"),
                    version=c.get("version", "1.0"),
                    reference_output=c.get("reference_output", ""),
                    strategy=c.get("strategy", None),
                    expected_strategy=c.get("expected_strategy", None),
                    dataset_group=dataset_group,
                    is_active=True,
                ))
        session.commit()
    finally:
        session.close()


def save_golden_to_json(cases: list[dict]):
    """Write golden dataset back to the JSON file."""
    with open(GOLDEN_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)


def get_active_cases(
    case_ids: list[str] = None,
    dataset_groups: list[str] | None = None,
) -> list[GoldenTestCase]:
    """Return active golden cases, optionally filtered by ids and dataset groups.

    ``dataset_groups`` works as a logical OR: a case matches if its
    ``dataset_group`` is in the list. ``None`` means "all groups" (legacy
    behaviour, fully backward-compatible).
    """
    session = get_session()
    try:
        q = session.query(GoldenTestCase).filter_by(is_active=True)
        if case_ids:
            q = q.filter(GoldenTestCase.id.in_(case_ids))
        if dataset_groups:
            q = q.filter(GoldenTestCase.dataset_group.in_(list(dataset_groups)))
        cases = q.all()
        session.expunge_all()
        return cases
    finally:
        session.close()
