"""
Golden dataset management.

Loads curated test cases from golden_dataset.json (git-tracked source of truth)
and syncs them to the GoldenTestCase DB table for UI access.
"""

import json
import os

from src.db.database import get_session
from src.db.models import GoldenTestCase


GOLDEN_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")


def load_golden_dataset() -> list[dict]:
    """Read golden dataset from JSON file."""
    if not os.path.exists(GOLDEN_DATASET_PATH):
        return []
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def sync_golden_to_db():
    """Upsert golden dataset from JSON into the database.

    Cases removed from the JSON (e.g. via v5 consolidation) are deactivated —
    their DB rows remain for historical regression results to reference, but
    they won't show up in `get_active_cases()` / the Studio picker.
    """
    cases = load_golden_dataset()
    if not cases:
        return

    json_ids = {c["id"] for c in cases}

    session = get_session()
    try:
        # Deactivate rows for ids no longer present in the JSON source of truth.
        stale = (
            session.query(GoldenTestCase)
            .filter(GoldenTestCase.is_active.is_(True))
            .filter(~GoldenTestCase.id.in_(json_ids))
            .all()
        )
        for row in stale:
            row.is_active = False
        for c in cases:
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
                    is_active=True,
                ))
        session.commit()
    finally:
        session.close()


def save_golden_to_json(cases: list[dict]):
    """Write golden dataset back to the JSON file."""
    with open(GOLDEN_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)


def get_active_cases(case_ids: list[str] = None) -> list[GoldenTestCase]:
    """Return active golden cases, optionally filtered by IDs."""
    session = get_session()
    try:
        q = session.query(GoldenTestCase).filter_by(is_active=True)
        if case_ids:
            q = q.filter(GoldenTestCase.id.in_(case_ids))
        cases = q.all()
        session.expunge_all()
        return cases
    finally:
        session.close()
