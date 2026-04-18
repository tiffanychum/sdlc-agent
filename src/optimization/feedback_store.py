"""
FeedbackStore — semantic archive of (role, version, tool_trace, scores) triples.

Every regression run is archived here so the PromptOptimizer agent can retrieve
semantically similar past failures as few-shot context when crafting improved prompts.

Storage: SQLite (PromptFeedback rows) + ChromaDB (for semantic retrieval).
The ChromaDB collection is populated lazily on first store/retrieve call.
"""

from __future__ import annotations

import json
import uuid
from typing import Optional


_CHROMA_COLLECTION = "prompt_feedback"


def _classify_failures(quality_scores: dict, deepeval_scores: dict) -> list[str]:
    """
    Heuristically label failure types based on metric scores.

    Returns a list of human-readable failure type strings used for both
    DB storage and ChromaDB document metadata.
    """
    thresholds = {
        "step_efficiency": 0.7,
        "tool_usage": 0.7,
        "correctness": 3.5,
        "completeness": 3.5,
        "coherence": 3.5,
        "faithfulness": 0.7,
        "answer_relevancy": 0.7,
    }
    failures = []
    combined = {**quality_scores, **deepeval_scores}
    for metric, threshold in thresholds.items():
        score = combined.get(metric)
        if score is not None and score < threshold:
            failures.append(f"low_{metric}")
    return failures


def _build_embedding_text(
    role: str,
    version: str,
    tool_trace_summary: str,
    failure_types: list[str],
    quality_scores: dict,
) -> str:
    """Build the text document that will be embedded in ChromaDB."""
    scores_str = ", ".join(
        f"{k}={v:.2f}" for k, v in quality_scores.items() if v is not None
    )
    failures_str = ", ".join(failure_types) or "none"
    return (
        f"role={role} version={version} "
        f"failures=[{failures_str}] scores=[{scores_str}] "
        f"tool_trace: {tool_trace_summary}"
    )


def _get_chroma_collection():
    """Return (or create) the ChromaDB feedback collection."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=".chroma_feedback")
        return client.get_or_create_collection(
            _CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        return None


class FeedbackStore:
    """
    Archive of prompt-run triples with semantic retrieval.

    The store is populated automatically by the RegressionRunner after
    each test run (see regression.py integration). The PromptOptimizer
    agent calls retrieve_similar_failures() as a tool to get few-shot
    context before generating improved prompts.
    """

    def store_run(
        self,
        role: str,
        prompt_version: str,
        golden_id: str,
        tool_trace: list[dict],
        quality_scores: dict,
        deepeval_scores: dict,
        overall_pass: bool,
    ) -> str:
        """
        Archive one regression run result. Returns the new feedback entry ID.

        Args:
            role: Agent role (e.g. "coder").
            prompt_version: Prompt version used (e.g. "v1").
            golden_id: Golden test case ID (e.g. "golden_004").
            tool_trace: List of tool call dicts from agent_trace.
            quality_scores: G-Eval metric scores dict.
            deepeval_scores: DeepEval metric scores dict.
            overall_pass: Whether the test passed overall.
        """
        from src.db.database import get_session
        from src.db.models import PromptFeedback

        # Build compact tool trace summary (tool names + count)
        tool_counts: dict[str, int] = {}
        for tc in tool_trace:
            tool_counts[tc.get("tool", "unknown")] = tool_counts.get(tc.get("tool", "unknown"), 0) + 1
        trace_summary = " → ".join(
            f"{t}×{n}" if n > 1 else t for t, n in tool_counts.items()
        ) or "no tools"

        failure_types = _classify_failures(quality_scores, deepeval_scores)
        embedding_text = _build_embedding_text(
            role, prompt_version, trace_summary, failure_types, quality_scores
        )

        entry_id = uuid.uuid4().hex[:12]

        # Persist to SQLite
        session = get_session()
        try:
            fb = PromptFeedback(
                id=entry_id,
                role=role,
                prompt_version=prompt_version,
                golden_id=golden_id,
                tool_trace_summary=trace_summary,
                quality_scores=quality_scores,
                deepeval_scores=deepeval_scores,
                failure_types=failure_types,
                overall_pass=overall_pass,
                embedding_text=embedding_text,
            )
            session.add(fb)
            session.commit()
        finally:
            session.close()

        # Index in ChromaDB for semantic retrieval
        col = _get_chroma_collection()
        if col:
            try:
                col.add(
                    ids=[entry_id],
                    documents=[embedding_text],
                    metadatas=[{
                        "role": role,
                        "version": prompt_version,
                        "golden_id": golden_id,
                        "overall_pass": str(overall_pass),
                        "failures": json.dumps(failure_types),
                    }],
                )
            except Exception:
                pass  # ChromaDB is best-effort; SQLite is the source of truth

        return entry_id

    def retrieve_similar_failures(
        self,
        role: str,
        query_text: str,
        limit: int = 5,
        only_failures: bool = True,
    ) -> list[dict]:
        """
        Retrieve semantically similar past failures for a given role.

        Falls back to SQL-based retrieval if ChromaDB is unavailable.

        Args:
            role: Filter to this agent role.
            query_text: Natural-language description of the current failure.
            limit: Max results to return.
            only_failures: If True, filter to runs that did not pass.

        Returns:
            List of dicts with keys: role, version, golden_id, tool_trace_summary,
            quality_scores, failure_types, overall_pass.
        """
        # Try ChromaDB semantic search first
        col = _get_chroma_collection()
        if col:
            try:
                where: dict = {"role": {"$eq": role}}
                if only_failures:
                    where["overall_pass"] = {"$eq": "False"}
                results = col.query(
                    query_texts=[query_text],
                    n_results=min(limit, 10),
                    where=where,
                )
                ids = results.get("ids", [[]])[0]
                if ids:
                    return self._fetch_by_ids(ids)
            except Exception:
                pass

        # SQL fallback
        return self._fetch_recent_failures(role, limit=limit, only_failures=only_failures)

    def _fetch_by_ids(self, ids: list[str]) -> list[dict]:
        from src.db.database import get_session
        from src.db.models import PromptFeedback
        session = get_session()
        try:
            rows = (
                session.query(PromptFeedback)
                .filter(PromptFeedback.id.in_(ids))
                .all()
            )
            return [self._row_to_dict(r) for r in rows]
        finally:
            session.close()

    def _fetch_recent_failures(
        self, role: str, limit: int = 5, only_failures: bool = True
    ) -> list[dict]:
        from src.db.database import get_session
        from src.db.models import PromptFeedback
        session = get_session()
        try:
            q = session.query(PromptFeedback).filter_by(role=role)
            if only_failures:
                q = q.filter_by(overall_pass=False)
            rows = q.order_by(PromptFeedback.created_at.desc()).limit(limit).all()
            return [self._row_to_dict(r) for r in rows]
        finally:
            session.close()

    @staticmethod
    def _row_to_dict(r) -> dict:
        return {
            "id": r.id,
            "role": r.role,
            "version": r.prompt_version,
            "golden_id": r.golden_id,
            "tool_trace_summary": r.tool_trace_summary,
            "quality_scores": r.quality_scores or {},
            "deepeval_scores": r.deepeval_scores or {},
            "failure_types": r.failure_types or [],
            "overall_pass": r.overall_pass,
            "created_at": r.created_at.isoformat() if r.created_at else "",
        }

    def recent_failure_summary(self, role: str, version: str, limit: int = 20) -> str:
        """
        Return a compact plain-text summary of recent failures for a role+version.
        Used as context in the optimizer's meta-prompt.
        """
        failures = self._fetch_recent_failures(role, limit=limit)
        if not failures:
            return f"No failure records found for role={role}."

        lines = [f"Recent failures for role={role} version={version} (last {len(failures)}):\n"]
        for fb in failures:
            scores = ", ".join(
                f"{k}={v:.2f}" for k, v in fb["quality_scores"].items() if v is not None
            )
            lines.append(
                f"  [{fb['golden_id']}] tools={fb['tool_trace_summary']} "
                f"failures={fb['failure_types']} scores=[{scores}]"
            )
        return "\n".join(lines)


# Module-level singleton
_store: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    global _store
    if _store is None:
        _store = FeedbackStore()
    return _store
