"""
RagClient — programmatic CRUD + ingestion for RAG configs.

Mirrors the FastAPI RAG routes: create / list / update RAG configs, ingest
documents (single or batch directory), retrieve, and query. All ops are
team-scoped via ``RagConfig.team_id``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Iterable, Optional

from src.db.database import get_session
from src.db.models import RagConfig, RagSource
from src.rag.pipeline import RAGConfig, get_pipeline


class RagClient:
    def __init__(self, team_id: str) -> None:
        self.team_id = team_id

    # ── Config CRUD ──────────────────────────────────────────────────

    def list_configs(self) -> list[dict[str, Any]]:
        session = get_session()
        try:
            rows = (
                session.query(RagConfig)
                .filter((RagConfig.team_id == self.team_id) | (RagConfig.team_id.is_(None)))
                .filter_by(is_active=True)
                .order_by(RagConfig.created_at.asc())
                .all()
            )
            return [self._cfg_to_dict(r) for r in rows]
        finally:
            session.close()

    def create_config(
        self,
        *,
        name: str,
        description: str = "",
        embedding_model: str = "openai/text-embedding-3-small",
        vector_store: str = "chroma",
        llm_model: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunk_strategy: str = "recursive",
        retrieval_strategy: str = "similarity",
        top_k: int = 5,
        mmr_lambda: float = 0.5,
        multi_query_n: int = 3,
        system_prompt: Optional[str] = None,
        reranker: str = "none",
        config_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create or upsert a RAG config bound to this team. Returns the row."""
        session = get_session()
        try:
            existing = None
            if config_id:
                existing = session.query(RagConfig).filter_by(id=config_id).one_or_none()
            if existing is None:
                row = RagConfig(
                    id=config_id,
                    name=name,
                    description=description,
                    embedding_model=embedding_model,
                    vector_store=vector_store,
                    llm_model=llm_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chunk_strategy=chunk_strategy,
                    retrieval_strategy=retrieval_strategy,
                    top_k=top_k,
                    mmr_lambda=mmr_lambda,
                    multi_query_n=multi_query_n,
                    system_prompt=system_prompt,
                    reranker=reranker,
                    team_id=self.team_id,
                )
                session.add(row)
            else:
                row = existing
                row.name = name
                row.description = description
                row.embedding_model = embedding_model
                row.vector_store = vector_store
                row.llm_model = llm_model
                row.chunk_size = chunk_size
                row.chunk_overlap = chunk_overlap
                row.chunk_strategy = chunk_strategy
                row.retrieval_strategy = retrieval_strategy
                row.top_k = top_k
                row.mmr_lambda = mmr_lambda
                row.multi_query_n = multi_query_n
                row.system_prompt = system_prompt
                row.reranker = reranker
                row.team_id = self.team_id
                row.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(row)
            return self._cfg_to_dict(row)
        finally:
            session.close()

    def list_sources(self, config_id: str) -> list[dict[str, Any]]:
        session = get_session()
        try:
            rows = (
                session.query(RagSource)
                .filter_by(config_id=config_id)
                .order_by(RagSource.created_at.asc())
                .all()
            )
            return [
                {
                    "id": r.id,
                    "source_type": r.source_type,
                    "content": r.content,
                    "label": r.label,
                    "chunks_count": r.chunks_count,
                    "tokens_estimated": r.tokens_estimated,
                    "status": r.status,
                    "error_message": r.error_message,
                    "ingested_at": r.ingested_at.isoformat() if r.ingested_at else None,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
        finally:
            session.close()

    # ── Ingest / retrieve / query ────────────────────────────────────

    def ingest_text(self, config_id: str, *, text: str, label: str = "text") -> dict[str, Any]:
        return asyncio.run(self._aingest(config_id, "text", text, label=label))

    def ingest_file(self, config_id: str, *, path: str, label: str = "") -> dict[str, Any]:
        return asyncio.run(self._aingest(config_id, "file", path, label=label or path))

    def ingest_url(self, config_id: str, *, url: str, label: str = "") -> dict[str, Any]:
        return asyncio.run(self._aingest(config_id, "url", url, label=label or url))

    def ingest_paths(
        self,
        config_id: str,
        *,
        paths: Iterable[str],
        recursive: bool = True,
        extensions: tuple[str, ...] = (".md", ".txt", ".rst", ".json", ".csv", ".pdf"),
    ) -> dict[str, Any]:
        """Batch-ingest a list of file/dir paths via ``RAGPipeline.ingest_paths``.

        Each ingested file is also recorded as a RagSource row so the UI's
        "Sources" panel reflects the directory ingestion.
        """
        async def _go() -> dict[str, Any]:
            cfg_dict = self._load_config_dict(config_id)
            pipeline = get_pipeline(self._dict_to_pipeline_config(cfg_dict))
            summary = await pipeline.ingest_paths(
                list(paths), recursive=recursive, extensions=extensions,
            )
            # Record one RagSource per directory root, summarised, so the UI
            # doesn't drown in 500 individual rows.
            session = get_session()
            try:
                for p in paths:
                    session.add(RagSource(
                        config_id=config_id,
                        source_type="batch",
                        content=str(p),
                        label=f"batch:{p}",
                        chunks_count=summary.get("total_chunks", 0),
                        status="ingested",
                        ingested_at=datetime.utcnow(),
                    ))
                session.commit()
            finally:
                session.close()
            return summary
        return asyncio.run(_go())

    def retrieve(self, config_id: str, *, query: str) -> list[dict[str, Any]]:
        async def _go():
            cfg_dict = self._load_config_dict(config_id)
            pipeline = get_pipeline(self._dict_to_pipeline_config(cfg_dict))
            results = await pipeline.retrieve(query)
            return [
                {
                    "source": r.source,
                    "chunk_index": r.chunk_index,
                    "total_chunks": r.total_chunks,
                    "page": r.page,
                    "score": r.score,
                    "snippet": (r.text or "")[:400],
                }
                for r in results
            ]
        return asyncio.run(_go())

    def query(
        self,
        config_id: str,
        *,
        query: str,
        use_compression: bool = False,
    ) -> dict[str, Any]:
        async def _go():
            cfg_dict = self._load_config_dict(config_id)
            pipeline = get_pipeline(self._dict_to_pipeline_config(cfg_dict))
            response = await pipeline.query(query, use_compression=use_compression)
            return {
                "answer": response.answer,
                "strategy_used": response.strategy_used,
                "chunks_retrieved": response.chunks_retrieved,
                "latency_ms": response.latency_ms,
                "tokens_in": response.tokens_in,
                "tokens_out": response.tokens_out,
                "citations": [
                    {
                        "source": c.source,
                        "chunk_index": c.chunk_index,
                        "total_chunks": c.total_chunks,
                        "page": c.page,
                        "score": c.score,
                        "snippet": c.snippet,
                    }
                    for c in response.citations
                ],
            }
        return asyncio.run(_go())

    # ── Internals ────────────────────────────────────────────────────

    async def _aingest(
        self, config_id: str, source_type: str, content: str, *, label: str
    ) -> dict[str, Any]:
        cfg_dict = self._load_config_dict(config_id)
        pipeline = get_pipeline(self._dict_to_pipeline_config(cfg_dict))
        summary = await pipeline.ingest(source_type, content)
        session = get_session()
        try:
            session.add(RagSource(
                config_id=config_id,
                source_type=source_type,
                content=content,
                label=label,
                chunks_count=summary.get("chunks", 0),
                tokens_estimated=summary.get("tokens_estimated", 0),
                status="ingested" if summary.get("chunks", 0) > 0 else "error",
                error_message=summary.get("error"),
                ingested_at=datetime.utcnow(),
            ))
            session.commit()
        finally:
            session.close()
        return summary

    def _load_config_dict(self, config_id: str) -> dict[str, Any]:
        session = get_session()
        try:
            row = session.query(RagConfig).filter_by(id=config_id).one_or_none()
            if row is None:
                raise LookupError(f"RAG config {config_id!r} not found")
            return self._cfg_to_dict(row)
        finally:
            session.close()

    @staticmethod
    def _dict_to_pipeline_config(d: dict[str, Any]) -> RAGConfig:
        return RAGConfig(
            config_id=d["id"],
            name=d["name"],
            embedding_model=d["embedding_model"],
            vector_store=d["vector_store"],
            llm_model=d.get("llm_model"),
            chunk_size=d["chunk_size"],
            chunk_overlap=d["chunk_overlap"],
            chunk_strategy=d["chunk_strategy"],
            retrieval_strategy=d["retrieval_strategy"],
            top_k=d["top_k"],
            mmr_lambda=d["mmr_lambda"],
            multi_query_n=d["multi_query_n"],
            system_prompt=d.get("system_prompt"),
            reranker=d.get("reranker", "none"),
        )

    @staticmethod
    def _cfg_to_dict(r: RagConfig) -> dict[str, Any]:
        return {
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "embedding_model": r.embedding_model,
            "vector_store": r.vector_store,
            "llm_model": r.llm_model,
            "chunk_size": r.chunk_size,
            "chunk_overlap": r.chunk_overlap,
            "chunk_strategy": r.chunk_strategy,
            "retrieval_strategy": r.retrieval_strategy,
            "top_k": r.top_k,
            "mmr_lambda": r.mmr_lambda,
            "multi_query_n": r.multi_query_n,
            "system_prompt": r.system_prompt,
            "reranker": r.reranker,
            "team_id": r.team_id,
            "is_active": r.is_active,
        }
