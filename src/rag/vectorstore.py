"""
Vector store abstraction for the RAG pipeline.

Three free/local options:
  - chroma  : ChromaDB — persistent SQLite-backed, richest metadata filtering
  - faiss   : FAISS    — in-memory, fastest similarity search, no server
  - qdrant  : Qdrant   — enterprise-grade, in-memory mode, best for production

All stores implement the same VectorStore interface:
  add(chunks, embeddings)          # idempotent upsert by (source, chunk_index)
  query(query_embedding, top_k, filter) -> list[SearchResult]
  delete_collection()              # drop the whole collection
  delete_by_source(source)         # drop just the rows whose source metadata matches
  list_sources() -> list[{source, count}]
  count() -> int
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

VECTOR_STORES = {
    "chroma": {
        "label": "ChromaDB",
        "description": "Persistent SQLite-backed store. Best for prototyping and dev.",
        "install": "chromadb",
    },
    "faiss": {
        "label": "FAISS",
        "description": "In-memory Facebook AI Similarity Search. Fastest retrieval.",
        "install": "faiss-cpu",
    },
    "qdrant": {
        "label": "Qdrant (in-memory)",
        "description": "Enterprise-grade, runs in-memory (no server needed in dev).",
        "install": "qdrant-client",
    },
}


@dataclass
class SearchResult:
    text: str
    source: str
    score: float
    chunk_index: int
    total_chunks: int
    page: Optional[int]
    metadata: dict


# ── Chroma ────────────────────────────────────────────────────────────────────

class ChromaStore:
    def __init__(self, collection_id: str, persist_dir: str = "./data/chroma"):
        import chromadb
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=collection_id,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        """Idempotent upsert.

        IDs are derived from (source, chunk_index) so re-ingesting the same
        content produces the same IDs; ``upsert`` then replaces rows instead
        of raising DuplicateIDError and avoids duplicate vectors.
        """
        if not texts:
            return
        ids = [_stable_chunk_id(metadatas[i].get("source", "src"), metadatas[i].get("chunk_index", i))
               for i in range(len(texts))]
        self._col.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[{k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in m.items()} for m in metadatas],
        )

    def query(self, embedding: list[float], top_k: int = 5, where: Optional[dict] = None) -> list[SearchResult]:
        kwargs: dict[str, Any] = dict(query_embeddings=[embedding], n_results=min(top_k, self.count() or top_k))
        if where:
            kwargs["where"] = where
        result = self._col.query(**kwargs)
        out = []
        for i, doc in enumerate(result["documents"][0]):
            meta = result["metadatas"][0][i] if result["metadatas"] else {}
            dist = result["distances"][0][i] if result["distances"] else 0.0
            out.append(SearchResult(
                text=doc,
                source=str(meta.get("source", "")),
                score=1.0 - float(dist),  # cosine: distance -> similarity
                chunk_index=int(meta.get("chunk_index", 0)),
                total_chunks=int(meta.get("total_chunks", 1)),
                page=int(meta["page"]) if meta.get("page") else None,
                metadata=dict(meta),
            ))
        return out

    def delete_collection(self) -> None:
        self._client.delete_collection(self._col.name)

    def delete_by_source(self, source: str) -> int:
        """Remove every chunk whose 'source' metadata matches. Returns deleted count."""
        before = self._col.count()
        # Chroma supports where-filtered delete natively.
        self._col.delete(where={"source": source})
        after = self._col.count()
        return max(0, before - after)

    def list_sources(self) -> list[dict]:
        """Return [{source, count}] grouped by the 'source' metadata field."""
        # chromadb has no group-by, so pull metadatas and tally in memory.
        # Good enough for dev / prototyping scale; for huge stores we'd
        # paginate with .get(offset=,limit=).
        got = self._col.get(include=["metadatas"])
        buckets: dict[str, int] = {}
        for m in got.get("metadatas") or []:
            s = str((m or {}).get("source", ""))
            buckets[s] = buckets.get(s, 0) + 1
        return [{"source": k, "count": v} for k, v in sorted(buckets.items())]

    def count(self) -> int:
        return self._col.count()


# ── FAISS ─────────────────────────────────────────────────────────────────────

class FAISSStore:
    def __init__(self, collection_id: str, persist_dir: str = "./data/faiss"):
        import faiss  # type: ignore
        self._faiss = faiss
        self._persist_dir = persist_dir
        self._collection_id = collection_id
        os.makedirs(persist_dir, exist_ok=True)
        self._index: Any = None          # lazy init after first add
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._load()

    def _index_path(self) -> str:
        return os.path.join(self._persist_dir, f"{self._collection_id}.index")

    def _meta_path(self) -> str:
        return os.path.join(self._persist_dir, f"{self._collection_id}.meta.json")

    def _load(self) -> None:
        ip = self._index_path()
        mp = self._meta_path()
        if os.path.exists(ip) and os.path.exists(mp):
            self._index = self._faiss.read_index(ip)
            with open(mp) as f:
                data = json.load(f)
                self._texts = data["texts"]
                self._metadatas = data["metadatas"]

    def _save(self) -> None:
        if self._index is not None:
            self._faiss.write_index(self._index, self._index_path())
            with open(self._meta_path(), "w") as f:
                json.dump({"texts": self._texts, "metadatas": self._metadatas}, f)

    def add(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        """Idempotent: drop any existing rows for each new source first.

        FAISS has no native upsert; we delete-then-append by rebuilding the
        index minus the old source. Cheap for prototype-sized stores.
        """
        if not texts:
            return
        import numpy as np  # type: ignore

        # Purge existing rows for any source in the new batch so re-ingest
        # doesn't duplicate vectors.
        new_sources = {str((m or {}).get("source", "")) for m in metadatas}
        if self._index is not None and self._texts and new_sources:
            keep_idx = [i for i, m in enumerate(self._metadatas)
                        if str((m or {}).get("source", "")) not in new_sources]
            if len(keep_idx) != len(self._texts):
                self._rebuild_from_indices(keep_idx)

        vecs = np.array(embeddings, dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-10, None)
        if self._index is None:
            dim = vecs.shape[1]
            self._index = self._faiss.IndexFlatIP(dim)  # Inner product = cosine after normalisation
        self._index.add(vecs)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)
        self._save()

    def _rebuild_from_indices(self, keep_idx: list[int]) -> None:
        """Rebuild the FAISS index keeping only the rows in ``keep_idx``."""
        import numpy as np  # type: ignore
        if not keep_idx:
            self._index = None
            self._texts = []
            self._metadatas = []
            return
        # Extract the kept vectors from the existing index.
        all_vecs = self._index.reconstruct_n(0, self._index.ntotal) if self._index is not None else np.zeros((0, 0))
        kept_vecs = all_vecs[keep_idx]
        kept_texts = [self._texts[i] for i in keep_idx]
        kept_metas = [self._metadatas[i] for i in keep_idx]
        dim = kept_vecs.shape[1]
        new_index = self._faiss.IndexFlatIP(dim)
        new_index.add(kept_vecs)
        self._index = new_index
        self._texts = kept_texts
        self._metadatas = kept_metas

    def query(self, embedding: list[float], top_k: int = 5, where: Optional[dict] = None) -> list[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []
        import numpy as np  # type: ignore
        vec = np.array([embedding], dtype="float32")
        norm = np.linalg.norm(vec)
        vec = vec / max(norm, 1e-10)
        scores, indices = self._index.search(vec, min(top_k, self._index.ntotal))
        out = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._texts):
                continue
            meta = self._metadatas[idx]
            out.append(SearchResult(
                text=self._texts[idx],
                source=str(meta.get("source", "")),
                score=float(score),
                chunk_index=int(meta.get("chunk_index", 0)),
                total_chunks=int(meta.get("total_chunks", 1)),
                page=int(meta["page"]) if meta.get("page") else None,
                metadata=dict(meta),
            ))
        return out

    def delete_collection(self) -> None:
        for f in [self._index_path(), self._meta_path()]:
            if os.path.exists(f):
                os.remove(f)
        self._index = None
        self._texts = []
        self._metadatas = []

    def delete_by_source(self, source: str) -> int:
        before = len(self._texts)
        keep_idx = [i for i, m in enumerate(self._metadatas)
                    if str((m or {}).get("source", "")) != source]
        if len(keep_idx) == before:
            return 0
        self._rebuild_from_indices(keep_idx)
        self._save()
        return before - len(self._texts)

    def list_sources(self) -> list[dict]:
        buckets: dict[str, int] = {}
        for m in self._metadatas:
            s = str((m or {}).get("source", ""))
            buckets[s] = buckets.get(s, 0) + 1
        return [{"source": k, "count": v} for k, v in sorted(buckets.items())]

    def count(self) -> int:
        return self._index.ntotal if self._index else 0


# ── Qdrant ────────────────────────────────────────────────────────────────────

class QdrantStore:
    def __init__(self, collection_id: str, dimensions: int = 1536):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        self._client = QdrantClient(":memory:")
        self._name = collection_id
        self._dimensions = dimensions
        if not self._client.collection_exists(self._name):
            self._client.create_collection(
                collection_name=self._name,
                vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
            )

    def add(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        """Idempotent upsert using stable (source, chunk_index)-derived IDs."""
        if not texts:
            return
        from qdrant_client.models import PointStruct
        points = []
        for text, emb, meta in zip(texts, embeddings, metadatas):
            safe_meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in meta.items()}
            safe_meta["_text"] = text
            pid = _stable_chunk_uint(meta.get("source", "src"), meta.get("chunk_index", 0))
            points.append(PointStruct(id=pid, vector=emb, payload=safe_meta))
        self._client.upsert(collection_name=self._name, points=points)

    def query(self, embedding: list[float], top_k: int = 5, where: Optional[dict] = None) -> list[SearchResult]:
        # qdrant-client 1.7+ uses query_points; older versions used search
        try:
            qr = self._client.query_points(
                collection_name=self._name,
                query=embedding,
                limit=top_k,
            )
            results = qr.points
        except AttributeError:
            results = self._client.search(  # type: ignore[attr-defined]
                collection_name=self._name,
                query_vector=embedding,
                limit=top_k,
            )
        out = []
        for r in results:
            pl = r.payload or {}
            out.append(SearchResult(
                text=str(pl.get("_text", "")),
                source=str(pl.get("source", "")),
                score=float(r.score),
                chunk_index=int(pl.get("chunk_index", 0)),
                total_chunks=int(pl.get("total_chunks", 1)),
                page=int(pl["page"]) if pl.get("page") else None,
                metadata={k: v for k, v in pl.items() if k != "_text"},
            ))
        return out

    def delete_collection(self) -> None:
        self._client.delete_collection(self._name)

    def delete_by_source(self, source: str) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
        before = self.count()
        self._client.delete(
            collection_name=self._name,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
            ),
        )
        after = self.count()
        return max(0, before - after)

    def list_sources(self) -> list[dict]:
        """Group-by source via scroll — fine for prototype scale."""
        buckets: dict[str, int] = {}
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                s = str((p.payload or {}).get("source", ""))
                buckets[s] = buckets.get(s, 0) + 1
            if not offset:
                break
        return [{"source": k, "count": v} for k, v in sorted(buckets.items())]

    def count(self) -> int:
        info = self._client.get_collection(self._name)
        return info.points_count or 0


# ── Stable ID helpers ─────────────────────────────────────────────────────────


def _stable_chunk_id(source: str, chunk_index: Any) -> str:
    """Deterministic string ID for a (source, chunk_index) pair.

    Used by Chroma. Same content → same ID → ``upsert`` replaces rather
    than duplicates.
    """
    key = f"{source}|{chunk_index}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _stable_chunk_uint(source: str, chunk_index: Any) -> int:
    """Deterministic uint64 ID for a (source, chunk_index) pair.

    Used by Qdrant — it only accepts int or uuid IDs for points.
    """
    key = f"{source}|{chunk_index}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    # Take the top 63 bits so the result fits in a signed int64 without
    # overflowing Qdrant's id field on the wire.
    return int.from_bytes(h[:8], "big") >> 1


# ── Factory / discovery ───────────────────────────────────────────────────────

_qdrant_instances: dict[str, QdrantStore] = {}  # keep in-memory Qdrant alive


def list_collections(store_type: str, persist_dir: str = "./data/vectorstore") -> list[dict]:
    """Enumerate collections known to a given store backend.

    Returns ``[{name, count}]``. For in-memory Qdrant we can only see
    collections created in this process.
    """
    if store_type == "chroma":
        import chromadb
        path = os.path.join(persist_dir, "chroma")
        if not os.path.isdir(path):
            return []
        client = chromadb.PersistentClient(path=path)
        out = []
        for c in client.list_collections():
            col = client.get_collection(c.name)
            out.append({"name": c.name, "count": col.count()})
        return out
    if store_type == "faiss":
        path = os.path.join(persist_dir, "faiss")
        if not os.path.isdir(path):
            return []
        names = sorted({
            f[: -len(".index")] for f in os.listdir(path)
            if f.endswith(".index")
        })
        out = []
        for n in names:
            s = FAISSStore(n, persist_dir=path)
            out.append({"name": n, "count": s.count()})
        return out
    if store_type == "qdrant":
        return [{"name": n, "count": s.count()} for n, s in _qdrant_instances.items()]
    raise ValueError(f"Unknown store_type '{store_type}'")


def create_store(
    store_type: str,
    collection_id: str,
    persist_dir: str = "./data/vectorstore",
    dimensions: int = 1536,
) -> ChromaStore | FAISSStore | QdrantStore:
    """
    Instantiate a vector store backend.

    Args:
        store_type: "chroma" | "faiss" | "qdrant"
        collection_id: unique name for this RAG config's collection
        persist_dir: base path for Chroma / FAISS persistence
        dimensions: embedding dimensions (must match embedding model)
    """
    if store_type == "chroma":
        return ChromaStore(collection_id, persist_dir=os.path.join(persist_dir, "chroma"))
    elif store_type == "faiss":
        return FAISSStore(collection_id, persist_dir=os.path.join(persist_dir, "faiss"))
    elif store_type == "qdrant":
        # Reuse existing in-memory instance to preserve data across calls in same process
        if collection_id not in _qdrant_instances:
            _qdrant_instances[collection_id] = QdrantStore(collection_id, dimensions=dimensions)
        return _qdrant_instances[collection_id]
    else:
        raise ValueError(f"Unknown store_type '{store_type}'. Choose: chroma, faiss, qdrant")
