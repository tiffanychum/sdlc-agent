"""
Vector store abstraction for the RAG pipeline.

Three free/local options:
  - chroma  : ChromaDB — persistent SQLite-backed, richest metadata filtering
  - faiss   : FAISS    — in-memory, fastest similarity search, no server
  - qdrant  : Qdrant   — enterprise-grade, in-memory mode, best for production

All stores implement the same VectorStore interface:
  add(chunks, embeddings)
  query(query_embedding, top_k, filter) -> list[SearchResult]
  delete_collection(collection_id)
  count() -> int
"""

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
        if not texts:
            return
        ids = [f"{metadatas[i].get('source','src')}_{i}_{hash(texts[i]) & 0xFFFFFF}" for i in range(len(texts))]
        self._col.add(
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
        if not texts:
            return
        import numpy as np  # type: ignore
        vecs = np.array(embeddings, dtype="float32")
        # Normalise for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-10, None)
        if self._index is None:
            dim = vecs.shape[1]
            self._index = self._faiss.IndexFlatIP(dim)  # Inner product = cosine after normalisation
        self._index.add(vecs)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)
        self._save()

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
        self._counter = 0

    def add(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        if not texts:
            return
        from qdrant_client.models import PointStruct
        points = []
        for text, emb, meta in zip(texts, embeddings, metadatas):
            safe_meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in meta.items()}
            safe_meta["_text"] = text
            points.append(PointStruct(id=self._counter, vector=emb, payload=safe_meta))
            self._counter += 1
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
        self._counter = 0

    def count(self) -> int:
        info = self._client.get_collection(self._name)
        return info.points_count or 0


# ── Factory ───────────────────────────────────────────────────────────────────

_qdrant_instances: dict[str, QdrantStore] = {}  # keep in-memory Qdrant alive


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
