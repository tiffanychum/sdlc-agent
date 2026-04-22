"""Layered topological executor for no-code workflows.

Design decisions (see notes in the workflow branch README):

1. We implement our own DAG traversal rather than lean on LangChain's
   LCEL because our DAG has side-effectful sinks (vector_store.add) and
   arbitrary fan-in that LCEL doesn't natively model.  LangChain is still
   used at the leaves: `ChatPromptTemplate`, `get_llm`, and the
   underlying text splitters / loaders.  This mirrors how LangFlow does
   it.

2. Topological sort is **layered**: every layer holds all nodes whose
   dependencies are satisfied, and the whole layer is launched together
   with `asyncio.gather`.  For a linear RAG chain the layers degenerate
   to size-1 (sequential), but the design already supports fan-out
   (e.g. hybrid BM25 + dense retrievers) without refactor.

3. Each node implementation is a plain async function
   `async def _run_<node_type>(cfg, inputs, ctx) -> Any`.  Downstream
   nodes read upstream outputs via a `state[node_id]` dict.

4. Mode-aware: `execute(mode="ingest")` filters the graph down to ingest
   nodes; `execute(mode="query")` to query nodes.  See
   `schema.NODE_TYPE_CATALOG` for the partition.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from src.workflow.schema import (
    Graph,
    GraphNode,
    NODE_TYPE_CATALOG,
    validate_graph,
    filter_graph_for_mode,
)

logger = logging.getLogger(__name__)


# ── Node output container ────────────────────────────────────────

@dataclass
class NodeResult:
    output: Any
    log: dict  # {node_id, node_type, status, duration_ms, preview, ...}


@dataclass
class ExecContext:
    """Per-run context threaded through every node.

    Holds mode, user input, and the persist_dir to use for vector-store
    creation.  `node_registry` maps node_id → GraphNode so mode-aware
    nodes (embedder / vector_store) can look up upstream configs.
    """
    mode: str                                           # "ingest" | "query"
    user_input: dict                                     # {query: str, ...}
    persist_dir: str = "./data/workflow_vs"
    node_registry: dict[str, GraphNode] = field(default_factory=dict)


# ── Layered topological sort ─────────────────────────────────────

def layered_topo_sort(graph: Graph) -> list[list[str]]:
    """Kahn's algorithm by level — nodes in the same layer have no
    inter-dependencies and can run concurrently.

    Raises ValueError on cycles (validate_graph() already checks this
    at save time; included here as a belt-and-braces guard).
    """
    indeg: dict[str, int] = {n.id: 0 for n in graph.nodes}
    adj: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    for e in graph.edges:
        indeg[e.target] += 1
        adj[e.source].append(e.target)

    layers: list[list[str]] = []
    ready = sorted([n for n, d in indeg.items() if d == 0])
    visited = 0
    while ready:
        layers.append(ready)
        visited += len(ready)
        next_layer: list[str] = []
        for nid in ready:
            for v in adj[nid]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    next_layer.append(v)
        ready = sorted(next_layer)

    if visited != len(graph.nodes):
        raise ValueError("Cycle detected in graph — cannot topologically sort")
    return layers


# ── Node implementations ─────────────────────────────────────────

async def _run_data_source(cfg: dict, inputs: dict, ctx: ExecContext):
    """Load raw text from source_type=text|file|url.  Returns list[(text, meta)]."""
    from src.rag.chunker import load_source

    source_type = cfg["source_type"]
    content = cfg.get("content", "")
    if source_type in ("file", "url") and not content:
        raise ValueError(f"data_source node needs 'content' for source_type={source_type}")
    # URL / file IO can block, so offload.  Plain text is in-memory and fine inline.
    if source_type in ("file", "url"):
        pages = await asyncio.to_thread(load_source, source_type, content)
    else:
        pages = load_source(source_type, content)
    return pages


async def _run_chunker(cfg: dict, inputs: dict, ctx: ExecContext):
    """Split loaded pages into Chunks using LangChain text splitters."""
    from src.rag.chunker import chunk_documents

    pages = _single_upstream_value(inputs)
    strategy = cfg.get("strategy", "recursive")
    chunk_size = int(cfg.get("chunk_size", 500))
    chunk_overlap = int(cfg.get("chunk_overlap", 50))
    # Splitting is pure Python but can be chunky for large inputs.
    chunks = await asyncio.to_thread(
        chunk_documents, pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
    )
    return chunks


async def _run_embedder(cfg: dict, inputs: dict, ctx: ExecContext):
    """
    Mode-polymorphic:
      - ingest: upstream is list[Chunk] → returns {chunks, embeddings}
      - query : inputs is empty (or has user_query) → returns {query, embedding}
    """
    from src.rag.embeddings import EmbeddingModel

    model_id = cfg["model"]
    em = EmbeddingModel(model_id=model_id)

    if ctx.mode == "ingest":
        chunks = _single_upstream_value(inputs)
        texts = [c.text for c in chunks]
        embeddings = await em.embed_batch(texts)
        return {"chunks": chunks, "embeddings": embeddings, "dimensions": em.dimensions}

    # query mode — embed the user query once
    query = ctx.user_input.get("query", "")
    if not query:
        raise ValueError("query mode requires user_input.query to embed")
    emb = await em.embed(query)
    return {"query": query, "embedding": emb, "dimensions": em.dimensions}


async def _run_vector_store(cfg: dict, inputs: dict, ctx: ExecContext):
    """
    Mode-polymorphic:
      - ingest: upstream embedder gave us {chunks, embeddings}. We add
                them to the store and return {count, collection_name}.
      - query : we open the same collection and return a handle for
                downstream retriever to query.  (No write.)
    """
    from src.rag.vectorstore import create_store

    store_type = cfg["store_type"]
    collection_name = cfg["collection_name"]
    persist_dir = cfg.get("persist_dir") or ctx.persist_dir

    # Dimensions are needed for Qdrant; pull from upstream embedder if possible.
    dimensions = 1536
    for up in inputs.values():
        if isinstance(up, dict) and "dimensions" in up:
            dimensions = int(up["dimensions"])
            break

    # Create/open the store.  Chroma / FAISS do disk IO → offload.
    def _open():
        return create_store(
            store_type=store_type,
            collection_id=collection_name,
            persist_dir=persist_dir,
            dimensions=dimensions,
        )
    store = await asyncio.to_thread(_open)

    if ctx.mode == "ingest":
        # Expect exactly one upstream — the embedder output dict.
        payload = _single_upstream_value(inputs)
        chunks = payload["chunks"]
        embeddings = payload["embeddings"]
        texts = [c.text for c in chunks]
        # Chroma stringifies None → "None" then int("None") blows up on
        # retrieve — drop None-valued metadata keys altogether.
        metadatas: list[dict] = []
        for c in chunks:
            m = {
                "source": c.source,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
            }
            if c.page is not None:
                m["page"] = c.page
            metadatas.append(m)
        await asyncio.to_thread(store.add, texts, embeddings, metadatas)
        return {"collection_name": collection_name, "store_type": store_type, "count": len(chunks)}

    # query mode — return the store handle so `retriever` can call query()
    return {"store": store, "collection_name": collection_name}


async def _run_retriever(cfg: dict, inputs: dict, ctx: ExecContext):
    """Run a similarity search against the upstream vector store.

    Upstream inputs must contain one vector_store payload (with `.store`)
    and one embedder payload (with `.embedding`).
    """
    top_k = int(cfg.get("top_k", 4))

    store = None
    query_emb = None
    query_text = ""
    for up in inputs.values():
        if isinstance(up, dict):
            if up.get("store") is not None:
                store = up["store"]
            if up.get("embedding") is not None:
                query_emb = up["embedding"]
                query_text = up.get("query", "")

    if store is None:
        raise ValueError("retriever needs an upstream vector_store (query mode)")
    if query_emb is None:
        raise ValueError("retriever needs an upstream embedder with a query embedding")

    results = await asyncio.to_thread(store.query, query_emb, top_k)
    return {"query": query_text, "results": results}


async def _run_prompt_template(cfg: dict, inputs: dict, ctx: ExecContext):
    """Render the template with {context} + {question} using LangChain."""
    from langchain_core.prompts import ChatPromptTemplate

    template = cfg["template"]
    retrieved = _find_retriever_output(inputs)
    question = ctx.user_input.get("query", "")

    context = "\n\n".join(
        f"[{i+1}] {r.text}" for i, r in enumerate(retrieved.get("results", []))
    ) if retrieved else ""

    prompt = ChatPromptTemplate.from_template(template)
    # ChatPromptTemplate.ainvoke is the canonical LangChain Runnable call.
    rendered = await prompt.ainvoke({"context": context, "question": question})
    return {"prompt_value": rendered, "context": context, "question": question}


async def _run_llm(cfg: dict, inputs: dict, ctx: ExecContext):
    """Invoke the LLM on the upstream rendered prompt."""
    from src.llm.client import get_llm

    temp = float(cfg.get("temperature", 0.2))
    llm = get_llm(model=cfg["model"], temperature=temp)
    payload = _single_upstream_value(inputs)
    prompt_value = payload["prompt_value"] if isinstance(payload, dict) else payload
    resp = await llm.ainvoke(prompt_value)
    answer = getattr(resp, "content", str(resp))
    return {"answer": answer, "context": payload.get("context", "") if isinstance(payload, dict) else ""}


async def _run_output(cfg: dict, inputs: dict, ctx: ExecContext):
    """Identity / terminal sink — flattens upstream to a consumable payload."""
    payload = _single_upstream_value(inputs)
    if isinstance(payload, dict):
        return {
            "answer": payload.get("answer", ""),
            "context": payload.get("context", ""),
        }
    return {"answer": str(payload), "context": ""}


# Dispatch table
_NODE_RUNNERS = {
    "data_source":     _run_data_source,
    "chunker":         _run_chunker,
    "embedder":        _run_embedder,
    "vector_store":    _run_vector_store,
    "retriever":       _run_retriever,
    "prompt_template": _run_prompt_template,
    "llm":             _run_llm,
    "output":          _run_output,
}


# ── Helpers ──────────────────────────────────────────────────────

def _single_upstream_value(inputs: dict) -> Any:
    """Get the one-and-only upstream value.  Raises if not exactly one."""
    if not inputs:
        raise ValueError("Node expected at least one upstream input, got none")
    if len(inputs) > 1:
        # For nodes that can take multiple upstreams we should merge,
        # but v1 semantics for chunker/embedder/llm is 1:1.  Fail loudly.
        raise ValueError(
            f"Node expected exactly one upstream input, got {len(inputs)}: "
            f"{list(inputs.keys())}"
        )
    return next(iter(inputs.values()))


def _find_retriever_output(inputs: dict) -> Optional[dict]:
    """Locate the upstream retriever payload (dict with 'results' key)."""
    for up in inputs.values():
        if isinstance(up, dict) and "results" in up:
            return up
    return None


def _preview(value: Any, limit: int = 300) -> str:
    """Short, safe string preview of a node output for the node log."""
    try:
        if hasattr(value, "__iter__") and not isinstance(value, (str, dict)):
            value = list(value)
            if value and hasattr(value[0], "text"):
                s = "; ".join(getattr(c, "text", str(c))[:80] for c in value[:3])
                return s[:limit] + ("…" if len(s) > limit else "")
        s = str(value)
    except Exception:
        s = repr(value)
    return s[:limit] + ("…" if len(s) > limit else "")


# ── Executor ─────────────────────────────────────────────────────

class WorkflowExecutor:
    """Stateless orchestrator — each `execute()` call takes a fresh graph.

    Kept as a class so future enhancements (shared embedder cache,
    streaming emitter, metrics sink) have a natural place to live.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or os.getenv(
            "WORKFLOW_PERSIST_DIR", "./data/workflow_vs"
        )

    async def execute(
        self,
        graph: Graph | dict,
        mode: str,
        user_input: Optional[dict] = None,
        on_event: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict:
        """Run the graph in the given mode.

        Returns: {status, output, node_log, duration_ms, terminal_node_id}.
        Raises on validation errors or node failures (sibling nodes in the
        same layer that already finished keep their results in node_log).

        ``on_event`` is an optional async callback.  When supplied, the
        executor streams progress events so callers (SSE endpoint) can
        render a live trajectory:

            {event: "run_start", mode, layers, total_nodes, timestamp}
            {event: "node_start", node_id, node_type, timestamp}
            {event: "node_end", node_id, node_type, status,
                                duration_ms, preview|error, timestamp}
            {event: "run_end",   status, output, error?, duration_ms,
                                 terminal_node_id, node_log, timestamp}
        """
        async def _emit(payload: dict) -> None:
            if on_event is None:
                return
            try:
                await on_event(payload)
            except Exception:
                # Never let a broken sink take down execution.
                logger.exception("on_event callback raised — ignoring")

        t0 = time.monotonic()
        g = validate_graph(graph if isinstance(graph, Graph) else Graph.model_validate(graph))
        subgraph = filter_graph_for_mode(g, mode)

        if not subgraph.nodes:
            result = {
                "status": "success",
                "output": {},
                "node_log": [],
                "duration_ms": 0,
                "terminal_node_id": None,
                "note": f"No nodes active for mode={mode}",
            }
            await _emit({"event": "run_end", **result, "timestamp": time.time()})
            return result

        layers = layered_topo_sort(subgraph)
        ctx = ExecContext(
            mode=mode,
            user_input=user_input or {},
            persist_dir=self.persist_dir,
            node_registry={n.id: n for n in g.nodes},
        )

        await _emit({
            "event": "run_start",
            "mode": mode,
            "layers": layers,
            "node_ids": [nid for layer in layers for nid in layer],
            "total_nodes": sum(len(l) for l in layers),
            "timestamp": time.time(),
        })

        state: dict[str, Any] = {}
        node_log: list[dict] = []
        predecessors = _predecessor_map(subgraph)

        for layer in layers:
            async def _one(nid: str):
                node = ctx.node_registry[nid]
                runner = _NODE_RUNNERS[node.type]
                upstream_inputs = {p: state[p] for p in predecessors.get(nid, [])}
                await _emit({
                    "event": "node_start",
                    "node_id": nid,
                    "node_type": node.type,
                    "timestamp": time.time(),
                })
                t_node = time.monotonic()
                try:
                    out = await runner(node.data, upstream_inputs, ctx)
                    dur_ms = int((time.monotonic() - t_node) * 1000)
                    log = {
                        "node_id": nid,
                        "node_type": node.type,
                        "status": "success",
                        "duration_ms": dur_ms,
                        "preview": _preview(out),
                    }
                    await _emit({"event": "node_end", **log, "timestamp": time.time()})
                    return NodeResult(output=out, log=log)
                except Exception as e:
                    dur_ms = int((time.monotonic() - t_node) * 1000)
                    log = {
                        "node_id": nid,
                        "node_type": node.type,
                        "status": "error",
                        "duration_ms": dur_ms,
                        "error": str(e),
                    }
                    await _emit({"event": "node_end", **log, "timestamp": time.time()})
                    return NodeResult(output=e, log=log)

            # Run the whole layer concurrently — siblings have no data
            # dependency on each other by construction (that's what "layer"
            # means in a topological level).  Async Practice #1 (parallelise
            # independent awaits) applied at the DAG level.
            results = await asyncio.gather(*(_one(nid) for nid in layer))
            for nid, res in zip(layer, results):
                node_log.append(res.log)
                if res.log["status"] == "error":
                    final = {
                        "status": "error",
                        "error": f"node '{nid}' failed: {res.log['error']}",
                        "output": {},
                        "node_log": node_log,
                        "duration_ms": int((time.monotonic() - t0) * 1000),
                        "terminal_node_id": None,
                    }
                    await _emit({"event": "run_end", **final, "timestamp": time.time()})
                    return final
                state[nid] = res.output

        terminal_id = _terminal_node_id(subgraph)
        output = state.get(terminal_id, {}) if terminal_id else {}
        final = {
            "status": "success",
            "output": output,
            "node_log": node_log,
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "terminal_node_id": terminal_id,
        }
        await _emit({"event": "run_end", **final, "timestamp": time.time()})
        return final


def _predecessor_map(graph: Graph) -> dict[str, list[str]]:
    preds: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    for e in graph.edges:
        preds[e.target].append(e.source)
    return preds


def _terminal_node_id(graph: Graph) -> Optional[str]:
    """Pick the single node with out-degree 0 (or the last of several)."""
    out_deg: dict[str, int] = {n.id: 0 for n in graph.nodes}
    for e in graph.edges:
        out_deg[e.source] += 1
    terminals = [nid for nid, d in out_deg.items() if d == 0]
    if not terminals:
        return None
    # Prefer an "output" node if one exists; else last by id order.
    by_type = {n.id: n.type for n in graph.nodes}
    for t in terminals:
        if by_type[t] == "output":
            return t
    return terminals[-1]
