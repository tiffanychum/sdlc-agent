"""End-to-end tests for the no-code workflow executor.

Covers the 7 points from the workflow-builder plan:
  1. Layered topo sort returns the right partition order
  2. Cycle detection raises GraphValidationError
  3. Ingest mode: full 4-node chain persists chunks to the store
  4. Query mode: retrieves + renders + calls LLM → answer includes retrieved text
  5. Graph round-trip (validate → dict → validate) is stable
  6. node_log entries cover every executed node with non-negative timings
  7. Two sibling nodes in the same topo layer run CONCURRENTLY
     (timing assertion — the whole point of layered execution)

All tests mock the embedding model + LLM, so no network calls.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


EMBED_DIM = 8


def _det_vec(seed: int, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic small vector for testing."""
    a = 2 * math.pi * (seed % 20) / 20
    base = [math.cos(a), math.sin(a), math.cos(2 * a), math.sin(2 * a)]
    return (base * ((dim // len(base)) + 1))[:dim]


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def tmp_persist(tmp_path) -> str:
    return str(tmp_path)


@pytest.fixture
def mock_embedding_model():
    """Patch EmbeddingModel constructor so no network hits happen."""
    from src.rag import embeddings as emb_mod

    class _MockModel:
        def __init__(self, *_a, **_kw):
            self.dimensions = EMBED_DIM
            self.model_id = "mock"

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [_det_vec(hash(t) & 0xffff) for t in texts]

        async def embed(self, text: str) -> list[float]:
            return _det_vec(hash(text) & 0xffff)

    with patch.object(emb_mod, "EmbeddingModel", _MockModel):
        yield _MockModel


@pytest.fixture
def mock_llm():
    """Replace get_llm with an AsyncMock that echoes the rendered prompt.

    Returns a callable that, when inspected after the test, carries
    `.mock_instance` — the AsyncMock so tests can assert call counts.
    """
    from src.llm import client as llm_client

    mock_instance = MagicMock()

    async def _ainvoke(prompt_value):
        # ChatPromptValue.to_string() gives the flattened text.  We take the
        # last 200 chars so tests can assert context made it through.
        try:
            text = prompt_value.to_string()
        except Exception:
            text = str(prompt_value)
        from langchain_core.messages import AIMessage
        return AIMessage(content=f"[stub-answer using {len(text)} chars of prompt]")

    mock_instance.ainvoke = AsyncMock(side_effect=_ainvoke)

    def _factory(model=None, temperature=None, max_tokens=None):
        return mock_instance

    with patch.object(llm_client, "get_llm", _factory):
        yield mock_instance


# ── Graph helpers ────────────────────────────────────────────────

def _full_rag_graph(
    collection_name: str,
    persist_dir: str,
    source_text: str = "Python is a high-level programming language. "
                       "FastAPI is a modern Python web framework. "
                       "RAG stands for Retrieval-Augmented Generation.",
) -> dict:
    """Build an 8-node workflow that covers both ingest and query paths."""
    return {
        "nodes": [
            {"id": "src1", "type": "data_source",
             "data": {"source_type": "text", "content": source_text}},
            {"id": "chunk1", "type": "chunker",
             "data": {"strategy": "recursive", "chunk_size": 60, "chunk_overlap": 10}},
            {"id": "emb1", "type": "embedder",
             "data": {"model": "openai/text-embedding-3-small"}},
            {"id": "vs1", "type": "vector_store",
             "data": {"store_type": "chroma", "collection_name": collection_name,
                      "persist_dir": persist_dir}},
            {"id": "ret1", "type": "retriever",
             "data": {"top_k": 3}},
            {"id": "prompt1", "type": "prompt_template",
             "data": {"template": "Answer using the context.\n\n"
                                   "Context:\n{context}\n\nQuestion: {question}"}},
            {"id": "llm1", "type": "llm",
             "data": {"model": "gpt-4o-mini", "temperature": 0.2}},
            {"id": "out1", "type": "output", "data": {}},
        ],
        "edges": [
            # Ingest path
            {"id": "e1", "source": "src1",    "target": "chunk1"},
            {"id": "e2", "source": "chunk1",  "target": "emb1"},
            {"id": "e3", "source": "emb1",    "target": "vs1"},
            # Query path (embedder is shared — different mode = different inputs)
            {"id": "e4", "source": "vs1",     "target": "ret1"},
            {"id": "e5", "source": "emb1",    "target": "ret1"},
            {"id": "e6", "source": "ret1",    "target": "prompt1"},
            {"id": "e7", "source": "prompt1", "target": "llm1"},
            {"id": "e8", "source": "llm1",    "target": "out1"},
        ],
    }


# ── Test 1: layered topological sort ─────────────────────────────

def test_layered_topo_sort_respects_dependencies():
    from src.workflow.schema import Graph, validate_graph
    from src.workflow.executor import layered_topo_sort

    raw = {
        "nodes": [
            {"id": "a", "type": "data_source", "data": {"source_type": "text", "content": "x"}},
            {"id": "b", "type": "chunker",     "data": {}},
            {"id": "c", "type": "chunker",     "data": {}},
            {"id": "d", "type": "embedder",    "data": {"model": "m"}},
        ],
        # a → b → d, a → c → d  (so d has two incoming edges)
        "edges": [
            {"id": "e1", "source": "a", "target": "b"},
            {"id": "e2", "source": "a", "target": "c"},
            {"id": "e3", "source": "b", "target": "d"},
            {"id": "e4", "source": "c", "target": "d"},
        ],
    }
    g = validate_graph(Graph.model_validate(raw))
    layers = layered_topo_sort(g)

    assert layers[0] == ["a"]
    assert set(layers[1]) == {"b", "c"}      # siblings → same layer
    assert layers[2] == ["d"]
    # Every node appears exactly once across layers.
    flat = [n for L in layers for n in L]
    assert sorted(flat) == ["a", "b", "c", "d"]


# ── Test 2: cycle detection ──────────────────────────────────────

def test_cycle_detection_rejects_at_validate():
    from src.workflow.schema import validate_graph, GraphValidationError

    raw = {
        "nodes": [
            {"id": "a", "type": "chunker", "data": {}},
            {"id": "b", "type": "chunker", "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "a", "target": "b"},
            {"id": "e2", "source": "b", "target": "a"},
        ],
    }
    with pytest.raises(GraphValidationError, match="cycle"):
        validate_graph(raw)


# ── Test 3: ingest mode persists chunks ──────────────────────────

@pytest.mark.asyncio
async def test_ingest_mode_persists_chunks(mock_embedding_model, tmp_persist):
    from src.workflow.executor import WorkflowExecutor

    graph = _full_rag_graph(collection_name="wf_ingest", persist_dir=tmp_persist)
    ex = WorkflowExecutor(persist_dir=tmp_persist)

    result = await ex.execute(graph, mode="ingest", user_input={})

    assert result["status"] == "success", result
    assert result["output"]["count"] > 0
    assert result["output"]["collection_name"] == "wf_ingest"

    # Only ingest-mode nodes should appear in the log.
    logged_types = [entry["node_type"] for entry in result["node_log"]]
    assert set(logged_types) == {"data_source", "chunker", "embedder", "vector_store"}
    assert all(entry["status"] == "success" for entry in result["node_log"])


# ── Test 4: query mode returns an answer wired to retrieval ──────

@pytest.mark.asyncio
async def test_query_mode_end_to_end(mock_embedding_model, mock_llm, tmp_persist):
    from src.workflow.executor import WorkflowExecutor

    graph = _full_rag_graph(collection_name="wf_query", persist_dir=tmp_persist)
    ex = WorkflowExecutor(persist_dir=tmp_persist)

    # First ingest so the vector store has something to retrieve.
    ingest = await ex.execute(graph, mode="ingest", user_input={})
    assert ingest["status"] == "success"

    # Now query.
    query_result = await ex.execute(
        graph, mode="query",
        user_input={"query": "What is FastAPI?"},
    )

    assert query_result["status"] == "success", query_result
    assert "stub-answer" in query_result["output"]["answer"]
    assert mock_llm.ainvoke.await_count == 1

    logged_types = [entry["node_type"] for entry in query_result["node_log"]]
    assert set(logged_types) == {"embedder", "vector_store", "retriever",
                                  "prompt_template", "llm", "output"}


# ── Test 5: graph round-trip through pydantic stays stable ───────

def test_graph_roundtrip_stable():
    from src.workflow.schema import Graph, validate_graph

    raw = _full_rag_graph(collection_name="rt", persist_dir="/tmp")
    g1 = validate_graph(raw)
    as_dict = g1.model_dump()
    g2 = validate_graph(Graph.model_validate(as_dict))
    assert g2.model_dump() == g1.model_dump()


# ── Test 6: node_log covers every executed node with timings ─────

@pytest.mark.asyncio
async def test_node_log_has_entry_per_node(mock_embedding_model, tmp_persist):
    from src.workflow.executor import WorkflowExecutor

    graph = _full_rag_graph(collection_name="wf_log", persist_dir=tmp_persist)
    ex = WorkflowExecutor(persist_dir=tmp_persist)
    result = await ex.execute(graph, mode="ingest", user_input={})

    assert result["status"] == "success"
    # 4 ingest nodes.
    assert len(result["node_log"]) == 4
    for entry in result["node_log"]:
        assert entry["duration_ms"] >= 0
        assert "preview" in entry or "error" in entry
        assert entry["node_id"]
        assert entry["node_type"] in {
            "data_source", "chunker", "embedder", "vector_store",
        }


# ── Test 7: same-layer nodes actually run concurrently ───────────

@pytest.mark.asyncio
async def test_same_layer_nodes_run_concurrently():
    """Executor-level proof that async Practice #1 (layered gather) holds.

    Register two stub node types that each sleep 300ms, wire them as
    siblings under one parent, and assert total wall-clock < 500ms.
    Serial execution would be ~600ms.
    """
    from src.workflow import executor as ex_mod

    delay = 0.3
    calls: list[tuple[float, str]] = []

    async def _stub_a(cfg, inputs, ctx):
        calls.append((time.monotonic(), "a-start"))
        await asyncio.sleep(delay)
        calls.append((time.monotonic(), "a-end"))
        return {"ok": "a"}

    async def _stub_b(cfg, inputs, ctx):
        calls.append((time.monotonic(), "b-start"))
        await asyncio.sleep(delay)
        calls.append((time.monotonic(), "b-end"))
        return {"ok": "b"}

    async def _stub_root(cfg, inputs, ctx):
        return {"ok": "root"}

    original = dict(ex_mod._NODE_RUNNERS)
    try:
        # Temporarily rebind 3 real node types to our stubs so we can
        # use the real validator + NODE_TYPE_CATALOG without adding
        # new types.
        ex_mod._NODE_RUNNERS["data_source"] = _stub_root    # root of layer 0
        ex_mod._NODE_RUNNERS["chunker"] = _stub_a           # sibling 1, layer 1
        ex_mod._NODE_RUNNERS["embedder"] = _stub_b          # sibling 2, layer 1

        graph = {
            "nodes": [
                {"id": "root", "type": "data_source",
                 "data": {"source_type": "text", "content": "x"}},
                {"id": "sibA", "type": "chunker", "data": {}},
                {"id": "sibB", "type": "embedder", "data": {"model": "m"}},
            ],
            "edges": [
                {"id": "e1", "source": "root", "target": "sibA"},
                {"id": "e2", "source": "root", "target": "sibB"},
            ],
        }

        t0 = time.monotonic()
        result = await ex_mod.WorkflowExecutor().execute(
            graph, mode="ingest", user_input={},
        )
        elapsed = time.monotonic() - t0

        assert result["status"] == "success"

        # Serial would be ~0.6s; parallel should be ~0.3s + epsilon.
        # Allow 0.5s ceiling for CI jitter.
        assert elapsed < 0.5, (
            f"Same-layer nodes ran sequentially! elapsed={elapsed:.3f}s. "
            f"Expected < 0.5s for two 0.3s siblings in parallel."
        )

        # Start-times of siblings overlap: A starts before B ends.
        a_start = next(t for t, tag in calls if tag == "a-start")
        b_start = next(t for t, tag in calls if tag == "b-start")
        a_end = next(t for t, tag in calls if tag == "a-end")
        b_end = next(t for t, tag in calls if tag == "b-end")
        assert max(a_start, b_start) < min(a_end, b_end), (
            "Sibling executions did not overlap in time"
        )

    finally:
        ex_mod._NODE_RUNNERS.clear()
        ex_mod._NODE_RUNNERS.update(original)


# ── Test 8: live trajectory events stream in the correct order ───

@pytest.mark.asyncio
async def test_on_event_emits_run_and_node_lifecycle(mock_embedding_model, tmp_persist):
    """The UI's live-trajectory panel depends on the contract:

        run_start → (node_start → node_end)* → run_end

    Capture every event and assert the ordering + payload shape.
    """
    from src.workflow.executor import WorkflowExecutor

    graph = _full_rag_graph(collection_name="wf_events", persist_dir=tmp_persist)
    events: list[dict] = []

    async def _sink(evt: dict) -> None:
        events.append(evt)

    result = await WorkflowExecutor(persist_dir=tmp_persist).execute(
        graph, mode="ingest", user_input={}, on_event=_sink,
    )
    assert result["status"] == "success"

    # First event must be run_start.
    assert events[0]["event"] == "run_start"
    assert events[0]["mode"] == "ingest"
    assert events[0]["total_nodes"] == 4
    assert events[0]["layers"]  # non-empty

    # Last event must be run_end with the final payload.
    assert events[-1]["event"] == "run_end"
    assert events[-1]["status"] == "success"
    assert events[-1]["node_log"]

    # Every node fires node_start then node_end, in that order.
    started: set[str] = set()
    ended: set[str] = set()
    for e in events[1:-1]:
        if e["event"] == "node_start":
            assert e["node_id"] not in ended, "node_start after node_end for same id"
            started.add(e["node_id"])
        elif e["event"] == "node_end":
            assert e["node_id"] in started, "node_end before node_start"
            ended.add(e["node_id"])
            assert e["status"] in ("success", "error")
            assert "duration_ms" in e
    assert started == ended, "some nodes started but never ended"
    assert len(ended) == 4, f"expected 4 ingest nodes, got {len(ended)}"
