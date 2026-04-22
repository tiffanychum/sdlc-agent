"""Pydantic schemas for the no-code workflow builder.

The JSON payload stored in `WorkflowDefinition.graph_json` uses the exact
shape React Flow emits on the frontend so we can round-trip without a
transform layer:

    {
      "nodes": [{"id": ..., "type": "chunker", "position": {"x":..,"y":..},
                 "data": {...typed config...}}],
      "edges": [{"id": ..., "source": ..., "target": ...}]
    }

`validate_graph()` enforces:
  - node ids are unique, every edge references existing nodes
  - node types are in the supported catalogue
  - DAG has no cycles (layered topo sort in executor.py is the authority;
    we just run a pre-check here so invalid graphs reject at save-time
    with a useful error)
  - required per-node config fields are present
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator

# ── Node type catalogue ──────────────────────────────────────────

# Maps node type → {ingest: runs during ingest mode, query: runs during query mode}.
# A node type that spans both modes (embedder / vector_store) knows how to
# behave in each based on its upstream inputs — see executor.py.
NODE_TYPE_CATALOG: dict[str, dict[str, bool]] = {
    "data_source":     {"ingest": True,  "query": False},
    "chunker":         {"ingest": True,  "query": False},
    "embedder":        {"ingest": True,  "query": True},
    "vector_store":    {"ingest": True,  "query": True},
    "retriever":       {"ingest": False, "query": True},
    "prompt_template": {"ingest": False, "query": True},
    "llm":             {"ingest": False, "query": True},
    "output":          {"ingest": False, "query": True},
}

NodeType = Literal[
    "data_source", "chunker", "embedder", "vector_store",
    "retriever", "prompt_template", "llm", "output",
]


# Per-node-type required keys in `data`.  Optional keys default in the executor.
REQUIRED_DATA_KEYS: dict[str, list[str]] = {
    "data_source":     ["source_type"],        # + content for text/file/url (checked at run-time)
    "chunker":         [],                     # all params default
    "embedder":        ["model"],
    "vector_store":    ["store_type", "collection_name"],
    "retriever":       [],                     # top_k defaults to 4
    "prompt_template": ["template"],
    "llm":             ["model"],
    "output":          [],
}


# ── Pydantic models ──────────────────────────────────────────────

class Position(BaseModel):
    x: float = 0
    y: float = 0


class GraphNode(BaseModel):
    id: str
    type: str                                  # NodeType — validated below
    data: dict[str, Any] = Field(default_factory=dict)
    position: Position = Field(default_factory=Position)

    @field_validator("type")
    @classmethod
    def _type_is_known(cls, v: str) -> str:
        if v not in NODE_TYPE_CATALOG:
            raise ValueError(
                f"Unknown node type '{v}'. "
                f"Choose one of: {sorted(NODE_TYPE_CATALOG.keys())}"
            )
        return v


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    # React Flow also emits sourceHandle / targetHandle but we ignore them —
    # every node has a single implicit input/output channel in this v1.


class Graph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


# ── Validation ───────────────────────────────────────────────────

class GraphValidationError(ValueError):
    pass


def validate_graph(graph: Graph | dict) -> Graph:
    """Parse + semantically validate a graph.

    Raises GraphValidationError on cycle / unknown node / missing
    required field / dangling edge.  Returns the parsed Graph on success.
    """
    if isinstance(graph, dict):
        try:
            g = Graph.model_validate(graph)
        except Exception as e:
            raise GraphValidationError(f"Malformed graph payload: {e}") from e
    else:
        g = graph

    ids = [n.id for n in g.nodes]
    if len(ids) != len(set(ids)):
        dups = [i for i in ids if ids.count(i) > 1]
        raise GraphValidationError(f"Duplicate node ids: {sorted(set(dups))}")

    id_set = set(ids)
    for e in g.edges:
        if e.source not in id_set:
            raise GraphValidationError(f"Edge {e.id}: source '{e.source}' is not a node")
        if e.target not in id_set:
            raise GraphValidationError(f"Edge {e.id}: target '{e.target}' is not a node")
        if e.source == e.target:
            raise GraphValidationError(f"Edge {e.id}: self-loop on node '{e.source}'")

    for n in g.nodes:
        missing = [k for k in REQUIRED_DATA_KEYS.get(n.type, []) if k not in n.data]
        if missing:
            raise GraphValidationError(
                f"Node '{n.id}' (type={n.type}) missing required field(s): {missing}"
            )

    if _has_cycle(g):
        raise GraphValidationError("Graph contains a cycle — workflow must be a DAG")

    return g


def _has_cycle(g: Graph) -> bool:
    """DFS-based cycle detection (three-color marking)."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n.id: WHITE for n in g.nodes}
    adj: dict[str, list[str]] = {n.id: [] for n in g.nodes}
    for e in g.edges:
        adj[e.source].append(e.target)

    def dfs(u: str) -> bool:
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    return any(color[n] == WHITE and dfs(n) for n in list(color.keys()))


def filter_graph_for_mode(g: Graph, mode: str) -> Graph:
    """Return a subgraph containing only nodes active for the given mode.

    Edges are kept iff both endpoints survive the filter, so the resulting
    graph is still a valid DAG for topological sort.
    """
    if mode not in ("ingest", "query"):
        raise GraphValidationError(f"Unknown mode '{mode}'. Use 'ingest' or 'query'.")
    keep = {n.id for n in g.nodes if NODE_TYPE_CATALOG[n.type][mode]}
    filtered_nodes = [n for n in g.nodes if n.id in keep]
    filtered_edges = [e for e in g.edges if e.source in keep and e.target in keep]
    return Graph(nodes=filtered_nodes, edges=filtered_edges)
