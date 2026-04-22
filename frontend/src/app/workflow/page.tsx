"use client";

/**
 * LangFlow-style no-code workflow builder with live trajectory.
 *
 * Layout:
 *   ┌── left rail ──────────┬── canvas ──────────────┬── inspector ──┐
 *   │ palette (drag source) │ React Flow DAG         │ node form     │
 *   │ saved workflows       │ + live trajectory log  │ run panel     │
 *   └───────────────────────┴────────────────────────┴───────────────┘
 *
 * Running a workflow uses the SSE stream (`/api/workflows/:id/run/stream`)
 * so each node lights up as it starts / succeeds / fails, and the
 * connecting edges animate while a node is in-flight.  The final run
 * payload is still persisted as a WorkflowRun row server-side.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  addEdge,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  useEdgesState,
  useNodesState,
  type Connection,
  type Edge,
  type Node,
  type NodeProps,
  Handle,
  Position,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { api } from "@/lib/api";
import { useTeam } from "@/contexts/TeamContext";

// ── Node-type metadata (icons + colours live on the frontend) ───
// LangFlow aesthetic: white card with a small coloured icon square; the
// accent colour shows up in the icon + handle glow, not a full header band.

type NodeGroup = "ingest" | "query" | "shared";

const NODE_META: Record<
  string,
  { label: string; color: string; accent: string; emoji: string; group: NodeGroup; description: string }
> = {
  data_source:     { label: "Data Source",   color: "#0ea5e9", accent: "#e0f2fe", emoji: "📥", group: "ingest", description: "Source documents: raw text, a file path or a URL." },
  chunker:         { label: "Chunker",       color: "#10b981", accent: "#d1fae5", emoji: "✂️", group: "ingest", description: "Split documents into overlapping text chunks." },
  embedder:        { label: "Embedder",      color: "#8b5cf6", accent: "#ede9fe", emoji: "🔢", group: "shared", description: "Convert text to dense vector embeddings." },
  vector_store:    { label: "Vector Store",  color: "#14b8a6", accent: "#ccfbf1", emoji: "🗄️", group: "shared", description: "Persist embeddings and retrieve by similarity." },
  retriever:       { label: "Retriever",     color: "#f59e0b", accent: "#fef3c7", emoji: "🔍", group: "query",  description: "Fetch the top-K chunks for a query." },
  prompt_template: { label: "Prompt",        color: "#ec4899", accent: "#fce7f3", emoji: "📝", group: "query",  description: "Compose a prompt from context and question." },
  llm:             { label: "LLM",           color: "#ef4444", accent: "#fee2e2", emoji: "🤖", group: "query",  description: "Invoke a language model to produce an answer." },
  output:          { label: "Output",        color: "#52525b", accent: "#f4f4f5", emoji: "📤", group: "query",  description: "Terminal sink — surface the final answer." },
};

const PALETTE_ORDER = [
  "data_source", "chunker", "embedder", "vector_store",
  "retriever", "prompt_template", "llm", "output",
];

// Handle colour is keyed by the *data type* flowing through the port so
// that a given edge has one consistent colour from producer to consumer.
const IO_COLOR: Record<string, string> = {
  docs:       "#0ea5e9",
  chunks:     "#10b981",
  embeddings: "#8b5cf6",
  store:      "#14b8a6",
  context:    "#f59e0b",
  prompt:     "#ec4899",
  answer:     "#ef4444",
};

// Each node exposes a typed IO schema so the LangFlow-style rows can
// render labelled input/output handles (left = inputs, right = outputs).
const NODE_IO: Record<string, { inputs: { key: string; label: string }[]; outputs: { key: string; label: string }[] }> = {
  data_source:     { inputs: [],                                               outputs: [{ key: "docs", label: "Documents" }] },
  chunker:         { inputs: [{ key: "docs", label: "Documents" }],            outputs: [{ key: "chunks", label: "Chunks" }] },
  embedder:        { inputs: [{ key: "chunks", label: "Chunks / Query" }],     outputs: [{ key: "embeddings", label: "Embeddings" }] },
  vector_store:    { inputs: [{ key: "embeddings", label: "Embeddings" }],     outputs: [{ key: "store", label: "Store Handle" }] },
  retriever:       { inputs: [{ key: "store", label: "Store + Query" }],       outputs: [{ key: "context", label: "Retrieved" }] },
  prompt_template: { inputs: [{ key: "context", label: "Retrieved" }],         outputs: [{ key: "prompt", label: "Prompt" }] },
  llm:             { inputs: [{ key: "prompt", label: "Prompt" }],             outputs: [{ key: "answer", label: "Answer" }] },
  output:          { inputs: [{ key: "answer", label: "Answer" }],             outputs: [] },
};

// Sensible defaults per node type when dragged onto the canvas.
function defaultDataFor(type: string): Record<string, any> {
  switch (type) {
    case "data_source":
      return { source_type: "text", content: "" };
    case "chunker":
      return { strategy: "recursive", chunk_size: 500, chunk_overlap: 50 };
    case "embedder":
      return { model: "baai/bge-m3" };
    case "vector_store":
      return { store_type: "chroma", collection_name: "my_collection" };
    case "retriever":
      return { top_k: 4 };
    case "prompt_template":
      return {
        template:
          "Answer the question using only the context.\n\n" +
          "Context:\n{context}\n\nQuestion: {question}",
      };
    case "llm":
      return { model: "claude-sonnet-4.6", temperature: 0.2 };
    case "output":
      return {};
    default:
      return {};
  }
}

// ── Minimalist line-art icons, one per node type ────────────────
// Kept small and flat (stroke-based, currentColor) so they inherit
// the node's accent color from CSS.

const NODE_ICONS: Record<string, React.ReactElement> = {
  // Inbox with an arrow dropping in
  data_source: (
    <>
      <path d="M3 4.5h14v9.2H3z" />
      <path d="M3 11h4l1 1.8h4l1-1.8h4" />
      <path d="M10 2.5v4.2M8 5.2l2 2 2-2" />
    </>
  ),
  // Two horizontal slabs (chunks of text)
  chunker: (
    <>
      <rect x="3" y="4.5" width="14" height="3.3" rx="0.8" />
      <rect x="3" y="12" width="14" height="3.3" rx="0.8" />
    </>
  ),
  // 3×3 dot grid (vector / embedding)
  embedder: (
    <>
      <circle cx="5"  cy="5"  r="1.1" />
      <circle cx="10" cy="5"  r="1.1" />
      <circle cx="15" cy="5"  r="1.1" />
      <circle cx="5"  cy="10" r="1.1" />
      <circle cx="10" cy="10" r="1.1" />
      <circle cx="15" cy="10" r="1.1" />
      <circle cx="5"  cy="15" r="1.1" />
      <circle cx="10" cy="15" r="1.1" />
      <circle cx="15" cy="15" r="1.1" />
    </>
  ),
  // Stacked database cylinders
  vector_store: (
    <>
      <ellipse cx="10" cy="5" rx="6" ry="1.8" />
      <path d="M4 5v10c0 1 2.7 1.8 6 1.8s6-0.8 6-1.8V5" />
      <path d="M4 10c0 1 2.7 1.8 6 1.8s6-0.8 6-1.8" />
    </>
  ),
  // Magnifying glass
  retriever: (
    <>
      <circle cx="8.5" cy="8.5" r="4.2" />
      <path d="M11.7 11.7L16 16" />
    </>
  ),
  // Document with lines
  prompt_template: (
    <>
      <path d="M5.5 3h7l3 3v11h-10z" />
      <path d="M12.5 3v3h3" />
      <path d="M7.5 10h5M7.5 13h5" />
    </>
  ),
  // Four-point sparkle (LLM)
  llm: (
    <>
      <path d="M10 3.2l1.5 3.8L15 8.5l-3.5 1.5L10 13.8 8.5 10 5 8.5 8.5 7z" />
      <path d="M15 13l0.6 1.5L17 15l-1.4 0.6L15 17l-0.6-1.4L13 15l1.4-0.5z" />
    </>
  ),
  // Send / arrow out
  output: (
    <>
      <path d="M3.5 10h9.5" />
      <path d="M10 6.5l3.5 3.5L10 13.5" />
      <path d="M16 4.5v11" />
    </>
  ),
};

function NodeIcon({ type, size = 14 }: { type: string; size?: number }) {
  const paths = NODE_ICONS[type] ?? NODE_ICONS.output;
  return (
    <svg
      viewBox="0 0 20 20"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      {paths}
    </svg>
  );
}

// ── LangFlow-style custom node ──────────────────────────────────

type RunStatus = "idle" | "running" | "success" | "error";

function FlowNode({ data, selected, id }: NodeProps) {
  const d = data as any;
  const nodeType = d?.nodeType as string;
  const meta = NODE_META[nodeType] ?? NODE_META.output;
  const io = NODE_IO[nodeType] ?? NODE_IO.output;
  const status: RunStatus = (d?.runStatus as RunStatus) ?? "idle";
  const duration = d?.runDurationMs as number | undefined;
  const preview = d?.runPreview as string | undefined;
  const error = d?.runError as string | undefined;

  const statusClass =
    status === "running" ? "running" :
    status === "success" ? "success" :
    status === "error"   ? "error"   : "";

  const inputRows = io.inputs;
  const outputRows = io.outputs;
  const configRows = configRowsFor(nodeType, d);

  // CSS custom property drives icon glow + edge-of-card accents.
  const rootStyle = { ["--wf-accent" as any]: meta.color } as React.CSSProperties;

  return (
    <div
      className={`wf-node ${selected ? "selected" : ""} ${statusClass}`}
      style={rootStyle}
    >
      {/* Header: icon square + title + play/status indicator */}
      <div className="wf-node__head">
        <span className="wf-node__icon">
          <NodeIcon type={nodeType} />
        </span>
        <div className="wf-node__head-text">
          <div className="wf-node__name">{meta.label}</div>
          <div className="wf-node__sub">{id}</div>
        </div>
        <RunIndicator status={status} />
      </div>

      {/* One-line description, LangFlow-style */}
      <div className="wf-node__desc">{meta.description}</div>

      {/* Body: typed input rows, config rows, then output rows.
          Each IO row renders its Handle inline so the handle's vertical
          position is anchored to the row's DOM box. */}
      <div className="wf-node__body">
        {inputRows.map((r) => (
          <div key={`in_${r.key}`} className="wf-row">
            <div className="wf-row__label">
              {r.label}
              <span className="wf-row__info" title="Input port">i</span>
            </div>
            <div className="wf-row__val locked">
              <span className="wf-row__val-icon">🔒</span>
              <span className="wf-row__val-truncate">Receiving input</span>
            </div>
            <Handle
              type="target"
              position={Position.Left}
              id={r.key}
              className="wf-handle wf-handle--in"
              style={{ ["--wf-handle" as any]: IO_COLOR[r.key] ?? "#a1a1aa" } as React.CSSProperties}
            />
          </div>
        ))}

        {configRows.map((row) => (
          <div key={`cfg_${row.label}`} className="wf-row">
            <div className="wf-row__label">
              {row.label}
              {row.required && <span className="wf-row__req">*</span>}
              <span className="wf-row__info" title="Config">i</span>
            </div>
            <div className={`wf-row__val ${row.mono ? "mono" : ""}`}>
              {row.icon && <span className="wf-row__val-icon">{row.icon}</span>}
              <span className="wf-row__val-truncate">{row.value}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Footer: the "Response" row — labelled output with colored dot handle.
          Also displays live preview / duration / error when running. */}
      {outputRows.map((r) => (
        <div key={`out_${r.key}`} className="wf-node__foot">
          <span className="wf-node__foot-label">
            {r.label}
            <span className="wf-row__info" title="Output port">i</span>
          </span>
          {(status !== "idle" || duration !== undefined) ? (
            <>
              <span className={`wf-node__foot-preview ${status === "error" ? "err" : ""}`}>
                {status === "error" ? error : preview ?? ""}
              </span>
              {duration !== undefined && (
                <span className="wf-node__foot-duration">{duration} ms</span>
              )}
            </>
          ) : null}
          <Handle
            type="source"
            position={Position.Right}
            id={r.key}
            className="wf-handle wf-handle--out"
            style={{ ["--wf-handle" as any]: IO_COLOR[r.key] ?? "#a1a1aa" } as React.CSSProperties}
          />
        </div>
      ))}

      {/* Edge case: terminal node (no outputs) — still surface run feedback */}
      {outputRows.length === 0 && (status !== "idle" || duration !== undefined) && (
        <div className="wf-node__foot">
          <span className="wf-node__foot-label">Result</span>
          <span className={`wf-node__foot-preview ${status === "error" ? "err" : ""}`}>
            {status === "error" ? error : preview ?? ""}
          </span>
          {duration !== undefined && (
            <span className="wf-node__foot-duration">{duration} ms</span>
          )}
        </div>
      )}
    </div>
  );
}

function RunIndicator({ status }: { status: RunStatus }) {
  if (status === "running") {
    return (
      <span className="wf-node__run running" title="Running">
        <span className="wf-node__spinner" />
      </span>
    );
  }
  if (status === "success") {
    return <span className="wf-node__run success" title="Succeeded">✓</span>;
  }
  if (status === "error") {
    return <span className="wf-node__run error" title="Failed">!</span>;
  }
  return <span className="wf-node__run" title="Run">▷</span>;
}

// Rows shown in the body for each node type — label/value pairs styled
// like LangFlow's input/dropdown containers. Kept short; the Inspector
// is where the full config lives.
type ConfigRow = { label: string; value: string; required?: boolean; mono?: boolean; icon?: string };

function configRowsFor(type: string, d: any): ConfigRow[] {
  if (!d) return [];
  switch (type) {
    case "data_source":
      return [
        { label: "Source Type", value: d.source_type ?? "text", required: true },
        { label: "Content", value: d.content ? String(d.content).slice(0, 32) + (String(d.content).length > 32 ? "…" : "") : "—", mono: true },
      ];
    case "chunker":
      return [
        { label: "Strategy", value: d.strategy ?? "fixed_size", required: true },
        { label: "Size / Overlap", value: `${d.chunk_size ?? 500} / ${d.chunk_overlap ?? 50}`, mono: true },
      ];
    case "embedder":
      return [
        { label: "Model", value: truncateModel(d.model), required: true },
      ];
    case "vector_store":
      return [
        { label: "Store", value: d.store_type ?? "chroma", required: true },
        { label: "Collection", value: d.collection_name ?? "—", mono: true },
      ];
    case "retriever":
      return [
        { label: "Top K", value: String(d.top_k ?? 5), required: true, mono: true },
      ];
    case "prompt_template":
      return [
        { label: "Template", value: (d.template || "").slice(0, 48) + ((d.template || "").length > 48 ? "…" : ""), required: true, mono: true },
      ];
    case "llm":
      return [
        { label: "Language Model", value: truncateModel(d.model), required: true },
        { label: "Temperature", value: String(d.temperature ?? 0), mono: true },
      ];
    case "output":
      return [
        { label: "Format", value: d.format ?? "text", mono: true },
      ];
    default:
      return [];
  }
}

function truncateModel(m: string | undefined): string {
  if (!m) return "—";
  const parts = m.split("/");
  return parts.length > 1 ? parts[parts.length - 1] : m;
}

const nodeTypes = { flow: FlowNode };

// ── Starter graph — what we seed on a brand-new workflow ────────

function starterGraph(): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [
    { id: "src1",    type: "flow", position: { x: 20,  y: 40  }, data: { nodeType: "data_source",     ...defaultDataFor("data_source") } },
    { id: "chunk1",  type: "flow", position: { x: 320, y: 40  }, data: { nodeType: "chunker",         ...defaultDataFor("chunker") } },
    { id: "emb1",    type: "flow", position: { x: 620, y: 40  }, data: { nodeType: "embedder",        ...defaultDataFor("embedder") } },
    { id: "vs1",     type: "flow", position: { x: 920, y: 40  }, data: { nodeType: "vector_store",    ...defaultDataFor("vector_store") } },
    { id: "ret1",    type: "flow", position: { x: 920, y: 280 }, data: { nodeType: "retriever",       ...defaultDataFor("retriever") } },
    { id: "prompt1", type: "flow", position: { x: 620, y: 280 }, data: { nodeType: "prompt_template", ...defaultDataFor("prompt_template") } },
    { id: "llm1",    type: "flow", position: { x: 320, y: 280 }, data: { nodeType: "llm",             ...defaultDataFor("llm") } },
    { id: "out1",    type: "flow", position: { x: 20,  y: 280 }, data: { nodeType: "output",          ...defaultDataFor("output") } },
  ];
  const edges: Edge[] = [
    { id: "e1", source: "src1",    target: "chunk1" },
    { id: "e2", source: "chunk1",  target: "emb1" },
    { id: "e3", source: "emb1",    target: "vs1" },
    { id: "e4", source: "vs1",     target: "ret1" },
    { id: "e5", source: "emb1",    target: "ret1" },
    { id: "e6", source: "ret1",    target: "prompt1" },
    { id: "e7", source: "prompt1", target: "llm1" },
    { id: "e8", source: "llm1",    target: "out1" },
  ].map((e) => ({ ...e, type: "smoothstep", markerEnd: { type: MarkerType.ArrowClosed, color: "#a1a1aa" } }));
  return { nodes, edges };
}

// Serialise React Flow nodes/edges → the backend wire shape (strip run-state fields).
function toWire(nodes: Node[], edges: Edge[]) {
  const RUN_KEYS = new Set(["nodeType", "runStatus", "runDurationMs", "runPreview", "runError"]);
  return {
    nodes: nodes.map((n) => ({
      id: n.id,
      type: (n.data as any)?.nodeType ?? "output",
      position: { x: n.position.x, y: n.position.y },
      data: Object.fromEntries(
        Object.entries(n.data as any).filter(([k]) => !RUN_KEYS.has(k)),
      ),
    })),
    edges: edges.map((e) => ({ id: e.id, source: e.source, target: e.target })),
  };
}

function fromWire(graph: any): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = (graph?.nodes ?? []).map((n: any, i: number) => ({
    id: n.id,
    type: "flow",
    position: n.position ?? { x: 40 + i * 300, y: 40 },
    data: { nodeType: n.type, ...(n.data ?? {}) },
  }));
  const edges: Edge[] = (graph?.edges ?? []).map((e: any) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    type: "smoothstep",
    markerEnd: { type: MarkerType.ArrowClosed, color: "#a1a1aa" },
  }));
  return { nodes, edges };
}

// ── Main page ───────────────────────────────────────────────────

export default function WorkflowPage() {
  return (
    <ReactFlowProvider>
      <WorkflowBuilder />
    </ReactFlowProvider>
  );
}

type Catalog = {
  node_types: Record<string, { ingest: boolean; query: boolean }>;
  embedding_models: Record<string, { label?: string; provider?: string; dimensions?: number }>;
  vector_stores: Record<string, { label: string; description: string }>;
  chunk_strategies: string[];
  source_types: string[];
};

type WorkflowSummary = {
  id: string;
  name: string;
  description: string;
  team_id: string | null;
  updated_at: string | null;
};

type RunResult = {
  run_id?: string;
  status: "success" | "error" | string;
  output?: any;
  node_log?: any[];
  error?: string;
  duration_ms?: number;
};

// An event entry in the live trajectory panel.
type TrajEvent = {
  t: number;                                                  // wall-clock (ms since run start)
  kind: "run_start" | "node_start" | "node_end" | "run_end";
  node_id?: string;
  node_type?: string;
  status?: "success" | "error";
  duration_ms?: number;
  preview?: string;
  error?: string;
  summary?: string;
};

function WorkflowBuilder() {
  const { teamId } = useTeam();
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [llmModels, setLlmModels] = useState<{ id: string; name: string }[]>([]);
  const [savedList, setSavedList] = useState<WorkflowSummary[]>([]);
  const [loaded, setLoaded] = useState<{ id: string; name: string; description: string } | null>(null);

  const [name, setName] = useState("Untitled Workflow");
  const [description, setDescription] = useState("");

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [jsonOpen, setJsonOpen] = useState(false);
  const [status, setStatus] = useState<string>("");
  const [statusKind, setStatusKind] = useState<"ok" | "err" | "">("");
  const [runResult, setRunResult] = useState<RunResult | null>(null);
  const [trajectory, setTrajectory] = useState<TrajEvent[]>([]);
  const [queryInput, setQueryInput] = useState("");
  const [busy, setBusy] = useState(false);

  const idCounterRef = useRef(0);
  const runAbortRef = useRef<AbortController | null>(null);
  const runStartRef = useRef<number>(0);

  // Initial load — catalog + saved workflows + LLM models
  useEffect(() => {
    (async () => {
      try {
        const [c, m] = await Promise.all([
          api.workflows.catalog(),
          api.models.list().catch(() => []),
        ]);
        setCatalog(c);
        setLlmModels(Array.isArray(m) ? m : (m?.models ?? []));
      } catch {
        setStatus("Failed to load catalog — is the backend running?");
        setStatusKind("err");
      }
      const { nodes: sn, edges: se } = starterGraph();
      setNodes(sn);
      setEdges(se);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const list = await api.workflows.list(teamId || undefined);
        setSavedList(list);
      } catch { /* non-fatal */ }
    })();
  }, [teamId || undefined]);

  // ── Drag from palette → drop on canvas ────────────────────────

  const onDragStart = (e: React.DragEvent, nodeType: string) => {
    e.dataTransfer.setData("application/reactflow", nodeType);
    e.dataTransfer.effectAllowed = "move";
  };

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const nodeType = e.dataTransfer.getData("application/reactflow");
      if (!nodeType || !NODE_META[nodeType]) return;
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      idCounterRef.current += 1;
      const baseId = nodeType.split("_")[0];
      const id = `${baseId}_${idCounterRef.current}`;
      const position = { x: e.clientX - rect.left - 100, y: e.clientY - rect.top - 30 };
      setNodes((nds) => [
        ...nds,
        {
          id,
          type: "flow",
          position,
          data: { nodeType, ...defaultDataFor(nodeType) },
        },
      ]);
      setSelectedId(id);
    },
    [setNodes],
  );

  const onConnect = useCallback(
    (conn: Connection) =>
      setEdges((eds) =>
        addEdge(
          {
            ...conn,
            id: `e_${conn.source}_${conn.target}_${Math.random().toString(36).slice(2, 6)}`,
            type: "smoothstep",
            markerEnd: { type: MarkerType.ArrowClosed, color: "#a1a1aa" },
          },
          eds,
        ),
      ),
    [setEdges],
  );

  // ── Run-state overlay helpers ────────────────────────────────
  //
  // These walk the *current* node/edge arrays and patch in live status
  // without forgetting existing config edits.

  const resetRunState = useCallback(() => {
    setNodes((nds) =>
      nds.map((n) => ({
        ...n,
        data: { ...(n.data as any), runStatus: "idle", runDurationMs: undefined, runPreview: undefined, runError: undefined },
      })),
    );
    setEdges((eds) =>
      eds.map((e) => ({ ...e, animated: false, className: "" })),
    );
  }, [setNodes, setEdges]);

  const markNodeRunning = useCallback((nid: string) => {
    setNodes((nds) =>
      nds.map((n) =>
        n.id === nid
          ? { ...n, data: { ...(n.data as any), runStatus: "running", runDurationMs: undefined, runPreview: undefined, runError: undefined } }
          : n,
      ),
    );
    // Animate incoming + outgoing edges while this node is live.
    setEdges((eds) =>
      eds.map((e) =>
        e.source === nid || e.target === nid
          ? { ...e, animated: true, className: "wf-edge-active" }
          : e,
      ),
    );
  }, [setNodes, setEdges]);

  const markNodeEnded = useCallback(
    (nid: string, payload: { status: "success" | "error"; duration_ms?: number; preview?: string; error?: string }) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === nid
            ? {
                ...n,
                data: {
                  ...(n.data as any),
                  runStatus: payload.status,
                  runDurationMs: payload.duration_ms,
                  runPreview: payload.preview,
                  runError: payload.error,
                },
              }
            : n,
        ),
      );
      setEdges((eds) =>
        eds.map((e) => {
          if (e.source !== nid && e.target !== nid) return e;
          const done = payload.status === "success" && e.source === nid;
          return {
            ...e,
            animated: false,
            className: payload.status === "error" ? "wf-edge-error" : done ? "wf-edge-done" : "",
          };
        }),
      );
    },
    [setNodes, setEdges],
  );

  // ── Save / Load ────────────────────────────────────────────────

  async function save() {
    setBusy(true);
    setStatus("");
    try {
      const wire = toWire(nodes, edges);
      if (loaded) {
        await api.workflows.update(loaded.id, { name, description, graph: wire });
        setStatus(`Updated '${name}'`);
      } else {
        const created = await api.workflows.create({
          name, description, graph: wire, team_id: teamId || null,
        });
        setLoaded({ id: created.id, name: created.name, description: created.description });
        setStatus(`Saved new workflow (id=${created.id})`);
      }
      setStatusKind("ok");
      setSavedList(await api.workflows.list(teamId || undefined));
    } catch (e: any) {
      setStatus(`Save failed: ${e.message ?? e}`);
      setStatusKind("err");
    } finally {
      setBusy(false);
    }
  }

  async function loadWorkflow(id: string) {
    setBusy(true);
    setRunResult(null);
    setTrajectory([]);
    try {
      const wf = await api.workflows.get(id);
      setName(wf.name);
      setDescription(wf.description ?? "");
      const { nodes: nn, edges: ne } = fromWire(wf.graph);
      setNodes(nn);
      setEdges(ne);
      setLoaded({ id: wf.id, name: wf.name, description: wf.description ?? "" });
      setStatus(`Loaded '${wf.name}'`);
      setStatusKind("ok");
    } catch (e: any) {
      setStatus(`Load failed: ${e.message ?? e}`);
      setStatusKind("err");
    } finally {
      setBusy(false);
    }
  }

  async function deleteLoaded() {
    if (!loaded) return;
    if (!confirm(`Delete workflow '${loaded.name}'?`)) return;
    try {
      await api.workflows.delete(loaded.id);
      const { nodes: sn, edges: se } = starterGraph();
      setNodes(sn); setEdges(se);
      setLoaded(null);
      setName("Untitled Workflow");
      setDescription("");
      setSavedList(await api.workflows.list(teamId || undefined));
      setStatus("Deleted");
      setStatusKind("ok");
    } catch (e: any) {
      setStatus(`Delete failed: ${e.message ?? e}`);
      setStatusKind("err");
    }
  }

  function newBlank() {
    const { nodes: sn, edges: se } = starterGraph();
    setNodes(sn); setEdges(se);
    setName("Untitled Workflow");
    setDescription("");
    setLoaded(null);
    setRunResult(null);
    setTrajectory([]);
    setStatus("");
  }

  // ── Run (streaming with live trajectory) ──────────────────────

  async function runMode(mode: "ingest" | "query") {
    if (!loaded) {
      setStatus("Save the workflow first, then run it.");
      setStatusKind("err");
      return;
    }
    // Cancel any prior run still in flight.
    if (runAbortRef.current) runAbortRef.current.abort();
    const ac = new AbortController();
    runAbortRef.current = ac;

    setBusy(true);
    setRunResult(null);
    setTrajectory([]);
    resetRunState();
    setStatus(`${mode === "ingest" ? "Ingesting" : "Querying"} (live)…`);
    setStatusKind("");
    runStartRef.current = Date.now();

    const input = mode === "query" ? { query: queryInput } : {};
    const nodeLog: any[] = [];
    let lastStatus: "success" | "error" = "success";
    let lastError: string | undefined;
    let lastOutput: any = undefined;

    try {
      await api.workflows.runStream(
        loaded.id,
        mode,
        input,
        (evt) => {
          const t = Date.now() - runStartRef.current;
          const d: any = evt.data ?? {};
          switch (evt.type) {
            case "run_start":
              setTrajectory((tr) => [...tr, { t, kind: "run_start", summary: `${d.total_nodes} nodes · ${d.layers?.length ?? 0} layers` }]);
              break;
            case "node_start":
              markNodeRunning(d.node_id);
              setTrajectory((tr) => [...tr, { t, kind: "node_start", node_id: d.node_id, node_type: d.node_type }]);
              break;
            case "node_end":
              markNodeEnded(d.node_id, {
                status: d.status,
                duration_ms: d.duration_ms,
                preview: d.preview,
                error: d.error,
              });
              nodeLog.push(d);
              setTrajectory((tr) => [...tr, {
                t, kind: "node_end", node_id: d.node_id, node_type: d.node_type,
                status: d.status, duration_ms: d.duration_ms,
                preview: d.preview, error: d.error,
              }]);
              break;
            case "run_end":
              lastStatus = d.status === "success" ? "success" : "error";
              lastError = d.error;
              lastOutput = d.output;
              setTrajectory((tr) => [...tr, {
                t, kind: "run_end", status: lastStatus, duration_ms: d.duration_ms,
                summary: lastStatus === "success" ? `done in ${d.duration_ms ?? 0} ms` : (d.error ?? "failed"),
              }]);
              break;
            case "run_persisted":
              // Attach run_id once we know it so users can see it persisted.
              setRunResult((r) => ({ ...(r ?? { status: lastStatus }), run_id: d.run_id }));
              break;
            case "done":
              break;
          }
        },
        ac.signal,
      );

      setRunResult({
        run_id: undefined,
        status: lastStatus,
        output: lastOutput,
        node_log: nodeLog,
        error: lastError,
        duration_ms: Date.now() - runStartRef.current,
      });
      setStatus(lastStatus === "success" ? `${mode} ok · ${Date.now() - runStartRef.current} ms` : `${mode} failed`);
      setStatusKind(lastStatus === "success" ? "ok" : "err");
    } catch (e: any) {
      if (e?.name !== "AbortError") {
        setStatus(`${mode} failed: ${e.message ?? e}`);
        setStatusKind("err");
      }
    } finally {
      setBusy(false);
      runAbortRef.current = null;
    }
  }

  // ── Inspector ─────────────────────────────────────────────────

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedId) ?? null,
    [nodes, selectedId],
  );

  function updateSelectedData(patch: Record<string, any>) {
    if (!selectedNode) return;
    setNodes((nds) =>
      nds.map((n) =>
        n.id === selectedNode.id
          ? { ...n, data: { ...(n.data as any), ...patch } }
          : n,
      ),
    );
  }

  function renameSelected(newId: string) {
    if (!selectedNode) return;
    if (!newId || nodes.some((n) => n.id === newId && n.id !== selectedNode.id)) return;
    const oldId = selectedNode.id;
    setNodes((nds) => nds.map((n) => (n.id === oldId ? { ...n, id: newId } : n)));
    setEdges((eds) =>
      eds.map((e) => ({
        ...e,
        source: e.source === oldId ? newId : e.source,
        target: e.target === oldId ? newId : e.target,
      })),
    );
    setSelectedId(newId);
  }

  function deleteSelected() {
    if (!selectedNode) return;
    const id = selectedNode.id;
    setNodes((nds) => nds.filter((n) => n.id !== id));
    setEdges((eds) => eds.filter((e) => e.source !== id && e.target !== id));
    setSelectedId(null);
  }

  // ── Render ────────────────────────────────────────────────────

  const wire = useMemo(() => toWire(nodes, edges), [nodes, edges]);

  return (
    <div className="flex h-[calc(100vh-56px)] gap-3 text-[13px]">
      {/* Left rail — palette + saved workflows */}
      <div className="w-60 flex flex-col gap-3 overflow-y-auto">
        <div className="wf-palette">
          <div className="wf-palette__title">
            <span>Components</span>
            <span className="wf-palette__hint">drag →</span>
          </div>
          <div className="wf-palette__list">
            {PALETTE_ORDER.map((t) => {
              const m = NODE_META[t];
              return (
                <div
                  key={t}
                  draggable
                  onDragStart={(e) => onDragStart(e, t)}
                  className="wf-palette-item"
                  style={{ ["--wf-accent" as any]: m.color } as React.CSSProperties}
                  title={m.description}
                >
                  <span className="wf-palette-item__icon">
                    <NodeIcon type={t} size={15} />
                  </span>
                  <div className="wf-palette-item__text">
                    <div className="wf-palette-item__label">{m.label}</div>
                    <div className="wf-palette-item__group">{m.group}</div>
                  </div>
                  <span className="wf-palette-item__grip" aria-hidden>⋮⋮</span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="wf-palette">
          <div className="wf-palette__title">
            <span>Saved ({savedList.length})</span>
            <button
              onClick={newBlank}
              className="wf-palette__hint hover:text-[var(--text)] transition"
              style={{ background: "transparent", border: 0, cursor: "pointer" }}
            >
              + New
            </button>
          </div>
          <div className="flex flex-col gap-1 max-h-80 overflow-y-auto">
            {savedList.length === 0 && (
              <div className="text-[11px] text-[var(--text-muted)]">No saved workflows yet.</div>
            )}
            {savedList.map((w) => {
              const active = loaded?.id === w.id;
              return (
                <button
                  key={w.id}
                  onClick={() => loadWorkflow(w.id)}
                  className={`wf-saved__item ${active ? "active" : ""}`}
                  title={w.description || w.name}
                >
                  <span className="wf-saved__dot" />
                  <span className="truncate flex-1">{w.name}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 flex flex-col">
        <div className="flex items-center gap-2 mb-2 flex-wrap">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="px-3 py-1.5 rounded-md border border-[var(--border)] bg-white text-[13px] font-medium flex-1 min-w-[180px]"
            placeholder="Workflow name"
          />
          <input
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="px-3 py-1.5 rounded-md border border-[var(--border)] bg-white text-[12px] text-[var(--text-muted)] flex-1 min-w-[200px]"
            placeholder="Short description"
          />
          <button
            onClick={save}
            disabled={busy}
            className="px-3 py-1.5 rounded-md bg-[var(--text)] text-white text-[12px] font-medium disabled:opacity-50"
          >
            {loaded ? "Update" : "Save"}
          </button>
          <button
            onClick={() => setJsonOpen(true)}
            className="px-3 py-1.5 rounded-md border border-[var(--border)] bg-white text-[12px]"
          >
            View JSON
          </button>
          {loaded && (
            <button
              onClick={deleteLoaded}
              className="px-3 py-1.5 rounded-md border border-[var(--border)] bg-white text-[12px] text-red-600"
            >
              Delete
            </button>
          )}
          {status && (
            <div
              className={`text-[12px] ml-1 truncate max-w-[280px] ${
                statusKind === "err" ? "text-red-600" : statusKind === "ok" ? "text-emerald-600" : "text-[var(--text-muted)]"
              }`}
              title={status}
            >
              {status}
            </div>
          )}
        </div>

        <div
          className="flex-1 min-h-[420px] border border-[var(--border)] rounded-lg bg-[#fafafa] relative overflow-hidden"
          onDragOver={onDragOver}
          onDrop={onDrop}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            onNodeClick={(_, n) => setSelectedId(n.id)}
            onPaneClick={() => setSelectedId(null)}
            fitView
            defaultEdgeOptions={{ type: "smoothstep" }}
            proOptions={{ hideAttribution: true }}
          >
            <MiniMap pannable zoomable maskColor="rgba(24,24,27,0.05)" />
            <Controls showInteractive={false} />
            <Background
              variant={BackgroundVariant.Dots}
              gap={18}
              size={1.2}
              color="#d4d4d8"
            />
          </ReactFlow>
        </div>

        {/* Trajectory & Result */}
        {(trajectory.length > 0 || runResult) && (
          <div className="mt-3 grid grid-cols-2 gap-3 h-56 shrink-0">
            {/* Live trajectory */}
            <div className="border border-[var(--border)] rounded-lg bg-white p-3 overflow-y-auto">
              <div className="flex items-center justify-between mb-2">
                <div className="text-[11px] font-semibold tracking-wide uppercase text-[var(--text-muted)]">
                  Live Trajectory
                </div>
                <button
                  onClick={() => { setTrajectory([]); setRunResult(null); resetRunState(); }}
                  className="text-[11px] text-[var(--text-muted)] hover:text-[var(--text)]"
                >
                  clear
                </button>
              </div>
              <div className="space-y-1">
                {trajectory.map((ev, i) => (
                  <TrajectoryRow key={i} ev={ev} />
                ))}
                {trajectory.length === 0 && (
                  <div className="text-[11px] text-[var(--text-muted)]">
                    Run the workflow to see a step-by-step trace.
                  </div>
                )}
              </div>
            </div>

            {/* Final run result */}
            <div className="border border-[var(--border)] rounded-lg bg-white p-3 overflow-y-auto">
              <div className="text-[11px] font-semibold tracking-wide uppercase text-[var(--text-muted)] mb-2">
                Result {runResult ? `· ${runResult.status}` : ""}
              </div>
              {!runResult && (
                <div className="text-[11px] text-[var(--text-muted)]">Waiting for run to finish…</div>
              )}
              {runResult?.error && (
                <div className="text-[12px] text-red-600 mb-2 whitespace-pre-wrap">{runResult.error}</div>
              )}
              {runResult?.output?.answer && (
                <div className="text-[12px] whitespace-pre-wrap mb-2">
                  <span className="font-semibold text-[var(--text)]">Answer: </span>
                  {runResult.output.answer}
                </div>
              )}
              {runResult?.output && !runResult.output.answer && (
                <pre className="text-[11px] bg-[var(--bg-hover)] p-2 rounded overflow-x-auto">
                  {JSON.stringify(runResult.output, null, 2)}
                </pre>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Inspector */}
      <div className="w-80 flex flex-col gap-3 overflow-y-auto">
        <div className="border border-[var(--border)] rounded-lg bg-white p-3">
          <div className="text-[11px] font-semibold tracking-wide uppercase text-[var(--text-muted)] mb-2">
            Inspector
          </div>
          {selectedNode ? (
            <Inspector
              node={selectedNode}
              onChange={updateSelectedData}
              onRename={renameSelected}
              onDelete={deleteSelected}
              catalog={catalog}
              llmModels={llmModels}
            />
          ) : (
            <div className="text-[12px] text-[var(--text-muted)]">
              Click a node on the canvas to edit it, or drag a new node from the palette.
            </div>
          )}
        </div>

        {/* Run panel */}
        <div className="border border-[var(--border)] rounded-lg bg-white p-3">
          <div className="text-[11px] font-semibold tracking-wide uppercase text-[var(--text-muted)] mb-2">
            Run
          </div>
          <button
            disabled={busy || !loaded}
            onClick={() => runMode("ingest")}
            className="w-full mb-2 px-3 py-2 rounded-md bg-[#059669] text-white text-[12px] font-medium disabled:opacity-50"
          >
            1 · Run Ingest
          </button>
          <textarea
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            rows={3}
            placeholder="User query (for query-mode run)"
            className="w-full mb-2 px-2 py-1.5 rounded-md border border-[var(--border)] text-[12px]"
          />
          <button
            disabled={busy || !loaded || !queryInput.trim()}
            onClick={() => runMode("query")}
            className="w-full px-3 py-2 rounded-md bg-[#dc2626] text-white text-[12px] font-medium disabled:opacity-50"
          >
            2 · Run Query
          </button>
          {!loaded && (
            <div className="text-[11px] text-[var(--text-muted)] mt-2">
              Save the workflow before running.
            </div>
          )}
          {busy && (
            <div className="text-[11px] text-[var(--text-muted)] mt-2 flex items-center gap-2">
              <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              streaming live events…
            </div>
          )}
        </div>
      </div>

      {/* JSON drawer */}
      {jsonOpen && (
        <div
          className="fixed inset-0 bg-black/20 z-40 flex justify-end"
          onClick={() => setJsonOpen(false)}
        >
          <div
            className="w-[560px] h-full bg-white shadow-xl p-5 overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="text-[13px] font-semibold">Flow JSON</div>
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(JSON.stringify(wire, null, 2));
                  }}
                  className="px-2 py-1 rounded-md border border-[var(--border)] text-[11px]"
                >
                  Copy
                </button>
                <button
                  onClick={() => setJsonOpen(false)}
                  className="px-2 py-1 rounded-md border border-[var(--border)] text-[11px]"
                >
                  Close
                </button>
              </div>
            </div>
            <div className="text-[11px] text-[var(--text-muted)] mb-2">
              This is the exact payload stored in <code>workflow_definitions.graph_json</code>.
            </div>
            <pre className="text-[11px] bg-[var(--bg-hover)] p-3 rounded overflow-x-auto">
              {JSON.stringify(wire, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Live trajectory row ─────────────────────────────────────────

function TrajectoryRow({ ev }: { ev: TrajEvent }) {
  const kindStyle: Record<string, string> = {
    run_start: "text-[var(--text-muted)]",
    node_start: "text-[var(--text)]",
    node_end: ev.status === "error" ? "text-red-600" : "text-emerald-700",
    run_end: ev.status === "error" ? "text-red-600" : "text-emerald-700",
  };
  const icon: Record<string, string> = {
    run_start: "▶",
    node_start: "•",
    node_end: ev.status === "error" ? "✗" : "✓",
    run_end: ev.status === "error" ? "■" : "■",
  };
  const t = (ev.t / 1000).toFixed(1);

  return (
    <div className={`flex items-start gap-2 text-[11px] ${kindStyle[ev.kind]}`}>
      <span className="w-9 tabular-nums text-right text-[var(--text-muted)]">{t}s</span>
      <span className="w-3 text-center">{icon[ev.kind]}</span>
      <span className="flex-1 min-w-0">
        {ev.kind === "run_start" && <span>run started · {ev.summary}</span>}
        {ev.kind === "node_start" && (
          <span>
            <span className="font-mono">{ev.node_id}</span>{" "}
            <span className="text-[var(--text-muted)]">({ev.node_type})</span> started
          </span>
        )}
        {ev.kind === "node_end" && (
          <span>
            <span className="font-mono">{ev.node_id}</span>{" "}
            <span className="text-[var(--text-muted)]">({ev.node_type})</span>{" "}
            {ev.status} · {ev.duration_ms} ms
            {ev.preview && (
              <span className="block truncate text-[var(--text-muted)] font-mono">
                → {ev.preview}
              </span>
            )}
            {ev.error && (
              <span className="block text-red-600 break-all">
                ⚠ {ev.error}
              </span>
            )}
          </span>
        )}
        {ev.kind === "run_end" && <span>run {ev.status} · {ev.summary}</span>}
      </span>
    </div>
  );
}

// ── Inspector form per node type ────────────────────────────────

function Inspector({
  node,
  onChange,
  onRename,
  onDelete,
  catalog,
  llmModels,
}: {
  node: Node;
  onChange: (patch: Record<string, any>) => void;
  onRename: (newId: string) => void;
  onDelete: () => void;
  catalog: Catalog | null;
  llmModels: { id: string; name: string }[];
}) {
  const data = node.data as any;
  const t = data?.nodeType as string;
  const meta = NODE_META[t] ?? NODE_META.output;

  return (
    <div
      className="wf-inspector"
      style={{ ["--wf-accent" as any]: meta.color } as React.CSSProperties}
    >
      <div className="wf-inspector__head">
        <span className="wf-node__icon">
          <NodeIcon type={t} size={15} />
        </span>
        <div className="wf-inspector__head-text">
          <div className="wf-inspector__title">{meta.label}</div>
          <div className="wf-inspector__sub">{node.id}</div>
        </div>
      </div>

      <div className="text-[11px] text-[#71717a] leading-snug -mt-1">
        {meta.description}
      </div>

      <Field label="Node ID" hint="Identifier used in the graph">
        <input
          defaultValue={node.id}
          onBlur={(e) => onRename(e.currentTarget.value.trim())}
          className="wf-input mono"
        />
      </Field>

      {t === "data_source" && (
        <>
          <Field label="Source Type" required>
            <select
              value={data.source_type ?? "text"}
              onChange={(e) => onChange({ source_type: e.target.value })}
              className="wf-select"
            >
              {(catalog?.source_types ?? ["text", "file", "url"]).map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </Field>
          <Field label={data.source_type === "url" ? "URL" : data.source_type === "file" ? "Path" : "Text"} required>
            <textarea
              value={data.content ?? ""}
              onChange={(e) => onChange({ content: e.target.value })}
              rows={4}
              className="wf-textarea mono"
            />
          </Field>
        </>
      )}

      {t === "chunker" && (
        <>
          <Field label="Strategy" required>
            <select
              value={data.strategy ?? "recursive"}
              onChange={(e) => onChange({ strategy: e.target.value })}
              className="wf-select"
            >
              {(catalog?.chunk_strategies ?? ["recursive", "fixed", "semantic", "code"]).map(
                (s) => (
                  <option key={s} value={s}>{s}</option>
                ),
              )}
            </select>
          </Field>
          <Field label="Chunk Size">
            <input
              type="number"
              value={data.chunk_size ?? 500}
              onChange={(e) => onChange({ chunk_size: Number(e.target.value) })}
              className="wf-input"
            />
          </Field>
          <Field label="Chunk Overlap">
            <input
              type="number"
              value={data.chunk_overlap ?? 50}
              onChange={(e) => onChange({ chunk_overlap: Number(e.target.value) })}
              className="wf-input"
            />
          </Field>
        </>
      )}

      {t === "embedder" && (
        <Field label="Embedding Model" required>
          <select
            value={data.model ?? ""}
            onChange={(e) => onChange({ model: e.target.value })}
            className="wf-select"
          >
            {Object.entries(catalog?.embedding_models ?? {}).map(([k]) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
        </Field>
      )}

      {t === "vector_store" && (
        <>
          <Field label="Store Type" required>
            <select
              value={data.store_type ?? "chroma"}
              onChange={(e) => onChange({ store_type: e.target.value })}
              className="wf-select"
            >
              {Object.entries(catalog?.vector_stores ?? {}).map(([k, v]: any) => (
                <option key={k} value={k}>{v.label ?? k}</option>
              ))}
            </select>
          </Field>
          <Field label="Collection Name">
            <input
              value={data.collection_name ?? ""}
              onChange={(e) => onChange({ collection_name: e.target.value })}
              className="wf-input mono"
            />
          </Field>
          <Field label="Persist Directory" hint="Optional — where embeddings are stored on disk">
            <input
              value={data.persist_dir ?? ""}
              onChange={(e) => onChange({ persist_dir: e.target.value })}
              placeholder="./data/workflow_vs"
              className="wf-input mono"
            />
          </Field>
        </>
      )}

      {t === "retriever" && (
        <Field label="Top K" hint="Number of chunks to retrieve">
          <input
            type="number"
            value={data.top_k ?? 4}
            onChange={(e) => onChange({ top_k: Number(e.target.value) })}
            className="wf-input"
          />
        </Field>
      )}

      {t === "prompt_template" && (
        <Field label="Template" required hint="Use {context} and {question} as placeholders">
          <textarea
            value={data.template ?? ""}
            onChange={(e) => onChange({ template: e.target.value })}
            rows={8}
            className="wf-textarea mono"
          />
        </Field>
      )}

      {t === "llm" && (
        <>
          <Field label="Language Model" required>
            <select
              value={data.model ?? ""}
              onChange={(e) => onChange({ model: e.target.value })}
              className="wf-select"
            >
              {llmModels.map((m) => (
                <option key={m.id} value={m.id}>{m.name || m.id}</option>
              ))}
            </select>
          </Field>
          <Field label="Temperature" hint="0 = deterministic, higher = more creative">
            <input
              type="number"
              step={0.1}
              min={0}
              max={2}
              value={data.temperature ?? 0.2}
              onChange={(e) => onChange({ temperature: Number(e.target.value) })}
              className="wf-input"
            />
          </Field>
        </>
      )}

      <div className="pt-1">
        <button onClick={onDelete} className="wf-delete-btn">
          Delete node
        </button>
      </div>
    </div>
  );
}

function Field({
  label,
  children,
  required,
  hint,
}: {
  label: string;
  children: React.ReactNode;
  required?: boolean;
  hint?: string;
}) {
  return (
    <div className="wf-field">
      <div className="wf-field__label">
        {label}
        {required && <span className="wf-row__req">*</span>}
        {hint && <span className="wf-row__info" title={hint}>i</span>}
      </div>
      {children}
    </div>
  );
}
