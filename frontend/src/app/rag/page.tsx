"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useTeam } from "@/contexts/TeamContext";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface EmbeddingModelMeta {
  provider: string; dimensions: number; max_tokens: number;
  cost_per_1m_tokens: number; description: string; recommended?: boolean;
}
interface VectorStoreMeta { label: string; description: string; }
interface RerankerMeta { label: string; description: string; hf_id: string | null; }
interface RagModels {
  embedding_models: Record<string, EmbeddingModelMeta>;
  vector_stores: Record<string, VectorStoreMeta>;
  chunk_strategies: string[];
  retrieval_strategies: string[];
  reranker_models?: Record<string, RerankerMeta>;
}
interface RagSource {
  id: string; source_type: string; content: string; label: string;
  chunks_count: number; tokens_estimated: number; status: string;
  error_message?: string; ingested_at?: string;
}
interface RagConfig {
  id: string; name: string; description: string;
  embedding_model: string; vector_store: string; llm_model?: string;
  chunk_size: number; chunk_overlap: number; chunk_strategy: string;
  retrieval_strategy: string; top_k: number; mmr_lambda: number;
  multi_query_n: number; system_prompt?: string; reranker: string;
  team_id?: string | null;
  is_active: boolean; created_at?: string; sources: RagSource[];
}
interface Citation {
  source: string; chunk_index: number; total_chunks: number;
  page?: number; score: number; snippet: string;
}
interface EvalScore { score: number; passed: boolean; reason: string; }
interface StoredQuery {
  id: string; query: string; answer: string;
  citations: Citation[]; strategy_used: string;
  chunks_retrieved: number; tokens_in: number; tokens_out: number;
  latency_ms: number; eval_scores: Record<string, EvalScore> | null;
  eval_status: string; eval_error?: string; trace_id?: string;
  created_at: string;
}
interface OtelSpan {
  span_id: string; parent_span_id?: string; name: string; span_type: string;
  start_time?: string; end_time?: string; duration_ms?: number; status: string;
  model?: string; tokens_in?: number; tokens_out?: number; cost?: number;
  attributes?: Record<string, unknown>; error?: string;
}
interface LLMModel { id: string; name: string; }

// ─────────────────────────────────────────────────────────────────────────────
// API helpers
// ─────────────────────────────────────────────────────────────────────────────

const API = "http://localhost:8000";
const apiFetch = async (path: string, opts?: RequestInit) => {
  const r = await fetch(`${API}${path}`, opts);
  if (!r.ok && r.status !== 204) throw new Error(await r.text());
  return r.status === 204 ? null : r.json();
};
const apiGet = <T,>(p: string) => apiFetch(p) as Promise<T>;
const apiPost = <T,>(p: string, b: unknown) =>
  apiFetch(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) }) as Promise<T>;
const apiPut = <T,>(p: string, b: unknown) =>
  apiFetch(p, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) }) as Promise<T>;
const apiDelete = (p: string) => apiFetch(p, { method: "DELETE" });

const isUrl = (s: string) => s.startsWith("http://") || s.startsWith("https://");
const domainOf = (s: string) => { try { return new URL(s).hostname; } catch { return s; } };

// ─────────────────────────────────────────────────────────────────────────────
// Primitives
// ─────────────────────────────────────────────────────────────────────────────

function Spinner({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className="animate-spin text-zinc-400 inline-block">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5"
        strokeLinecap="round" strokeDasharray="31.4" strokeDashoffset="10" />
    </svg>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  async function copy(e: React.MouseEvent) {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* ignore */ }
  }
  return (
    <button
      onClick={copy}
      title="Copy response"
      className="absolute -bottom-6 right-0 opacity-0 group-hover:opacity-100 transition-opacity
                 text-[10px] text-zinc-400 hover:text-zinc-600 flex items-center gap-1
                 bg-white border border-zinc-100 rounded-md px-2 py-0.5 shadow-sm"
    >
      {copied ? (
        <><span>✓</span> Copied</>
      ) : (
        <><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy</>
      )}
    </button>
  );
}

function Badge({ children, variant = "default" }: {
  children: React.ReactNode;
  variant?: "default" | "ok" | "warn" | "error" | "err" | "blue" | "emerald";
}) {
  const cls: Record<string, string> = {
    default: "bg-zinc-100 text-zinc-500",
    ok: "bg-emerald-50 text-emerald-700",
    warn: "bg-amber-50 text-amber-600",
    error: "bg-red-50 text-red-500",
    err: "bg-red-50 text-red-500",
    blue: "bg-blue-50 text-blue-600",
    emerald: "bg-emerald-50 text-emerald-700",
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium ${cls[variant]}`}>
      {children}
    </span>
  );
}

function ScoreBar({ label, score, passed, reason }: { label: string; score: number; passed: boolean; reason: string }) {
  const [open, setOpen] = useState(false);
  const pct = Math.round(score * 100);
  const isErr = reason?.startsWith("ERROR");
  return (
    <div className="space-y-1">
      <button className="w-full" onClick={() => setOpen(o => !o)}>
        <div className="flex items-center justify-between">
          <span className="text-xs text-zinc-500 capitalize">{label.replace(/_/g, " ")}</span>
          <div className="flex items-center gap-1.5">
            <span className={`text-xs font-mono font-medium ${passed ? "text-emerald-600" : isErr ? "text-red-500" : "text-amber-500"}`}>{pct}%</span>
            <Badge variant={passed ? "ok" : isErr ? "err" : "warn"}>{passed ? "pass" : "fail"}</Badge>
            {reason && <span className="text-[9px] text-zinc-300">{open ? "▲" : "▼"}</span>}
          </div>
        </div>
        <div className="h-1.5 bg-zinc-100 rounded-full overflow-hidden mt-1">
          <div className={`h-full rounded-full transition-all ${passed ? "bg-emerald-400" : isErr ? "bg-red-300" : "bg-amber-400"}`}
            style={{ width: `${pct}%` }} />
        </div>
      </button>
      {open && reason && (
        <div className={`text-[10px] leading-relaxed px-2 py-1.5 rounded ${isErr ? "bg-red-50 text-red-500" : "bg-zinc-50 text-zinc-500"}`}>
          {reason}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Citations panel — shows each retrieved chunk with click-to-expand metric reasons
// ─────────────────────────────────────────────────────────────────────────────

const CONTEXT_METRICS = ["contextual_relevancy", "contextual_precision", "contextual_recall", "faithfulness"];
const METRIC_LABELS: Record<string, string> = {
  contextual_relevancy: "Contextual Relevancy",
  contextual_precision: "Contextual Precision",
  contextual_recall: "Contextual Recall",
  faithfulness: "Faithfulness",
  answer_relevancy: "Answer Relevancy",
};

function CitationsPanel({ citations, evalScores }: {
  citations: Citation[] | undefined;
  evalScores: Record<string, EvalScore> | null | undefined;
}) {
  const [expanded, setExpanded] = useState<number | null>(null);

  if (!citations || citations.length === 0) {
    return <p className="text-xs text-zinc-400">No citations for this query.</p>;
  }

  // Context-level metric reasons (shown when expanding a citation)
  const ctxReasons = evalScores
    ? CONTEXT_METRICS.filter(k => evalScores[k]).map(k => ({
        key: k,
        label: METRIC_LABELS[k] || k,
        score: evalScores[k].score,
        passed: evalScores[k].passed,
        reason: evalScores[k].reason,
      }))
    : [];

  return (
    <div className="space-y-2">
      {citations.map((c, i) => {
        const isOpen = expanded === i;
        const pct = Math.max(4, Math.round(Math.abs(c.score) * 100));
        const good = c.score > 0.65;
        return (
          <div key={i}
            className={`border rounded-lg overflow-hidden transition-all cursor-pointer ${
              isOpen ? "border-zinc-300 bg-zinc-50" : "border-zinc-100 hover:border-zinc-200"
            }`}
            onClick={() => setExpanded(isOpen ? null : i)}
          >
            {/* Citation header */}
            <div className="p-3 space-y-1.5">
              <div className="flex items-start gap-2">
                <span className="text-[11px] font-mono font-bold text-zinc-400 shrink-0 mt-0.5">[{i + 1}]</span>
                <div className="flex-1 min-w-0">
                  {isUrl(c.source) ? (
                    <a href={c.source} target="_blank" rel="noopener noreferrer"
                      onClick={e => e.stopPropagation()}
                      className="text-xs text-zinc-700 hover:underline break-all leading-snug">
                      {domainOf(c.source)}<span className="ml-1 text-zinc-400">↗</span>
                    </a>
                  ) : (
                    <span className="text-xs text-zinc-600 break-all leading-snug">
                      {c.source.split("/").pop() || c.source}
                    </span>
                  )}
                  {c.page && <span className="ml-1.5 text-[10px] text-zinc-400">p.{c.page}</span>}
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  <span className="text-[11px] font-mono text-zinc-400">{Math.round(c.score * 100)}%</span>
                  <span className="text-[9px] text-zinc-300">{isOpen ? "▲" : "▼"}</span>
                </div>
              </div>
              <div className="h-1 bg-zinc-100 rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${good ? "bg-emerald-400" : "bg-amber-300"}`}
                  style={{ width: `${pct}%` }} />
              </div>
              <p className="text-[11px] text-zinc-400 leading-relaxed italic line-clamp-3">"{c.snippet}"</p>
            </div>

            {/* Expanded: show metric reasons for this context */}
            {isOpen && (
              <div className="px-3 pb-3 border-t border-zinc-100 pt-2.5 space-y-2.5">
                <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wide">Scoring Reasons</p>
                {ctxReasons.length === 0 && (
                  <p className="text-[10px] text-zinc-300 italic">
                    No evaluation run yet — enable auto-evaluate when sending queries.
                  </p>
                )}
                {ctxReasons.map(m => (
                  <div key={m.key} className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] text-zinc-500">{m.label}</span>
                      <div className="flex items-center gap-1">
                        <span className={`text-[10px] font-mono font-medium ${m.passed ? "text-emerald-600" : "text-amber-500"}`}>
                          {Math.round(m.score * 100)}%
                        </span>
                        <span className={`text-[9px] px-1 py-0.5 rounded font-medium ${m.passed ? "bg-emerald-50 text-emerald-600" : "bg-amber-50 text-amber-600"}`}>
                          {m.passed ? "pass" : "fail"}
                        </span>
                      </div>
                    </div>
                    {m.reason && !m.reason.startsWith("ERROR") && (
                      <p className="text-[10px] text-zinc-400 leading-relaxed bg-zinc-50 rounded px-2 py-1.5">
                        {m.reason}
                      </p>
                    )}
                    {m.reason?.startsWith("ERROR") && (
                      <p className="text-[10px] text-red-400 leading-relaxed bg-red-50 rounded px-2 py-1.5">
                        {m.reason}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Right detail panel — always visible, content driven by selected query
// ─────────────────────────────────────────────────────────────────────────────

function DetailPanel({ query, onRunEval, onClose }: {
  query: StoredQuery | null;
  onRunEval: (q: StoredQuery) => void;
  onClose: () => void;
}) {
  const [spans, setSpans] = useState<OtelSpan[]>([]);
  const [traceLoaded, setTraceLoaded] = useState(false);
  const [traceTab, setTraceTab] = useState<"eval" | "trace" | "citations">("eval");
  const [loadingTrace, setLoadingTrace] = useState(false);
  const hasSpans = spans.length > 0;

  useEffect(() => {
    setSpans([]);
    setTraceLoaded(false);
    setTraceTab("eval");
  }, [query?.id]);

  useEffect(() => {
    if (!query?.trace_id || traceTab !== "trace" || traceLoaded) return;
    setLoadingTrace(true);
    apiGet<OtelSpan[]>(`/api/rag/traces/${query.trace_id}`)
      .then((s) => { setSpans(s); setTraceLoaded(true); })
      .catch(() => setTraceLoaded(true))
      .finally(() => setLoadingTrace(false));
  }, [query?.trace_id, traceTab, traceLoaded]);

  if (!query) {
    return (
      <aside className="w-72 shrink-0 border-l border-zinc-100 bg-zinc-50/50 flex flex-col items-center
                        justify-center text-zinc-300 text-sm gap-2">
        <div className="text-3xl">↖</div>
        <p>Click a response to inspect</p>
      </aside>
    );
  }

  // Available tabs (hide Trace if no trace data and already loaded)
  const showTrace = !!query.trace_id;
  const tabs = [
    { key: "eval" as const, label: "Evaluation" },
    ...(showTrace ? [{ key: "trace" as const, label: "Trace" }] : []),
    { key: "citations" as const, label: `Citations (${query.citations?.length ?? 0})` },
  ];

  const avgScore = query.eval_scores
    ? Object.values(query.eval_scores).reduce((a, v) => a + v.score, 0) /
      Object.keys(query.eval_scores).length
    : null;

  return (
    <aside className="w-72 shrink-0 border-l border-zinc-100 bg-white flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b border-zinc-100">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium text-zinc-700 leading-snug line-clamp-2">{query.query}</p>
            <div className="flex items-center gap-2 mt-1 flex-wrap">
              <span className="text-[10px] text-zinc-400">
                {new Date(query.created_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
              </span>
              <span className="text-[10px] text-zinc-400">{query.latency_ms}ms</span>
              {avgScore !== null && (
                <Badge variant={avgScore >= 0.5 ? "ok" : "warn"}>
                  avg {Math.round(avgScore * 100)}%
                </Badge>
              )}
            </div>
          </div>
          <button onClick={onClose} className="text-zinc-300 hover:text-zinc-600 text-lg leading-none shrink-0">×</button>
        </div>
      </div>

      {/* Sub-tabs */}
      <div className="flex border-b border-zinc-100">
        {tabs.map((t) => (
          <button key={t.key} onClick={() => setTraceTab(t.key)}
            className={`flex-1 py-2 text-[11px] font-medium transition-colors border-b-2 -mb-px whitespace-nowrap px-1 ${
              traceTab === t.key
                ? "border-zinc-900 text-zinc-900"
                : "border-transparent text-zinc-400 hover:text-zinc-600"
            }`}>
            {t.label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* ── Evaluation ── */}
        {traceTab === "eval" && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Badge variant={
                query.eval_status === "done" ? "ok" :
                query.eval_status === "running" ? "blue" :
                query.eval_status === "error" ? "error" : "default"
              }>
                {query.eval_status === "running" && <Spinner size={10} />}
                {query.eval_status === "done" ? "Evaluated" :
                  query.eval_status === "running" ? "Running…" :
                  query.eval_status === "error" ? "Failed" : "Pending"}
              </Badge>
              <button onClick={() => onRunEval(query)}
                className="text-xs text-zinc-400 hover:text-zinc-900 border border-zinc-200
                           hover:border-zinc-400 px-2 py-1 rounded-md transition-colors">
                {query.eval_status === "done" ? "Re-run" : "Evaluate"}
              </button>
            </div>

            {query.eval_status === "error" && query.eval_error && (
              <div className="p-2 bg-red-50 border border-red-100 rounded-lg">
                <p className="text-xs text-red-600 leading-snug">{query.eval_error}</p>
              </div>
            )}

            {query.eval_scores && (
              <div className="space-y-3">
                {Object.entries(query.eval_scores).map(([k, v]) => (
                  <ScoreBar key={k} label={k} score={v.score} passed={v.passed} reason={v.reason} />
                ))}
              </div>
            )}

            {!query.eval_scores && query.eval_status === "pending" && (
              <p className="text-xs text-zinc-400 leading-relaxed">
                Auto-evaluation is enabled by default. Toggle the checkbox in the chat toolbar, or click "Evaluate" to run manually.
              </p>
            )}

            {/* Stats grid */}
            <div className="border-t border-zinc-100 pt-3 grid grid-cols-2 gap-x-4 gap-y-3">
              {[
                { l: "Chunks", v: query.chunks_retrieved },
                { l: "Strategy", v: query.strategy_used },
                { l: "Tokens in", v: query.tokens_in },
                { l: "Tokens out", v: query.tokens_out },
              ].map(({ l, v }) => (
                <div key={l}>
                  <p className="text-[10px] uppercase tracking-wide text-zinc-400 mb-0.5">{l}</p>
                  <p className="text-sm font-medium text-zinc-700">{v || "—"}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Trace ── */}
        {traceTab === "trace" && (
          <div className="space-y-2">
            {loadingTrace && (
              <div className="flex items-center gap-2 text-xs text-zinc-400"><Spinner />Loading spans…</div>
            )}
            {!loadingTrace && spans.length === 0 && (
              <p className="text-xs text-zinc-400">No spans recorded for this query.</p>
            )}
            {spans.map((s) => (
              <div key={s.span_id} className="border border-zinc-100 rounded-lg p-3 space-y-1">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-medium text-zinc-700 truncate">{s.name}</span>
                  <Badge variant={s.status === "ok" ? "ok" : "error"}>{s.status}</Badge>
                </div>
                <div className="flex flex-wrap gap-2">
                  {s.duration_ms != null && (
                    <span className="text-[10px] text-zinc-400">{Math.round(s.duration_ms)}ms</span>
                  )}
                  {s.model && <span className="text-[10px] text-zinc-400 truncate max-w-[120px]">{s.model}</span>}
                  {(s.tokens_in || s.tokens_out) && (
                    <span className="text-[10px] text-zinc-400">↑{s.tokens_in ?? 0} ↓{s.tokens_out ?? 0}</span>
                  )}
                </div>
                {s.error && <p className="text-[10px] text-red-500 leading-snug">{s.error}</p>}
              </div>
            ))}
          </div>
        )}

        {/* ── Citations ── */}
        {traceTab === "citations" && (
          <CitationsPanel citations={query.citations} evalScores={query.eval_scores} />
        )}
      </div>
    </aside>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat tab
// ─────────────────────────────────────────────────────────────────────────────

function ChatTab({ config }: { config: RagConfig }) {
  const [history, setHistory] = useState<StoredQuery[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<StoredQuery | null>(null);
  const [autoEval, setAutoEval] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadHistory = useCallback(async () => {
    try {
      const rows = await apiGet<StoredQuery[]>(`/api/rag/configs/${config.id}/history`);
      setHistory(rows);
      setSelected((prev) => {
        if (!prev) return rows.length > 0 ? rows[rows.length - 1] : null;
        const updated = rows.find((r) => r.id === prev.id);
        return updated ?? prev;
      });
    } catch { /* ignore */ }
  }, [config.id]);

  useEffect(() => { loadHistory(); }, [loadHistory]);

  // Poll while any query is being evaluated
  useEffect(() => {
    const hasPending = history.some((q) => q.eval_status === "pending" || q.eval_status === "running");
    if (hasPending && !pollRef.current) {
      pollRef.current = setInterval(loadHistory, 3000);
    } else if (!hasPending && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; } };
  }, [history, loadHistory]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [history.length]);

  async function send() {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setLoading(true);

    // Optimistically show the user's question immediately — replaced by real data after response
    const optimisticId = "pending-" + Date.now();
    const optimistic: StoredQuery = {
      id: optimisticId, query: q, answer: "",
      citations: [], strategy_used: "", chunks_retrieved: 0,
      tokens_in: 0, tokens_out: 0, latency_ms: 0,
      eval_scores: null, eval_status: "pending", trace_id: undefined,
      created_at: new Date().toISOString(),
    };
    setHistory((h) => [...h, optimistic]);

    try {
      await apiPost("/api/rag/chat", { query: q, config_id: config.id, auto_evaluate: autoEval });
      await loadHistory(); // replaces optimistic entry
    } catch (e: unknown) {
      const err: StoredQuery = {
        id: optimisticId, query: q,
        answer: `**Error:** ${e instanceof Error ? e.message : String(e)}`,
        citations: [], strategy_used: "", chunks_retrieved: 0,
        tokens_in: 0, tokens_out: 0, latency_ms: 0,
        eval_scores: null, eval_status: "error", trace_id: undefined,
        created_at: new Date().toISOString(),
      };
      setHistory((h) => h.map((m) => m.id === optimisticId ? err : m));
      setSelected(err);
    } finally { setLoading(false); }
  }

  async function clearHistory() {
    if (!confirm("Clear all conversation history?")) return;
    await apiDelete(`/api/rag/configs/${config.id}/history`);
    setHistory([]);
    setSelected(null);
  }

  async function runEval(q: StoredQuery) {
    await apiPost(`/api/rag/queries/${q.id}/evaluate`, {});
    // optimistically mark running
    setHistory((h) => h.map((m) => m.id === q.id ? { ...m, eval_status: "running" } : m));
    setSelected((s) => s?.id === q.id ? { ...s, eval_status: "running" } : s);
    await loadHistory();
  }

  const totalChunks = config.sources.reduce((a, s) => a + s.chunks_count, 0);

  return (
    <div className="flex h-[calc(100vh-172px)] overflow-hidden">
      {/* Messages column */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Toolbar */}
        <div className="flex items-center gap-3 pb-2 mb-3 border-b border-zinc-100 shrink-0">
          <span className="text-xs text-zinc-400">{totalChunks.toLocaleString()} chunks · {history.length} messages</span>
          <div className="flex-1" />
          <label className="flex items-center gap-1.5 text-xs text-zinc-400 cursor-pointer select-none">
            <input type="checkbox" className="accent-zinc-900"
              checked={autoEval} onChange={(e) => setAutoEval(e.target.checked)} />
            Auto-evaluate
          </label>
          {history.length > 0 && (
            <button onClick={clearHistory}
              className="text-xs text-zinc-400 hover:text-red-500 transition-colors border border-zinc-200
                         hover:border-red-200 px-2 py-1 rounded-md">
              Clear history
            </button>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto space-y-5 pr-2">
          {history.length === 0 && !loading && (
            <div className="flex flex-col items-center justify-center h-full text-zinc-300 gap-2 pb-20">
              <div className="text-4xl mb-1">⌕</div>
              <p className="text-sm text-zinc-500 font-medium">Ask anything about your knowledge base</p>
              <p className="text-xs text-zinc-400">Responses grounded in documents · click any reply to inspect</p>
            </div>
          )}

          {history.map((msg) => (
            <div key={msg.id} className="space-y-2">
              {/* User message */}
              <div className="flex justify-end">
                <div className="bg-zinc-900 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 max-w-[80%]">
                  <p className="text-sm leading-relaxed">{msg.query}</p>
                </div>
              </div>

              {/* Assistant message */}
              <div className="flex justify-start">
                {/* Loading skeleton while optimistic entry has no answer yet */}
                {!msg.answer && msg.id.startsWith("pending-") ? (
                  <div className="bg-white border border-zinc-100 rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm">
                    <div className="flex items-center gap-1.5">
                      {[0, 150, 300].map((d) => (
                        <span key={d} className="w-1.5 h-1.5 rounded-full bg-zinc-300 animate-bounce"
                          style={{ animationDelay: `${d}ms` }} />
                      ))}
                    </div>
                  </div>
                ) : (
                <div className="group relative max-w-[85%]">
                  <button
                    onClick={() => setSelected(selected?.id === msg.id ? null : msg)}
                    className={`w-full text-left rounded-2xl rounded-tl-sm border shadow-sm
                                 transition-all overflow-hidden
                                 ${selected?.id === msg.id
                                   ? "border-zinc-400 bg-white ring-1 ring-zinc-200"
                                   : "border-zinc-150 bg-white hover:border-zinc-300"}`}
                  >
                    {/* Formatted answer */}
                    <div className="px-5 pt-4 pb-3">
                      <div className="prose prose-sm prose-zinc max-w-none
                        [&_p]:leading-relaxed [&_p]:mb-2 [&_p:last-child]:mb-0
                        [&_ul]:my-2 [&_ol]:my-2
                        [&_li]:my-0.5 [&_li]:leading-relaxed
                        [&_code]:bg-zinc-100 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-xs [&_code]:font-mono
                        [&_pre]:bg-zinc-900 [&_pre]:text-zinc-100 [&_pre]:rounded-lg [&_pre]:p-4 [&_pre]:my-3 [&_pre]:overflow-x-auto
                        [&_pre_code]:bg-transparent [&_pre_code]:p-0
                        [&_h1]:text-base [&_h1]:font-semibold [&_h2]:text-sm [&_h2]:font-semibold [&_h3]:text-sm [&_h3]:font-medium
                        [&_blockquote]:border-l-2 [&_blockquote]:border-zinc-300 [&_blockquote]:pl-3 [&_blockquote]:text-zinc-500
                        [&_strong]:font-semibold [&_strong]:text-zinc-800
                        [&_a]:text-blue-600 [&_a:hover]:underline
                        [&_table]:text-xs [&_th]:font-medium [&_th]:text-zinc-700
                        [&_hr]:border-zinc-100">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.answer}</ReactMarkdown>
                      </div>
                    </div>

                    {/* Citations bar */}
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="px-5 pb-3 flex flex-wrap gap-1.5 border-t border-zinc-50 pt-3">
                        {msg.citations.map((c, i) => (
                          isUrl(c.source) ? (
                            <a key={i} href={c.source} target="_blank" rel="noopener noreferrer"
                              onClick={(e) => e.stopPropagation()}
                              className="inline-flex items-center gap-1 text-[11px] text-blue-600
                                         bg-blue-50 border border-blue-100 rounded-full px-2.5 py-0.5
                                         hover:bg-blue-100 transition-colors font-medium">
                              [{i + 1}] {domainOf(c.source)}
                              <span className="text-blue-400 text-[10px]">↗</span>
                            </a>
                          ) : (
                            <span key={i}
                              className="inline-flex items-center gap-1 text-[11px] text-zinc-500
                                         bg-zinc-50 border border-zinc-100 rounded-full px-2.5 py-0.5 font-medium">
                              [{i + 1}] {c.source.split("/").pop() || c.source}
                            </span>
                          )
                        ))}
                      </div>
                    )}

                    {/* Footer metadata */}
                    <div className="px-5 pb-3 flex items-center gap-3 border-t border-zinc-50 pt-2">
                      <span className="text-[10px] text-zinc-400">{msg.latency_ms}ms</span>
                      <span className="text-[10px] text-zinc-400">{msg.chunks_retrieved} chunks</span>
                      {msg.eval_status === "done" && msg.eval_scores && (
                        <Badge variant={
                          Object.values(msg.eval_scores).every(v => v.passed) ? "ok" : "warn"
                        }>
                          eval {Math.round(
                            Object.values(msg.eval_scores).reduce((a, v) => a + v.score, 0) /
                            Object.keys(msg.eval_scores).length * 100
                          )}%
                        </Badge>
                      )}
                      {msg.eval_status === "running" && (
                        <span className="text-[10px] text-zinc-400 flex items-center gap-1">
                          <Spinner size={10} /> evaluating
                        </span>
                      )}
                      <span className="ml-auto text-[10px] text-zinc-300">
                        {new Date(msg.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                      </span>
                    </div>
                  </button>

                  {/* Copy button — appears on hover */}
                  <CopyButton text={msg.answer} />
                </div>
                )}
              </div>
            </div>
          ))}

          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="mt-3 pt-3 border-t border-zinc-100 flex gap-2 shrink-0">
          <input
            className="flex-1 border border-zinc-200 rounded-xl px-4 py-3 text-sm outline-none
                       focus:border-zinc-400 transition-colors placeholder:text-zinc-300 bg-white"
            placeholder="Ask a question about your documents…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
            disabled={loading}
          />
          <button onClick={send} disabled={loading || !input.trim()}
            className="px-5 py-3 text-sm font-medium bg-zinc-900 text-white rounded-xl
                       hover:bg-zinc-700 disabled:opacity-30 transition-colors shrink-0">
            Send
          </button>
        </div>
      </div>

      {/* Right panel — always shown */}
      <div className="ml-4 shrink-0 w-72 h-full flex flex-col">
        <DetailPanel
          query={selected}
          onRunEval={runEval}
          onClose={() => setSelected(null)}
        />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Data sources (multi-source + file picker)
// ─────────────────────────────────────────────────────────────────────────────

function DataSources({ config, onRefresh }: { config: RagConfig; onRefresh: () => void }) {
  const [rows, setRows] = useState<Array<{ type: "url" | "file" | "text"; content: string; label: string }>>([
    { type: "url", content: "", label: "" },
  ]);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function addRow() { setRows((r) => [...r, { type: "url", content: "", label: "" }]); }
  function removeRow(i: number) { setRows((r) => r.filter((_, j) => j !== i)); }
  function setRow(i: number, key: string, val: string) {
    setRows((r) => r.map((row, j) => j === i ? { ...row, [key]: val } : row));
  }

  function handleFileInput(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    const newRows = files.map((f) => ({
      type: "file" as const,
      content: (f as File & { path?: string }).path || f.name,
      label: f.name,
    }));
    setRows((r) => [...r.filter((x) => x.content), ...newRows]);
    e.target.value = "";
  }

  async function ingestAll() {
    const valid = rows.filter((r) => r.content.trim());
    if (!valid.length) return;
    setLoading(true);
    const errs: string[] = [];
    await Promise.all(
      valid.map(async (row) => {
        try {
          await apiPost(`/api/rag/configs/${config.id}/ingest`, {
            source_type: row.type, content: row.content,
            label: row.label || row.content.slice(0, 60),
          });
        } catch (e: unknown) {
          errs.push(`${row.content.slice(0, 40)}: ${e instanceof Error ? e.message : String(e)}`);
        }
      })
    );
    setErrors(errs);
    setRows([{ type: "url", content: "", label: "" }]);
    setLoading(false);
    setTimeout(onRefresh, 600);
  }

  async function deleteSource(id: string) {
    await apiDelete(`/api/rag/configs/${config.id}/sources/${id}`);
    onRefresh();
  }

  const statusColor = (s: string) =>
    s === "ingested" ? "text-emerald-600 bg-emerald-50 border-emerald-100" :
    s === "error" ? "text-red-500 bg-red-50 border-red-100" :
    s === "ingesting" ? "text-blue-500 bg-blue-50 border-blue-100" :
    "text-zinc-400 bg-zinc-50 border-zinc-100";

  return (
    <div className="space-y-6 max-w-2xl">
      {config.sources.some((s) => s.status === "ingested") && (
        <div className="flex gap-8">
          {[
            { l: "Ingested", v: config.sources.filter(s => s.status === "ingested").length },
            { l: "Total chunks", v: config.sources.reduce((a, s) => a + s.chunks_count, 0).toLocaleString() },
          ].map(({ l, v }) => (
            <div key={l}>
              <p className="text-2xl font-semibold text-zinc-800 tabular-nums">{v}</p>
              <p className="text-xs text-zinc-400 mt-0.5">{l}</p>
            </div>
          ))}
        </div>
      )}

      {/* Add sources */}
      <div className="border border-zinc-200 rounded-xl p-4 space-y-3">
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium text-zinc-700">Add sources</p>
          <div className="flex gap-2">
            <button onClick={() => fileInputRef.current?.click()}
              className="text-xs text-zinc-500 hover:text-zinc-900 border border-zinc-200
                         hover:border-zinc-400 px-2 py-1 rounded-md transition-colors">
              Browse files
            </button>
            <input ref={fileInputRef} type="file" multiple
              accept=".txt,.md,.pdf,.py,.js,.ts,.json,.csv,.html" className="hidden"
              onChange={handleFileInput} />
            <button onClick={addRow}
              className="text-xs text-zinc-500 hover:text-zinc-900 border border-zinc-200
                         hover:border-zinc-400 px-2 py-1 rounded-md transition-colors">
              + Add row
            </button>
          </div>
        </div>

        {rows.map((row, i) => (
          <div key={i} className="flex gap-2 items-start">
            <select value={row.type} onChange={(e) => setRow(i, "type", e.target.value)}
              className="border border-zinc-200 rounded-lg px-2 py-2 text-xs outline-none
                         focus:border-zinc-400 shrink-0 bg-white">
              <option value="url">URL</option>
              <option value="file">File</option>
              <option value="text">Text</option>
            </select>
            {row.type === "text" ? (
              <textarea value={row.content} onChange={(e) => setRow(i, "content", e.target.value)}
                className="flex-1 border border-zinc-200 rounded-lg px-3 py-2 text-sm outline-none
                           focus:border-zinc-400 resize-none placeholder:text-zinc-300 bg-white"
                rows={2} placeholder="Paste text content…" />
            ) : (
              <input value={row.content} onChange={(e) => setRow(i, "content", e.target.value)}
                className="flex-1 border border-zinc-200 rounded-lg px-3 py-2 text-sm outline-none
                           focus:border-zinc-400 placeholder:text-zinc-300 bg-white"
                placeholder={row.type === "url" ? "https://..." : "/path/to/file.pdf"} />
            )}
            <input value={row.label} onChange={(e) => setRow(i, "label", e.target.value)}
              className="w-28 border border-zinc-200 rounded-lg px-2 py-2 text-xs outline-none
                         focus:border-zinc-400 placeholder:text-zinc-300 bg-white"
              placeholder="Label" />
            {rows.length > 1 && (
              <button onClick={() => removeRow(i)}
                className="text-zinc-300 hover:text-red-400 text-base leading-none pt-2">×</button>
            )}
          </div>
        ))}
        {errors.map((e, i) => <p key={i} className="text-xs text-red-500">{e}</p>)}
        <button onClick={ingestAll} disabled={loading || rows.every((r) => !r.content.trim())}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-zinc-900 text-white
                     rounded-lg hover:bg-zinc-700 disabled:opacity-30 transition-colors">
          {loading && <Spinner size={14} />}
          {loading ? `Ingesting…` : "Ingest all"}
        </button>
      </div>

      {config.sources.length === 0 ? (
        <div className="text-center py-12 text-zinc-300">
          <p className="text-sm text-zinc-400">No sources yet</p>
        </div>
      ) : (
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <p className="text-xs text-zinc-400 uppercase tracking-wider font-medium">
              Sources ({config.sources.length})
            </p>
            <button onClick={onRefresh} className="text-xs text-zinc-400 hover:text-zinc-700">↻</button>
          </div>
          {config.sources.map((src) => (
            <div key={src.id} className="flex items-start gap-3 p-3 border border-zinc-100 rounded-xl
                                         hover:border-zinc-200 transition-colors group">
              <span className="text-sm text-zinc-300 mt-0.5 shrink-0">
                {src.source_type === "url" ? "↗" : src.source_type === "file" ? "◻" : "≡"}
              </span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  {isUrl(src.content) ? (
                    <a href={src.content} target="_blank" rel="noopener noreferrer"
                      className="text-sm font-medium text-blue-600 hover:underline truncate max-w-sm">
                      {src.label || domainOf(src.content)}
                    </a>
                  ) : (
                    <p className="text-sm font-medium text-zinc-700 truncate max-w-sm">
                      {src.label || src.content}
                    </p>
                  )}
                  <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium border ${statusColor(src.status)}`}>
                    {src.status}
                  </span>
                </div>
                {src.status === "ingested" && (
                  <p className="text-xs text-zinc-400 mt-1">
                    {src.chunks_count} chunks · ~{src.tokens_estimated.toLocaleString()} tokens
                  </p>
                )}
                {src.error_message && <p className="text-xs text-red-500 mt-1">{src.error_message}</p>}
              </div>
              <button onClick={() => deleteSource(src.id)}
                className="opacity-0 group-hover:opacity-100 text-zinc-300 hover:text-red-400
                           transition-all text-base shrink-0 pt-0.5">×</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Compare tab — side-by-side with inline config editing + DeepEval radar chart
// ─────────────────────────────────────────────────────────────────────────────

type PaneConfig = {
  pipeline_id: string;
  retrieval_strategy: string;
  reranker: string;
  llm_model: string;
};

interface PaneResult {
  answer: string;
  citations: Citation[];
  strategy_used: string;
  reranker_used: string;
  chunks_retrieved: number;
  latency_ms: number;
  eval_scores: Record<string, { score: number; passed: boolean; reason: string }> | null;
  error?: string;
}

const EVAL_METRIC_LABELS: Record<string, string> = {
  answer_relevancy: "Relevancy",
  faithfulness: "Faithfulness",
  contextual_relevancy: "Ctx Relevancy",
  contextual_precision: "Ctx Precision",
  contextual_recall: "Ctx Recall",
};

function CompareRadar({ resultA, resultB, labelA, labelB }: {
  resultA: PaneResult | null; resultB: PaneResult | null;
  labelA: string; labelB: string;
}) {
  const { RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend, ResponsiveContainer } =
    require("recharts");

  const metrics = Object.keys(EVAL_METRIC_LABELS);
  const data = metrics.map((k) => ({
    metric: EVAL_METRIC_LABELS[k],
    A: resultA?.eval_scores?.[k]?.score ?? 0,
    B: resultB?.eval_scores?.[k]?.score ?? 0,
  }));

  return (
    <div className="border border-zinc-200 rounded-xl p-4 bg-white">
      <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-3">DeepEval Comparison</p>
      <ResponsiveContainer width="100%" height={220}>
        <RadarChart data={data} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
          <PolarGrid stroke="#e4e4e7" />
          <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: "#71717a" }} />
          <PolarRadiusAxis angle={90} domain={[0, 1]} tick={{ fontSize: 9, fill: "#a1a1aa" }} tickCount={3} />
          <Radar name={labelA} dataKey="A" stroke="#2563eb" fill="#2563eb" fillOpacity={0.15} strokeWidth={2} />
          <Radar name={labelB} dataKey="B" stroke="#16a34a" fill="#16a34a" fillOpacity={0.15} strokeWidth={2} />
          <Legend iconSize={8} wrapperStyle={{ fontSize: 11 }} />
        </RadarChart>
      </ResponsiveContainer>
      {/* Numeric comparison table */}
      <div className="mt-3 space-y-1">
        {metrics.map((k) => {
          const aScore = resultA?.eval_scores?.[k]?.score ?? null;
          const bScore = resultB?.eval_scores?.[k]?.score ?? null;
          const winner = aScore !== null && bScore !== null
            ? aScore > bScore ? "A" : bScore > aScore ? "B" : "tie"
            : null;
          return (
            <div key={k} className="flex items-center gap-2 text-[11px]">
              <span className="w-28 text-zinc-400 truncate">{EVAL_METRIC_LABELS[k]}</span>
              <span className={`w-10 text-right font-mono ${winner === "A" ? "text-blue-600 font-semibold" : "text-zinc-500"}`}>
                {aScore !== null ? (aScore * 100).toFixed(0) + "%" : "—"}
              </span>
              <span className="text-zinc-200">vs</span>
              <span className={`w-10 font-mono ${winner === "B" ? "text-emerald-600 font-semibold" : "text-zinc-500"}`}>
                {bScore !== null ? (bScore * 100).toFixed(0) + "%" : "—"}
              </span>
              {winner && winner !== "tie" && (
                <span className={`text-[9px] px-1.5 py-0.5 rounded-full font-medium
                  ${winner === "A" ? "bg-blue-50 text-blue-600" : "bg-emerald-50 text-emerald-600"}`}>
                  {winner} wins
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ComparePane({ cfg, result, loading, label, color }: {
  cfg: PaneConfig; result: PaneResult | null; loading: boolean;
  label: string; color: "blue" | "green";
}) {
  const borderCls = color === "blue" ? "border-blue-200" : "border-emerald-200";
  const headerCls = color === "blue" ? "bg-blue-50 border-blue-100" : "bg-emerald-50 border-emerald-100";
  const dotCls = color === "blue" ? "bg-blue-500" : "bg-emerald-500";

  return (
    <div className={`flex flex-col border ${borderCls} rounded-xl overflow-hidden`}>
      <div className={`px-4 py-3 border-b ${headerCls} flex items-center gap-2`}>
        <span className={`h-2.5 w-2.5 rounded-full ${dotCls} flex-shrink-0`} />
        <p className="text-sm font-medium text-zinc-800">{label}</p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-white min-h-0">
        {loading && !result && (
          <div className="flex items-center gap-1.5 py-4 justify-center">
            {[0, 150, 300].map((d) => (
              <span key={d} className="w-2 h-2 rounded-full bg-zinc-300 animate-bounce"
                style={{ animationDelay: `${d}ms` }} />
            ))}
          </div>
        )}
        {result?.error && (
          <div className="text-sm text-red-500 bg-red-50 border border-red-100 rounded-lg p-3">{result.error}</div>
        )}
        {result && !result.error && (
          <>
            {/* Answer */}
            <div className="prose prose-sm prose-zinc max-w-none text-[13px]
              [&_p]:leading-relaxed [&_code]:bg-zinc-50 [&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-xs">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.answer}</ReactMarkdown>
            </div>
            {/* Citations */}
            {result.citations.length > 0 && (
              <div className="flex flex-wrap gap-1 pt-1">
                {result.citations.map((c, ci) => (
                  isUrl(c.source) ? (
                    <a key={ci} href={c.source} target="_blank" rel="noopener noreferrer"
                      className="text-[10px] text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full border border-blue-100 hover:bg-blue-100 font-medium">
                      [{ci + 1}] {domainOf(c.source)} ↗
                    </a>
                  ) : (
                    <span key={ci} className="text-[10px] text-zinc-500 bg-zinc-100 px-2 py-0.5 rounded-full font-medium">
                      [{ci + 1}] {c.source.split("/").pop()}
                    </span>
                  )
                ))}
              </div>
            )}
            {/* Stats */}
            <div className="flex flex-wrap gap-2 pt-1">
              <Badge>{result.strategy_used}</Badge>
              {result.reranker_used && result.reranker_used !== "none" && (
                <Badge variant="blue">↑ {result.reranker_used}</Badge>
              )}
              <Badge>{result.latency_ms}ms</Badge>
              <Badge>{result.chunks_retrieved} chunks</Badge>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function CompareTab({ configs, llmModels, models }: {
  configs: RagConfig[]; llmModels: LLMModel[]; models: RagModels;
}) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [paneA, setPaneA] = useState<PaneConfig>({
    pipeline_id: configs[0]?.id ?? "",
    retrieval_strategy: "", reranker: "none", llm_model: "",
  });
  const [paneB, setPaneB] = useState<PaneConfig>({
    pipeline_id: configs[1]?.id ?? configs[0]?.id ?? "",
    retrieval_strategy: "", reranker: "bge-reranker-base", llm_model: "",
  });
  const [resultA, setResultA] = useState<PaneResult | null>(null);
  const [resultB, setResultB] = useState<PaneResult | null>(null);
  const [history, setHistory] = useState<Array<{ q: string; a: PaneResult; b: PaneResult }>>([]);

  const strategies = ["", "similarity", "mmr", "multi_query", "hybrid"];
  const rerankers = Object.keys(models.reranker_models ?? { none: {} });

  async function sendBoth() {
    const q = query.trim();
    if (!q || loading) return;
    setLoading(true);
    setQuery("");
    setResultA(null);
    setResultB(null);

    try {
      const res = await apiPost<{ pane_a: PaneResult; pane_b: PaneResult }>(
        "/api/rag/compare",
        {
          query: q,
          pane_a: {
            config_id: paneA.pipeline_id,
            ...(paneA.retrieval_strategy && { retrieval_strategy_override: paneA.retrieval_strategy }),
            reranker_override: paneA.reranker || null,
            ...(paneA.llm_model && { llm_model_override: paneA.llm_model }),
          },
          pane_b: {
            config_id: paneB.pipeline_id,
            ...(paneB.retrieval_strategy && { retrieval_strategy_override: paneB.retrieval_strategy }),
            reranker_override: paneB.reranker || null,
            ...(paneB.llm_model && { llm_model_override: paneB.llm_model }),
          },
          auto_evaluate: true,
        }
      );
      setResultA(res.pane_a);
      setResultB(res.pane_b);
      setHistory(h => [{ q, a: res.pane_a, b: res.pane_b }, ...h.slice(0, 9)]);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setResultA({ answer: "", citations: [], strategy_used: "", reranker_used: "", chunks_retrieved: 0, latency_ms: 0, eval_scores: null, error: msg });
      setResultB({ answer: "", citations: [], strategy_used: "", reranker_used: "", chunks_retrieved: 0, latency_ms: 0, eval_scores: null, error: msg });
    }
    setLoading(false);
  }

  const cfgA = configs.find((c) => c.id === paneA.pipeline_id);
  const cfgB = configs.find((c) => c.id === paneB.pipeline_id);
  const labelA = `A: ${cfgA?.name ?? "—"}${paneA.reranker && paneA.reranker !== "none" ? ` + ${paneA.reranker}` : ""}`;
  const labelB = `B: ${cfgB?.name ?? "—"}${paneB.reranker && paneB.reranker !== "none" ? ` + ${paneB.reranker}` : ""}`;

  function PaneSettings({ pane, setPane, label, color }: {
    pane: PaneConfig; setPane: (p: PaneConfig) => void;
    label: string; color: "blue" | "green";
  }) {
    const accentBorder = color === "blue" ? "border-blue-200 bg-blue-50/30" : "border-emerald-200 bg-emerald-50/30";
    return (
      <div className={`border ${accentBorder} rounded-xl p-4 space-y-3`}>
        <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">{label}</p>
        <div>
          <label className="block text-xs text-zinc-500 mb-1">Pipeline</label>
          <select value={pane.pipeline_id} onChange={(e) => setPane({ ...pane, pipeline_id: e.target.value })}
            className="w-full border border-zinc-200 rounded-lg px-2 py-1.5 text-sm outline-none focus:border-zinc-400 bg-white">
            {configs.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
        </div>
        <div className="grid grid-cols-3 gap-2">
          <div>
            <label className="block text-xs text-zinc-500 mb-1">Retrieval</label>
            <select value={pane.retrieval_strategy}
              onChange={(e) => setPane({ ...pane, retrieval_strategy: e.target.value })}
              className="w-full border border-zinc-200 rounded-lg px-2 py-1.5 text-xs outline-none focus:border-zinc-400 bg-white">
              {strategies.map((s) => <option key={s} value={s}>{s || "default"}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs text-zinc-500 mb-1">Reranker</label>
            <select value={pane.reranker}
              onChange={(e) => setPane({ ...pane, reranker: e.target.value })}
              className="w-full border border-zinc-200 rounded-lg px-2 py-1.5 text-xs outline-none focus:border-zinc-400 bg-white">
              {rerankers.map((r) => <option key={r} value={r}>{r === "none" ? "none" : r.replace("bge-reranker-", "bge-")}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs text-zinc-500 mb-1">LLM</label>
            <select value={pane.llm_model}
              onChange={(e) => setPane({ ...pane, llm_model: e.target.value })}
              className="w-full border border-zinc-200 rounded-lg px-2 py-1.5 text-xs outline-none focus:border-zinc-400 bg-white">
              <option value="">default</option>
              {llmModels.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </div>
        </div>
      </div>
    );
  }

  const hasEval = !!(resultA?.eval_scores && resultB?.eval_scores);
  const hasResults = !!(resultA || resultB);

  return (
    <div className="flex flex-col gap-4 h-[calc(100vh-200px)]">
      {/* Config settings */}
      <div className="grid grid-cols-2 gap-4 shrink-0">
        <PaneSettings pane={paneA} setPane={setPaneA} label="Pipeline A" color="blue" />
        <PaneSettings pane={paneB} setPane={setPaneB} label="Pipeline B" color="green" />
      </div>

      {/* Current question */}
      {history.length > 0 && (
        <div className="shrink-0 px-1 text-sm text-zinc-500">
          <span className="font-medium text-zinc-800">Q: </span>{history[0].q}
        </div>
      )}

      {/* Results area — answers + radar */}
      <div className={`flex-1 min-h-0 grid gap-4 ${hasEval ? "grid-cols-5" : "grid-cols-2"}`}>
        <div className={hasEval ? "col-span-2" : "col-span-1"}>
          <ComparePane cfg={paneA} result={resultA} loading={loading} label={labelA} color="blue" />
        </div>
        <div className={hasEval ? "col-span-2" : "col-span-1"}>
          <ComparePane cfg={paneB} result={resultB} loading={loading} label={labelB} color="green" />
        </div>
        {hasEval && (
          <div className="col-span-1 overflow-y-auto">
            <CompareRadar resultA={resultA} resultB={resultB} labelA="A" labelB="B" />
          </div>
        )}
      </div>

      {/* Shared input */}
      <div className="flex gap-2 shrink-0">
        <input value={query} onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendBoth()}
          disabled={loading} placeholder="Ask the same question to both pipelines…"
          className="flex-1 border border-zinc-200 rounded-xl px-4 py-3 text-sm outline-none
                     focus:border-zinc-400 placeholder:text-zinc-300 bg-white" />
        <button onClick={sendBoth} disabled={loading || !query.trim()}
          className="px-5 py-3 text-sm font-medium bg-zinc-900 text-white rounded-xl
                     hover:bg-zinc-700 disabled:opacity-30 transition-colors flex items-center gap-2">
          {loading && <Spinner size={14} />}
          {loading ? "Running…" : "Compare"}
        </button>
      </div>

      {/* History list */}
      {history.length > 1 && (
        <div className="shrink-0 border-t border-zinc-100 pt-3">
          <p className="text-xs text-zinc-400 mb-2">Previous comparisons</p>
          <div className="space-y-1 max-h-24 overflow-y-auto">
            {history.slice(1).map((h, i) => (
              <button key={i}
                onClick={() => { setResultA(h.a); setResultB(h.b); }}
                className="w-full text-left text-xs text-zinc-500 hover:text-zinc-800 truncate px-2 py-1 rounded hover:bg-zinc-50">
                {h.q}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluate tab
// ─────────────────────────────────────────────────────────────────────────────

function EvaluateTab({ config }: { config: RagConfig }) {
  const [samples, setSamples] = useState([{ query: "", expected_answer: "" }]);
  const [results, setResults] = useState<Array<{
    query: string; actual_answer: string; overall_pass: boolean;
    avg_score: number; latency_ms: number; error?: string;
    metrics: Array<{ name: string; score: number; passed: boolean; reason: string }>;
  }>>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function run() {
    const valid = samples.filter((s) => s.query.trim() && s.expected_answer.trim());
    if (!valid.length) { setError("Add at least one sample with both fields."); return; }
    setLoading(true); setError("");
    try {
      const res = await apiPost<typeof results>("/api/rag/evaluate", {
        config_id: config.id, samples: valid,
      });
      setResults(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setLoading(false); }
  }

  return (
    <div className="space-y-6 max-w-2xl">
      <div className="border border-zinc-200 rounded-xl p-4 space-y-3">
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium text-zinc-700">Q&A evaluation samples</p>
          <button onClick={() => setSamples((s) => [...s, { query: "", expected_answer: "" }])}
            className="text-xs text-zinc-400 hover:text-zinc-700">+ Add</button>
        </div>
        {samples.map((s, i) => (
          <div key={i} className="grid grid-cols-2 gap-2">
            <textarea rows={2} placeholder={`Question ${i + 1}…`} value={s.query}
              onChange={(e) => { const n = [...samples]; n[i] = { ...n[i], query: e.target.value }; setSamples(n); }}
              className="border border-zinc-200 rounded-lg px-3 py-2 text-sm outline-none
                         focus:border-zinc-400 resize-none placeholder:text-zinc-300 bg-white" />
            <textarea rows={2} placeholder="Expected answer…" value={s.expected_answer}
              onChange={(e) => { const n = [...samples]; n[i] = { ...n[i], expected_answer: e.target.value }; setSamples(n); }}
              className="border border-zinc-200 rounded-lg px-3 py-2 text-sm outline-none
                         focus:border-zinc-400 resize-none placeholder:text-zinc-300 bg-white" />
          </div>
        ))}
      </div>
      {error && <p className="text-sm text-red-500">{error}</p>}
      <button onClick={run} disabled={loading}
        className="flex items-center gap-2 px-5 py-2.5 text-sm font-medium bg-zinc-900 text-white
                   rounded-lg hover:bg-zinc-700 disabled:opacity-30 transition-colors">
        {loading && <Spinner size={14} />}
        {loading ? "Evaluating…" : "Run DeepEval RAG Metrics"}
      </button>
      <div className="space-y-3">
        {results.map((r, i) => (
          <div key={i} className="border border-zinc-100 rounded-xl p-4 space-y-3">
            <div className="flex items-start gap-2">
              <span className={`mt-0.5 text-sm ${r.overall_pass ? "text-emerald-500" : "text-amber-400"}`}>●</span>
              <p className="text-sm font-medium text-zinc-800 flex-1">{r.query}</p>
              <span className="text-xs text-zinc-400 shrink-0">{r.latency_ms}ms</span>
            </div>
            {r.error ? (
              <p className="text-sm text-red-500 pl-5">{r.error}</p>
            ) : (
              <div className="space-y-2.5 pl-5">
                {r.metrics.map((m) => (
                  <ScoreBar key={m.name} label={m.name} score={m.score} passed={m.passed} reason={m.reason} />
                ))}
              </div>
            )}
            {r.actual_answer && (
              <details className="pl-5">
                <summary className="text-xs text-zinc-400 cursor-pointer hover:text-zinc-700">View generated answer</summary>
                <p className="mt-2 text-xs text-zinc-600 bg-zinc-50 rounded-lg p-3 whitespace-pre-wrap leading-relaxed">
                  {r.actual_answer}
                </p>
              </details>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Config wizard
// ─────────────────────────────────────────────────────────────────────────────

const WIZARD_STEPS = ["Identity", "Embedding", "Storage", "Retrieval", "LLM & Prompt"];

type FormState = {
  name: string; description: string; embedding_model: string; vector_store: string;
  llm_model: string; chunk_size: number; chunk_overlap: number; chunk_strategy: string;
  retrieval_strategy: string; top_k: number; mmr_lambda: number; multi_query_n: number;
  system_prompt: string; reranker: string;
};
const DEFAULT_FORM: FormState = {
  name: "", description: "", embedding_model: "qwen/qwen3-embedding-8b", vector_store: "chroma",
  llm_model: "", chunk_size: 1000, chunk_overlap: 200, chunk_strategy: "recursive",
  retrieval_strategy: "similarity", top_k: 5, mmr_lambda: 0.5, multi_query_n: 3, system_prompt: "",
  reranker: "none",
};

function ConfigWizard({ models, llmModels, onCreated, teamId }: {
  models: RagModels; llmModels: LLMModel[]; onCreated: (c: RagConfig) => void;
  teamId?: string;
}) {
  const [step, setStep] = useState(0);
  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const set = (k: string, v: unknown) => setForm((f) => ({ ...f, [k]: v }));

  async function submit() {
    setLoading(true); setError("");
    try {
      // Tag the new pipeline with the currently-selected team so it stays
      // scoped to that team going forward (and doesn't leak into others).
      const body = {
        ...form,
        llm_model: form.llm_model || null,
        system_prompt: form.system_prompt || null,
        team_id: teamId || null,
      };
      const cfg = await apiPost<RagConfig>("/api/rag/configs", body);
      onCreated(cfg);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setLoading(false); }
  }

  const recommended = Object.entries(models.embedding_models).filter(([, v]) => v.recommended);
  const others = Object.entries(models.embedding_models).filter(([, v]) => !v.recommended);

  return (
    <div className="max-w-xl">
      {/* Progress */}
      <div className="flex items-center gap-2 mb-8">
        {WIZARD_STEPS.map((s, i) => (
          <div key={i} className="flex items-center gap-2">
            <button onClick={() => i < step && setStep(i)}
              className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium transition-colors
                ${i < step ? "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 cursor-pointer" :
                  i === step ? "bg-zinc-900 text-white" : "bg-zinc-100 text-zinc-400"}`}>
              {i < step ? "✓" : i + 1}
            </button>
            {i < WIZARD_STEPS.length - 1 && <div className={`w-6 h-px ${i < step ? "bg-zinc-300" : "bg-zinc-100"}`} />}
          </div>
        ))}
        <span className="ml-2 text-sm text-zinc-500">{WIZARD_STEPS[step]}</span>
      </div>

      {step === 0 && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1.5">Pipeline name</label>
            <input autoFocus value={form.name} onChange={(e) => set("name", e.target.value)}
              className="w-full border border-zinc-200 rounded-lg px-3 py-2 text-sm outline-none focus:border-zinc-400 placeholder:text-zinc-300 bg-white"
              placeholder="e.g. Product Docs, Code Search" />
          </div>
          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1.5">Description <span className="text-zinc-400 font-normal">(optional)</span></label>
            <textarea rows={3} value={form.description} onChange={(e) => set("description", e.target.value)}
              className="w-full border border-zinc-200 rounded-lg px-3 py-2 text-sm outline-none focus:border-zinc-400 resize-none placeholder:text-zinc-300 bg-white"
              placeholder="What documents does this search?" />
          </div>
        </div>
      )}

      {step === 1 && (
        <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
          <p className="text-xs text-zinc-400">All via <strong className="text-zinc-700">OpenRouter</strong> — uses <code className="bg-zinc-100 px-1 py-0.5 rounded text-[11px]">OPENROUTER_KEY</code>.</p>
          {[{ label: "Recommended", items: recommended }, { label: "Others", items: others }].map(({ label: l, items }) =>
            items.length > 0 && (
              <div key={l} className="space-y-2">
                <p className="text-[10px] uppercase tracking-widest text-zinc-400 font-semibold">{l}</p>
                {items.map(([k, v]) => (
                  <label key={k} className={`flex items-start gap-3 p-3 border rounded-lg cursor-pointer transition-colors
                    ${form.embedding_model === k ? "border-zinc-900 bg-zinc-50" : "border-zinc-200 hover:border-zinc-300"}`}>
                    <input type="radio" name="emb" className="mt-0.5 accent-zinc-900 shrink-0"
                      checked={form.embedding_model === k} onChange={() => set("embedding_model", k)} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-sm font-medium text-zinc-800 font-mono">{k}</span>
                        {v.recommended && <Badge variant="ok">Recommended</Badge>}
                        {v.cost_per_1m_tokens === 0 && <Badge variant="blue">Free</Badge>}
                      </div>
                      <div className="flex gap-3 mt-0.5">
                        <span className="text-xs text-zinc-400">{v.dimensions}d</span>
                        <span className="text-xs text-zinc-400">{(v.max_tokens / 1000).toFixed(0)}K ctx</span>
                        <span className="text-xs text-zinc-400">{v.cost_per_1m_tokens === 0 ? "Free" : `$${v.cost_per_1m_tokens}/1M`}</span>
                      </div>
                      <p className="text-xs text-zinc-400 mt-1">{v.description}</p>
                    </div>
                  </label>
                ))}
              </div>
            )
          )}
        </div>
      )}

      {step === 2 && (
        <div className="space-y-5">
          <div>
            <p className="text-sm font-medium text-zinc-700 mb-2">Vector store</p>
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(models.vector_stores).map(([k, v]) => (
                <button key={k} onClick={() => set("vector_store", k)}
                  className={`p-3 border rounded-lg text-left transition-colors ${
                    form.vector_store === k ? "border-zinc-900 bg-zinc-900 text-white" : "border-zinc-200 hover:border-zinc-400"
                  }`}>
                  <p className="text-sm font-semibold">{v.label}</p>
                  <p className={`text-xs mt-1 leading-snug ${form.vector_store === k ? "text-zinc-300" : "text-zinc-400"}`}>
                    {v.description.split(".")[0]}
                  </p>
                </button>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-zinc-700 mb-1">Chunk strategy</label>
              <select value={form.chunk_strategy} onChange={(e) => set("chunk_strategy", e.target.value)}
                className="w-full border border-zinc-200 rounded-lg px-3 py-2 text-sm outline-none focus:border-zinc-400 bg-white">
                {models.chunk_strategies.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-zinc-700 mb-1">Chunk size · {form.chunk_size}</label>
              <input type="range" min={200} max={4000} step={100} value={form.chunk_size}
                onChange={(e) => set("chunk_size", Number(e.target.value))} className="w-full accent-zinc-900" />
            </div>
            <div>
              <label className="block text-sm font-medium text-zinc-700 mb-1">Overlap · {form.chunk_overlap}</label>
              <input type="range" min={0} max={Math.round(form.chunk_size * 0.5)} step={50} value={form.chunk_overlap}
                onChange={(e) => set("chunk_overlap", Number(e.target.value))} className="w-full accent-zinc-900" />
            </div>
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="space-y-4">
          <div className="space-y-2">
            {[
              { k: "similarity", l: "Similarity", d: "Cosine top-K. Fast and deterministic." },
              { k: "mmr", l: "MMR", d: "Max Marginal Relevance — diversifies results." },
              { k: "multi_query", l: "Multi-Query", d: "Generates N query variants for better recall." },
              { k: "hybrid", l: "Hybrid", d: "Dense vector + BM25 keyword scoring." },
            ].map(({ k, l, d }) => (
              <label key={k} className={`flex items-start gap-3 p-3 border rounded-lg cursor-pointer transition-colors
                ${form.retrieval_strategy === k ? "border-zinc-900 bg-zinc-50" : "border-zinc-200 hover:border-zinc-300"}`}>
                <input type="radio" name="ret" className="mt-0.5 accent-zinc-900"
                  checked={form.retrieval_strategy === k} onChange={() => set("retrieval_strategy", k)} />
                <div>
                  <p className="text-sm font-medium text-zinc-800">{l}</p>
                  <p className="text-xs text-zinc-400">{d}</p>
                </div>
              </label>
            ))}
          </div>
          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1">Top-K · {form.top_k}</label>
            <input type="range" min={1} max={20} value={form.top_k}
              onChange={(e) => set("top_k", Number(e.target.value))} className="w-full accent-zinc-900" />
          </div>
          {/* Reranker */}
          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-2">
              Reranker <span className="text-zinc-400 font-normal text-xs">(BGE cross-encoder, CPU)</span>
            </label>
            <div className="space-y-2">
              {Object.entries(models.reranker_models ?? {}).map(([k, v]) => (
                <label key={k} className={`flex items-start gap-3 p-3 border rounded-lg cursor-pointer transition-colors
                  ${form.reranker === k ? "border-zinc-900 bg-zinc-50" : "border-zinc-200 hover:border-zinc-300"}`}>
                  <input type="radio" name="reranker" className="mt-0.5 accent-zinc-900"
                    checked={form.reranker === k} onChange={() => set("reranker", k)} />
                  <div>
                    <p className="text-sm font-medium text-zinc-800">{v.label}</p>
                    <p className="text-xs text-zinc-400">{v.description}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>
      )}

      {step === 4 && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1.5">LLM model</label>
            <select value={form.llm_model} onChange={(e) => set("llm_model", e.target.value)}
              className="w-full border border-zinc-200 rounded-lg px-3 py-2 text-sm outline-none focus:border-zinc-400 bg-white">
              <option value="">— Use system default —</option>
              {llmModels.map((m) => <option key={m.id} value={m.id}>{m.name} ({m.id})</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1.5">System prompt <span className="text-zinc-400 font-normal">(optional)</span></label>
            <textarea rows={5} value={form.system_prompt} onChange={(e) => set("system_prompt", e.target.value)}
              className="w-full border border-zinc-200 rounded-lg px-3 py-2 text-sm font-mono outline-none focus:border-zinc-400 resize-none placeholder:text-zinc-300 bg-white"
              placeholder="You are a helpful assistant…" />
          </div>
        </div>
      )}

      {error && <p className="mt-4 text-sm text-red-500 bg-red-50 border border-red-100 rounded-lg p-3">{error}</p>}

      <div className="flex justify-between mt-8">
        <button onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}
          className="px-4 py-2 text-sm text-zinc-500 border border-zinc-200 rounded-lg hover:text-zinc-700 disabled:opacity-30">
          Back
        </button>
        {step < WIZARD_STEPS.length - 1 ? (
          <button onClick={() => setStep((s) => s + 1)} disabled={step === 0 && !form.name.trim()}
            className="px-5 py-2 text-sm font-medium bg-zinc-900 text-white rounded-lg hover:bg-zinc-700 disabled:opacity-30">
            Continue
          </button>
        ) : (
          <button onClick={submit} disabled={loading || !form.name.trim()}
            className="flex items-center gap-2 px-5 py-2 text-sm font-medium bg-zinc-900 text-white rounded-lg hover:bg-zinc-700 disabled:opacity-30">
            {loading && <Spinner size={14} />}
            {loading ? "Creating…" : "Create Pipeline"}
          </button>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Edit modal
// ─────────────────────────────────────────────────────────────────────────────

function EditModal({ config, models, llmModels, onSave, onClose }: {
  config: RagConfig; models: RagModels; llmModels: LLMModel[];
  onSave: (c: RagConfig) => void; onClose: () => void;
}) {
  const [form, setForm] = useState<FormState>({
    name: config.name, description: config.description,
    embedding_model: config.embedding_model, vector_store: config.vector_store,
    llm_model: config.llm_model ?? "", chunk_size: config.chunk_size,
    chunk_overlap: config.chunk_overlap, chunk_strategy: config.chunk_strategy,
    retrieval_strategy: config.retrieval_strategy, top_k: config.top_k,
    mmr_lambda: config.mmr_lambda, multi_query_n: config.multi_query_n,
    system_prompt: config.system_prompt ?? "", reranker: config.reranker ?? "none",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const set = (k: string, v: unknown) => setForm((f) => ({ ...f, [k]: v }));

  async function save() {
    setLoading(true); setError("");
    try {
      const body = { ...form, llm_model: form.llm_model || null, system_prompt: form.system_prompt || null };
      const updated = await apiPut<RagConfig>(`/api/rag/configs/${config.id}`, body);
      onSave(updated);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setLoading(false); }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg mx-4 max-h-[90vh] flex flex-col">
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-100">
          <h2 className="text-base font-semibold text-zinc-900">Edit pipeline — {config.name}</h2>
          <button onClick={onClose} className="text-zinc-400 hover:text-zinc-700 text-xl leading-none">×</button>
        </div>
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1">Name</label>
            <input value={form.name} onChange={(e) => set("name", e.target.value)}
              className="w-full border border-zinc-200 rounded-lg px-3 py-2 text-sm outline-none focus:border-zinc-400 bg-white" />
          </div>
          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1">Description</label>
            <textarea rows={2} value={form.description} onChange={(e) => set("description", e.target.value)}
              className="w-full border border-zinc-200 rounded-lg px-3 py-2 text-sm outline-none focus:border-zinc-400 resize-none bg-white" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            {[
              { l: "Embedding model", k: "embedding_model", opts: Object.keys(models.embedding_models), xs: true },
              { l: "Vector store", k: "vector_store", opts: Object.keys(models.vector_stores) },
              { l: "Retrieval strategy", k: "retrieval_strategy", opts: models.retrieval_strategies },
              { l: "Chunk strategy", k: "chunk_strategy", opts: models.chunk_strategies },
            ].map(({ l, k, opts, xs }) => (
              <div key={k}>
                <label className="block text-sm font-medium text-zinc-700 mb-1">{l}</label>
                <select value={(form as Record<string, unknown>)[k] as string}
                  onChange={(e) => set(k, e.target.value)}
                  className={`w-full border border-zinc-200 rounded-lg px-2 py-2 outline-none focus:border-zinc-400 bg-white ${xs ? "text-[11px]" : "text-sm"}`}>
                  {opts.map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
              </div>
            ))}
            <div>
              <label className="block text-sm font-medium text-zinc-700 mb-1">Top-K · {form.top_k}</label>
              <input type="range" min={1} max={20} value={form.top_k}
                onChange={(e) => set("top_k", Number(e.target.value))} className="w-full accent-zinc-900 mt-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-zinc-700 mb-1">LLM model</label>
              <select value={form.llm_model} onChange={(e) => set("llm_model", e.target.value)}
                className="w-full border border-zinc-200 rounded-lg px-2 py-2 text-xs outline-none focus:border-zinc-400 bg-white">
                <option value="">System default</option>
                {llmModels.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
              </select>
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1">System prompt</label>
            <textarea rows={3} value={form.system_prompt} onChange={(e) => set("system_prompt", e.target.value)}
              className="w-full border border-zinc-200 rounded-lg px-3 py-2 text-sm font-mono outline-none focus:border-zinc-400 resize-none placeholder:text-zinc-300 bg-white"
              placeholder="You are a helpful assistant…" />
          </div>
          {error && <p className="text-sm text-red-500">{error}</p>}
        </div>
        <div className="flex justify-end gap-2 px-6 py-4 border-t border-zinc-100">
          <button onClick={onClose} className="px-4 py-2 text-sm border border-zinc-200 rounded-lg hover:bg-zinc-50">Cancel</button>
          <button onClick={save} disabled={loading || !form.name.trim()}
            className="flex items-center gap-2 px-5 py-2 text-sm font-medium bg-zinc-900 text-white rounded-lg hover:bg-zinc-700 disabled:opacity-30">
            {loading && <Spinner size={14} />}
            Save changes
          </button>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar
// ─────────────────────────────────────────────────────────────────────────────

function Sidebar({ configs, selectedId, onSelect, onNew }: {
  configs: RagConfig[]; selectedId: string | null;
  onSelect: (id: string) => void; onNew: () => void;
}) {
  return (
    <aside className="w-52 shrink-0 border-r border-zinc-100 flex flex-col bg-white">
      <div className="px-4 pt-5 pb-3">
        <p className="text-[10px] uppercase tracking-widest text-zinc-400 font-semibold mb-1">Pipelines</p>
      </div>
      <div className="flex-1 overflow-y-auto px-2 space-y-0.5">
        {configs.map((cfg) => {
          const ready = cfg.sources.some((s) => s.status === "ingested");
          const active = selectedId === cfg.id;
          return (
            <button key={cfg.id} onClick={() => onSelect(cfg.id)}
              className={`w-full text-left px-3 py-2.5 rounded-lg transition-colors ${
                active ? "bg-zinc-900 text-white" : "hover:bg-zinc-50 text-zinc-700"
              }`}>
              <div className="flex items-center gap-2">
                <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${ready ? "bg-emerald-400" : "bg-zinc-300"}`} />
                <span className="text-sm font-medium truncate">{cfg.name}</span>
              </div>
              <p className={`text-[11px] mt-0.5 ml-3.5 ${active ? "text-zinc-400" : "text-zinc-400"}`}>
                {cfg.sources.filter(s => s.status === "ingested").length} docs · {cfg.vector_store}
              </p>
            </button>
          );
        })}
      </div>
      <div className="p-3 border-t border-zinc-100">
        <button onClick={onNew}
          className="w-full py-2 text-sm font-medium text-zinc-600 border border-zinc-200 rounded-lg
                     hover:bg-zinc-900 hover:text-white hover:border-zinc-900 transition-all">
          + New Pipeline
        </button>
      </div>
    </aside>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main page
// ─────────────────────────────────────────────────────────────────────────────

type Tab = "sources" | "chat" | "compare" | "evaluate";

export default function RAGPage() {
  // Team scoping: only show pipelines belonging to the currently-selected
  // team (plus legacy NULL-team rows, which the backend also returns).
  const { teamId } = useTeam();
  const [ragModels, setRagModels] = useState<RagModels | null>(null);
  const [llmModels, setLlmModels] = useState<LLMModel[]>([]);
  const [configs, setConfigs] = useState<RagConfig[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("sources");
  const [loading, setLoading] = useState(true);
  const [showWizard, setShowWizard] = useState(false);
  const [showEdit, setShowEdit] = useState(false);

  const selected = configs.find((c) => c.id === selectedId) ?? null;

  const load = useCallback(async () => {
    try {
      // Scope the config list to the currently-selected team so sdlc_2_0's
      // pipelines don't show up when the user is on the dev team (and vice
      // versa). Pipelines with NULL team_id (pre-migration rows) still
      // surface in every team — see the server-side filter.
      const cfgsPath = teamId
        ? `/api/rag/configs?team_id=${encodeURIComponent(teamId)}`
        : "/api/rag/configs";
      const [m, cfgs, llms] = await Promise.all([
        apiGet<RagModels>("/api/rag/models"),
        apiGet<RagConfig[]>(cfgsPath),
        apiGet<LLMModel[]>("/api/models").catch(() => []),
      ]);
      setRagModels(m);
      setConfigs(cfgs);
      setLlmModels(llms);
      if (!selectedId && cfgs.length > 0) { setSelectedId(cfgs[0].id); setTab("sources"); }
      if (cfgs.length === 0) setShowWizard(true);
    } catch { /* ignore */ }
    finally { setLoading(false); }
  }, [selectedId, teamId]);

  useEffect(() => {
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, [load]);

  // Reset per-team view state whenever the team changes so we don't keep
  // showing a pipeline that no longer belongs to the active team.
  useEffect(() => {
    setSelectedId(null);
    setShowWizard(false);
    setShowEdit(false);
  }, [teamId]);

  async function deleteConfig(id: string) {
    if (!confirm("Delete pipeline and all its data?")) return;
    await apiDelete(`/api/rag/configs/${id}`);
    const rest = configs.filter((x) => x.id !== id);
    setConfigs(rest);
    setSelectedId(rest[0]?.id ?? null);
    if (!rest.length) setShowWizard(true);
  }

  function handleCreated(cfg: RagConfig) {
    setConfigs((c) => [cfg, ...c]);
    setSelectedId(cfg.id);
    setShowWizard(false);
    setTab("sources");
  }

  function handleSaved(cfg: RagConfig) {
    setConfigs((c) => c.map((x) => x.id === cfg.id ? cfg : x));
    setShowEdit(false);
  }

  if (loading) {
    return <div className="flex items-center justify-center h-96"><Spinner size={24} /></div>;
  }

  const TABS: { key: Tab; label: string }[] = [
    { key: "sources", label: "Data Sources" },
    { key: "chat", label: "Chat" },
    { key: "compare", label: "Compare" },
    { key: "evaluate", label: "Evaluate" },
  ];

  return (
    <div className="flex h-[calc(100vh-64px)] overflow-hidden -m-7">
      <Sidebar configs={configs} selectedId={selectedId}
        onSelect={(id) => { setSelectedId(id); setShowWizard(false); setTab("sources"); }}
        onNew={() => { setSelectedId(null); setShowWizard(true); }} />

      <div className="flex-1 overflow-y-auto min-w-0">
        <div className="px-8 py-8">
          {/* Header */}
          {(showWizard || !selected) ? (
            <div className="mb-8">
              <h1 className="text-xl font-semibold text-zinc-900">New RAG Pipeline</h1>
              <p className="text-sm text-zinc-400 mt-1">Configure your pipeline in 5 steps.</p>
            </div>
          ) : (
            <div className="flex items-start justify-between mb-5">
              <div>
                <h1 className="text-xl font-semibold text-zinc-900">{selected.name}</h1>
                {selected.description && <p className="text-sm text-zinc-400 mt-0.5">{selected.description}</p>}
                <div className="flex items-center gap-1.5 mt-2 flex-wrap">
                  <Badge>{selected.embedding_model.split("/").pop()}</Badge>
                  <Badge>{selected.vector_store}</Badge>
                  <Badge>{selected.retrieval_strategy}</Badge>
                  {selected.llm_model && <Badge variant="blue">{selected.llm_model}</Badge>}
                </div>
              </div>
              <div className="flex gap-2 shrink-0">
                <button onClick={() => setShowEdit(true)}
                  className="text-xs text-zinc-500 hover:text-zinc-900 border border-zinc-200
                             hover:border-zinc-400 px-3 py-1.5 rounded-lg transition-colors">
                  Edit
                </button>
                <button onClick={() => deleteConfig(selected.id)}
                  className="text-xs text-zinc-400 hover:text-red-500 border border-zinc-200
                             hover:border-red-200 px-3 py-1.5 rounded-lg transition-colors">
                  Delete
                </button>
              </div>
            </div>
          )}

          {/* Tab bar */}
          {selected && !showWizard && (
            <div className="flex items-center border-b border-zinc-100 mb-6 -mx-8 px-8">
              {TABS.map((t) => (
                <button key={t.key} onClick={() => setTab(t.key)}
                  className={`px-4 py-2.5 text-sm font-medium transition-colors border-b-2 -mb-px ${
                    tab === t.key ? "border-zinc-900 text-zinc-900" : "border-transparent text-zinc-400 hover:text-zinc-700"
                  }`}>
                  {t.label}
                </button>
              ))}
            </div>
          )}

          {/* Content */}
          {(showWizard || !selected) && ragModels && (
            <ConfigWizard models={ragModels} llmModels={llmModels} onCreated={handleCreated} teamId={teamId || undefined} />
          )}
          {selected && !showWizard && tab === "sources" && (
            <DataSources config={selected} onRefresh={load} />
          )}
          {selected && !showWizard && tab === "chat" && (
            <ChatTab config={selected} />
          )}
          {selected && !showWizard && tab === "compare" && ragModels && (
            <CompareTab configs={configs} llmModels={llmModels} models={ragModels} />
          )}
          {selected && !showWizard && tab === "evaluate" && (
            <EvaluateTab config={selected} />
          )}
        </div>
      </div>

      {showEdit && selected && ragModels && (
        <EditModal config={selected} models={ragModels} llmModels={llmModels}
          onSave={handleSaved} onClose={() => setShowEdit(false)} />
      )}
    </div>
  );
}
