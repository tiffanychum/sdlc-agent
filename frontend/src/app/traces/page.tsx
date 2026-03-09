"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

const TYPE_COLORS: Record<string, string> = {
  routing: "text-blue-400 bg-blue-500/10",
  llm_call: "text-purple-400 bg-purple-500/10",
  tool_call: "text-green-400 bg-green-500/10",
};

export default function TracesPage() {
  const [traces, setTraces] = useState<any[]>([]);
  const [detail, setDetail] = useState<any>(null);

  useEffect(() => { api.traces.list(100).then(setTraces); }, []);

  async function select(id: string) { setDetail(await api.traces.get(id)); }

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Traces</h1>

      <div className="grid grid-cols-3 gap-6" style={{ height: "calc(100vh - 10rem)" }}>
        {/* List */}
        <div className="space-y-1.5 overflow-y-auto pr-2">
          {traces.map(t => (
            <button key={t.id} onClick={() => select(t.id)}
              className={`w-full text-left p-3 rounded-lg text-sm transition-all ${detail?.id === t.id ? "bg-[var(--accent)]/10 border border-[var(--accent)]/30" : "bg-[var(--bg-card)] border border-[var(--border)] hover:bg-[var(--bg-hover)]"}`}>
              <div className="flex items-center justify-between">
                <span className="font-mono text-[10px] text-[var(--text-muted)]">{t.id}</span>
                <span className={`badge ${t.status === "completed" ? "bg-[var(--success)]/10 text-[var(--success)]" : "bg-[var(--error)]/10 text-[var(--error)]"}`}>{t.status}</span>
              </div>
              <div className="mt-1 truncate text-[var(--text)]">{t.user_prompt}</div>
              <div className="flex gap-3 text-[10px] text-[var(--text-muted)] mt-1.5">
                <span>{t.total_latency_ms.toFixed(0)}ms</span>
                <span>{t.total_tokens} tok</span>
                <span>${t.total_cost.toFixed(4)}</span>
              </div>
            </button>
          ))}
          {traces.length === 0 && <div className="text-[var(--text-muted)] text-center py-12">No traces yet</div>}
        </div>

        {/* Detail */}
        <div className="col-span-2 overflow-y-auto">
          {detail ? (
            <div className="space-y-4">
              <div className="card">
                <div className="flex items-center justify-between">
                  <div className="font-mono text-sm">{detail.id}</div>
                  <span className={`badge ${detail.status === "completed" ? "bg-[var(--success)]/10 text-[var(--success)]" : "bg-[var(--error)]/10 text-[var(--error)]"}`}>{detail.status}</span>
                </div>
                <p className="text-sm text-[var(--text-muted)] mt-2">&quot;{detail.user_prompt}&quot;</p>
                <div className="flex gap-6 mt-3 text-sm">
                  <span><span className="text-[var(--text-muted)]">Latency:</span> {detail.total_latency_ms.toFixed(0)}ms</span>
                  <span><span className="text-[var(--text-muted)]">Tokens:</span> {detail.total_tokens}</span>
                  <span><span className="text-[var(--text-muted)]">Cost:</span> ${detail.total_cost.toFixed(4)}</span>
                  <span><span className="text-[var(--text-muted)]">Spans:</span> {detail.spans.length}</span>
                </div>
              </div>

              <div className="card space-y-2">
                <h3 className="text-xs text-[var(--text-muted)] uppercase tracking-wide mb-2">Span Waterfall</h3>
                {detail.spans.map((s: any, i: number) => (
                  <details key={s.id} className={`rounded-lg p-3 border border-[var(--border)] ${TYPE_COLORS[s.span_type] || "bg-[var(--bg)]"}`}>
                    <summary className="flex items-center justify-between cursor-pointer">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono font-bold opacity-50">{i + 1}</span>
                        <span className="text-sm font-medium">{s.name}</span>
                        <span className="badge bg-[var(--bg)]/50 text-[var(--text-muted)]">{s.span_type}</span>
                      </div>
                      <div className="flex gap-3 text-[10px] opacity-60">
                        {s.tokens_in + s.tokens_out > 0 && <span>{s.tokens_in + s.tokens_out} tok</span>}
                        {s.cost > 0 && <span>${s.cost.toFixed(4)}</span>}
                        {s.model && <span>{s.model}</span>}
                        <span className={s.status === "completed" ? "text-[var(--success)]" : "text-[var(--error)]"}>{s.status}</span>
                      </div>
                    </summary>
                    <div className="mt-2 text-xs font-mono bg-[var(--bg)]/50 rounded p-2 space-y-1">
                      {s.input_data && Object.keys(s.input_data).length > 0 && <div><span className="text-[var(--text-muted)]">Input:</span> {JSON.stringify(s.input_data).slice(0, 300)}</div>}
                      {s.output_data && Object.keys(s.output_data).length > 0 && <div><span className="text-[var(--text-muted)]">Output:</span> {JSON.stringify(s.output_data).slice(0, 300)}</div>}
                      {s.error && <div className="text-[var(--error)]">Error: {s.error}</div>}
                    </div>
                  </details>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-[var(--text-muted)] text-center py-20">Select a trace</div>
          )}
        </div>
      </div>
    </div>
  );
}
