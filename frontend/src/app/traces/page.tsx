"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

const TYPE_COLORS: Record<string, string> = {
  routing: "border-l-blue-500 bg-blue-50",
  llm_call: "border-l-purple-500 bg-purple-50",
  tool_call: "border-l-green-500 bg-green-50",
};

export default function TracesPage() {
  const [traces, setTraces] = useState<any[]>([]);
  const [detail, setDetail] = useState<any>(null);

  useEffect(() => { api.traces.list(100).then(setTraces); }, []);
  async function select(id: string) { setDetail(await api.traces.get(id)); }

  return (
    <div className="space-y-5">
      <h1 className="text-xl font-semibold">Trace History</h1>
      <div className="grid grid-cols-3 gap-5" style={{ height: "calc(100vh - 9rem)" }}>
        <div className="space-y-1.5 overflow-y-auto pr-1">
          {traces.map(t => (
            <button key={t.id} onClick={() => select(t.id)}
              className={`w-full text-left p-2.5 rounded-lg text-sm transition-all ${detail?.id === t.id ? "bg-[var(--accent-light)] border border-[var(--accent)]/20" : "bg-[var(--bg-card)] border border-[var(--border)] hover:bg-[var(--bg-hover)]"}`}>
              <div className="flex items-center justify-between">
                <span className="font-mono text-[10px] text-[var(--text-muted)]">{t.id}</span>
                <span className={`badge ${t.status === "completed" ? "bg-[var(--success-light)] text-[var(--success)]" : "bg-[var(--error-light)] text-[var(--error)]"}`}>{t.status}</span>
              </div>
              <div className="mt-0.5 truncate">{t.user_prompt}</div>
              <div className="flex gap-3 text-[10px] text-[var(--text-muted)] mt-1">
                <span>{t.total_latency_ms.toFixed(0)}ms</span>
                <span>{t.total_tokens} tok</span>
                <span>${t.total_cost.toFixed(4)}</span>
              </div>
            </button>
          ))}
          {traces.length === 0 && <div className="text-[var(--text-muted)] text-center py-10">No traces yet</div>}
        </div>

        <div className="col-span-2 overflow-y-auto">
          {detail ? (
            <div className="space-y-3">
              <div className="card">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm">{detail.id}</span>
                  <span className={`badge ${detail.status === "completed" ? "bg-[var(--success-light)] text-[var(--success)]" : "bg-[var(--error-light)] text-[var(--error)]"}`}>{detail.status}</span>
                </div>
                <p className="text-sm text-[var(--text-muted)] mt-1.5">&quot;{detail.user_prompt}&quot;</p>
                <div className="flex gap-5 mt-2 text-sm">
                  <span>{detail.total_latency_ms.toFixed(0)}ms</span>
                  <span>{detail.total_tokens} tokens</span>
                  <span>${detail.total_cost.toFixed(4)}</span>
                  <span>{detail.spans.length} spans</span>
                </div>
              </div>
              <div className="space-y-1.5">
                {detail.spans.map((s: any, i: number) => (
                  <details key={s.id} className={`rounded-lg p-3 border-l-4 border border-[var(--border)] ${TYPE_COLORS[s.span_type] || "bg-[var(--bg)]"}`}>
                    <summary className="flex items-center justify-between cursor-pointer text-sm">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono opacity-40">{i + 1}</span>
                        <span className="font-medium">{s.name}</span>
                        <span className="badge bg-[var(--bg-card)] text-[var(--text-muted)]">{s.span_type}</span>
                      </div>
                      <div className="flex gap-3 text-[10px] text-[var(--text-muted)]">
                        {s.tokens_in + s.tokens_out > 0 && <span>{s.tokens_in + s.tokens_out} tok</span>}
                        {s.cost > 0 && <span>${s.cost.toFixed(4)}</span>}
                      </div>
                    </summary>
                    <div className="mt-2 text-xs font-mono bg-white/60 rounded p-2 space-y-1">
                      {s.input_data && Object.keys(s.input_data).length > 0 && <div><span className="text-[var(--text-muted)]">In:</span> {JSON.stringify(s.input_data).slice(0, 300)}</div>}
                      {s.output_data && Object.keys(s.output_data).length > 0 && <div><span className="text-[var(--text-muted)]">Out:</span> {JSON.stringify(s.output_data).slice(0, 300)}</div>}
                      {s.error && <div className="text-[var(--error)]">Error: {s.error}</div>}
                    </div>
                  </details>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-[var(--text-muted)] text-center py-16">Select a trace</div>
          )}
        </div>
      </div>
    </div>
  );
}
