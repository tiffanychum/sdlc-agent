"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

interface TraceSummary {
  id: string; team_id: string; user_prompt: string;
  total_latency_ms: number; total_tokens: number; total_cost: number;
  status: string; created_at: string;
}
interface SpanDetail {
  id: string; parent_span_id: string | null; name: string; span_type: string;
  start_time: string; end_time: string | null;
  input_data: any; output_data: any;
  tokens_in: number; tokens_out: number; cost: number;
  model: string; status: string; error: string | null;
}
interface TraceDetail extends TraceSummary { spans: SpanDetail[] }

export default function TracesPage() {
  const [traces, setTraces] = useState<TraceSummary[]>([]);
  const [selected, setSelected] = useState<TraceDetail | null>(null);

  useEffect(() => { api.traces.list(100).then(setTraces); }, []);

  async function selectTrace(id: string) {
    setSelected(await api.traces.get(id));
  }

  const spanTypeColors: Record<string, string> = {
    routing: "bg-blue-600/20 text-blue-400 border-blue-500/30",
    llm_call: "bg-purple-600/20 text-purple-400 border-purple-500/30",
    tool_call: "bg-green-600/20 text-green-400 border-green-500/30",
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Traces</h1>

      <div className="grid grid-cols-3 gap-6">
        {/* Trace List */}
        <div className="space-y-2 max-h-[70vh] overflow-y-auto">
          {traces.map(t => (
            <button key={t.id} onClick={() => selectTrace(t.id)}
              className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-colors ${selected?.id === t.id ? "bg-blue-600/20 border border-blue-500/30" : "bg-gray-800/50 border border-gray-700 hover:bg-gray-800"}`}>
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs text-gray-400">{t.id}</span>
                <span className={`text-xs px-1.5 py-0.5 rounded ${t.status === "completed" ? "bg-green-600/20 text-green-400" : "bg-red-600/20 text-red-400"}`}>
                  {t.status}
                </span>
              </div>
              <div className="text-gray-300 mt-1 truncate">{t.user_prompt}</div>
              <div className="flex gap-3 text-xs text-gray-500 mt-1">
                <span>{t.total_latency_ms.toFixed(0)}ms</span>
                <span>{t.total_tokens} tok</span>
                <span>${t.total_cost.toFixed(4)}</span>
              </div>
            </button>
          ))}
          {traces.length === 0 && <div className="text-gray-500 text-center py-10">No traces yet. Send a chat message first.</div>}
        </div>

        {/* Trace Detail */}
        <div className="col-span-2">
          {selected ? (
            <div className="space-y-4">
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
                <div className="flex items-center justify-between">
                  <h2 className="font-semibold">Trace {selected.id}</h2>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${selected.status === "completed" ? "bg-green-600/20 text-green-400" : "bg-red-600/20 text-red-400"}`}>
                    {selected.status}
                  </span>
                </div>
                <p className="text-sm text-gray-400 mt-2">&quot;{selected.user_prompt}&quot;</p>
                <div className="flex gap-6 mt-3 text-sm">
                  <div><span className="text-gray-500">Latency:</span> <span className="text-white">{selected.total_latency_ms.toFixed(0)}ms</span></div>
                  <div><span className="text-gray-500">Tokens:</span> <span className="text-white">{selected.total_tokens}</span></div>
                  <div><span className="text-gray-500">Cost:</span> <span className="text-white">${selected.total_cost.toFixed(4)}</span></div>
                  <div><span className="text-gray-500">Spans:</span> <span className="text-white">{selected.spans.length}</span></div>
                </div>
              </div>

              {/* Waterfall View */}
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-2">
                <h3 className="text-sm font-semibold text-gray-400">Span Waterfall</h3>
                {selected.spans.map((span, i) => {
                  const colorClass = spanTypeColors[span.span_type] || "bg-gray-700/50 text-gray-400 border-gray-600";
                  return (
                    <div key={span.id} className={`border rounded-lg p-3 ${colorClass}`}>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-mono font-bold">{i + 1}</span>
                          <span className="text-sm font-medium">{span.name}</span>
                          <span className="text-xs px-1.5 py-0.5 bg-gray-700/50 rounded">{span.span_type}</span>
                        </div>
                        <div className="flex gap-3 text-xs opacity-70">
                          {span.tokens_in + span.tokens_out > 0 && <span>{span.tokens_in + span.tokens_out} tok</span>}
                          {span.cost > 0 && <span>${span.cost.toFixed(4)}</span>}
                          {span.model && <span>{span.model}</span>}
                          <span className={span.status === "completed" ? "text-green-400" : "text-red-400"}>
                            {span.status}
                          </span>
                        </div>
                      </div>

                      {/* Expandable Details */}
                      {(span.input_data && Object.keys(span.input_data).length > 0) && (
                        <details className="mt-2">
                          <summary className="text-xs cursor-pointer opacity-60 hover:opacity-100">Input / Output</summary>
                          <div className="mt-1 text-xs font-mono bg-gray-900/50 rounded p-2 space-y-1">
                            <div><span className="text-gray-500">Input:</span> {JSON.stringify(span.input_data).slice(0, 200)}</div>
                            {span.output_data && <div><span className="text-gray-500">Output:</span> {JSON.stringify(span.output_data).slice(0, 200)}</div>}
                            {span.error && <div className="text-red-400">Error: {span.error}</div>}
                          </div>
                        </details>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="text-gray-500 text-center py-20">Select a trace to view details</div>
          )}
        </div>
      </div>
    </div>
  );
}
