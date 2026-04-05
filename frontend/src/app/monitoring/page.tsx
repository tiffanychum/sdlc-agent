"use client";
import React, { useEffect, useState, useCallback, useMemo } from "react";
import { api } from "@/lib/api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area,
} from "recharts";

// ── Constants & helpers ──────────────────────────────────────────────────────

const SPAN_COLORS: Record<string, string> = {
  routing: "#3b82f6", agent_execution: "#8b5cf6", tool_call: "#10b981",
  supervisor: "#f59e0b", llm_call: "#6366f1", unknown: "#6b7280",
  "rag.generate": "#0891b2", "rag.retrieve": "#0d9488", "rag.ingest": "#64748b",
  "rag.embed": "#a16207", "rag.evaluate": "#7c3aed",
};

const MODEL_COLORS = ["#6366f1", "#f59e0b", "#10b981", "#3b82f6", "#ec4899", "#14b8a6"];

function Metric({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: string }) {
  return (
    <div className="card text-center">
      <div className={`text-2xl font-semibold ${accent || ""}`}>{value}</div>
      <div className="text-xs text-[var(--text-muted)] mt-1">{label}</div>
      {sub && <div className="text-[11px] text-[var(--text-muted)] mt-0.5">{sub}</div>}
    </div>
  );
}

const tip = { contentStyle: { background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 } };

// ── Main component ────────────────────────────────────────────────────────────

export default function MonitoringPage() {
  const [activeTab, setActiveTab] = useState<"overview" | "otel">("overview");
  const [teams, setTeams] = useState<any[]>([]);
  const [teamId, setTeamId] = useState("");
  const [stats, setStats] = useState<any>(null);
  const [traces, setTraces] = useState<any[]>([]);
  const [otelStats, setOtelStats] = useState<any>(null);
  const [evalRuns, setEvalRuns] = useState<any[]>([]);
  const [ragStats, setRagStats] = useState<any>(null);

  useEffect(() => {
    api.teams.list().then(t => { setTeams(t); if (t.length) setTeamId(t[0].id); });
  }, []);

  const loadAll = useCallback(async () => {
    try { const s = await api.traces.stats(); setStats(s); } catch { /* ignore */ }
    try { const t = await api.traces.list(50); setTraces(t); } catch { /* ignore */ }
    try { const o = await api.otel.spanStats(30); setOtelStats(o); } catch { /* ignore */ }
    try { const e = await api.eval.runs(); setEvalRuns(e); } catch { /* ignore */ }
    try { const r = await (api as any).rag.stats(30); setRagStats(r); } catch { /* ignore */ }
  }, []);

  useEffect(() => { if (teamId) loadAll(); }, [teamId, loadAll]);

  const selectedTeam = teams.find(t => t.id === teamId);
  const totalRuns = stats?.total_runs || traces.length || 0;
  const failures = stats?.failures || traces.filter((t: any) => t.status === "error").length;
  const successRate = totalRuns > 0 ? ((totalRuns - failures) / totalRuns) : 0;
  const avgLatency = stats?.avg_latency_ms || 0;
  const totalCost = stats?.total_cost || 0;
  const avgCostPerRun = totalRuns > 0 ? totalCost / totalRuns : 0;
  const avgTokensPerRun = totalRuns > 0 ? Math.round((stats?.total_tokens || 0) / totalRuns) : 0;

  // ── Derived data ───────────────────────────────────────────────────────────

  const latencyData = traces.slice(0, 30).reverse().map((t: any, i: number) => ({
    idx: i + 1,
    latency: +(t.total_latency_ms || 0).toFixed(0),
    tokens: t.total_tokens || 0,
    cost: +(t.total_cost || 0).toFixed(5),
  }));

  const qualityData = evalRuns.slice(0, 15).reverse().map((r: any) => ({
    run: (r.prompt_version || r.id?.slice(0, 6)),
    success: +((r.task_success_rate || 0) * 100).toFixed(1),
    toolAcc: +((r.tool_accuracy || 0) * 100).toFixed(1),
    routing: +((r.routing_accuracy || 0) * 100).toFixed(1),
    faithfulness: +((r.faithfulness || 0) * 100).toFixed(1),
  }));

  const spanTypeData = useMemo(() => otelStats?.by_type ? Object.entries(otelStats.by_type).map(([type, data]: [string, any]) => ({
    name: type.replace(/_/g, " "),
    type,
    value: data.count,
    tokens: (data.tokens_in || 0) + (data.tokens_out || 0),
    cost: data.cost || 0,
    avgLatency: data.avg_latency_ms || 0,
    fill: SPAN_COLORS[type] || SPAN_COLORS.unknown,
  })) : [], [otelStats]);

  const modelData = useMemo(() => otelStats?.by_model ? Object.entries(otelStats.by_model).map(([model, data]: [string, any], i) => ({
    model: model || "unknown",
    shortModel: (model || "unknown").split("/").pop() || model || "unknown",
    count: data.count || 0,
    tokens_in: data.tokens_in || 0,
    tokens_out: data.tokens_out || 0,
    cost: data.cost || 0,
    color: MODEL_COLORS[i % MODEL_COLORS.length],
  })).filter(m => m.model && m.model !== "") : [], [otelStats]);

  const tokenFlow = otelStats?.token_flow || [];

  // Per-agent stats derived from trace spans
  const agentStats = useMemo(() => {
    const map: Record<string, { calls: number; totalLatency: number; totalCost: number; totalTokens: number; tools: Record<string, number> }> = {};
    for (const trace of traces) {
      for (const span of trace.spans || []) {
        if (span.span_type === "agent_execution") {
          const name = (span.name || "").replace(/^agent_execution:/, "").replace(/^agent:/, "") || "unknown";
          if (!map[name]) map[name] = { calls: 0, totalLatency: 0, totalCost: 0, totalTokens: 0, tools: {} };
          map[name].calls++;
          const dur = span.start_time && span.end_time
            ? new Date(span.end_time).getTime() - new Date(span.start_time).getTime() : 0;
          map[name].totalLatency += dur;
          map[name].totalCost += span.cost || 0;
          map[name].totalTokens += (span.tokens_in || 0) + (span.tokens_out || 0);
        }
        if (span.span_type === "tool_call") {
          // Attribute tool call to the most recently started agent in this trace
          const toolName = (span.name || "").replace(/^tool:/, "").replace(/^tool_call:/, "") || "unknown";
          // Find agent from same trace
          const agentSpan = (trace.spans || []).find((s: any) => s.span_type === "agent_execution");
          const agentName = agentSpan
            ? (agentSpan.name || "").replace(/^agent_execution:/, "").replace(/^agent:/, "") : "unknown";
          if (map[agentName]) {
            map[agentName].tools[toolName] = (map[agentName].tools[toolName] || 0) + 1;
          }
        }
      }
    }
    return Object.entries(map).map(([agent, d]) => ({
      agent,
      calls: d.calls,
      avgLatency: d.calls > 0 ? Math.round(d.totalLatency / d.calls) : 0,
      avgCost: d.calls > 0 ? d.totalCost / d.calls : 0,
      avgTokens: d.calls > 0 ? Math.round(d.totalTokens / d.calls) : 0,
      topTools: Object.entries(d.tools).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([t]) => t),
    })).sort((a, b) => b.calls - a.calls);
  }, [traces]);

  // Tool usage frequency from tool_call spans
  const toolUsage = useMemo(() => {
    const map: Record<string, number> = {};
    for (const trace of traces) {
      for (const span of trace.spans || []) {
        if (span.span_type === "tool_call") {
          const name = (span.name || "").replace(/^tool:/, "").replace(/^tool_call:/, "") || "unknown";
          map[name] = (map[name] || 0) + 1;
        }
      }
    }
    return Object.entries(map).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([name, count]) => ({ name, count }));
  }, [traces]);

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-5 max-w-6xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Monitoring & Observability</h1>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">Agent performance, cost, latency and OTel tracing</p>
        </div>
        <div className="flex items-center gap-2">
          <select value={teamId} onChange={e => setTeamId(e.target.value)} className="input !w-auto">
            {teams.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
          </select>
          <button onClick={loadAll} className="btn-secondary">Refresh</button>
        </div>
      </div>

      {/* Tabs */}
        <div className="flex gap-1 border-b border-[var(--border)]">
        {([
          { id: "overview" as const, label: "Overview" },
          { id: "otel" as const, label: "OTel Observability" },
        ]).map(tab => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-all ${
              activeTab === tab.id ? "border-zinc-900 text-zinc-900" : "border-transparent text-[var(--text-muted)] hover:text-[var(--text)]"
            }`}>
            {tab.label}
          </button>
        ))}
      </div>

      {/* ═══════════════════ OVERVIEW ═══════════════════ */}
      {activeTab === "overview" && (
        <div className="space-y-4">
          {/* System status row */}
          <div className="card !p-0 overflow-hidden">
            <table className="w-full text-sm">
              <thead><tr className="text-[var(--text-muted)] text-xs bg-[var(--bg)]">
                <th className="text-left px-4 py-2.5">Team</th>
                <th className="text-left px-4 py-2.5">Strategy</th>
                <th className="text-left px-4 py-2.5">Status</th>
                <th className="text-right px-4 py-2.5">Eval Runs</th>
                <th className="text-right px-4 py-2.5">OTel Spans (30d)</th>
              </tr></thead>
              <tbody><tr className="border-t border-[var(--border)]">
                <td className="px-4 py-2.5 font-medium">{selectedTeam?.name || "—"}</td>
                <td className="px-4 py-2.5">
                  {selectedTeam?.decision_strategy && (
                    <span className="px-2 py-0.5 rounded bg-zinc-100 text-zinc-600 text-xs border border-zinc-200 font-medium">
                      {selectedTeam.decision_strategy === "auto" ? "auto" : selectedTeam.decision_strategy}
                    </span>
                  )}
                </td>
                <td className="px-4 py-2.5"><span className="badge bg-[var(--success-light)] text-[var(--success)]">Active</span></td>
                <td className="text-right px-4 py-2.5 font-medium">{evalRuns.length}</td>
                <td className="text-right px-4 py-2.5 font-medium">{otelStats?.total_spans || 0}</td>
              </tr></tbody>
            </table>
          </div>

          {/* KPI row */}
          <div className="grid grid-cols-6 gap-3">
            <Metric label="Total Requests" value={String(totalRuns)} />
            <Metric label="Success Rate" value={`${(successRate * 100).toFixed(1)}%`}
              accent={successRate >= 0.9 ? "text-emerald-600" : successRate >= 0.7 ? "text-amber-600" : "text-red-600"} />
            <Metric label="Avg Latency" value={`${avgLatency.toFixed(0)}ms`}
              sub={`P95: ${(stats?.p95_latency_ms || 0).toFixed(0)}ms`} />
            <Metric label="Avg Cost/Run" value={`$${avgCostPerRun.toFixed(4)}`}
              sub={`Total: $${totalCost.toFixed(4)}`} />
            <Metric label="Avg Tokens/Run" value={String(avgTokensPerRun)}
              sub={`Total: ${(stats?.total_tokens || 0).toLocaleString()}`} />
            <Metric label="Errors" value={String(failures)}
              accent={failures > 0 ? "text-red-600" : "text-emerald-600"} />
          </div>

          {/* RAG stats row */}
          {ragStats && ragStats.total_queries > 0 && (
            <div className="card space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-medium">RAG Pipeline</h2>
                  <p className="text-[11px] text-[var(--text-muted)]">{ragStats.total_queries} queries · {ragStats.queries_with_eval} evaluated · {ragStats.pending_eval} pending</p>
                </div>
              </div>
              <div className="grid grid-cols-4 gap-3">
                <div className="bg-zinc-50 rounded-lg p-3 text-center">
                  <div className="text-lg font-semibold">{ragStats.total_queries}</div>
                  <div className="text-[10px] text-zinc-400 mt-0.5">Total Queries</div>
                </div>
                <div className="bg-zinc-50 rounded-lg p-3 text-center">
                  <div className="text-lg font-semibold">{ragStats.avg_latency_ms?.toFixed(0)}ms</div>
                  <div className="text-[10px] text-zinc-400 mt-0.5">Avg Latency</div>
                </div>
                <div className="bg-zinc-50 rounded-lg p-3 text-center">
                  <div className="text-lg font-semibold">{ragStats.avg_tokens?.toLocaleString()}</div>
                  <div className="text-[10px] text-zinc-400 mt-0.5">Avg Tokens</div>
                </div>
                <div className="bg-zinc-50 rounded-lg p-3 text-center">
                  <div className="text-lg font-semibold">{ragStats.queries_with_eval}</div>
                  <div className="text-[10px] text-zinc-400 mt-0.5">Evaluated</div>
                </div>
              </div>
              {Object.keys(ragStats.avg_scores || {}).length > 0 && (
                <div>
                  <div className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wide mb-2">Avg DeepEval Scores</div>
                  <div className="grid grid-cols-5 gap-2">
                    {[
                      { key: "answer_relevancy", label: "Answer Rel." },
                      { key: "faithfulness", label: "Faithfulness" },
                      { key: "contextual_relevancy", label: "Ctx. Rel." },
                      { key: "contextual_precision", label: "Ctx. Prec." },
                      { key: "contextual_recall", label: "Ctx. Rec." },
                    ].map(m => {
                      const s = ragStats.avg_scores[m.key];
                      const pct = s != null ? Math.round(s * 100) : null;
                      return (
                        <div key={m.key} className={`rounded-lg p-2 text-center border ${pct == null ? "border-zinc-100 bg-zinc-50" : pct >= 70 ? "border-emerald-100 bg-emerald-50" : pct >= 40 ? "border-amber-100 bg-amber-50" : "border-red-100 bg-red-50"}`}>
                          <div className={`text-sm font-semibold ${pct == null ? "text-zinc-300" : pct >= 70 ? "text-emerald-700" : pct >= 40 ? "text-amber-600" : "text-red-600"}`}>
                            {pct != null ? `${pct}%` : "—"}
                          </div>
                          <div className="text-[9px] text-zinc-400 mt-0.5 leading-tight">{m.label}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Charts row */}
          <div className="grid grid-cols-2 gap-4">
            <div className="card">
              <h2 className="text-sm font-medium mb-0.5">Latency per Request</h2>
              <p className="text-[11px] text-[var(--text-muted)] mb-3">Per-request end-to-end latency (last 30 requests)</p>
              {latencyData.length > 0 ? (
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={latencyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="idx" stroke="var(--text-muted)" fontSize={10} />
                    <YAxis stroke="var(--text-muted)" fontSize={10} />
                    <Tooltip {...tip} formatter={(v: any) => `${v}ms`} />
                    <Area type="monotone" dataKey="latency" name="Latency (ms)" stroke="#6366f1" fill="#6366f1" fillOpacity={0.12} strokeWidth={2} dot={{ r: 2 }} />
                  </AreaChart>
                </ResponsiveContainer>
              ) : <div className="text-sm text-[var(--text-muted)] text-center py-10">No traces yet</div>}
            </div>

            <div className="card">
              <h2 className="text-sm font-medium mb-0.5">Cost per Request</h2>
              <p className="text-[11px] text-[var(--text-muted)] mb-3">Estimated cost per run based on token usage × model rates</p>
              {latencyData.length > 0 ? (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={latencyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="idx" stroke="var(--text-muted)" fontSize={10} />
                    <YAxis stroke="var(--text-muted)" fontSize={10} tickFormatter={v => `$${v}`} />
                    <Tooltip {...tip} formatter={(v: any) => `$${Number(v).toFixed(5)}`} />
                    <Bar dataKey="cost" name="Cost ($)" fill="#059669" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : <div className="text-sm text-[var(--text-muted)] text-center py-10">No traces yet</div>}
            </div>
          </div>

          {/* Per-Model Performance */}
          {modelData.length > 0 && (
            <div className="card">
              <h2 className="text-sm font-medium mb-0.5">Per-Model Performance</h2>
              <p className="text-[11px] text-[var(--text-muted)] mb-3">Aggregated cost, token usage and call volume per LLM model (30-day window)</p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-[10px] text-[var(--text-muted)] uppercase bg-[var(--bg)]">
                      <th className="text-left px-3 py-2">Model</th>
                      <th className="text-right px-3 py-2">LLM Calls</th>
                      <th className="text-right px-3 py-2">Tokens In</th>
                      <th className="text-right px-3 py-2">Tokens Out</th>
                      <th className="text-right px-3 py-2">Total Cost</th>
                      <th className="text-right px-3 py-2">Cost/Call</th>
                      <th className="px-3 py-2">Distribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelData.map((m, i) => {
                      const totalCalls = modelData.reduce((s, x) => s + x.count, 0);
                      const pct = totalCalls > 0 ? m.count / totalCalls : 0;
                      return (
                        <tr key={m.model} className="border-t border-[var(--border)]">
                          <td className="px-3 py-2">
                            <div className="flex items-center gap-2">
                              <span className="w-2.5 h-2.5 rounded-sm" style={{ background: m.color }} />
                              <span className="font-mono">{m.shortModel}</span>
                            </div>
                          </td>
                          <td className="text-right px-3 py-2 font-semibold">{m.count.toLocaleString()}</td>
                          <td className="text-right px-3 py-2">{m.tokens_in.toLocaleString()}</td>
                          <td className="text-right px-3 py-2">{m.tokens_out.toLocaleString()}</td>
                          <td className="text-right px-3 py-2 font-semibold">${m.cost.toFixed(4)}</td>
                          <td className="text-right px-3 py-2">${m.count > 0 ? (m.cost / m.count).toFixed(5) : "—"}</td>
                          <td className="px-3 py-2">
                            <div className="flex items-center gap-1.5">
                              <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                                <div className="h-full rounded-full" style={{ width: `${pct * 100}%`, background: m.color }} />
                              </div>
                              <span className="text-[10px] text-[var(--text-muted)] w-8 shrink-0">{(pct * 100).toFixed(0)}%</span>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Per-Agent Activity + Tool Usage */}
          <div className="grid grid-cols-2 gap-4">
            {/* Per-Agent */}
            <div className="card">
              <h2 className="text-sm font-medium mb-0.5">Agent Activity Breakdown</h2>
              <p className="text-[11px] text-[var(--text-muted)] mb-3">Calls, avg latency, avg cost and top tools per agent (derived from traces)</p>
              {agentStats.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-[10px] text-[var(--text-muted)] uppercase bg-[var(--bg)]">
                        <th className="text-left px-2 py-1.5">Agent</th>
                        <th className="text-right px-2 py-1.5">Calls</th>
                        <th className="text-right px-2 py-1.5">Avg Latency</th>
                        <th className="text-right px-2 py-1.5">Avg Cost</th>
                        <th className="text-right px-2 py-1.5">Avg Tokens</th>
                        <th className="px-2 py-1.5">Top Tools</th>
                      </tr>
                    </thead>
                    <tbody>
                      {agentStats.map(a => (
                        <tr key={a.agent} className="border-t border-[var(--border)]">
                          <td className="px-2 py-1.5">
                            <span className="px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-600 text-[10px] border border-zinc-200 font-medium">{a.agent}</span>
                          </td>
                          <td className="text-right px-2 py-1.5 font-semibold">{a.calls}</td>
                          <td className="text-right px-2 py-1.5">{a.avgLatency > 0 ? `${a.avgLatency.toLocaleString()}ms` : "—"}</td>
                          <td className="text-right px-2 py-1.5">{a.avgCost > 0 ? `$${a.avgCost.toFixed(4)}` : "—"}</td>
                          <td className="text-right px-2 py-1.5">{a.avgTokens > 0 ? a.avgTokens.toLocaleString() : "—"}</td>
                          <td className="px-2 py-1.5">
                            <div className="flex flex-wrap gap-0.5">
                              {a.topTools.map(t => (
                                <span key={t} className="px-1 py-0.5 rounded bg-zinc-100 text-zinc-500 text-[9px] border border-zinc-200">{t}</span>
                              ))}
                              {a.topTools.length === 0 && <span className="text-[var(--text-muted)]">—</span>}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-sm text-[var(--text-muted)] text-center py-8">No agent span data yet — run some chat requests</div>
              )}
            </div>

            {/* Tool Usage */}
            <div className="card">
              <h2 className="text-sm font-medium mb-0.5">Tool Call Frequency</h2>
              <p className="text-[11px] text-[var(--text-muted)] mb-3">Most used tools across all recent requests</p>
              {toolUsage.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={toolUsage} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis type="number" stroke="var(--text-muted)" fontSize={10} allowDecimals={false} />
                    <YAxis type="category" dataKey="name" stroke="var(--text-muted)" fontSize={9} width={110} />
                    <Tooltip {...tip} />
                    <Bar dataKey="count" name="Calls" fill="#10b981" radius={[0, 3, 3, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-sm text-[var(--text-muted)] text-center py-8">No tool call data yet</div>
              )}
            </div>
          </div>

          {/* Operational percentiles */}
          <div className="card">
            <h2 className="text-sm font-medium mb-3">Operational Efficiency (P-Percentiles)</h2>
            <table className="w-full text-sm">
              <thead><tr className="text-[var(--text-muted)] text-xs">
                <th className="text-left py-2">Metric</th>
                <th className="text-right">Mean</th>
                <th className="text-right">P50</th>
                <th className="text-right">P95</th>
                <th className="text-right">P99</th>
                <th className="text-right">Total</th>
              </tr></thead>
              <tbody>
                <tr className="border-t border-[var(--border)]">
                  <td className="py-2 font-medium">End-to-End Latency</td>
                  <td className="text-right">{avgLatency.toFixed(0)}ms</td>
                  <td className="text-right">{(stats?.p50_latency_ms || 0).toFixed(0)}ms</td>
                  <td className="text-right">{(stats?.p95_latency_ms || 0).toFixed(0)}ms</td>
                  <td className="text-right">{(stats?.p99_latency_ms || 0).toFixed(0)}ms</td>
                  <td className="text-right text-[var(--text-muted)]">—</td>
                </tr>
                <tr className="border-t border-[var(--border)]">
                  <td className="py-2 font-medium">Tokens per Run</td>
                  <td className="text-right">{avgTokensPerRun.toLocaleString()}</td>
                  <td className="text-right" colSpan={3}><span className="text-[var(--text-muted)]">—</span></td>
                  <td className="text-right">{(stats?.total_tokens || 0).toLocaleString()}</td>
                </tr>
                <tr className="border-t border-[var(--border)]">
                  <td className="py-2 font-medium">Cost per Run</td>
                  <td className="text-right">${avgCostPerRun.toFixed(4)}</td>
                  <td className="text-right" colSpan={3}><span className="text-[var(--text-muted)]">—</span></td>
                  <td className="text-right">${totalCost.toFixed(4)}</td>
                </tr>
                <tr className="border-t border-[var(--border)]">
                  <td className="py-2 font-medium">Error Rate</td>
                  <td className="text-right" colSpan={4}>
                    <span className={failures > 0 ? "text-red-600 font-semibold" : "text-emerald-600"}>
                      {totalRuns > 0 ? ((failures / totalRuns) * 100).toFixed(1) : "0"}% ({failures} / {totalRuns} requests)
                    </span>
                  </td>
                  <td />
                </tr>
              </tbody>
            </table>
          </div>

          {/* Quality Trends */}
          {qualityData.length > 0 && (
            <div className="card">
              <h2 className="text-sm font-medium mb-0.5">Quality Trends (Eval Runs)</h2>
              <p className="text-[11px] text-[var(--text-muted)] mb-3">Task success, tool accuracy, routing accuracy and faithfulness over evaluation runs</p>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={qualityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="run" stroke="var(--text-muted)" fontSize={10} />
                  <YAxis stroke="var(--text-muted)" fontSize={10} domain={[0, 100]} tickFormatter={v => `${v}%`} />
                  <Tooltip {...tip} formatter={(v: any) => `${v}%`} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Line type="monotone" dataKey="success" name="Success" stroke="#16a34a" strokeWidth={1.5} dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="toolAcc" name="Tool Acc." stroke="#2563eb" strokeWidth={1.5} dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="routing" name="Routing" stroke="#7c3aed" strokeWidth={1.5} dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="faithfulness" name="Faithfulness" stroke="#0891b2" strokeWidth={1.5} dot={{ r: 2 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Recent requests */}
          <div className="card">
            <h2 className="text-sm font-medium mb-3">Recent Requests</h2>
            <div className="space-y-0">
              {traces.slice(0, 15).map((t: any) => {
                const agentSpan = (t.spans || []).find((s: any) => s.span_type === "agent_execution");
                const agentName = agentSpan ? (agentSpan.name || "").replace(/^agent_execution:/, "").replace(/^agent:/, "") : null;
                const toolCount = (t.spans || []).filter((s: any) => s.span_type === "tool_call").length;
                const llmCount = (t.spans || []).filter((s: any) => s.span_type === "llm_call").length;
                return (
                  <div key={t.id} className="flex items-center gap-3 py-2 border-b border-[var(--border)] text-xs">
                    <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${t.status === "completed" ? "bg-emerald-500" : "bg-red-500"}`} />
                    <span className="flex-1 truncate text-[var(--text-muted)]">{t.user_prompt}</span>
                    {agentName && (
                      <span className="px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-600 text-[10px] border border-zinc-200 shrink-0 font-medium">{agentName}</span>
                    )}
                    <div className="flex items-center gap-2 shrink-0 text-[var(--text-muted)]">
                      {llmCount > 0 && <span className="text-[10px]">{llmCount} LLM</span>}
                      {toolCount > 0 && <span className="text-[10px]">{toolCount} tools</span>}
                      <span>{(t.total_latency_ms || 0).toFixed(0)}ms</span>
                      <span>${(t.total_cost || 0).toFixed(4)}</span>
                    </div>
                  </div>
                );
              })}
              {traces.length === 0 && (
                <div className="text-[var(--text-muted)] text-center py-8">No traces yet — send chat messages to generate data</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ═══════════════════ OTEL OBSERVABILITY ═══════════════════ */}
      {activeTab === "otel" && (
        <div className="space-y-4">
          {/* KPIs */}
          <div className="grid grid-cols-5 gap-3">
            <Metric label="Total Spans" value={String(otelStats?.total_spans || 0)} sub="Across all traces" />
            <Metric label="Total Traces" value={String(otelStats?.total_traces || 0)} />
            <Metric label="Error Spans" value={String(otelStats?.error_spans || 0)}
              accent={(otelStats?.error_spans || 0) > 0 ? "text-red-600" : "text-emerald-600"} />
            <Metric label="Span Types" value={String(spanTypeData.length)} sub="llm, tool, routing, etc." />
            <Metric label="Models Seen" value={String(modelData.length)}
              sub={modelData.slice(0, 2).map(m => m.shortModel).join(", ")} />
          </div>

          {/* Span type + model breakdown */}
          <div className="grid grid-cols-2 gap-4">
            <div className="card">
              <h3 className="text-sm font-medium mb-0.5">Span Type Distribution</h3>
              <p className="text-[11px] text-[var(--text-muted)] mb-3">GenAI semantic conventions: routing, llm_call, tool_call, agent_execution, supervisor</p>
              {spanTypeData.length > 0 ? (
                <div className="flex items-center gap-4">
                  <ResponsiveContainer width="45%" height={200}>
                    <PieChart>
                      <Pie data={spanTypeData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80}
                        label={({ name, percent }: any) => `${name ?? ""} ${((percent ?? 0) * 100).toFixed(0)}%`}
                        labelLine={false} fontSize={9}>
                        {spanTypeData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
                      </Pie>
                      <Tooltip {...tip} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="flex-1 space-y-1.5">
                    {spanTypeData.map(s => (
                      <div key={s.name} className="flex items-center justify-between text-xs py-1 border-b border-[var(--border)]">
                        <div className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded-sm" style={{ background: s.fill }} />
                          <span className="capitalize">{s.name}</span>
                        </div>
                        <div className="flex gap-2 text-[var(--text-muted)]">
                          <span>{s.value} spans</span>
                          <span>{s.tokens.toLocaleString()} tok</span>
                          <span>${s.cost.toFixed(4)}</span>
                          <span>{s.avgLatency.toFixed(0)}ms</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : <div className="text-sm text-[var(--text-muted)] text-center py-10">No span data yet</div>}
            </div>

            <div className="card">
              <h3 className="text-sm font-medium mb-0.5">Model Usage Breakdown</h3>
              <p className="text-[11px] text-[var(--text-muted)] mb-3">LLM call counts per model (gen_ai.request.model attribute)</p>
              {modelData.length > 0 ? (
                <div className="space-y-3">
                  <ResponsiveContainer width="100%" height={130}>
                    <BarChart data={modelData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" stroke="var(--text-muted)" fontSize={10} allowDecimals={false} />
                      <YAxis type="category" dataKey="shortModel" stroke="var(--text-muted)" fontSize={9} width={110} />
                      <Tooltip {...tip} />
                      <Bar dataKey="count" name="LLM Calls" radius={[0, 3, 3, 0]}>
                        {modelData.map((m, i) => <Cell key={i} fill={m.color} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <div className="space-y-1">
                    {modelData.map(m => (
                      <div key={m.model} className="flex items-center justify-between text-xs border-b border-[var(--border)] py-1">
                        <div className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded-sm" style={{ background: m.color }} />
                          <span className="font-mono">{m.shortModel}</span>
                        </div>
                        <div className="flex gap-3 text-[var(--text-muted)]">
                          <span>In: {m.tokens_in.toLocaleString()}</span>
                          <span>Out: {m.tokens_out.toLocaleString()}</span>
                          <span className="font-medium text-[var(--text)]">${m.cost.toFixed(4)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : <div className="text-sm text-[var(--text-muted)] text-center py-10">No model data yet</div>}
            </div>
          </div>

          {/* Token flow + cost per trace */}
          {tokenFlow.length > 0 && (
            <div className="grid grid-cols-2 gap-4">
              <div className="card">
                <h3 className="text-sm font-medium mb-0.5">Token Flow Over Time</h3>
                <p className="text-[11px] text-[var(--text-muted)] mb-3">gen_ai.usage.input_tokens / output_tokens per trace</p>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={tokenFlow}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="id" stroke="var(--text-muted)" fontSize={10} />
                    <YAxis stroke="var(--text-muted)" fontSize={10} />
                    <Tooltip {...tip} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Area type="monotone" dataKey="tokens_in" name="Input Tokens" stroke="#2563eb" fill="#2563eb" fillOpacity={0.15} />
                    <Area type="monotone" dataKey="tokens_out" name="Output Tokens" stroke="#7c3aed" fill="#7c3aed" fillOpacity={0.15} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="card">
                <h3 className="text-sm font-medium mb-0.5">Cost per Trace</h3>
                <p className="text-[11px] text-[var(--text-muted)] mb-3">Estimated via DBSpanProcessor: tokens × model pricing rates</p>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={tokenFlow}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="id" stroke="var(--text-muted)" fontSize={10} />
                    <YAxis stroke="var(--text-muted)" fontSize={10} tickFormatter={v => `$${v}`} />
                    <Tooltip {...tip} formatter={(v: any) => `$${Number(v ?? 0).toFixed(5)}`} />
                    <Bar dataKey="cost" name="Cost ($)" fill="#059669" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Recent traces with span breakdown */}
          <div className="card">
            <h3 className="text-sm font-medium mb-3">Recent Traces (Span Breakdown)</h3>
            <div className="space-y-0 max-h-[350px] overflow-y-auto">
              {traces.slice(0, 20).map((t: any) => {
                const spanCounts: Record<string, number> = {};
                for (const s of t.spans || []) spanCounts[s.span_type] = (spanCounts[s.span_type] || 0) + 1;
                const models = [...new Set((t.spans || []).filter((s: any) => s.model).map((s: any) => s.model as string))];
                return (
                  <div key={t.id} className="py-2 border-b border-[var(--border)]">
                    <div className="flex items-start gap-2 text-xs">
                      <span className={`w-1.5 h-1.5 rounded-full mt-1 shrink-0 ${t.status === "completed" ? "bg-emerald-500" : "bg-red-500"}`} />
                      <div className="flex-1 min-w-0">
                        <div className="truncate text-[var(--text-muted)]">{t.user_prompt}</div>
                        {models.length > 0 && (
                          <div className="flex flex-wrap gap-0.5 mt-0.5">
                            {(models as string[]).map((m: string) => (
                              <span key={m} className="px-1 py-0.5 rounded bg-zinc-100 text-zinc-500 text-[9px] border border-zinc-200 font-mono">
                                {m.split("/").pop() || m}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0 flex-wrap justify-end">
                        {Object.entries(spanCounts).map(([type, count]) => (
                          <span key={type} className="px-1.5 py-0.5 rounded text-[9px] border"
                            style={{ borderColor: SPAN_COLORS[type] || "#6b7280", color: SPAN_COLORS[type] || "#6b7280" }}>
                            {type.replace(/_/g, " ")} ×{count}
                          </span>
                        ))}
                        <span className="text-[10px] text-[var(--text-muted)] ml-1">{(t.total_latency_ms || 0).toFixed(0)}ms</span>
                        <span className="text-[10px] text-[var(--text-muted)]">${(t.total_cost || 0).toFixed(4)}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
              {traces.length === 0 && <div className="text-[var(--text-muted)] text-center py-8">Send chat messages to generate traces</div>}
            </div>
          </div>

          {/* OTel architecture info */}
          <div className="card">
            <h3 className="text-sm font-medium mb-3">OpenTelemetry Architecture</h3>
            <div className="grid grid-cols-4 gap-2 text-xs">
              {[
                { title: "GenAI Semantic Conventions", body: "Spans use gen_ai.* namespace: gen_ai.request.model, gen_ai.usage.input_tokens, gen_ai.usage.output_tokens, gen_ai.operation.name" },
                { title: "OpenInference Auto-Instrumentation", body: "LangChainInstrumentor captures all LLM calls, tool invocations, and chain executions as OTel spans automatically." },
                { title: "Custom DBSpanProcessor", body: "Persists every span to SQLite with cost estimation (tokens × model rates). Enriches spans with span.type classification." },
                { title: "OTLP Export", body: "Dual export: local DB + optional OTLP HTTP to Phoenix, Langfuse, Jaeger, or Datadog via BatchSpanProcessor." },
              ].map(item => (
                <div key={item.title} className="p-2.5 rounded bg-zinc-50 border border-zinc-200 text-[var(--text-muted)]">
                  <div className="font-medium text-zinc-700 mb-1">{item.title}</div>
                  <div>{item.body}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
