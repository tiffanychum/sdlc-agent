"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area,
} from "recharts";

const REGRESSION_METRICS = [
  { key: "task_success_rate", label: "Task Success", color: "#16a34a" },
  { key: "tool_accuracy", label: "Tool Accuracy", color: "#2563eb" },
  { key: "routing_accuracy", label: "Routing Acc.", color: "#7c3aed" },
  { key: "reasoning_quality", label: "Reasoning", color: "#ca8a04" },
  { key: "step_efficiency", label: "Step Effic.", color: "#0891b2" },
  { key: "faithfulness", label: "Faithfulness", color: "#059669" },
  { key: "safety_compliance", label: "Safety", color: "#dc2626" },
];

const SPAN_TYPE_COLORS: Record<string, string> = {
  routing: "#3b82f6",
  agent_execution: "#8b5cf6",
  tool_call: "#10b981",
  supervisor: "#f59e0b",
  llm_call: "#6366f1",
  unknown: "#6b7280",
};

const REGRESSION_THRESHOLD = -0.05;

function Metric({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: string }) {
  return (
    <div className="card text-center">
      <div className={`text-2xl font-semibold ${accent || ""}`}>{value}</div>
      <div className="text-xs text-[var(--text-muted)] mt-1">{label}</div>
      {sub && <div className="text-[11px] text-[var(--text-muted)] mt-0.5">{sub}</div>}
    </div>
  );
}

function DeltaBadge({ delta }: { delta: number }) {
  const isRegression = delta < REGRESSION_THRESHOLD;
  const isImproved = delta > 0.05;
  const color = isRegression ? "bg-red-50 text-red-700 border-red-200"
    : isImproved ? "bg-emerald-50 text-emerald-700 border-emerald-200"
    : "bg-gray-50 text-gray-600 border-gray-200";
  const icon = isRegression ? "▼" : isImproved ? "▲" : "—";
  return (
    <span className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded border text-[10px] font-medium ${color}`}>
      {icon} {(delta * 100).toFixed(1)}%
    </span>
  );
}

export default function MonitoringPage() {
  const [activeTab, setActiveTab] = useState<"overview" | "regression" | "otel">("overview");
  const [teams, setTeams] = useState<any[]>([]);
  const [teamId, setTeamId] = useState("");
  const [stats, setStats] = useState<any>(null);
  const [evalRuns, setEvalRuns] = useState<any[]>([]);
  const [traces, setTraces] = useState<any[]>([]);
  const [otelStats, setOtelStats] = useState<any>(null);

  // Regression state
  const [runA, setRunA] = useState<string>("");
  const [runB, setRunB] = useState<string>("");
  const [comparison, setComparison] = useState<any>(null);
  const [comparing, setComparing] = useState(false);
  const [runningEval, setRunningEval] = useState(false);
  const [evalMsg, setEvalMsg] = useState("");

  useEffect(() => {
    api.teams.list().then(t => { setTeams(t); if (t.length) setTeamId(t[0].id); });
  }, []);

  useEffect(() => {
    if (!teamId) return;
    loadAll();
  }, [teamId]);

  async function loadAll() {
    const [s, e, t, o] = await Promise.all([
      api.traces.stats(30),
      api.eval.runs(),
      api.traces.list(50),
      api.otel.spanStats(30),
    ]);
    setStats(s);
    setEvalRuns(e);
    setTraces(t);
    setOtelStats(o);
    if (e.length >= 2) {
      setRunA(e[1].id);
      setRunB(e[0].id);
    } else if (e.length === 1) {
      setRunA(e[0].id);
    }
  }

  async function runEvaluation() {
    setRunningEval(true);
    setEvalMsg("Running evaluation scenarios...");
    try {
      await api.eval.run(teamId);
      setEvalMsg("Evaluation complete. Reloading...");
      await loadAll();
      setEvalMsg("");
    } catch (e: any) {
      setEvalMsg(`Error: ${e.message}`);
    } finally {
      setRunningEval(false);
    }
  }

  async function compareRuns() {
    if (!runA || !runB) return;
    setComparing(true);
    try {
      const result = await api.eval.compareRuns(runA, runB);
      setComparison(result);
    } catch {
      setComparison(null);
    } finally {
      setComparing(false);
    }
  }

  const selectedTeam = teams.find(t => t.id === teamId);
  const totalRuns = stats?.total_runs || traces.length || 0;
  const failures = stats?.failures || traces.filter((t: any) => t.status === "error").length;
  const successRate = totalRuns > 0 ? ((totalRuns - failures) / totalRuns) : 0;
  const avgLatency = stats?.avg_latency_ms || 0;
  const totalCost = stats?.total_cost || 0;

  const qualityData = evalRuns.slice(0, 15).reverse().map((r: any) => ({
    run: (r.prompt_version || r.id?.slice(0, 6)),
    success: +((r.task_success_rate || 0) * 100).toFixed(1),
    toolAcc: +((r.tool_accuracy || 0) * 100).toFixed(1),
    routing: +((r.routing_accuracy || 0) * 100).toFixed(1),
    reasoning: +((r.reasoning_quality || 0) * 100).toFixed(1),
    safety: +((r.safety_compliance || 0) * 100).toFixed(1),
    faithfulness: +((r.faithfulness || 0) * 100).toFixed(1),
  }));

  const latencyData = traces.slice(0, 25).reverse().map((t: any, i: number) => ({
    idx: i + 1,
    latency: +(t.total_latency_ms || 0).toFixed(0),
    tokens: t.total_tokens || 0,
    cost: +(t.total_cost || 0).toFixed(4),
  }));

  const tip = { contentStyle: { background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 } };

  // OTel data transforms
  const spanTypeData = otelStats?.by_type ? Object.entries(otelStats.by_type).map(([type, data]: [string, any]) => ({
    name: type.replace(/_/g, " "),
    value: data.count,
    tokens: data.tokens_in + data.tokens_out,
    cost: data.cost,
    avgLatency: data.avg_latency_ms,
    fill: SPAN_TYPE_COLORS[type] || SPAN_TYPE_COLORS.unknown,
  })) : [];

  const modelData = otelStats?.by_model ? Object.entries(otelStats.by_model).map(([model, data]: [string, any]) => ({
    model: model || "unknown",
    count: data.count,
    tokens_in: data.tokens_in,
    tokens_out: data.tokens_out,
    cost: data.cost,
  })) : [];

  const tokenFlow = otelStats?.token_flow || [];

  const regressionCount = comparison?.comparison
    ? Object.values(comparison.comparison).filter((v: any) => v.regression).length
    : 0;

  return (
    <div className="space-y-5 max-w-6xl">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Monitoring & Observability</h1>
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
          { id: "regression" as const, label: "Regression Testing" },
          { id: "otel" as const, label: "OTel Observability" },
        ]).map(tab => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-all ${
              activeTab === tab.id ? "border-[var(--accent)] text-[var(--accent)]" : "border-transparent text-[var(--text-muted)] hover:text-[var(--text)]"
            }`}>
            {tab.label}
          </button>
        ))}
      </div>

      {/* ═══════════════════ OVERVIEW TAB ═══════════════════ */}
      {activeTab === "overview" && (
        <div className="space-y-4">
          {/* App Info */}
          <div className="card !p-0 overflow-hidden">
            <table className="w-full text-sm">
              <thead><tr className="text-[var(--text-muted)] text-xs bg-[var(--bg)]">
                <th className="text-left px-4 py-2.5">Project</th>
                <th className="text-left px-4 py-2.5">Application</th>
                <th className="text-left px-4 py-2.5">Status</th>
                <th className="text-right px-4 py-2.5">Eval Runs</th>
                <th className="text-right px-4 py-2.5">OTel Spans</th>
              </tr></thead>
              <tbody><tr className="border-t border-[var(--border)]">
                <td className="px-4 py-2.5">SDLC Agent</td>
                <td className="px-4 py-2.5">{selectedTeam?.name || "—"}</td>
                <td className="px-4 py-2.5"><span className="badge bg-[var(--success-light)] text-[var(--success)]">Active</span></td>
                <td className="text-right px-4 py-2.5 font-medium">{evalRuns.length}</td>
                <td className="text-right px-4 py-2.5 font-medium">{otelStats?.total_spans || 0}</td>
              </tr></tbody>
            </table>
          </div>

          {/* Overview KPIs */}
          <div className="grid grid-cols-5 gap-3">
            <Metric label="Total Runs" value={String(totalRuns)} />
            <Metric label="Pass Rate" value={`${(successRate * 100).toFixed(1)}%`}
              accent={successRate >= 0.9 ? "text-emerald-600" : successRate >= 0.7 ? "text-amber-600" : "text-red-600"} />
            <Metric label="Avg Latency" value={`${avgLatency.toFixed(0)}ms`} sub={`P95: ${(stats?.p95_latency_ms || 0).toFixed(0)}ms`} />
            <Metric label="Total Cost" value={`$${totalCost.toFixed(4)}`} sub={`$${totalRuns > 0 ? (totalCost/totalRuns).toFixed(4) : "0"}/run`} />
            <Metric label="Failures" value={String(failures)}
              accent={failures > 0 ? "text-red-600" : "text-emerald-600"} />
          </div>

          {/* Quality Trends + Latency */}
          <div className="grid grid-cols-2 gap-4">
            <div className="card">
              <h2 className="text-sm font-medium mb-1">Quality Trends (Eval Runs)</h2>
              <p className="text-[11px] text-[var(--text-muted)] mb-3">CLASSic framework: Cost, Latency, Accuracy, Stability, Security</p>
              {qualityData.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={qualityData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="run" stroke="var(--text-muted)" fontSize={10} />
                    <YAxis stroke="var(--text-muted)" fontSize={10} domain={[0, 100]} />
                    <Tooltip {...tip} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Line type="monotone" dataKey="success" name="Success" stroke="#16a34a" strokeWidth={1.5} dot={{ r: 2 }} />
                    <Line type="monotone" dataKey="toolAcc" name="Tool Acc." stroke="#2563eb" strokeWidth={1.5} dot={{ r: 2 }} />
                    <Line type="monotone" dataKey="safety" name="Safety" stroke="#dc2626" strokeWidth={1.5} dot={{ r: 2 }} />
                    <Line type="monotone" dataKey="faithfulness" name="Faithfulness" stroke="#0891b2" strokeWidth={1.5} dot={{ r: 2 }} />
                    <Line type="monotone" dataKey="routing" name="Routing" stroke="#7c3aed" strokeWidth={1.5} dot={{ r: 2 }} />
                  </LineChart>
                </ResponsiveContainer>
              ) : <div className="text-sm text-[var(--text-muted)] text-center py-10">Run evaluations to see quality trends</div>}
            </div>

            <div className="card">
              <h2 className="text-sm font-medium mb-1">Latency Distribution</h2>
              <p className="text-[11px] text-[var(--text-muted)] mb-3">Per-request latency (last 25 requests)</p>
              {latencyData.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={latencyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="idx" stroke="var(--text-muted)" fontSize={10} />
                    <YAxis stroke="var(--text-muted)" fontSize={10} />
                    <Tooltip {...tip} />
                    <Bar dataKey="latency" name="Latency (ms)" fill="#2563eb" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : <div className="text-sm text-[var(--text-muted)] text-center py-10">No trace data yet</div>}
            </div>
          </div>

          {/* Percentile Table */}
          <div className="card">
            <h2 className="text-sm font-medium mb-2">Operational Efficiency</h2>
            <table className="w-full text-sm">
              <thead><tr className="text-[var(--text-muted)] text-xs">
                <th className="text-left py-2">Metric</th><th className="text-right">Avg</th><th className="text-right">P50</th><th className="text-right">P95</th><th className="text-right">P99</th>
              </tr></thead>
              <tbody>
                <tr className="border-t border-[var(--border)]">
                  <td className="py-2 font-medium">Latency</td>
                  <td className="text-right">{avgLatency.toFixed(0)}ms</td>
                  <td className="text-right">{(stats?.p50_latency_ms || 0).toFixed(0)}ms</td>
                  <td className="text-right">{(stats?.p95_latency_ms || 0).toFixed(0)}ms</td>
                  <td className="text-right">{(stats?.p99_latency_ms || 0).toFixed(0)}ms</td>
                </tr>
                <tr className="border-t border-[var(--border)]">
                  <td className="py-2 font-medium">Tokens/Run</td>
                  <td className="text-right">{stats?.avg_tokens || 0}</td>
                  <td className="text-right" colSpan={3}>Total: {stats?.total_tokens || 0}</td>
                </tr>
                <tr className="border-t border-[var(--border)]">
                  <td className="py-2 font-medium">Cost/Run</td>
                  <td className="text-right">${totalRuns > 0 ? (totalCost / totalRuns).toFixed(4) : "0.00"}</td>
                  <td className="text-right" colSpan={3}>Total: ${totalCost.toFixed(4)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ═══════════════════ REGRESSION TESTING TAB ═══════════════════ */}
      {activeTab === "regression" && (
        <div className="space-y-4">
          {/* Header */}
          <div className="card">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h2 className="text-sm font-medium">Regression Testing</h2>
                <p className="text-[11px] text-[var(--text-muted)]">
                  Run standardized evaluation scenarios, then compare runs to detect performance regressions.
                  Follows the CLASSic framework (Cost, Latency, Accuracy, Stability, Security).
                </p>
              </div>
              <div className="flex items-center gap-2">
                {evalMsg && <span className="text-xs text-[var(--accent)]">{evalMsg}</span>}
                <button onClick={runEvaluation} disabled={runningEval} className="btn-primary">
                  {runningEval ? (
                    <span className="flex items-center gap-1.5">
                      <span className="h-3 w-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Running...
                    </span>
                  ) : "Run Evaluation Suite"}
                </button>
              </div>
            </div>

            {/* Run Comparison Selector */}
            <div className="border-t border-[var(--border)] pt-3">
              <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-2">Compare Two Runs</div>
              <div className="flex items-center gap-3">
                <div className="flex-1">
                  <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">Baseline (Run A)</label>
                  <select value={runA} onChange={e => setRunA(e.target.value)} className="input !py-1.5 text-xs w-full">
                    <option value="">Select baseline...</option>
                    {evalRuns.map(r => (
                      <option key={r.id} value={r.id}>
                        {r.prompt_version} — {r.model} ({r.num_tasks} tasks, {((r.task_success_rate || 0) * 100).toFixed(0)}% success)
                        {r.created_at ? ` — ${new Date(r.created_at).toLocaleDateString()}` : ""}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="text-lg text-[var(--text-muted)] mt-4">→</div>
                <div className="flex-1">
                  <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">Candidate (Run B)</label>
                  <select value={runB} onChange={e => setRunB(e.target.value)} className="input !py-1.5 text-xs w-full">
                    <option value="">Select candidate...</option>
                    {evalRuns.map(r => (
                      <option key={r.id} value={r.id}>
                        {r.prompt_version} — {r.model} ({r.num_tasks} tasks, {((r.task_success_rate || 0) * 100).toFixed(0)}% success)
                        {r.created_at ? ` — ${new Date(r.created_at).toLocaleDateString()}` : ""}
                      </option>
                    ))}
                  </select>
                </div>
                <button onClick={compareRuns} disabled={comparing || !runA || !runB || runA === runB}
                  className="btn-primary mt-4">
                  {comparing ? "Comparing..." : "Compare"}
                </button>
              </div>
            </div>
          </div>

          {/* Comparison Results */}
          {comparison?.comparison && (
            <div className="space-y-3">
              {/* Regression Summary */}
              <div className={`card !p-3 border-l-4 ${
                regressionCount > 0 ? "border-l-red-500 bg-red-50/30" : "border-l-emerald-500 bg-emerald-50/30"
              }`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-lg">{regressionCount > 0 ? "⚠️" : "✅"}</span>
                    <div>
                      <div className="text-sm font-medium">
                        {regressionCount > 0
                          ? `${regressionCount} regression(s) detected across ${Object.keys(comparison.comparison).length} metrics`
                          : `All ${Object.keys(comparison.comparison).length} metrics passed — safe to deploy`}
                      </div>
                      <div className="text-[11px] text-[var(--text-muted)]">
                        Threshold: {(REGRESSION_THRESHOLD * 100).toFixed(0)}% drop triggers regression flag
                        {comparison.regressions?.length > 0 && (
                          <span className="text-red-600 ml-2">
                            Failed: {comparison.regressions.map((r: string) =>
                              REGRESSION_METRICS.find(m => m.key === r)?.label || r
                            ).join(", ")}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <span className={`text-xs font-semibold px-3 py-1 rounded ${
                    comparison.pass ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"
                  }`}>
                    {comparison.pass ? "PASS" : "FAIL"}
                  </span>
                </div>
                {comparison.meta_a && comparison.meta_b && (
                  <div className="mt-2 pt-2 border-t border-[var(--border)] flex gap-6 text-[10px] text-[var(--text-muted)]">
                    <div><span className="font-medium">Baseline:</span> {comparison.meta_a.model} / {comparison.meta_a.prompt_version} ({comparison.meta_a.num_tasks} tasks)</div>
                    <div><span className="font-medium">Candidate:</span> {comparison.meta_b.model} / {comparison.meta_b.prompt_version} ({comparison.meta_b.num_tasks} tasks)</div>
                  </div>
                )}
              </div>

              {/* Detailed Metric Table */}
              <div className="card !p-0 overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-[var(--text-muted)] text-xs bg-[var(--bg)]">
                      <th className="text-left px-4 py-2.5">Metric</th>
                      <th className="text-right px-4 py-2.5">Baseline</th>
                      <th className="text-right px-4 py-2.5">Candidate</th>
                      <th className="text-right px-4 py-2.5">Delta</th>
                      <th className="text-center px-4 py-2.5">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(comparison.comparison).map(([metric, data]: [string, any]) => {
                      const label = REGRESSION_METRICS.find(m => m.key === metric)?.label || metric.replace(/_/g, " ");
                      return (
                        <tr key={metric} className={`border-t border-[var(--border)] ${data.regression ? "bg-red-50/50" : ""}`}>
                          <td className="px-4 py-2 font-medium capitalize">{label}</td>
                          <td className="text-right px-4 py-2">{(data.before * 100).toFixed(1)}%</td>
                          <td className="text-right px-4 py-2">{(data.after * 100).toFixed(1)}%</td>
                          <td className="text-right px-4 py-2"><DeltaBadge delta={data.delta} /></td>
                          <td className="text-center px-4 py-2">
                            {data.regression
                              ? <span className="text-xs font-medium text-red-700 bg-red-100 px-2 py-0.5 rounded">REGRESSION</span>
                              : <span className="text-xs font-medium text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded">PASS</span>
                            }
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Visual Comparison */}
              <div className="card">
                <h3 className="text-xs text-[var(--text-muted)] mb-2">Visual Comparison: Baseline vs Candidate</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={Object.entries(comparison.comparison).map(([metric, data]: [string, any]) => ({
                    metric: REGRESSION_METRICS.find(m => m.key === metric)?.label || metric.replace(/_/g, " "),
                    Baseline: +(data.before * 100).toFixed(1),
                    Candidate: +(data.after * 100).toFixed(1),
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="metric" stroke="var(--text-muted)" fontSize={9} angle={-15} textAnchor="end" height={50} />
                    <YAxis stroke="var(--text-muted)" fontSize={10} domain={[0, 100]} />
                    <Tooltip {...tip} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Bar dataKey="Baseline" fill="#6b7280" radius={[2, 2, 0, 0]} />
                    <Bar dataKey="Candidate" fill="#2563eb" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Eval Run History */}
          <div className="card">
            <h3 className="text-xs text-[var(--text-muted)] mb-2">Evaluation Run History ({evalRuns.length} runs)</h3>
            {evalRuns.length > 0 ? (
              <div className="space-y-1 max-h-[300px] overflow-y-auto">
                {evalRuns.map((r: any) => (
                  <div key={r.id} className="flex items-center justify-between text-xs py-2 border-b border-[var(--border)]">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[var(--text-muted)]">{r.id}</span>
                      <span className="badge bg-[var(--accent-light)] text-[var(--accent)]">{r.prompt_version}</span>
                      <span className="text-[var(--text-muted)]">{r.model}</span>
                    </div>
                    <div className="flex items-center gap-3 text-[var(--text-muted)]">
                      <span>{r.num_tasks} tasks</span>
                      <span className={`font-medium ${(r.task_success_rate || 0) >= 0.7 ? "text-emerald-600" : "text-red-600"}`}>
                        {((r.task_success_rate || 0) * 100).toFixed(0)}% success
                      </span>
                      {r.created_at && <span>{new Date(r.created_at).toLocaleDateString()}</span>}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-[var(--text-muted)] text-center py-6">
                No evaluation runs yet. Click &quot;Run Evaluation Suite&quot; to create a baseline.
              </div>
            )}
          </div>

          {/* Methodology */}
          <div className="card">
            <h3 className="text-xs text-[var(--text-muted)] mb-2">Regression Testing Methodology</h3>
            <div className="grid grid-cols-3 gap-3 text-xs text-[var(--text-muted)]">
              <div className="p-2.5 rounded bg-blue-50 border border-blue-100">
                <div className="font-medium text-blue-700 mb-1">1. Run Scenarios</div>
                <div>Execute {evalRuns.length > 0 ? "16" : "N"} standardized scenarios (quick, medium, complex) against the agent. Each tests routing, tool selection, and output quality.</div>
              </div>
              <div className="p-2.5 rounded bg-blue-50 border border-blue-100">
                <div className="font-medium text-blue-700 mb-1">2. Compare Runs</div>
                <div>Compare any two runs (before/after a prompt change, model swap, or config update). 7 metrics are compared with a {(REGRESSION_THRESHOLD * 100).toFixed(0)}% threshold.</div>
              </div>
              <div className="p-2.5 rounded bg-blue-50 border border-blue-100">
                <div className="font-medium text-blue-700 mb-1">3. Gate Deployment</div>
                <div>Any metric dropping &gt;5% flags a regression. Use as a CI/CD gate: no regressions = safe to ship. Follows the pass@k / pass^k stochastic framework.</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ═══════════════════ OTEL OBSERVABILITY TAB ═══════════════════ */}
      {activeTab === "otel" && (
        <div className="space-y-4">
          {/* OTel KPIs */}
          <div className="grid grid-cols-5 gap-3">
            <Metric label="Total Spans" value={String(otelStats?.total_spans || 0)} sub="Across all traces" />
            <Metric label="Total Traces" value={String(otelStats?.total_traces || 0)} />
            <Metric label="Error Spans" value={String(otelStats?.error_spans || 0)}
              accent={otelStats?.error_spans > 0 ? "text-red-600" : "text-emerald-600"} />
            <Metric label="Span Types" value={String(spanTypeData.length)} sub="routing, tool, llm, etc." />
            <Metric label="Models Used" value={String(modelData.length)} sub={modelData.map(m => m.model).join(", ").slice(0, 30)} />
          </div>

          {/* Span Type Breakdown + Model Breakdown */}
          <div className="grid grid-cols-2 gap-4">
            <div className="card">
              <h3 className="text-xs text-[var(--text-muted)] mb-2">Span Type Distribution (GenAI Semantic Conventions)</h3>
              {spanTypeData.length > 0 ? (
                <div className="flex items-center gap-4">
                  <ResponsiveContainer width="50%" height={200}>
                    <PieChart>
                      <Pie data={spanTypeData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                        outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        labelLine={false} fontSize={9}>
                        {spanTypeData.map((entry, i) => (
                          <Cell key={i} fill={entry.fill} />
                        ))}
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
                        <div className="flex gap-3 text-[var(--text-muted)]">
                          <span>{s.value} spans</span>
                          <span>{s.tokens} tok</span>
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
              <h3 className="text-xs text-[var(--text-muted)] mb-2">Model Usage (gen_ai.request.model)</h3>
              {modelData.length > 0 ? (
                <div className="space-y-2">
                  <ResponsiveContainer width="100%" height={140}>
                    <BarChart data={modelData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" stroke="var(--text-muted)" fontSize={10} />
                      <YAxis type="category" dataKey="model" stroke="var(--text-muted)" fontSize={9} width={100} />
                      <Tooltip {...tip} />
                      <Bar dataKey="count" name="Spans" fill="#6366f1" radius={[0, 3, 3, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                  <div className="space-y-1">
                    {modelData.map(m => (
                      <div key={m.model} className="flex items-center justify-between text-xs border-b border-[var(--border)] py-1">
                        <span className="font-mono">{m.model}</span>
                        <div className="flex gap-3 text-[var(--text-muted)]">
                          <span>In: {m.tokens_in}</span>
                          <span>Out: {m.tokens_out}</span>
                          <span>${m.cost.toFixed(4)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : <div className="text-sm text-[var(--text-muted)] text-center py-10">No model data</div>}
            </div>
          </div>

          {/* Token Flow + Cost Over Time */}
          {tokenFlow.length > 0 && (
            <div className="grid grid-cols-2 gap-4">
              <div className="card">
                <h3 className="text-xs text-[var(--text-muted)] mb-2">Token Flow (gen_ai.usage.*)</h3>
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
                <h3 className="text-xs text-[var(--text-muted)] mb-2">Cost per Request (DBSpanProcessor Cost Estimation)</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={tokenFlow}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="id" stroke="var(--text-muted)" fontSize={10} />
                    <YAxis stroke="var(--text-muted)" fontSize={10} />
                    <Tooltip {...tip} formatter={(value: number) => `$${value.toFixed(4)}`} />
                    <Bar dataKey="cost" name="Cost ($)" fill="#059669" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Recent Traces with Span Details */}
          <div className="card">
            <h3 className="text-xs text-[var(--text-muted)] mb-2">Recent Traces (with OTel Span Breakdown)</h3>
            <div className="space-y-1 max-h-[350px] overflow-y-auto">
              {traces.slice(0, 15).map((t: any) => {
                const spanCounts: Record<string, number> = {};
                for (const s of t.spans || []) {
                  spanCounts[s.span_type] = (spanCounts[s.span_type] || 0) + 1;
                }
                return (
                  <div key={t.id} className="flex items-center justify-between text-xs py-2 border-b border-[var(--border)]">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${t.status === "completed" ? "bg-emerald-500" : "bg-red-500"}`} />
                      <span className="truncate">{t.user_prompt}</span>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0 ml-2 text-[var(--text-muted)]">
                      {Object.entries(spanCounts).map(([type, count]) => (
                        <span key={type} className="px-1.5 py-0.5 rounded text-[9px] border"
                          style={{ borderColor: SPAN_TYPE_COLORS[type] || "#6b7280", color: SPAN_TYPE_COLORS[type] || "#6b7280" }}>
                          {type.replace(/_/g, " ")} ×{count}
                        </span>
                      ))}
                      <span className="ml-1">{(t.total_latency_ms || 0).toFixed(0)}ms</span>
                      <span>{t.total_tokens || 0} tok</span>
                      <span>${(t.total_cost || 0).toFixed(4)}</span>
                    </div>
                  </div>
                );
              })}
              {traces.length === 0 && <div className="text-[var(--text-muted)] text-center py-6">Send chat messages to generate traces</div>}
            </div>
          </div>

          {/* OTel Architecture Info */}
          <div className="card">
            <h3 className="text-xs text-[var(--text-muted)] mb-2">OpenTelemetry Architecture</h3>
            <div className="grid grid-cols-4 gap-2 text-xs text-[var(--text-muted)]">
              <div className="p-2.5 rounded bg-indigo-50 border border-indigo-100">
                <div className="font-medium text-indigo-700 mb-1">GenAI Semantic Conventions</div>
                <div>Spans follow <code className="text-[10px]">gen_ai.*</code> namespace: <code className="text-[10px]">gen_ai.request.model</code>, <code className="text-[10px]">gen_ai.usage.input_tokens</code>, <code className="text-[10px]">gen_ai.operation.name</code></div>
              </div>
              <div className="p-2.5 rounded bg-indigo-50 border border-indigo-100">
                <div className="font-medium text-indigo-700 mb-1">OpenInference Auto-Instrumentation</div>
                <div>LangChainInstrumentor automatically captures all LLM calls, tool invocations, and chain executions as OTel spans.</div>
              </div>
              <div className="p-2.5 rounded bg-indigo-50 border border-indigo-100">
                <div className="font-medium text-indigo-700 mb-1">Custom DBSpanProcessor</div>
                <div>Persists every span to SQLite with cost estimation (tokens × model rates). Enriches spans with <code className="text-[10px]">span.type</code> classification.</div>
              </div>
              <div className="p-2.5 rounded bg-indigo-50 border border-indigo-100">
                <div className="font-medium text-indigo-700 mb-1">OTLP Export</div>
                <div>Dual export: local DB + optional OTLP HTTP to Phoenix, Langfuse, Jaeger, or Datadog via <code className="text-[10px]">BatchSpanProcessor</code>.</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
