"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, BarChart, Bar } from "recharts";

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card text-center">
      <div className="text-2xl font-semibold">{value}</div>
      <div className="text-xs text-[var(--text-muted)] mt-1">{label}</div>
      {sub && <div className="text-[11px] text-[var(--text-muted)] mt-0.5">{sub}</div>}
    </div>
  );
}

export default function MonitoringPage() {
  const [teams, setTeams] = useState<any[]>([]);
  const [teamId, setTeamId] = useState("");
  const [stats, setStats] = useState<any>(null);
  const [evalRuns, setEvalRuns] = useState<any[]>([]);
  const [traces, setTraces] = useState<any[]>([]);

  useEffect(() => {
    api.teams.list().then(t => { setTeams(t); if (t.length) setTeamId(t[0].id); });
  }, []);

  useEffect(() => {
    if (!teamId) return;
    Promise.all([
      api.traces.stats(30),
      api.eval.runs(),
      api.traces.list(50),
    ]).then(([s, e, t]) => { setStats(s); setEvalRuns(e); setTraces(t); });
  }, [teamId]);

  const selectedTeam = teams.find(t => t.id === teamId);
  const totalRuns = stats?.total_runs || traces.length || 0;
  const failures = stats?.failures || traces.filter((t: any) => t.status === "error").length;
  const successRate = totalRuns > 0 ? ((totalRuns - failures) / totalRuns) : 0;
  const avgLatency = stats?.avg_latency_ms || (traces.length > 0 ? traces.reduce((s: number, t: any) => s + (t.total_latency_ms || 0), 0) / traces.length : 0);
  const totalCost = stats?.total_cost || traces.reduce((s: number, t: any) => s + (t.total_cost || 0), 0);

  const qualityData = evalRuns.slice(0, 12).reverse().map((r: any) => ({
    run: r.id?.slice(0, 6) || "?",
    success: +((r.task_success_rate || 0) * 100).toFixed(1),
    toolAcc: +((r.tool_accuracy || 0) * 100).toFixed(1),
    routing: +((r.routing_accuracy || 0) * 100).toFixed(1),
    reasoning: +((r.reasoning_quality || 0) * 100).toFixed(1),
    safety: +((r.safety_compliance || 0) * 100).toFixed(1),
    faithfulness: +((r.faithfulness || 0) * 100).toFixed(1),
  }));

  const latencyData = traces.slice(0, 20).reverse().map((t: any, i: number) => ({
    idx: i + 1,
    latency: +(t.total_latency_ms || 0).toFixed(0),
    tokens: t.total_tokens || 0,
  }));

  const tip = { contentStyle: { background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 } };

  return (
    <div className="space-y-5 max-w-6xl">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Monitoring</h1>
        <select value={teamId} onChange={e => setTeamId(e.target.value)} className="input !w-auto">
          {teams.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
      </div>

      {/* App Info */}
      <div className="card !p-0 overflow-hidden">
        <table className="w-full text-sm">
          <thead><tr className="text-[var(--text-muted)] text-xs bg-[var(--bg)]">
            <th className="text-left px-4 py-2.5">Project</th>
            <th className="text-left px-4 py-2.5">Application</th>
            <th className="text-left px-4 py-2.5">Status</th>
            <th className="text-right px-4 py-2.5">Last Updated</th>
          </tr></thead>
          <tbody><tr className="border-t border-[var(--border)]">
            <td className="px-4 py-2.5">SDLC Agent</td>
            <td className="px-4 py-2.5">{selectedTeam?.name || "—"}</td>
            <td className="px-4 py-2.5"><span className="badge bg-[var(--success-light)] text-[var(--success)]">Active</span></td>
            <td className="text-right px-4 py-2.5 text-[var(--text-muted)] text-xs">{new Date().toLocaleDateString()}</td>
          </tr></tbody>
        </table>
      </div>

      {/* Overview */}
      <div className="grid grid-cols-4 gap-3">
        <Metric label="Application Runs" value={String(totalRuns)} />
        <Metric label="Runs With Failures" value={String(failures)} />
        <Metric label="Overall Pass Rate" value={`${(successRate * 100).toFixed(1)}%`} />
        <Metric label="Total Cost" value={`$${totalCost.toFixed(4)}`} />
      </div>

      {/* Task Quality + Efficiency */}
      <div className="grid grid-cols-2 gap-4">
        <div className="card">
          <h2 className="text-sm font-medium mb-1">Task Quality</h2>
          <p className="text-[11px] text-[var(--text-muted)] mb-3">Scores from evaluation runs</p>
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
          ) : (
            <div className="text-sm text-[var(--text-muted)] text-center py-10">Run evaluations to see quality trends</div>
          )}
        </div>

        <div className="card">
          <h2 className="text-sm font-medium mb-1">Operational Efficiency</h2>
          <p className="text-[11px] text-[var(--text-muted)] mb-3">Resource usage metrics</p>
          <table className="w-full text-sm">
            <thead><tr className="text-[var(--text-muted)] text-xs">
              <th className="text-left py-2"></th><th className="text-right">Avg</th><th className="text-right">P50</th><th className="text-right">P95</th><th className="text-right">P99</th>
            </tr></thead>
            <tbody>
              <tr className="border-t border-[var(--border)]">
                <td className="py-2 font-medium">Latency</td>
                <td className="text-right">{avgLatency.toFixed(0)}ms</td>
                <td className="text-right">{(stats?.p50_latency_ms || avgLatency * 0.9).toFixed(0)}ms</td>
                <td className="text-right">{(stats?.p95_latency_ms || avgLatency * 1.5).toFixed(0)}ms</td>
                <td className="text-right">{(stats?.p99_latency_ms || avgLatency * 2).toFixed(0)}ms</td>
              </tr>
              <tr className="border-t border-[var(--border)]">
                <td className="py-2 font-medium">Tokens</td>
                <td className="text-right">{stats?.avg_tokens || 0}</td>
                <td className="text-right">—</td>
                <td className="text-right">—</td>
                <td className="text-right">—</td>
              </tr>
              <tr className="border-t border-[var(--border)]">
                <td className="py-2 font-medium">Cost/Run</td>
                <td className="text-right">${totalRuns > 0 ? (totalCost / totalRuns).toFixed(4) : "0.00"}</td>
                <td className="text-right">—</td>
                <td className="text-right">—</td>
                <td className="text-right">—</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Latency + Recent Traces */}
      {latencyData.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          <div className="card">
            <h3 className="text-xs text-[var(--text-muted)] mb-2">Latency Per Request (ms)</h3>
            <ResponsiveContainer width="100%" height={150}>
              <BarChart data={latencyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="idx" stroke="var(--text-muted)" fontSize={10} />
                <YAxis stroke="var(--text-muted)" fontSize={10} />
                <Tooltip {...tip} />
                <Bar dataKey="latency" fill="#2563eb" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <h3 className="text-xs text-[var(--text-muted)] mb-2">Recent Traces</h3>
            <div className="space-y-1 max-h-[150px] overflow-y-auto">
              {traces.slice(0, 10).map((t: any) => (
                <div key={t.id} className="flex items-center justify-between text-xs py-1 border-b border-[var(--border)]">
                  <div className="truncate flex-1 mr-2">{t.user_prompt}</div>
                  <div className="flex gap-2 text-[var(--text-muted)] flex-shrink-0">
                    <span>{(t.total_latency_ms || 0).toFixed(0)}ms</span>
                    <span className={`badge ${t.status === "completed" ? "bg-[var(--success-light)] text-[var(--success)]" : "bg-[var(--error-light)] text-[var(--error)]"}`}>{t.status}</span>
                  </div>
                </div>
              ))}
              {traces.length === 0 && <div className="text-[var(--text-muted)] text-center py-4">Send chat messages to see traces</div>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
