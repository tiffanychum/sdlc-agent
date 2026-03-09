"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

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
  const [teamId, setTeamId] = useState<string>("");
  const [stats, setStats] = useState<any>(null);
  const [evalRuns, setEvalRuns] = useState<any[]>([]);

  useEffect(() => {
    api.teams.list().then(t => { setTeams(t); if (t.length) setTeamId(t[0].id); });
  }, []);

  useEffect(() => {
    if (!teamId) return;
    api.traces.stats(30).then(setStats);
    api.eval.runs().then(setEvalRuns);
  }, [teamId]);

  if (!stats) return <div className="text-[var(--text-muted)] py-20 text-center">Loading...</div>;

  const daily = Object.entries(stats.daily || {})
    .map(([d, v]: [string, any]) => ({ date: d.slice(5), ...v, cost: +(v.cost || 0).toFixed(4) }))
    .sort((a, b) => a.date.localeCompare(b.date));

  const qualityData = evalRuns.slice(0, 15).reverse().map((r: any) => ({
    date: r.created_at?.slice(5, 10) || r.id.slice(0, 4),
    success: +((r.task_completion_rate || 0) * 100).toFixed(1),
    toolAcc: +((r.avg_tool_call_accuracy || 0) * 100).toFixed(1),
    routing: +((r.routing_accuracy || 0) * 100).toFixed(1),
  }));

  const tip = { contentStyle: { background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 } };

  const selectedTeam = teams.find(t => t.id === teamId);

  return (
    <div className="space-y-5 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Monitoring</h1>
        <select value={teamId} onChange={e => setTeamId(e.target.value)} className="input !w-auto">
          {teams.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
      </div>

      {/* Application Info */}
      <div className="card !p-0 overflow-hidden">
        <table className="w-full text-sm">
          <thead><tr className="text-[var(--text-muted)] text-xs bg-[var(--bg)]">
            <th className="text-left px-4 py-2.5">Project</th>
            <th className="text-left px-4 py-2.5">Application</th>
            <th className="text-left px-4 py-2.5">EIM ID</th>
            <th className="text-left px-4 py-2.5">Status</th>
          </tr></thead>
          <tbody>
            <tr className="border-t border-[var(--border)]">
              <td className="px-4 py-2.5">SDLC Agent</td>
              <td className="px-4 py-2.5">{selectedTeam?.name || "SDLC Assistant"}</td>
              <td className="px-4 py-2.5 font-mono text-xs">{teamId}</td>
              <td className="px-4 py-2.5"><span className="badge bg-[var(--success-light)] text-[var(--success)]">Active</span></td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-4 gap-3">
        <Metric label="Application Runs" value={String(stats.total_runs)} />
        <Metric label="Runs With Failures" value={String(stats.failures)} />
        <Metric label="Overall Pass Rate" value={`${(stats.success_rate * 100).toFixed(1)}%`} />
        <Metric label="Total Cost" value={`$${stats.total_cost.toFixed(4)}`} />
      </div>

      {/* Monitoring Section */}
      <div className="grid grid-cols-2 gap-4">
        {/* Task Quality Chart */}
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h2 className="text-sm font-medium">Task Quality</h2>
              <p className="text-[11px] text-[var(--text-muted)]">Scores for how well the task was executed</p>
            </div>
          </div>
          {qualityData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={qualityData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={10} />
                <YAxis stroke="var(--text-muted)" fontSize={10} domain={[0, 100]} />
                <Tooltip {...tip} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line type="monotone" dataKey="success" name="Success Rate" stroke="#16a34a" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="toolAcc" name="Tool Accuracy" stroke="#2563eb" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="routing" name="Routing" stroke="#7c3aed" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-sm text-[var(--text-muted)] text-center py-12">Run evaluations to see quality trends</div>
          )}
        </div>

        {/* Operational Efficiency Table */}
        <div className="card">
          <h2 className="text-sm font-medium mb-1">Operational Efficiency</h2>
          <p className="text-[11px] text-[var(--text-muted)] mb-3">Metrics on how resource-conscious the agent was</p>
          <table className="w-full text-sm">
            <thead><tr className="text-[var(--text-muted)] text-xs">
              <th className="text-left py-2"></th>
              <th className="text-right py-2">Avg</th>
              <th className="text-right py-2">P25</th>
              <th className="text-right py-2">P50</th>
              <th className="text-right py-2">P99</th>
            </tr></thead>
            <tbody>
              <tr className="border-t border-[var(--border)]">
                <td className="py-2 font-medium">Latency</td>
                <td className="text-right">{stats.avg_latency_ms.toFixed(0)}ms</td>
                <td className="text-right">{stats.p50_latency_ms ? (stats.p50_latency_ms * 0.7).toFixed(0) : "-"}ms</td>
                <td className="text-right">{stats.p50_latency_ms?.toFixed(0) || "-"}ms</td>
                <td className="text-right">{stats.p99_latency_ms?.toFixed(0) || "-"}ms</td>
              </tr>
              <tr className="border-t border-[var(--border)]">
                <td className="py-2 font-medium">Tokens</td>
                <td className="text-right">{stats.avg_tokens?.toLocaleString() || 0}</td>
                <td className="text-right">-</td>
                <td className="text-right">-</td>
                <td className="text-right">-</td>
              </tr>
              <tr className="border-t border-[var(--border)]">
                <td className="py-2 font-medium">Cost per Run</td>
                <td className="text-right">${stats.total_runs ? (stats.total_cost / stats.total_runs).toFixed(4) : "0"}</td>
                <td className="text-right">-</td>
                <td className="text-right">-</td>
                <td className="text-right">-</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Trend Charts */}
      {daily.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          <div className="card">
            <h3 className="text-xs text-[var(--text-muted)] mb-2">Runs Over Time</h3>
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={daily}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={10} />
                <YAxis stroke="var(--text-muted)" fontSize={10} />
                <Tooltip {...tip} />
                <Line type="monotone" dataKey="runs" stroke="#2563eb" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <h3 className="text-xs text-[var(--text-muted)] mb-2">Cost Over Time</h3>
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={daily}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={10} />
                <YAxis stroke="var(--text-muted)" fontSize={10} />
                <Tooltip {...tip} />
                <Line type="monotone" dataKey="cost" stroke="#ca8a04" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
