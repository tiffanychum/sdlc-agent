"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts";

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card">
      <div className="text-[11px] text-[var(--text-muted)] uppercase tracking-wide">{label}</div>
      <div className="text-2xl font-semibold mt-1.5">{value}</div>
      {sub && <div className="text-xs text-[var(--text-muted)] mt-1">{sub}</div>}
    </div>
  );
}

export default function MonitoringPage() {
  const [stats, setStats] = useState<any>(null);
  useEffect(() => { api.traces.stats(30).then(setStats); }, []);

  if (!stats) return <div className="text-[var(--text-muted)] py-20 text-center">Loading...</div>;

  const daily = Object.entries(stats.daily || {})
    .map(([d, v]: [string, any]) => ({ date: d.slice(5), ...v }))
    .sort((a, b) => a.date.localeCompare(b.date));

  const tip = { contentStyle: { background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 } };

  return (
    <div className="space-y-6 max-w-5xl">
      <h1 className="text-xl font-semibold">Monitoring</h1>

      <div className="grid grid-cols-4 gap-4">
        <Metric label="Total Runs" value={String(stats.total_runs)} sub={`${stats.failures} failures`} />
        <Metric label="Success Rate" value={`${(stats.success_rate * 100).toFixed(1)}%`} />
        <Metric label="Avg Latency" value={`${stats.avg_latency_ms.toFixed(0)}ms`} sub={`P95: ${stats.p95_latency_ms.toFixed(0)}ms`} />
        <Metric label="Total Cost" value={`$${stats.total_cost.toFixed(4)}`} sub={`${stats.total_tokens.toLocaleString()} tokens`} />
      </div>

      <div className="card">
        <h2 className="text-sm font-medium mb-4">Latency Percentiles</h2>
        <table className="w-full text-sm">
          <thead><tr className="text-[var(--text-muted)] text-xs">
            <th className="text-left py-2">Metric</th><th className="text-right">Avg</th><th className="text-right">P50</th><th className="text-right">P95</th><th className="text-right">P99</th>
          </tr></thead>
          <tbody>
            <tr className="border-t border-[var(--border)]">
              <td className="py-2">Latency</td>
              <td className="text-right">{stats.avg_latency_ms.toFixed(0)}ms</td>
              <td className="text-right">{stats.p50_latency_ms.toFixed(0)}ms</td>
              <td className="text-right">{stats.p95_latency_ms.toFixed(0)}ms</td>
              <td className="text-right">{stats.p99_latency_ms.toFixed(0)}ms</td>
            </tr>
            <tr className="border-t border-[var(--border)]">
              <td className="py-2">Tokens</td>
              <td className="text-right">{stats.avg_tokens}</td>
              <td className="text-right" colSpan={3}>-</td>
            </tr>
          </tbody>
        </table>
      </div>

      {daily.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          {[
            { title: "Runs", key: "runs", color: "#3b82f6", type: "bar" },
            { title: "Avg Latency (ms)", key: "avg_latency", color: "#22c55e", type: "line" },
            { title: "Cost ($)", key: "cost", color: "#eab308", type: "line" },
            { title: "Tokens", key: "tokens", color: "#8b5cf6", type: "bar" },
          ].map(chart => (
            <div key={chart.key} className="card">
              <h3 className="text-xs text-[var(--text-muted)] mb-3">{chart.title}</h3>
              <ResponsiveContainer width="100%" height={160}>
                {chart.type === "bar" ? (
                  <BarChart data={daily}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={10} />
                    <YAxis stroke="var(--text-muted)" fontSize={10} />
                    <Tooltip {...tip} />
                    <Bar dataKey={chart.key} fill={chart.color} radius={[3, 3, 0, 0]} />
                  </BarChart>
                ) : (
                  <LineChart data={daily}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={10} />
                    <YAxis stroke="var(--text-muted)" fontSize={10} />
                    <Tooltip {...tip} />
                    <Line type="monotone" dataKey={chart.key} stroke={chart.color} strokeWidth={2} dot={false} />
                  </LineChart>
                )}
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
