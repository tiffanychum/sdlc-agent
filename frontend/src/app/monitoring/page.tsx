"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts";

interface Stats {
  total_runs: number; failures: number; success_rate: number;
  avg_latency_ms: number; p50_latency_ms: number; p95_latency_ms: number; p99_latency_ms: number;
  total_cost: number; total_tokens: number; avg_tokens: number;
  daily: Record<string, { runs: number; cost: number; tokens: number; avg_latency: number }>;
}

function MetricCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
      <div className="text-2xl font-bold mt-1">{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-1">{sub}</div>}
    </div>
  );
}

export default function MonitoringPage() {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => { api.traces.stats(30).then(setStats); }, []);

  if (!stats) return <div className="text-gray-500 py-20 text-center">Loading monitoring data...</div>;

  const dailyData = Object.entries(stats.daily)
    .map(([date, d]) => ({ date: date.slice(5), ...d }))
    .sort((a, b) => a.date.localeCompare(b.date));

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Monitoring</h1>

      {/* Overview Cards */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Total Runs" value={String(stats.total_runs)} sub={`${stats.failures} failures`} />
        <MetricCard label="Success Rate" value={`${(stats.success_rate * 100).toFixed(1)}%`} />
        <MetricCard label="Avg Latency" value={`${stats.avg_latency_ms.toFixed(0)}ms`} sub={`P95: ${stats.p95_latency_ms.toFixed(0)}ms`} />
        <MetricCard label="Total Cost" value={`$${stats.total_cost.toFixed(4)}`} sub={`${stats.total_tokens.toLocaleString()} tokens`} />
      </div>

      {/* Latency Distribution */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
        <h2 className="font-semibold mb-1">Operational Efficiency</h2>
        <table className="w-full text-sm mt-3">
          <thead>
            <tr className="text-gray-500 text-xs uppercase">
              <th className="text-left py-2">Metric</th>
              <th className="text-right py-2">Avg</th>
              <th className="text-right py-2">P50</th>
              <th className="text-right py-2">P95</th>
              <th className="text-right py-2">P99</th>
            </tr>
          </thead>
          <tbody className="text-gray-300">
            <tr className="border-t border-gray-800">
              <td className="py-2">Latency</td>
              <td className="text-right">{stats.avg_latency_ms.toFixed(0)}ms</td>
              <td className="text-right">{stats.p50_latency_ms.toFixed(0)}ms</td>
              <td className="text-right">{stats.p95_latency_ms.toFixed(0)}ms</td>
              <td className="text-right">{stats.p99_latency_ms.toFixed(0)}ms</td>
            </tr>
            <tr className="border-t border-gray-800">
              <td className="py-2">Tokens</td>
              <td className="text-right">{stats.avg_tokens}</td>
              <td className="text-right">-</td>
              <td className="text-right">-</td>
              <td className="text-right">-</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Charts */}
      {dailyData.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold mb-3 text-gray-400">Runs Over Time</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={dailyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#6B7280" fontSize={11} />
                <YAxis stroke="#6B7280" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: "#1F2937", border: "1px solid #374151", borderRadius: 8 }} />
                <Bar dataKey="runs" fill="#3B82F6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold mb-3 text-gray-400">Avg Latency Over Time</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={dailyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#6B7280" fontSize={11} />
                <YAxis stroke="#6B7280" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: "#1F2937", border: "1px solid #374151", borderRadius: 8 }} />
                <Line type="monotone" dataKey="avg_latency" stroke="#10B981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold mb-3 text-gray-400">Cost Over Time</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={dailyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#6B7280" fontSize={11} />
                <YAxis stroke="#6B7280" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: "#1F2937", border: "1px solid #374151", borderRadius: 8 }} />
                <Line type="monotone" dataKey="cost" stroke="#F59E0B" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold mb-3 text-gray-400">Token Usage Over Time</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={dailyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#6B7280" fontSize={11} />
                <YAxis stroke="#6B7280" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: "#1F2937", border: "1px solid #374151", borderRadius: 8 }} />
                <Bar dataKey="tokens" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
