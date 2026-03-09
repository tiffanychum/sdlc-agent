"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

export default function EvaluationPage() {
  const [runs, setRuns] = useState<any[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [comparison, setComparison] = useState<any>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => { api.eval.runs().then(setRuns); }, []);

  async function runEval() {
    setRunning(true);
    try { await api.eval.run("default"); setRuns(await api.eval.runs()); }
    finally { setRunning(false); }
  }

  function toggle(id: string) {
    setSelected(p => p.includes(id) ? p.filter(x => x !== id) : p.length < 2 ? [...p, id] : [p[1], id]);
    setComparison(null);
  }

  async function compare() {
    if (selected.length !== 2) return;
    setComparison(await api.eval.compareRuns(selected[0], selected[1]));
  }

  const chart = runs.slice(0, 8).reverse().map(r => ({
    label: `${r.model.slice(0, 10)} (${r.id.slice(0, 4)})`,
    completion: +(r.task_completion_rate * 100).toFixed(1),
    routing: +(r.routing_accuracy * 100).toFixed(1),
    tool_acc: +(r.avg_tool_call_accuracy * 100).toFixed(1),
  }));

  const tip = { contentStyle: { background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 } };

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Evaluation</h1>
        <div className="flex gap-2">
          {selected.length === 2 && <button onClick={compare} className="btn-ghost border border-[var(--warning)]/30 text-[var(--warning)]">Compare ({selected.length})</button>}
          <button onClick={runEval} disabled={running} className="btn-primary">{running ? "Running..." : "Run Evaluation"}</button>
        </div>
      </div>

      {chart.length > 0 && (
        <div className="card">
          <h3 className="text-xs text-[var(--text-muted)] mb-3">Performance Across Runs</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chart}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="label" stroke="var(--text-muted)" fontSize={10} />
              <YAxis stroke="var(--text-muted)" fontSize={10} domain={[0, 100]} />
              <Tooltip {...tip} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="completion" name="Completion" fill="#22c55e" radius={[2, 2, 0, 0]} />
              <Bar dataKey="routing" name="Routing" fill="#3b82f6" radius={[2, 2, 0, 0]} />
              <Bar dataKey="tool_acc" name="Tool Acc." fill="#8b5cf6" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {comparison && (
        <div className="card">
          <h3 className="text-sm font-medium mb-3">Regression: {comparison.run_a} vs {comparison.run_b}</h3>
          <table className="w-full text-sm">
            <thead><tr className="text-[var(--text-muted)] text-xs">
              <th className="text-left py-2">Metric</th><th className="text-right">Before</th><th className="text-right">After</th><th className="text-right">Delta</th><th className="text-right">Status</th>
            </tr></thead>
            <tbody>
              {Object.entries(comparison.comparison).map(([k, d]: [string, any]) => (
                <tr key={k} className="border-t border-[var(--border)]">
                  <td className="py-2">{k.replace(/_/g, " ")}</td>
                  <td className="text-right">{(d.before * 100).toFixed(1)}%</td>
                  <td className="text-right">{(d.after * 100).toFixed(1)}%</td>
                  <td className={`text-right ${d.delta >= 0 ? "text-[var(--success)]" : "text-[var(--error)]"}`}>{d.delta >= 0 ? "+" : ""}{(d.delta * 100).toFixed(1)}%</td>
                  <td className="text-right"><span className={`badge ${d.regression ? "bg-[var(--error)]/10 text-[var(--error)]" : "bg-[var(--success)]/10 text-[var(--success)]"}`}>{d.regression ? "REGRESSION" : "OK"}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="card !p-0 overflow-hidden">
        <table className="w-full text-sm">
          <thead><tr className="text-[var(--text-muted)] text-xs bg-[var(--bg)]">
            <th className="w-10 px-4 py-3"></th><th className="text-left px-4 py-3">ID</th><th className="text-left px-4 py-3">Model</th>
            <th className="text-right px-4 py-3">Tasks</th><th className="text-right px-4 py-3">Completion</th>
            <th className="text-right px-4 py-3">Routing</th><th className="text-right px-4 py-3">Tool Acc.</th>
            <th className="text-right px-4 py-3">Latency</th><th className="text-right px-4 py-3">Date</th>
          </tr></thead>
          <tbody>
            {runs.map(r => (
              <tr key={r.id} className={`border-t border-[var(--border)] hover:bg-[var(--bg-hover)] transition-colors ${selected.includes(r.id) ? "bg-[var(--accent)]/5" : ""}`}>
                <td className="px-4 py-3"><input type="checkbox" checked={selected.includes(r.id)} onChange={() => toggle(r.id)} /></td>
                <td className="px-4 py-3 font-mono text-xs">{r.id}</td>
                <td className="px-4 py-3">{r.model}</td>
                <td className="text-right px-4 py-3">{r.num_tasks}</td>
                <td className="text-right px-4 py-3">{(r.task_completion_rate * 100).toFixed(0)}%</td>
                <td className="text-right px-4 py-3">{(r.routing_accuracy * 100).toFixed(0)}%</td>
                <td className="text-right px-4 py-3">{(r.avg_tool_call_accuracy * 100).toFixed(0)}%</td>
                <td className="text-right px-4 py-3">{r.avg_latency_ms ? `${r.avg_latency_ms.toFixed(0)}ms` : "-"}</td>
                <td className="text-right px-4 py-3 text-[var(--text-muted)] text-xs">{r.created_at?.slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {runs.length === 0 && <div className="text-center py-12 text-[var(--text-muted)]">No runs yet. Click &quot;Run Evaluation&quot;.</div>}
      </div>
    </div>
  );
}
