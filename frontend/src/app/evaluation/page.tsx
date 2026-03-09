"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from "recharts";

const METRICS = [
  { key: "task_success_rate", label: "Success", color: "#16a34a" },
  { key: "tool_accuracy", label: "Tool Acc.", color: "#2563eb" },
  { key: "reasoning_quality", label: "Reasoning", color: "#7c3aed" },
  { key: "step_efficiency", label: "Efficiency", color: "#ca8a04" },
  { key: "faithfulness", label: "Faithful", color: "#0891b2" },
  { key: "safety_compliance", label: "Safety", color: "#dc2626" },
  { key: "routing_accuracy", label: "Routing", color: "#ea580c" },
];

export default function EvaluationPage() {
  const [runs, setRuns] = useState<any[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [comparison, setComparison] = useState<any>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState("");

  useEffect(() => { api.eval.runs().then(setRuns); }, []);

  async function runEval() {
    setRunning(true);
    setProgress("Running evaluation (5 scenarios)...");
    try {
      const result = await api.eval.run("default");
      setProgress(`Done! Success: ${((result.task_success_rate || 0) * 100).toFixed(0)}%`);
      setRuns(await api.eval.runs());
    } catch (e: any) {
      setProgress(`Error: ${e.message}`);
    } finally {
      setRunning(false);
      setTimeout(() => setProgress(""), 5000);
    }
  }

  function toggle(id: string) {
    setSelected(p => p.includes(id) ? p.filter(x => x !== id) : p.length < 2 ? [...p, id] : [p[1], id]);
    setComparison(null);
  }

  async function compare() {
    if (selected.length !== 2) return;
    setComparison(await api.eval.compareRuns(selected[0], selected[1]));
  }

  const latest = runs[0];
  const radarData = latest ? METRICS.map(m => ({
    metric: m.label,
    score: +((latest[m.key] || 0) * 100).toFixed(1),
  })) : [];

  const barData = runs.slice(0, 6).reverse().map(r => {
    const d: any = { label: `${(r.model || "").slice(0, 8)}(${r.id.slice(0, 4)})` };
    METRICS.slice(0, 4).forEach(m => { d[m.key] = +((r[m.key] || 0) * 100).toFixed(1); });
    return d;
  });

  const tip = { contentStyle: { background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 } };

  return (
    <div className="space-y-5 max-w-6xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Evaluation</h1>
          <p className="text-[13px] text-[var(--text-muted)]">7 metrics: success, tool accuracy, reasoning, efficiency, faithfulness, safety, routing</p>
        </div>
        <div className="flex items-center gap-3">
          {progress && <span className={`text-xs ${running ? "text-[var(--accent)]" : progress.startsWith("Error") ? "text-[var(--error)]" : "text-[var(--success)]"}`}>{progress}</span>}
          {selected.length === 2 && <button onClick={compare} className="btn-secondary">Compare</button>}
          <button onClick={runEval} disabled={running} className="btn-primary">
            {running ? (
              <span className="flex items-center gap-1.5">
                <span className="h-3 w-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Running...
              </span>
            ) : "Run Evaluation"}
          </button>
        </div>
      </div>

      {/* Latest Run Summary */}
      {latest && (
        <div className="grid grid-cols-7 gap-2">
          {METRICS.map(m => {
            const val = latest[m.key] || 0;
            return (
              <div key={m.key} className="card !p-3 text-center">
                <div className="text-lg font-semibold" style={{ color: m.color }}>{(val * 100).toFixed(0)}%</div>
                <div className="text-[10px] text-[var(--text-muted)] mt-0.5">{m.label}</div>
              </div>
            );
          })}
        </div>
      )}

      {/* Charts */}
      {runs.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          <div className="card">
            <h3 className="text-xs text-[var(--text-muted)] mb-2">Latest Run — Metric Radar</h3>
            <ResponsiveContainer width="100%" height={250}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="var(--border)" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: "var(--text-muted)" }} />
                <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 9 }} />
                <Radar name="Score" dataKey="score" stroke="#2563eb" fill="#2563eb" fillOpacity={0.15} strokeWidth={2} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <h3 className="text-xs text-[var(--text-muted)] mb-2">Across Runs</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="label" stroke="var(--text-muted)" fontSize={9} />
                <YAxis stroke="var(--text-muted)" fontSize={10} domain={[0, 100]} />
                <Tooltip {...tip} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                {METRICS.slice(0, 4).map(m => (
                  <Bar key={m.key} dataKey={m.key} name={m.label} fill={m.color} radius={[2, 2, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Regression */}
      {comparison && (
        <div className="card">
          <h3 className="text-sm font-medium mb-2">Regression: {comparison.run_a} vs {comparison.run_b}</h3>
          <table className="w-full text-sm">
            <thead><tr className="text-[var(--text-muted)] text-xs">
              <th className="text-left py-2">Metric</th><th className="text-right">Before</th><th className="text-right">After</th><th className="text-right">Delta</th><th className="text-right">Status</th>
            </tr></thead>
            <tbody>
              {Object.entries(comparison.comparison).map(([k, d]: [string, any]) => (
                <tr key={k} className="border-t border-[var(--border)]">
                  <td className="py-1.5">{k.replace(/_/g, " ")}</td>
                  <td className="text-right">{(d.before * 100).toFixed(1)}%</td>
                  <td className="text-right">{(d.after * 100).toFixed(1)}%</td>
                  <td className={`text-right ${d.delta >= 0 ? "text-[var(--success)]" : "text-[var(--error)]"}`}>{d.delta >= 0 ? "+" : ""}{(d.delta * 100).toFixed(1)}%</td>
                  <td className="text-right"><span className={`badge ${d.regression ? "bg-[var(--error-light)] text-[var(--error)]" : "bg-[var(--success-light)] text-[var(--success)]"}`}>{d.regression ? "REGR" : "OK"}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Runs Table */}
      <div className="card !p-0 overflow-hidden">
        <table className="w-full text-sm">
          <thead><tr className="text-[var(--text-muted)] text-xs bg-[var(--bg)]">
            <th className="w-8 px-3 py-2.5"></th><th className="text-left px-3">ID</th><th className="text-left px-3">Model</th>
            {METRICS.map(m => <th key={m.key} className="text-right px-2 py-2.5">{m.label}</th>)}
            <th className="text-right px-3 py-2.5">Latency</th>
          </tr></thead>
          <tbody>
            {runs.map(r => (
                <tr key={r.id} className={`border-t border-[var(--border)] hover:bg-[var(--bg-hover)] ${selected.includes(r.id) ? "bg-[var(--accent-light)]" : ""}`}>
                  <td className="px-3 py-2"><input type="checkbox" checked={selected.includes(r.id)} onChange={() => toggle(r.id)} /></td>
                  <td className="px-3 py-2 font-mono text-[11px]">{r.id}</td>
                  <td className="px-3 py-2 text-xs">{r.model}</td>
                  {METRICS.map(m => {
                    const val = r[m.key] ?? 0;
                    return <td key={m.key} className="text-right px-2 py-2 text-xs">{(val * 100).toFixed(0)}%</td>;
                  })}
                  <td className="text-right px-3 py-2 text-xs">{r.avg_latency_ms ? `${r.avg_latency_ms.toFixed(0)}ms` : "—"}</td>
                </tr>
            ))}
          </tbody>
        </table>
        {runs.length === 0 && <div className="text-center py-10 text-[var(--text-muted)]">No runs yet. Click &quot;Run Evaluation&quot;.</div>}
      </div>
    </div>
  );
}
