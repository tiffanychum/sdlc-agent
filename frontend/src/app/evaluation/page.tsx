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
  const [detail, setDetail] = useState<any>(null);

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

  async function viewDetail(id: string) {
    const d = await api.eval.get(id);
    setDetail(d);
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

  // ── Detail View ──
  if (detail) {
    return (
      <div className="space-y-5 max-w-6xl">
        <div className="flex items-center gap-3">
          <button onClick={() => setDetail(null)} className="btn-ghost">&larr; Back</button>
          <h1 className="text-xl font-semibold">Run {detail.id}</h1>
          <span className="text-sm text-[var(--text-muted)]">{detail.model}</span>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-7 gap-2">
          {METRICS.map(m => (
            <div key={m.key} className="card !p-3 text-center">
              <div className="text-lg font-semibold" style={{ color: m.color }}>{((detail[m.key] || 0) * 100).toFixed(0)}%</div>
              <div className="text-[10px] text-[var(--text-muted)]">{m.label}</div>
            </div>
          ))}
        </div>

        {/* Per-Request Details */}
        <div className="space-y-3">
          <h2 className="text-sm font-medium">Requests ({(detail.tasks || []).length})</h2>
          {(detail.tasks || []).map((t: any, i: number) => {
            const scores = t.scores || {};
            return (
              <details key={i} className={`card !p-0 overflow-hidden border-l-4 ${t.completed ? "border-l-green-500" : "border-l-red-500"}`}>
                <summary className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-[var(--bg-hover)]">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <span className={`badge flex-shrink-0 ${t.completed ? "bg-[var(--success-light)] text-[var(--success)]" : "bg-[var(--error-light)] text-[var(--error)]"}`}>
                      {t.completed ? "PASS" : "FAIL"}
                    </span>
                    <span className="text-sm truncate">{t.prompt || t.scenario}</span>
                  </div>
                  <div className="flex gap-2 text-xs text-[var(--text-muted)] flex-shrink-0 ml-3">
                    {t.actual_agent && <span className="badge bg-[var(--accent-light)] text-[var(--accent)]">{t.actual_agent}</span>}
                    {t.latency_ms && <span>{t.latency_ms.toFixed(0)}ms</span>}
                  </div>
                </summary>

                <div className="border-t border-[var(--border)] bg-[var(--bg)]">
                  {/* User Question */}
                  <div className="px-4 py-2.5 border-b border-[var(--border)]">
                    <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1">User Request</div>
                    <div className="text-sm">{t.prompt || "—"}</div>
                  </div>

                  {/* Agent Trace (tool calls) */}
                  {t.tool_calls && t.tool_calls.length > 0 && (
                    <div className="px-4 py-2.5 border-b border-[var(--border)]">
                      <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1.5">Agent Trace</div>
                      <div className="space-y-1">
                        {t.routing_correct !== undefined && (
                          <div className="flex items-center gap-2 text-xs">
                            <span className={t.routing_correct ? "text-[var(--success)]" : "text-[var(--error)]"}>
                              {t.routing_correct ? "✓" : "✗"}
                            </span>
                            <span className="text-[var(--text-muted)]">Route to</span>
                            <span className="font-medium">{t.actual_agent || "?"}</span>
                            {!t.routing_correct && <span className="text-[var(--text-muted)]">(expected: {t.expected_agent})</span>}
                          </div>
                        )}
                        {t.tool_calls.map((tc: any, j: number) => (
                          <div key={j} className="flex items-center gap-2 text-xs">
                            <span className={tc.correct ? "text-[var(--success)]" : "text-[var(--warning)]"}>
                              {tc.correct ? "✓" : "~"}
                            </span>
                            <span className="font-mono">{tc.tool}</span>
                            <span className="text-[var(--text-muted)] truncate">
                              ({Object.entries(tc.args || {}).map(([k,v]) => `${k}="${String(v).slice(0,30)}"`).join(", ")})
                            </span>
                            {tc.error && <span className="text-[var(--error)]">err</span>}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Agent Response */}
                  {t.agent_response && (
                    <div className="px-4 py-2.5 border-b border-[var(--border)]">
                      <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1">Agent Output</div>
                      <div className="text-xs bg-[var(--bg-card)] border border-[var(--border)] rounded p-2 max-h-32 overflow-y-auto whitespace-pre-wrap">
                        {t.agent_response.slice(0, 800)}
                        {t.agent_response.length > 800 && "..."}
                      </div>
                    </div>
                  )}

                  {/* Evaluation Scores */}
                  <div className="px-4 py-2.5">
                    <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1.5">Evaluation Scores</div>
                    <div className="grid grid-cols-6 gap-1.5">
                      {Object.entries(scores).map(([k, v]: [string, any]) => (
                        <div key={k} className="text-center p-1.5 rounded bg-[var(--bg-card)] border border-[var(--border)]">
                          <div className={`text-sm font-semibold ${(v || 0) >= 0.7 ? "text-[var(--success)]" : (v || 0) >= 0.4 ? "text-[var(--warning)]" : "text-[var(--error)]"}`}>
                            {((v || 0) * 100).toFixed(0)}%
                          </div>
                          <div className="text-[9px] text-[var(--text-muted)]">{k.replace(/_/g, " ")}</div>
                        </div>
                      ))}
                    </div>

                    {/* LLM Judge */}
                    {t.llm_judge && Object.keys(t.llm_judge).length > 0 && (
                      <div className="mt-2">
                        <div className="text-[10px] text-[var(--text-muted)] mb-1">LLM Judge</div>
                        <div className="flex gap-1.5 flex-wrap">
                          {Object.entries(t.llm_judge).map(([k, v]: [string, any]) => (
                            <span key={k} className="badge bg-[var(--accent-light)] text-[var(--accent)]">{k}: {((v||0)*100).toFixed(0)}%</span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Trajectory */}
                    {t.trajectory && t.trajectory.reasoning && (
                      <div className="mt-2 text-xs bg-[var(--bg-card)] border border-[var(--border)] rounded p-2">
                        <span className="font-medium">Trajectory: {((t.trajectory.trajectory_score||0)*100).toFixed(0)}%</span>
                        <span className="text-[var(--text-muted)] ml-2">{t.trajectory.reasoning}</span>
                      </div>
                    )}
                  </div>

                  {t.error && (
                    <div className="px-4 py-2 text-xs text-[var(--error)] bg-[var(--error-light)] border-t border-[var(--border)]">
                      Error: {t.error}
                    </div>
                  )}
                </div>
              </details>
            );
          })}
          {(!detail.tasks || detail.tasks.length === 0) && (
            <div className="text-[var(--text-muted)] text-center py-6">No details available. Run a new evaluation to see per-request breakdowns.</div>
          )}
        </div>
      </div>
    );
  }

  // ── List View ──
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

      {latest && (
        <div className="grid grid-cols-7 gap-2">
          {METRICS.map(m => (
            <div key={m.key} className="card !p-3 text-center">
              <div className="text-lg font-semibold" style={{ color: m.color }}>{((latest[m.key] || 0) * 100).toFixed(0)}%</div>
              <div className="text-[10px] text-[var(--text-muted)]">{m.label}</div>
            </div>
          ))}
        </div>
      )}

      {runs.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          <div className="card">
            <h3 className="text-xs text-[var(--text-muted)] mb-2">Latest Run — Radar</h3>
            <ResponsiveContainer width="100%" height={250}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="var(--border)" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: "var(--text-muted)" }} />
                <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 9 }} />
                <Radar dataKey="score" stroke="#2563eb" fill="#2563eb" fillOpacity={0.15} strokeWidth={2} />
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
            <th className="text-right px-3 py-2.5">Latency</th><th className="px-3 py-2.5"></th>
          </tr></thead>
          <tbody>
            {runs.map(r => (
              <tr key={r.id} className={`border-t border-[var(--border)] hover:bg-[var(--bg-hover)] ${selected.includes(r.id) ? "bg-[var(--accent-light)]" : ""}`}>
                <td className="px-3 py-2"><input type="checkbox" checked={selected.includes(r.id)} onChange={() => toggle(r.id)} /></td>
                <td className="px-3 py-2 font-mono text-[11px]">{r.id}</td>
                <td className="px-3 py-2 text-xs">{r.model}</td>
                {METRICS.map(m => (
                  <td key={m.key} className="text-right px-2 py-2 text-xs">{((r[m.key] ?? 0) * 100).toFixed(0)}%</td>
                ))}
                <td className="text-right px-3 py-2 text-xs">{r.avg_latency_ms ? `${r.avg_latency_ms.toFixed(0)}ms` : "—"}</td>
                <td className="px-3 py-2"><button onClick={() => viewDetail(r.id)} className="text-[11px] text-[var(--accent)] hover:underline">Details</button></td>
              </tr>
            ))}
          </tbody>
        </table>
        {runs.length === 0 && <div className="text-center py-10 text-[var(--text-muted)]">No runs yet. Click &quot;Run Evaluation&quot;.</div>}
      </div>
    </div>
  );
}
