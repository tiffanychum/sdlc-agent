"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";

const SCORE_KEYS = [
  { key: "tool_accuracy", label: "Tool Acc.", color: "#2563eb" },
  { key: "step_efficiency", label: "Efficiency", color: "#ca8a04" },
  { key: "faithfulness", label: "Faithfulness", color: "#0891b2" },
  { key: "safety", label: "Safety", color: "#dc2626" },
  { key: "reasoning_quality", label: "Reasoning", color: "#7c3aed" },
];

const JUDGE_KEYS = [
  { key: "judge_correctness", label: "Correctness" },
  { key: "judge_relevance", label: "Relevance" },
  { key: "judge_coherence", label: "Coherence" },
  { key: "judge_tool_usage_quality", label: "Tool Quality" },
  { key: "judge_completeness", label: "Completeness" },
];

export default function EvaluationPage() {
  const [traces, setTraces] = useState<any[]>([]);
  const [evaluating, setEvaluating] = useState(false);
  const [evalResult, setEvalResult] = useState("");

  useEffect(() => { load(); }, []);

  async function load() {
    const t = await api.traces.list(100);
    setTraces(t.filter((tr: any) => tr.agent_response));
  }

  async function runEval() {
    setEvaluating(true);
    setEvalResult("Evaluating traces with LLM-Judge...");
    try {
      const r = await api.traces.evaluate();
      setEvalResult(`Evaluated ${r.evaluated} trace(s). ${r.remaining} remaining.`);
      await load();
    } catch (e: any) {
      setEvalResult(`Error: ${e.message}`);
    } finally {
      setEvaluating(false);
      setTimeout(() => setEvalResult(""), 8000);
    }
  }

  const pendingCount = traces.filter(t => t.eval_status !== "evaluated").length;

  const avgScores = SCORE_KEYS.map(sk => {
    const vals = traces.map(t => t.eval_scores?.[sk.key] || 0).filter(v => v > 0);
    return { metric: sk.label, score: vals.length > 0 ? +(vals.reduce((a, b) => a + b, 0) / vals.length * 100).toFixed(1) : 0 };
  });

  const judgeAvgs = JUDGE_KEYS.map(jk => {
    const vals = traces.map(t => t.eval_scores?.[jk.key] || 0).filter(v => v > 0);
    return { metric: jk.label, score: vals.length > 0 ? +(vals.reduce((a, b) => a + b, 0) / vals.length * 100).toFixed(1) : 0 };
  });

  const hasJudge = judgeAvgs.some(j => j.score > 0);

  const barData = traces.slice(0, 10).reverse().map(t => ({
    label: (t.user_prompt || "").slice(0, 15) + "...",
    ...Object.fromEntries(SCORE_KEYS.map(sk => [sk.key, +((t.eval_scores?.[sk.key] || 0) * 100).toFixed(1)])),
  }));

  const tip = { contentStyle: { background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 } };

  return (
    <div className="space-y-5 max-w-6xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Evaluation</h1>
          <p className="text-[13px] text-[var(--text-muted)]">
            {traces.length} requests tracked. {pendingCount > 0 ? `${pendingCount} pending full evaluation.` : "All evaluated."}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {evalResult && <span className={`text-xs ${evaluating ? "text-[var(--accent)]" : evalResult.startsWith("Error") ? "text-[var(--error)]" : "text-[var(--success)]"}`}>{evalResult}</span>}
          <button onClick={load} className="btn-secondary">Refresh</button>
          <button onClick={runEval} disabled={evaluating} className="btn-primary">
            {evaluating ? (
              <span className="flex items-center gap-1.5">
                <span className="h-3 w-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Evaluating...
              </span>
            ) : `Evaluate (${pendingCount})`}
          </button>
        </div>
      </div>

      {/* Aggregate Scores */}
      {traces.length > 0 && (
        <div className="grid grid-cols-5 gap-2">
          {avgScores.map(s => (
            <div key={s.metric} className="card !p-3 text-center">
              <div className="text-lg font-semibold" style={{ color: SCORE_KEYS.find(k => k.label === s.metric)?.color }}>
                {s.score.toFixed(0)}%
              </div>
              <div className="text-[10px] text-[var(--text-muted)]">{s.metric} (avg)</div>
            </div>
          ))}
        </div>
      )}

      {/* Charts */}
      {traces.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          <div className="card">
            <h3 className="text-xs text-[var(--text-muted)] mb-2">{hasJudge ? "LLM Judge Scores (avg)" : "Rule-Based Scores (avg)"}</h3>
            <ResponsiveContainer width="100%" height={230}>
              <RadarChart data={hasJudge ? judgeAvgs : avgScores}>
                <PolarGrid stroke="var(--border)" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: "var(--text-muted)" }} />
                <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 9 }} />
                <Radar dataKey="score" stroke="#2563eb" fill="#2563eb" fillOpacity={0.15} strokeWidth={2} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <h3 className="text-xs text-[var(--text-muted)] mb-2">Scores Per Request (recent 10)</h3>
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="label" stroke="var(--text-muted)" fontSize={8} />
                <YAxis stroke="var(--text-muted)" fontSize={10} domain={[0, 100]} />
                <Tooltip {...tip} />
                <Legend wrapperStyle={{ fontSize: 9 }} />
                {SCORE_KEYS.slice(0, 3).map(sk => (
                  <Bar key={sk.key} dataKey={sk.key} name={sk.label} fill={sk.color} radius={[2, 2, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Per-Request List */}
      <div className="space-y-2">
        <h2 className="text-sm font-medium">All Requests (newest first)</h2>
        {traces.map(t => (
          <details key={t.id} className={`card !p-0 overflow-hidden border-l-4 ${t.status === "completed" ? "border-l-green-500" : "border-l-red-500"}`}>
            <summary className="flex items-center justify-between px-4 py-2.5 cursor-pointer hover:bg-[var(--bg-hover)]">
              <div className="flex items-center gap-2 flex-1 min-w-0">
                <span className={`badge flex-shrink-0 ${t.eval_status === "evaluated" ? "bg-[var(--success-light)] text-[var(--success)]" : t.eval_status === "quick" ? "bg-[var(--warning-light)] text-[var(--warning)]" : "bg-[var(--bg-hover)] text-[var(--text-muted)]"}`}>
                  {t.eval_status === "evaluated" ? "EVAL" : t.eval_status === "quick" ? "QUICK" : "PEND"}
                </span>
                <span className="text-sm truncate">{t.user_prompt}</span>
              </div>
              <div className="flex gap-2 text-xs text-[var(--text-muted)] flex-shrink-0 ml-3">
                {t.agent_used && <span className="badge bg-[var(--accent-light)] text-[var(--accent)]">{t.agent_used}</span>}
                <span>{t.total_latency_ms.toFixed(0)}ms</span>
              </div>
            </summary>

            <div className="border-t border-[var(--border)] bg-[var(--bg)]">
              {/* Request */}
              <div className="px-4 py-2 border-b border-[var(--border)]">
                <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-0.5">Request</div>
                <div className="text-sm">{t.user_prompt}</div>
              </div>

              {/* Agent Trace */}
              {t.tool_calls && t.tool_calls.length > 0 && (
                <div className="px-4 py-2 border-b border-[var(--border)]">
                  <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1">Trace ({t.tool_calls.length} tool calls)</div>
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-1.5 text-xs">
                      <span className="text-[var(--success)]">&#10003;</span>
                      <span className="text-[var(--text-muted)]">Route to</span>
                      <span className="font-medium">{t.agent_used}</span>
                    </div>
                    {t.tool_calls.map((tc: any, j: number) => (
                      <div key={j} className="flex items-center gap-1.5 text-xs">
                        <span className="text-[var(--success)]">&#10003;</span>
                        <span className="font-mono">{tc.tool}</span>
                        <span className="text-[var(--text-muted)] truncate text-[10px]">
                          ({Object.entries(tc.args || {}).map(([k,v]) => `${k}="${String(v).slice(0,25)}"`).join(", ")})
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Agent Output */}
              {t.agent_response && (
                <div className="px-4 py-2 border-b border-[var(--border)]">
                  <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-0.5">Agent Output</div>
                  <div className="text-xs bg-[var(--bg-card)] border border-[var(--border)] rounded p-2 max-h-24 overflow-y-auto whitespace-pre-wrap">
                    {t.agent_response.slice(0, 500)}{t.agent_response.length > 500 && "..."}
                  </div>
                </div>
              )}

              {/* Scores */}
              {t.eval_scores && Object.keys(t.eval_scores).length > 0 && (
                <div className="px-4 py-2">
                  <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1">Scores</div>
                  <div className="flex gap-1.5 flex-wrap">
                    {Object.entries(t.eval_scores).map(([k, v]: [string, any]) => (
                      <div key={k} className={`text-center px-2 py-1 rounded border text-xs ${
                        (v || 0) >= 0.7 ? "bg-[var(--success-light)] text-[var(--success)] border-[var(--success)]/20"
                        : (v || 0) >= 0.4 ? "bg-[var(--warning-light)] text-[var(--warning)] border-[var(--warning)]/20"
                        : "bg-[var(--error-light)] text-[var(--error)] border-[var(--error)]/20"
                      }`}>
                        <div className="font-semibold">{((v || 0) * 100).toFixed(0)}%</div>
                        <div className="text-[9px] opacity-70">{k.replace(/^judge_/, "").replace(/_/g, " ")}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </details>
        ))}
        {traces.length === 0 && <div className="text-[var(--text-muted)] text-center py-10">No requests yet. Send messages in Chat first.</div>}
      </div>
    </div>
  );
}
