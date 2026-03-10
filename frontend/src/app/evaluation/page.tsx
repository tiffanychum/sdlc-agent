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

const SPAN_COLORS: Record<string, string> = {
  routing: "border-l-blue-500 bg-blue-50",
  agent_execution: "border-l-purple-500 bg-purple-50",
  tool_call: "border-l-green-500 bg-green-50",
  supervisor: "border-l-amber-500 bg-amber-50",
  llm_call: "border-l-indigo-500 bg-indigo-50",
};

const SPAN_ICONS: Record<string, string> = {
  routing: "🔀",
  agent_execution: "🤖",
  tool_call: "🔧",
  supervisor: "👁",
  llm_call: "💬",
};

function ScoreBadge({ value, label }: { value: number; label: string }) {
  const pct = (value || 0) * 100;
  const color = pct >= 70 ? "bg-[var(--success-light)] text-[var(--success)] border-[var(--success)]/20"
    : pct >= 40 ? "bg-[var(--warning-light)] text-[var(--warning)] border-[var(--warning)]/20"
    : "bg-[var(--error-light)] text-[var(--error)] border-[var(--error)]/20";
  return (
    <div className={`text-center px-2.5 py-1.5 rounded border text-xs ${color}`}>
      <div className="font-semibold text-sm">{pct.toFixed(0)}%</div>
      <div className="text-[9px] opacity-70">{label}</div>
    </div>
  );
}

function SpanTimeline({ spans }: { spans: any[] }) {
  if (!spans || spans.length === 0) return <div className="text-xs text-[var(--text-muted)]">No span data available</div>;

  return (
    <div className="space-y-1">
      {spans.map((s: any, i: number) => {
        const typeClass = SPAN_COLORS[s.span_type] || "border-l-gray-400 bg-gray-50";
        const icon = SPAN_ICONS[s.span_type] || "•";
        const duration = s.start_time && s.end_time
          ? Math.round(new Date(s.end_time).getTime() - new Date(s.start_time).getTime())
          : null;
        const tokens = (s.tokens_in || 0) + (s.tokens_out || 0);

        return (
          <details key={s.id || i} className={`border-l-[3px] rounded-r-md ${typeClass} group`}>
            <summary className="flex items-center gap-2 px-3 py-1.5 cursor-pointer hover:opacity-80">
              <span className="text-xs">{icon}</span>
              <span className="text-xs font-medium flex-1 truncate">{s.name}</span>
              <div className="flex items-center gap-2 text-[10px] text-[var(--text-muted)]">
                <span className="uppercase tracking-wide">{s.span_type.replace("_", " ")}</span>
                {duration !== null && <span>{duration}ms</span>}
                {tokens > 0 && <span>{tokens} tok</span>}
                {s.cost > 0 && <span>${s.cost.toFixed(4)}</span>}
                <span className={`h-1.5 w-1.5 rounded-full ${s.status === "completed" ? "bg-green-500" : "bg-red-500"}`} />
              </div>
            </summary>
            <div className="px-3 pb-2 space-y-1">
              {s.input_data && Object.keys(s.input_data).length > 0 && (
                <div>
                  <div className="text-[9px] text-[var(--text-muted)] uppercase tracking-wide">Input</div>
                  <pre className="text-[10px] bg-white/50 rounded p-1.5 overflow-x-auto max-h-24 overflow-y-auto border border-[var(--border)]">
                    {JSON.stringify(s.input_data, null, 2)}
                  </pre>
                </div>
              )}
              {s.output_data && Object.keys(s.output_data).length > 0 && (
                <div>
                  <div className="text-[9px] text-[var(--text-muted)] uppercase tracking-wide">Output</div>
                  <pre className="text-[10px] bg-white/50 rounded p-1.5 overflow-x-auto max-h-24 overflow-y-auto border border-[var(--border)]">
                    {JSON.stringify(s.output_data, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </details>
        );
      })}
    </div>
  );
}

function AgentTraceTimeline({ agentTrace }: { agentTrace: any[] }) {
  if (!agentTrace || agentTrace.length === 0) return null;

  return (
    <div className="space-y-1.5">
      {agentTrace.map((entry: any, i: number) => {
        if (entry.step === "routing") {
          return (
            <div key={i} className="flex items-center gap-2 text-xs p-2 rounded-md bg-blue-50 border-l-[3px] border-l-blue-500">
              <span>🔀</span>
              <span className="text-[var(--text-muted)]">Route to</span>
              <span className="font-semibold text-blue-700">{entry.selected_agent}</span>
              {entry.reasoning && <span className="text-[10px] text-[var(--text-muted)] truncate">({entry.reasoning.slice(0, 50)})</span>}
            </div>
          );
        }
        if (entry.step === "supervisor") {
          return (
            <div key={i} className="flex items-center gap-2 text-xs p-2 rounded-md bg-amber-50 border-l-[3px] border-l-amber-500">
              <span>👁</span>
              <span className="text-[var(--text-muted)]">Supervisor</span>
              <span className="font-semibold text-amber-700">{entry.decision === "done" ? "DONE" : `→ ${entry.decision}`}</span>
            </div>
          );
        }
        if (entry.step === "execution") {
          return (
            <details key={i} className="border-l-[3px] border-l-purple-500 bg-purple-50 rounded-r-md">
              <summary className="flex items-center gap-2 px-3 py-1.5 cursor-pointer hover:opacity-80 text-xs">
                <span>🤖</span>
                <span className="font-semibold text-purple-700">{entry.agent}</span>
                <span className="text-[var(--text-muted)]">({entry.tool_calls?.length || 0} tool calls)</span>
              </summary>
              <div className="px-3 pb-2 space-y-0.5">
                {entry.tool_calls?.map((tc: any, j: number) => (
                  <div key={j} className="flex items-center gap-1.5 text-[11px] py-0.5">
                    <span className="text-green-600">🔧</span>
                    <span className="font-mono text-green-700">{tc.tool}</span>
                    <span className="text-[var(--text-muted)] truncate text-[10px]">
                      ({Object.entries(tc.args || {}).map(([k, v]) => `${k}="${String(v).slice(0, 30)}"`).join(", ")})
                    </span>
                  </div>
                ))}
              </div>
            </details>
          );
        }
        return (
          <div key={i} className="flex items-center gap-2 text-xs p-2 rounded-md bg-gray-50 border-l-[3px] border-l-gray-400">
            <span className="text-[var(--text-muted)]">{entry.tool || entry.step || "unknown"}</span>
          </div>
        );
      })}
    </div>
  );
}

export default function EvaluationPage() {
  const [traces, setTraces] = useState<any[]>([]);
  const [evaluating, setEvaluating] = useState(false);
  const [evalResult, setEvalResult] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);

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
    return { metric: sk.label, score: vals.length > 0 ? +(vals.reduce((a: number, b: number) => a + b, 0) / vals.length * 100).toFixed(1) : 0 };
  });

  const judgeAvgs = JUDGE_KEYS.map(jk => {
    const vals = traces.map(t => t.eval_scores?.[jk.key] || 0).filter(v => v > 0);
    return { metric: jk.label, score: vals.length > 0 ? +(vals.reduce((a: number, b: number) => a + b, 0) / vals.length * 100).toFixed(1) : 0 };
  });

  const hasJudge = judgeAvgs.some(j => j.score > 0);

  const barData = traces.slice(0, 10).reverse().map(t => ({
    label: (t.user_prompt || "").slice(0, 15) + "...",
    ...Object.fromEntries(SCORE_KEYS.map(sk => [sk.key, +((t.eval_scores?.[sk.key] || 0) * 100).toFixed(1)])),
  }));

  const tip = { contentStyle: { background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 } };

  function hasAgentTraceFormat(arr: any[]): boolean {
    return arr.length > 0 && arr[0]?.step !== undefined;
  }

  function flatToolCalls(traceData: any): any[] {
    const tc = traceData.agent_trace || traceData.tool_calls || [];
    if (hasAgentTraceFormat(tc)) {
      const flat: any[] = [];
      for (const entry of tc) {
        if (entry.step === "execution") {
          for (const c of entry.tool_calls || []) {
            flat.push({ ...c, agent: entry.agent });
          }
        }
      }
      return flat;
    }
    return tc;
  }

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
                {Number(s.score).toFixed(0)}%
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

      {/* Per-Request Task List */}
      <div className="space-y-2">
        <h2 className="text-sm font-medium">All Requests (newest first)</h2>
        {traces.map(t => {
          const isExpanded = expandedId === t.id;
          const allTools = flatToolCalls(t);
          const agentTrace = t.agent_trace || t.tool_calls || [];
          const isRichTrace = hasAgentTraceFormat(agentTrace);

          return (
            <div key={t.id} className={`card !p-0 overflow-hidden border-l-4 transition-all ${
              t.status === "completed" ? "border-l-green-500" : "border-l-red-500"
            } ${isExpanded ? "ring-1 ring-[var(--accent)]/30" : ""}`}>
              {/* Summary Row */}
              <div
                onClick={() => setExpandedId(isExpanded ? null : t.id)}
                className="flex items-center justify-between px-4 py-2.5 cursor-pointer hover:bg-[var(--bg-hover)] transition-colors"
              >
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <span className={`badge flex-shrink-0 ${
                    t.eval_status === "evaluated" ? "bg-[var(--success-light)] text-[var(--success)]"
                    : t.eval_status === "quick" ? "bg-[var(--warning-light)] text-[var(--warning)]"
                    : "bg-[var(--bg-hover)] text-[var(--text-muted)]"
                  }`}>
                    {t.eval_status === "evaluated" ? "EVAL" : t.eval_status === "quick" ? "QUICK" : "PEND"}
                  </span>
                  <span className="text-sm truncate">{t.user_prompt}</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-[var(--text-muted)] flex-shrink-0 ml-3">
                  {t.agent_used && <span className="badge bg-[var(--accent-light)] text-[var(--accent)]">{t.agent_used}</span>}
                  <span>{allTools.length} tools</span>
                  <span>{t.spans?.length || 0} spans</span>
                  <span>{t.total_latency_ms.toFixed(0)}ms</span>
                  <span className="text-[10px]">{isExpanded ? "▲" : "▼"}</span>
                </div>
              </div>

              {/* Expanded Detail View */}
              {isExpanded && (
                <div className="border-t border-[var(--border)] bg-[var(--bg)]">
                  <div className="grid grid-cols-[1fr_1fr] divide-x divide-[var(--border)]">
                    {/* Left: Trace + Agent Output */}
                    <div className="divide-y divide-[var(--border)]">
                      {/* Request */}
                      <div className="px-4 py-2.5">
                        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1">User Request</div>
                        <div className="text-sm">{t.user_prompt}</div>
                      </div>

                      {/* Agent Trace (rich format) */}
                      {isRichTrace && (
                        <div className="px-4 py-2.5">
                          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1.5">
                            Agent Execution Flow ({agentTrace.length} steps)
                          </div>
                          <AgentTraceTimeline agentTrace={agentTrace} />
                        </div>
                      )}

                      {/* Span Timeline */}
                      {t.spans && t.spans.length > 0 && (
                        <div className="px-4 py-2.5">
                          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1.5">
                            Full Span Timeline ({t.spans.length} spans)
                          </div>
                          <SpanTimeline spans={t.spans} />
                        </div>
                      )}

                      {/* Flat tool calls fallback for old traces */}
                      {!isRichTrace && allTools.length > 0 && (
                        <div className="px-4 py-2.5">
                          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1">Tool Calls ({allTools.length})</div>
                          <div className="space-y-0.5">
                            {allTools.map((tc: any, j: number) => (
                              <div key={j} className="flex items-center gap-1.5 text-xs">
                                <span className="text-[var(--success)]">&#10003;</span>
                                <span className="font-mono">{tc.tool}</span>
                                <span className="text-[var(--text-muted)] truncate text-[10px]">
                                  ({Object.entries(tc.args || {}).map(([k, v]) => `${k}="${String(v).slice(0, 25)}"`).join(", ")})
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Agent Output */}
                      {t.agent_response && (
                        <div className="px-4 py-2.5">
                          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1">Agent Output</div>
                          <div className="text-xs bg-[var(--bg-card)] border border-[var(--border)] rounded p-2.5 max-h-40 overflow-y-auto whitespace-pre-wrap leading-relaxed">
                            {t.agent_response}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Right: Evaluation Scores */}
                    <div className="divide-y divide-[var(--border)]">
                      {/* Overview Stats */}
                      <div className="px-4 py-2.5">
                        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1.5">Overview</div>
                        <div className="grid grid-cols-4 gap-2 text-center">
                          <div className="p-1.5 rounded bg-[var(--bg-card)] border border-[var(--border)]">
                            <div className="text-sm font-semibold">{t.total_latency_ms.toFixed(0)}<span className="text-[9px] text-[var(--text-muted)]">ms</span></div>
                            <div className="text-[9px] text-[var(--text-muted)]">Latency</div>
                          </div>
                          <div className="p-1.5 rounded bg-[var(--bg-card)] border border-[var(--border)]">
                            <div className="text-sm font-semibold">{t.total_tokens || 0}</div>
                            <div className="text-[9px] text-[var(--text-muted)]">Tokens</div>
                          </div>
                          <div className="p-1.5 rounded bg-[var(--bg-card)] border border-[var(--border)]">
                            <div className="text-sm font-semibold">${(t.total_cost || 0).toFixed(4)}</div>
                            <div className="text-[9px] text-[var(--text-muted)]">Cost</div>
                          </div>
                          <div className="p-1.5 rounded bg-[var(--bg-card)] border border-[var(--border)]">
                            <div className="text-sm font-semibold">{allTools.length}</div>
                            <div className="text-[9px] text-[var(--text-muted)]">Tool Calls</div>
                          </div>
                        </div>
                      </div>

                      {/* Rule-Based Scores */}
                      {t.eval_scores && Object.keys(t.eval_scores).some(k => !k.startsWith("judge_")) && (
                        <div className="px-4 py-2.5">
                          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1.5">Rule-Based Scores</div>
                          <div className="flex gap-1.5 flex-wrap">
                            {SCORE_KEYS.map(sk => (
                              t.eval_scores?.[sk.key] !== undefined && (
                                <ScoreBadge key={sk.key} value={t.eval_scores[sk.key]} label={sk.label} />
                              )
                            ))}
                          </div>
                        </div>
                      )}

                      {/* LLM Judge Scores */}
                      {t.eval_scores && Object.keys(t.eval_scores).some(k => k.startsWith("judge_")) && (
                        <div className="px-4 py-2.5">
                          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1.5">LLM Judge Scores</div>
                          <div className="flex gap-1.5 flex-wrap">
                            {JUDGE_KEYS.map(jk => (
                              t.eval_scores?.[jk.key] !== undefined && (
                                <ScoreBadge key={jk.key} value={t.eval_scores[jk.key]} label={jk.label} />
                              )
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Raw eval_scores dump for any other scores */}
                      {t.eval_scores && Object.keys(t.eval_scores).length > 0 && (
                        <div className="px-4 py-2.5">
                          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1">All Scores (raw)</div>
                          <div className="grid grid-cols-2 gap-1">
                            {Object.entries(t.eval_scores).map(([k, v]: [string, any]) => (
                              <div key={k} className="flex justify-between text-[10px] py-0.5">
                                <span className="text-[var(--text-muted)]">{k.replace(/^judge_/, "").replace(/_/g, " ")}</span>
                                <span className="font-medium">{typeof v === "number" ? `${(v * 100).toFixed(0)}%` : String(v)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Trace ID + Meta */}
                      <div className="px-4 py-2.5">
                        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1">Meta</div>
                        <div className="space-y-0.5 text-[10px]">
                          <div><span className="text-[var(--text-muted)]">Trace ID:</span> <span className="font-mono">{t.id}</span></div>
                          <div><span className="text-[var(--text-muted)]">Agent(s):</span> {t.agent_used || "unknown"}</div>
                          <div><span className="text-[var(--text-muted)]">Status:</span> {t.status}</div>
                          <div><span className="text-[var(--text-muted)]">Eval Status:</span> {t.eval_status}</div>
                          {t.created_at && <div><span className="text-[var(--text-muted)]">Time:</span> {new Date(t.created_at).toLocaleString()}</div>}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
        {traces.length === 0 && <div className="text-[var(--text-muted)] text-center py-10">No requests yet. Send messages in Chat first.</div>}
      </div>
    </div>
  );
}
