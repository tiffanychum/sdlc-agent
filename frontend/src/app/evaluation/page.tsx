"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";

const RULE_KEYS = [
  { key: "tool_accuracy", label: "Tool Acc.", color: "#2563eb" },
  { key: "step_efficiency", label: "Efficiency", color: "#ca8a04" },
  { key: "faithfulness", label: "Faithfulness", color: "#0891b2" },
  { key: "safety", label: "Safety", color: "#dc2626" },
  { key: "reasoning_quality", label: "Reasoning", color: "#7c3aed" },
];

const GEVAL_KEYS = [
  { key: "correctness", label: "Correctness", color: "#2563eb" },
  { key: "relevance", label: "Relevance", color: "#059669" },
  { key: "coherence", label: "Coherence", color: "#7c3aed" },
  { key: "tool_usage_quality", label: "Tool Quality", color: "#ca8a04" },
  { key: "completeness", label: "Completeness", color: "#dc2626" },
];

const DEEPEVAL_KEYS = [
  { key: "deepeval_relevancy", label: "Relevancy", color: "#0891b2" },
  { key: "deepeval_faithfulness", label: "Faithfulness", color: "#7c3aed" },
  { key: "tool_correctness", label: "Tool Correct.", color: "#059669" },
  { key: "argument_correctness", label: "Arg Correct.", color: "#2563eb" },
  { key: "task_completion", label: "Task Complete", color: "#dc2626" },
  { key: "step_efficiency_de", label: "Step Effic.", color: "#ca8a04" },
  { key: "plan_quality", label: "Plan Quality", color: "#0d9488" },
  { key: "plan_adherence", label: "Plan Adhere.", color: "#7c3aed" },
];

const SPAN_COLORS: Record<string, string> = {
  routing: "border-l-blue-500 bg-blue-50",
  agent_execution: "border-l-purple-500 bg-purple-50",
  tool_call: "border-l-green-500 bg-green-50",
  supervisor: "border-l-amber-500 bg-amber-50",
  llm_call: "border-l-indigo-500 bg-indigo-50",
};

const SPAN_ICONS: Record<string, string> = {
  routing: "🔀", agent_execution: "🤖", tool_call: "🔧", supervisor: "👁", llm_call: "💬",
};

function ScoreBadge({ value, label, size = "sm" }: { value: number; label: string; size?: string }) {
  const pct = (value || 0) * 100;
  const color = pct >= 70 ? "bg-emerald-50 text-emerald-700 border-emerald-200"
    : pct >= 40 ? "bg-amber-50 text-amber-700 border-amber-200"
    : "bg-red-50 text-red-700 border-red-200";
  return (
    <div className={`text-center rounded border ${color} ${size === "lg" ? "px-3 py-2" : "px-2 py-1"}`}>
      <div className={`font-semibold ${size === "lg" ? "text-base" : "text-sm"}`}>{pct.toFixed(0)}%</div>
      <div className="text-[9px] opacity-70">{label}</div>
    </div>
  );
}

function SpanTimeline({ spans }: { spans: any[] }) {
  if (!spans || spans.length === 0) return <div className="text-xs text-[var(--text-muted)]">No span data</div>;
  return (
    <div className="space-y-1">
      {spans.map((s: any, i: number) => {
        const typeClass = SPAN_COLORS[s.span_type] || "border-l-gray-400 bg-gray-50";
        const icon = SPAN_ICONS[s.span_type] || "•";
        const duration = s.start_time && s.end_time
          ? Math.round(new Date(s.end_time).getTime() - new Date(s.start_time).getTime()) : null;
        const tokens = (s.tokens_in || 0) + (s.tokens_out || 0);
        return (
          <details key={s.id || i} className={`border-l-[3px] rounded-r-md ${typeClass}`}>
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
                  <pre className="text-[10px] bg-white/50 rounded p-1.5 overflow-x-auto max-h-20 overflow-y-auto border border-[var(--border)]">
                    {JSON.stringify(s.input_data, null, 2)}
                  </pre>
                </div>
              )}
              {s.output_data && Object.keys(s.output_data).length > 0 && (
                <div>
                  <div className="text-[9px] text-[var(--text-muted)] uppercase tracking-wide">Output</div>
                  <pre className="text-[10px] bg-white/50 rounded p-1.5 overflow-x-auto max-h-20 overflow-y-auto border border-[var(--border)]">
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
            </div>
          );
        }
        if (entry.step === "supervisor") {
          return (
            <div key={i} className="flex items-center gap-2 text-xs p-2 rounded-md bg-amber-50 border-l-[3px] border-l-amber-500">
              <span>👁</span><span className="text-[var(--text-muted)]">Supervisor</span>
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
                      ({Object.entries(tc.args || {}).map(([k, v]) => `${k}="${String(v).slice(0, 25)}"`).join(", ")})
                    </span>
                  </div>
                ))}
              </div>
            </details>
          );
        }
        return null;
      })}
    </div>
  );
}

export default function EvaluationPage() {
  const [traces, setTraces] = useState<any[]>([]);
  const [evaluating, setEvaluating] = useState(false);
  const [evalResult, setEvalResult] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"combined" | "geval" | "deepeval">("combined");

  useEffect(() => { load(); }, []);

  async function load() {
    const t = await api.traces.list(100);
    setTraces(t.filter((tr: any) => tr.agent_response));
  }

  async function runEval() {
    setEvaluating(true);
    setEvalResult("Running G-Eval + DeepEval...");
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

  function getGEvalScores(t: any) {
    return t.eval_scores?.geval_scores || {};
  }
  function getGEvalReasoning(t: any) {
    return t.eval_scores?.geval_reasoning || {};
  }
  function getDeepEvalScores(t: any) {
    return t.eval_scores?.deepeval_scores || {};
  }

  const gevalAvgs = GEVAL_KEYS.map(gk => {
    const vals = traces.map(t => getGEvalScores(t)[gk.key] || 0).filter(v => v > 0);
    return { metric: gk.label, score: vals.length > 0 ? +(vals.reduce((a: number, b: number) => a + b, 0) / vals.length * 100).toFixed(1) : 0 };
  });

  const deepevalAvgs = DEEPEVAL_KEYS.map(dk => {
    const vals = traces.map(t => getDeepEvalScores(t)[dk.key] || 0).filter(v => v > 0);
    return { metric: dk.label, score: vals.length > 0 ? +(vals.reduce((a: number, b: number) => a + b, 0) / vals.length * 100).toFixed(1) : 0 };
  });

  const ruleAvgs = RULE_KEYS.map(rk => {
    const vals = traces.map(t => t.eval_scores?.[rk.key] || 0).filter(v => v > 0);
    return { metric: rk.label, score: vals.length > 0 ? +(vals.reduce((a: number, b: number) => a + b, 0) / vals.length * 100).toFixed(1) : 0 };
  });

  const combinedRadar = [
    ...GEVAL_KEYS.map(gk => {
      const gVal = gevalAvgs.find(a => a.metric === gk.label)?.score || 0;
      return { metric: gk.label, "G-Eval": gVal, "DeepEval": 0 };
    }),
    ...DEEPEVAL_KEYS.map(dk => {
      const dVal = deepevalAvgs.find(a => a.metric === dk.label)?.score || 0;
      return { metric: dk.label, "G-Eval": 0, "DeepEval": dVal };
    }),
  ];

  const hasGEval = gevalAvgs.some(g => g.score > 0);
  const hasDeepEval = deepevalAvgs.some(d => d.score > 0);

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
          for (const c of entry.tool_calls || []) flat.push({ ...c, agent: entry.agent });
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
            {traces.length} requests. {pendingCount > 0 ? `${pendingCount} pending.` : "All evaluated."}
            {hasGEval && " G-Eval ✓"}{hasDeepEval && " DeepEval ✓"}
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

      {/* Evaluation Method Tabs */}
      <div className="flex gap-1 border-b border-[var(--border)]">
        {(["combined", "geval", "deepeval"] as const).map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-all ${
              activeTab === tab ? "border-[var(--accent)] text-[var(--accent)]" : "border-transparent text-[var(--text-muted)] hover:text-[var(--text)]"
            }`}>
            {tab === "combined" ? "Combined View" : tab === "geval" ? "G-Eval (LLM Judge)" : "DeepEval (External)"}
          </button>
        ))}
      </div>

      {/* Combined View */}
      {activeTab === "combined" && traces.length > 0 && (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            {/* Rule-Based */}
            <div className="card">
              <h3 className="text-xs font-medium text-[var(--text-muted)] mb-2">Rule-Based (avg)</h3>
              <div className="grid grid-cols-2 gap-1.5">
                {ruleAvgs.map(s => (
                  <ScoreBadge key={s.metric} value={Number(s.score) / 100} label={s.metric} />
                ))}
              </div>
            </div>
            {/* G-Eval */}
            <div className="card">
              <h3 className="text-xs font-medium text-[var(--text-muted)] mb-2">G-Eval / LLM Judge (avg)</h3>
              {hasGEval ? (
                <div className="grid grid-cols-2 gap-1.5">
                  {gevalAvgs.filter(s => s.score > 0).map(s => (
                    <ScoreBadge key={s.metric} value={Number(s.score) / 100} label={s.metric} />
                  ))}
                </div>
              ) : <div className="text-xs text-[var(--text-muted)] py-4 text-center">Click Evaluate to run G-Eval</div>}
            </div>
            {/* DeepEval */}
            <div className="card">
              <h3 className="text-xs font-medium text-[var(--text-muted)] mb-2">DeepEval Agentic (avg)</h3>
              {hasDeepEval ? (
                <div className="grid grid-cols-3 gap-1.5">
                  {deepevalAvgs.filter(s => s.score > 0).map(s => (
                    <ScoreBadge key={s.metric} value={Number(s.score) / 100} label={s.metric} />
                  ))}
                </div>
              ) : <div className="text-xs text-[var(--text-muted)] py-4 text-center">Click Evaluate to run DeepEval</div>}
            </div>
          </div>

          {/* Cross-Validation Comparison Chart */}
          {(hasGEval || hasDeepEval) && (
            <div className="card">
              <h3 className="text-xs text-[var(--text-muted)] mb-2">Cross-Validation: G-Eval vs DeepEval vs Rule-Based</h3>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={[
                  ...ruleAvgs.map(r => ({ metric: r.metric, "Rule-Based": r.score, "G-Eval": 0, "DeepEval": 0 })),
                  ...gevalAvgs.map(g => ({ metric: g.metric, "Rule-Based": 0, "G-Eval": g.score, "DeepEval": 0 })),
                  ...deepevalAvgs.map(d => ({ metric: d.metric, "Rule-Based": 0, "G-Eval": 0, "DeepEval": d.score })),
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="metric" stroke="var(--text-muted)" fontSize={9} angle={-20} textAnchor="end" height={50} />
                  <YAxis stroke="var(--text-muted)" fontSize={10} domain={[0, 100]} />
                  <Tooltip {...tip} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Bar dataKey="Rule-Based" fill="#6b7280" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="G-Eval" fill="#2563eb" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="DeepEval" fill="#7c3aed" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* G-Eval Tab */}
      {activeTab === "geval" && traces.length > 0 && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="card">
              <h3 className="text-xs text-[var(--text-muted)] mb-2">G-Eval Radar (CoT + Per-Criterion)</h3>
              <ResponsiveContainer width="100%" height={250}>
                <RadarChart data={gevalAvgs}>
                  <PolarGrid stroke="var(--border)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: "var(--text-muted)" }} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 9 }} />
                  <Radar dataKey="score" stroke="#2563eb" fill="#2563eb" fillOpacity={0.15} strokeWidth={2} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
            <div className="card">
              <h3 className="text-xs text-[var(--text-muted)] mb-2">G-Eval Method</h3>
              <div className="text-xs space-y-2 text-[var(--text-muted)]">
                <div className="p-2 rounded bg-blue-50 border border-blue-100">
                  <div className="font-medium text-blue-700 mb-1">Phase 1: CoT Step Generation</div>
                  <div>Auto-generates evaluation steps from criteria before scoring. Each criterion gets its own reasoning chain.</div>
                </div>
                <div className="p-2 rounded bg-blue-50 border border-blue-100">
                  <div className="font-medium text-blue-700 mb-1">Phase 2: Per-Criterion Scoring</div>
                  <div>Each criterion evaluated independently (5 parallel LLM calls) to prevent cross-criterion interference.</div>
                </div>
                <div className="p-2 rounded bg-blue-50 border border-blue-100">
                  <div className="font-medium text-blue-700 mb-1">Reasoning Before Score</div>
                  <div>LLM reasons through evaluation steps before committing to a 1-5 score. Returns both score and explanation.</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DeepEval Tab */}
      {activeTab === "deepeval" && traces.length > 0 && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="card">
              <h3 className="text-xs text-[var(--text-muted)] mb-2">DeepEval Scores (External Validation)</h3>
              <ResponsiveContainer width="100%" height={250}>
                <RadarChart data={deepevalAvgs}>
                  <PolarGrid stroke="var(--border)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: "var(--text-muted)" }} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 9 }} />
                  <Radar dataKey="score" stroke="#7c3aed" fill="#7c3aed" fillOpacity={0.15} strokeWidth={2} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
            <div className="card space-y-2">
              <h3 className="text-xs text-[var(--text-muted)] mb-2">DeepEval Agentic Metrics</h3>
              <div className="text-xs space-y-1.5 text-[var(--text-muted)]">
                <div className="p-2 rounded bg-purple-50 border border-purple-100">
                  <div className="font-medium text-purple-700 mb-0.5">Task Completion</div>
                  <div>Evaluates whether the agent accomplished the user&apos;s task by analyzing execution trace and final output.</div>
                </div>
                <div className="p-2 rounded bg-purple-50 border border-purple-100">
                  <div className="font-medium text-purple-700 mb-0.5">Tool Correctness</div>
                  <div>Assesses whether the right tools were called, in the right order, with optimal selection.</div>
                </div>
                <div className="p-2 rounded bg-purple-50 border border-purple-100">
                  <div className="font-medium text-purple-700 mb-0.5">Argument Correctness</div>
                  <div>Evaluates whether tool arguments (input parameters) were correctly generated from the user&apos;s request.</div>
                </div>
                <div className="p-2 rounded bg-purple-50 border border-purple-100">
                  <div className="font-medium text-purple-700 mb-0.5">Step Efficiency</div>
                  <div>Scores execution efficiency — penalizes redundant or unnecessary steps in the agent trace.</div>
                </div>
                <div className="p-2 rounded bg-purple-50 border border-purple-100">
                  <div className="font-medium text-purple-700 mb-0.5">Plan Quality</div>
                  <div>Assesses whether the agent&apos;s generated plan is logical, complete, and efficient for the task.</div>
                </div>
                <div className="p-2 rounded bg-purple-50 border border-purple-100">
                  <div className="font-medium text-purple-700 mb-0.5">Plan Adherence</div>
                  <div>Checks if the agent followed its own plan. Detects deviations and evaluates if they were justified.</div>
                </div>
                <div className="p-2 rounded bg-purple-50 border border-purple-100">
                  <div className="font-medium text-purple-700 mb-0.5">Answer Relevancy + Faithfulness</div>
                  <div>Cross-validates against self-evaluation bias using DeepEval&apos;s independent LLM for hallucination and relevancy checks.</div>
                </div>
              </div>
            </div>
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
          const gScores = getGEvalScores(t);
          const gReasoning = getGEvalReasoning(t);
          const dScores = getDeepEvalScores(t);
          const hasG = Object.keys(gScores).length > 0;
          const hasD = Object.keys(dScores).length > 0;

          return (
            <div key={t.id} className={`card !p-0 overflow-hidden border-l-4 transition-all ${
              t.status === "completed" ? "border-l-green-500" : "border-l-red-500"
            } ${isExpanded ? "ring-1 ring-[var(--accent)]/30" : ""}`}>
              <div onClick={() => setExpandedId(isExpanded ? null : t.id)}
                className="flex items-center justify-between px-4 py-2.5 cursor-pointer hover:bg-[var(--bg-hover)] transition-colors">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <span className={`badge flex-shrink-0 ${
                    t.eval_status === "evaluated" ? "bg-emerald-50 text-emerald-700"
                    : t.eval_status === "quick" ? "bg-amber-50 text-amber-700"
                    : "bg-[var(--bg-hover)] text-[var(--text-muted)]"
                  }`}>
                    {t.eval_status === "evaluated" ? "EVAL" : t.eval_status === "quick" ? "QUICK" : "PEND"}
                  </span>
                  <span className="text-sm truncate">{t.user_prompt}</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-[var(--text-muted)] flex-shrink-0 ml-3">
                  {t.agent_used && <span className="badge bg-[var(--accent-light)] text-[var(--accent)]">{t.agent_used}</span>}
                  {hasG && <span className="badge bg-blue-50 text-blue-700">G-Eval</span>}
                  {hasD && <span className="badge bg-purple-50 text-purple-700">DeepEval</span>}
                  <span>{t.total_latency_ms.toFixed(0)}ms</span>
                  <span className="text-[10px]">{isExpanded ? "▲" : "▼"}</span>
                </div>
              </div>

              {isExpanded && (
                <div className="border-t border-[var(--border)] bg-[var(--bg)]">
                  <div className="grid grid-cols-[1fr_1fr] divide-x divide-[var(--border)]">
                    {/* Left: Trace */}
                    <div className="divide-y divide-[var(--border)]">
                      <div className="px-4 py-2.5">
                        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1">User Request</div>
                        <div className="text-sm">{t.user_prompt}</div>
                      </div>

                      {isRichTrace && (
                        <div className="px-4 py-2.5">
                          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1.5">Agent Execution Flow</div>
                          <AgentTraceTimeline agentTrace={agentTrace} />
                        </div>
                      )}

                      {t.spans && t.spans.length > 0 && (
                        <div className="px-4 py-2.5">
                          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1.5">OTel Span Timeline ({t.spans.length})</div>
                          <SpanTimeline spans={t.spans} />
                        </div>
                      )}

                      {t.agent_response && (
                        <div className="px-4 py-2.5">
                          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1">Agent Output</div>
                          <div className="text-xs bg-[var(--bg-card)] border border-[var(--border)] rounded p-2.5 max-h-32 overflow-y-auto whitespace-pre-wrap">
                            {t.agent_response}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Right: Scores */}
                    <div className="divide-y divide-[var(--border)]">
                      {/* Overview */}
                      <div className="px-4 py-2.5">
                        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1.5">Overview</div>
                        <div className="grid grid-cols-4 gap-2 text-center">
                          {[
                            { v: `${t.total_latency_ms.toFixed(0)}ms`, l: "Latency" },
                            { v: t.total_tokens || 0, l: "Tokens" },
                            { v: `$${(t.total_cost || 0).toFixed(4)}`, l: "Cost" },
                            { v: allTools.length, l: "Tools" },
                          ].map(s => (
                            <div key={s.l} className="p-1.5 rounded bg-[var(--bg-card)] border border-[var(--border)]">
                              <div className="text-sm font-semibold">{s.v}</div>
                              <div className="text-[9px] text-[var(--text-muted)]">{s.l}</div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Rule-Based */}
                      <div className="px-4 py-2.5">
                        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1.5">
                          <span className="inline-block w-2 h-2 rounded-full bg-gray-500 mr-1" />Rule-Based (Heuristic)
                        </div>
                        <div className="flex gap-1.5 flex-wrap">
                          {RULE_KEYS.map(rk => t.eval_scores?.[rk.key] !== undefined && (
                            <ScoreBadge key={rk.key} value={t.eval_scores[rk.key]} label={rk.label} />
                          ))}
                        </div>
                      </div>

                      {/* G-Eval */}
                      <div className="px-4 py-2.5">
                        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1.5">
                          <span className="inline-block w-2 h-2 rounded-full bg-blue-500 mr-1" />G-Eval (CoT + Per-Criterion LLM Judge)
                        </div>
                        {hasG ? (
                          <div className="space-y-2">
                            <div className="flex gap-1.5 flex-wrap">
                              {GEVAL_KEYS.map(gk => gScores[gk.key] !== undefined && (
                                <ScoreBadge key={gk.key} value={gScores[gk.key]} label={gk.label} />
                              ))}
                            </div>
                            {Object.keys(gReasoning).length > 0 && (
                              <details className="text-[10px]">
                                <summary className="cursor-pointer text-[var(--accent)] hover:underline">Show G-Eval reasoning</summary>
                                <div className="mt-1 space-y-1">
                                  {Object.entries(gReasoning).map(([k, v]) => (
                                    <div key={k} className="p-1.5 rounded bg-blue-50 border border-blue-100">
                                      <div className="font-medium text-blue-700 capitalize">{k.replace(/_/g, " ")}</div>
                                      <div className="text-[var(--text-muted)] mt-0.5">{String(v).slice(0, 300)}</div>
                                    </div>
                                  ))}
                                </div>
                              </details>
                            )}
                          </div>
                        ) : <div className="text-[10px] text-[var(--text-muted)]">Not yet evaluated</div>}
                      </div>

                      {/* DeepEval */}
                      <div className="px-4 py-2.5">
                        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1.5">
                          <span className="inline-block w-2 h-2 rounded-full bg-purple-500 mr-1" />DeepEval Agentic Metrics
                        </div>
                        {hasD ? (
                          <div className="space-y-2">
                            <div className="flex gap-1.5 flex-wrap">
                              {DEEPEVAL_KEYS.map(dk => dScores[dk.key] !== undefined && (
                                <ScoreBadge key={dk.key} value={dScores[dk.key] as number} label={dk.label} />
                              ))}
                            </div>
                            {Object.keys(dScores).some(k => k.endsWith("_reason") && dScores[k]) && (
                              <details className="text-[10px]">
                                <summary className="cursor-pointer text-purple-600 hover:underline">Show DeepEval reasoning</summary>
                                <div className="mt-1 space-y-1">
                                  {Object.entries(dScores).filter(([k, v]) => k.endsWith("_reason") && v).map(([k, v]) => (
                                    <div key={k} className="p-1.5 rounded bg-purple-50 border border-purple-100">
                                      <div className="font-medium text-purple-700 capitalize">{k.replace(/_reason$/, "").replace(/_/g, " ")}</div>
                                      <div className="text-[var(--text-muted)] mt-0.5">{String(v).slice(0, 400)}</div>
                                    </div>
                                  ))}
                                </div>
                              </details>
                            )}
                          </div>
                        ) : <div className="text-[10px] text-[var(--text-muted)]">Not yet evaluated</div>}
                      </div>

                      {/* Cross-Validation Insight */}
                      {hasG && hasD && (
                        <div className="px-4 py-2.5">
                          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1.5">Cross-Validation Insight</div>
                          {(() => {
                            const gOverall = t.eval_scores?.geval_overall || 0;
                            const dAvg = Object.values(dScores as Record<string, number>).reduce((a: number, b: number) => a + b, 0) / Math.max(Object.keys(dScores).length, 1);
                            const gap = Math.abs(gOverall - dAvg);
                            const biasFlag = gOverall > dAvg + 0.15;
                            return (
                              <div className={`p-2 rounded border text-xs ${biasFlag ? "bg-amber-50 border-amber-200 text-amber-800" : "bg-emerald-50 border-emerald-200 text-emerald-800"}`}>
                                {biasFlag ? (
                                  <div><span className="font-medium">Self-evaluation bias detected.</span> G-Eval ({(gOverall * 100).toFixed(0)}%) rated significantly higher than DeepEval ({(dAvg * 100).toFixed(0)}%). Gap: {(gap * 100).toFixed(0)}%. The agent may be overrating its own output quality.</div>
                                ) : (
                                  <div><span className="font-medium">Scores aligned.</span> G-Eval ({(gOverall * 100).toFixed(0)}%) and DeepEval ({(dAvg * 100).toFixed(0)}%) are consistent. Gap: {(gap * 100).toFixed(0)}%.</div>
                                )}
                              </div>
                            );
                          })()}
                        </div>
                      )}

                      {/* Meta */}
                      <div className="px-4 py-2.5">
                        <div className="text-[10px] text-[var(--text-muted)] space-y-0.5">
                          <div><span className="opacity-60">Trace:</span> <span className="font-mono">{t.id}</span></div>
                          <div><span className="opacity-60">Agent(s):</span> {t.agent_used}</div>
                          {t.created_at && <div><span className="opacity-60">Time:</span> {new Date(t.created_at).toLocaleString()}</div>}
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
