"use client";
import React, { useEffect, useState, useCallback, useMemo } from "react";
import { api } from "@/lib/api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area, RadarChart,
  Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";

const REGRESSION_METRICS = [
  { key: "task_success_rate", label: "Task Success", color: "#16a34a" },
  { key: "tool_accuracy", label: "Tool Accuracy", color: "#2563eb" },
  { key: "routing_accuracy", label: "Routing Acc.", color: "#7c3aed" },
  { key: "reasoning_quality", label: "Reasoning", color: "#ca8a04" },
  { key: "step_efficiency", label: "Step Effic.", color: "#0891b2" },
  { key: "faithfulness", label: "Faithfulness", color: "#059669" },
  { key: "safety_compliance", label: "Safety", color: "#dc2626" },
];

const DEEPEVAL_METRICS: { key: string; label: string; reasonKey: string; description: string }[] = [
  { key: "deepeval_relevancy", label: "Relevancy", reasonKey: "deepeval_relevancy_reason", description: "Does the response directly address the user's query?" },
  { key: "deepeval_faithfulness", label: "Faithfulness", reasonKey: "deepeval_faithfulness_reason", description: "Is the response grounded in retrieved/tool outputs?" },
  { key: "tool_correctness", label: "Tool Correct.", reasonKey: "tool_correctness_reason", description: "Were the correct tools selected for the task?" },
  { key: "argument_correctness", label: "Arg Correct.", reasonKey: "argument_correctness_reason", description: "Were tool arguments well-formed and appropriate?" },
  { key: "task_completion", label: "Task Complete", reasonKey: "task_completion_reason", description: "Did the agent fully accomplish the user's task?" },
  { key: "step_efficiency_de", label: "Step Effic.", reasonKey: "step_efficiency_de_reason", description: "Were execution steps minimal and necessary?" },
  { key: "plan_quality", label: "Plan Quality", reasonKey: "plan_quality_reason", description: "Was the agent's plan logical, complete, and actionable?" },
  { key: "plan_adherence", label: "Plan Adhere.", reasonKey: "plan_adherence_reason", description: "Did the agent follow its own plan during execution?" },
];

const SPAN_TYPE_COLORS: Record<string, string> = {
  routing: "#3b82f6",
  agent_execution: "#8b5cf6",
  tool_call: "#10b981",
  supervisor: "#f59e0b",
  llm_call: "#6366f1",
  unknown: "#6b7280",
};

const REGRESSION_THRESHOLD = -0.05;

function Metric({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: string }) {
  return (
    <div className="card text-center">
      <div className={`text-2xl font-semibold ${accent || ""}`}>{value}</div>
      <div className="text-xs text-[var(--text-muted)] mt-1">{label}</div>
      {sub && <div className="text-[11px] text-[var(--text-muted)] mt-0.5">{sub}</div>}
    </div>
  );
}

function DeltaBadge({ delta }: { delta: number }) {
  const isRegression = delta < REGRESSION_THRESHOLD;
  const isImproved = delta > 0.05;
  const color = isRegression ? "bg-red-50 text-red-700 border-red-200"
    : isImproved ? "bg-emerald-50 text-emerald-700 border-emerald-200"
    : "bg-gray-50 text-gray-600 border-gray-200";
  const icon = isRegression ? "▼" : isImproved ? "▲" : "—";
  return (
    <span className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded border text-[10px] font-medium ${color}`}>
      {icon} {(delta * 100).toFixed(1)}%
    </span>
  );
}

function PassFailBadge({ pass }: { pass: boolean }) {
  return (
    <span className={`text-[10px] font-semibold px-2 py-0.5 rounded ${pass ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"}`}>
      {pass ? "PASS" : "FAIL"}
    </span>
  );
}

export default function MonitoringPage() {
  const [activeTab, setActiveTab] = useState<"overview" | "regression" | "otel">("overview");
  const [teams, setTeams] = useState<any[]>([]);
  const [teamId, setTeamId] = useState("");
  const [stats, setStats] = useState<any>(null);
  const [evalRuns, setEvalRuns] = useState<any[]>([]);
  const [traces, setTraces] = useState<any[]>([]);
  const [otelStats, setOtelStats] = useState<any>(null);

  // Legacy regression state
  const [runA, setRunA] = useState<string>("");
  const [runB, setRunB] = useState<string>("");
  const [comparison, setComparison] = useState<any>(null);
  const [comparing, setComparing] = useState(false);
  const [runningEval, setRunningEval] = useState(false);
  const [evalMsg, setEvalMsg] = useState("");

  // Golden dataset regression state
  const [regSubTab, setRegSubTab] = useState<"golden" | "run" | "results" | "compare">("golden");
  const [goldenCases, setGoldenCases] = useState<any[]>([]);
  const [selectedCaseIds, setSelectedCaseIds] = useState<Set<string>>(new Set());
  const [expandedCaseId, setExpandedCaseId] = useState<string | null>(null);
  const [regModel, setRegModel] = useState("");
  const [regPromptVer, setRegPromptVer] = useState("v1");
  const [regBaselineRunId, setRegBaselineRunId] = useState("");
  const [regRunning, setRegRunning] = useState(false);
  const [regRunResult, setRegRunResult] = useState<any>(null);
  const [regRuns, setRegRuns] = useState<any[]>([]);
  const [regSelectedRunId, setRegSelectedRunId] = useState("");
  const [regResults, setRegResults] = useState<any>(null);
  const [regCaseDetail, setRegCaseDetail] = useState<any>(null);
  const [regDiff, setRegDiff] = useState<any>(null);
  const [regRCA, setRegRCA] = useState<any>(null);
  const [regDiffRunA, setRegDiffRunA] = useState("");
  const [regDiffRunB, setRegDiffRunB] = useState("");
  const [regDiffCaseId, setRegDiffCaseId] = useState("");
  const [rcaLoading, setRcaLoading] = useState(false);

  // Models & prompt versioning
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [promptVersions, setPromptVersions] = useState<any[]>([]);
  const [showPromptEditor, setShowPromptEditor] = useState(false);
  const [editingPrompts, setEditingPrompts] = useState<Record<string, string>>({});
  const [editingPvLabel, setEditingPvLabel] = useState("");
  const [editingPvDesc, setEditingPvDesc] = useState("");
  const [editingPvStrategy, setEditingPvStrategy] = useState("");
  const [pvSaving, setPvSaving] = useState(false);

  const comparableCases = useMemo(() => {
    const runAData = regRuns.find(r => r.id === regDiffRunA);
    const runBData = regRuns.find(r => r.id === regDiffRunB);
    if (!runAData?.case_ids || !runBData?.case_ids) return [];
    const setB = new Set(runBData.case_ids as string[]);
    const intersection = (runAData.case_ids as string[]).filter(id => setB.has(id));
    return goldenCases.filter(c => intersection.includes(c.id));
  }, [regDiffRunA, regDiffRunB, regRuns, goldenCases]);

  useEffect(() => {
    if (regDiffCaseId && comparableCases.length > 0 && !comparableCases.some(c => c.id === regDiffCaseId)) {
      setRegDiffCaseId("");
    }
  }, [comparableCases, regDiffCaseId]);

  useEffect(() => {
    api.teams.list().then(t => { setTeams(t); if (t.length) setTeamId(t[0].id); });
  }, []);

  useEffect(() => {
    if (!teamId) return;
    loadAll();
  }, [teamId]);

  const loadGolden = useCallback(async () => {
    try {
      const cases = await api.golden.list();
      setGoldenCases(cases);
    } catch { /* ignore */ }
  }, []);

  const loadRegRuns = useCallback(async () => {
    try {
      const runs = await api.regression.runs();
      setRegRuns(runs);
    } catch { /* ignore */ }
  }, []);

  const loadModels = useCallback(async () => {
    try {
      const models = await api.models.list();
      setAvailableModels(models);
    } catch { /* ignore */ }
  }, []);

  const loadPromptVersions = useCallback(async () => {
    try {
      const pvs = await api.promptVersions.list();
      setPromptVersions(pvs);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    if (activeTab === "regression") {
      loadGolden();
      loadRegRuns();
      loadModels();
      loadPromptVersions();
    }
  }, [activeTab, loadGolden, loadRegRuns, loadModels, loadPromptVersions]);

  async function loadAll() {
    const [s, e, t, o] = await Promise.all([
      api.traces.stats(30),
      api.eval.runs(),
      api.traces.list(50),
      api.otel.spanStats(30),
    ]);
    setStats(s);
    setEvalRuns(e);
    setTraces(t);
    setOtelStats(o);
    if (e.length >= 2) {
      setRunA(e[1].id);
      setRunB(e[0].id);
    } else if (e.length === 1) {
      setRunA(e[0].id);
    }
  }

  async function runEvaluation() {
    setRunningEval(true);
    setEvalMsg("Running evaluation scenarios...");
    try {
      await api.eval.run(teamId);
      setEvalMsg("Evaluation complete. Reloading...");
      await loadAll();
      setEvalMsg("");
    } catch (e: any) {
      setEvalMsg(`Error: ${e.message}`);
    } finally {
      setRunningEval(false);
    }
  }

  async function compareRuns() {
    if (!runA || !runB) return;
    setComparing(true);
    try {
      const result = await api.eval.compareRuns(runA, runB);
      setComparison(result);
    } catch {
      setComparison(null);
    } finally {
      setComparing(false);
    }
  }

  async function runRegression() {
    setRegRunning(true);
    setRegRunResult(null);
    try {
      const result = await api.regression.run({
        team_id: teamId,
        case_ids: selectedCaseIds.size > 0 ? Array.from(selectedCaseIds) : undefined,
        model: regModel || undefined,
        prompt_version: regPromptVer,
        baseline_run_id: regBaselineRunId || undefined,
      });
      setRegRunResult(result);
      setRegSubTab("results");
      await loadRegRuns();
    } catch (e: any) {
      setRegRunResult({ error: e.message });
    } finally {
      setRegRunning(false);
    }
  }

  async function loadRegResults(runId: string) {
    setRegSelectedRunId(runId);
    try {
      const r = await api.regression.results(runId);
      setRegResults(r);
      setRegSubTab("results");
    } catch { /* ignore */ }
  }

  async function loadCaseDetail(runId: string, caseId: string) {
    try {
      const d = await api.regression.caseDetail(runId, caseId);
      setRegCaseDetail(d);
    } catch { /* ignore */ }
  }

  async function loadTraceDiff() {
    if (!regDiffRunA || !regDiffRunB || !regDiffCaseId) return;
    try {
      const d = await api.regression.diff(regDiffRunA, regDiffRunB, regDiffCaseId);
      setRegDiff(d);
    } catch { /* ignore */ }
  }

  async function runRCA(runId: string, caseId: string, baselineRunId?: string) {
    setRcaLoading(true);
    try {
      const r = await api.regression.rca(runId, caseId, baselineRunId);
      setRegRCA(r);
    } catch { /* ignore */ }
    setRcaLoading(false);
  }

  function toggleCase(id: string) {
    setSelectedCaseIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  function selectAllCases() {
    if (selectedCaseIds.size === goldenCases.filter(c => c.is_active).length) {
      setSelectedCaseIds(new Set());
    } else {
      setSelectedCaseIds(new Set(goldenCases.filter(c => c.is_active).map(c => c.id)));
    }
  }

  async function openPromptEditor() {
    try {
      const current = await api.promptVersions.current();
      setEditingPrompts(current.agent_prompts || {});
      setEditingPvStrategy(current.team_strategy || "router_decides");
      setEditingPvLabel("");
      setEditingPvDesc("");
      setShowPromptEditor(true);
    } catch { /* ignore */ }
  }

  async function loadPromptFromVersion(pvId: string) {
    const pv = promptVersions.find(p => p.id === pvId);
    if (pv) {
      setEditingPrompts(pv.agent_prompts || {});
      setEditingPvLabel(pv.version_label);
      setEditingPvDesc(pv.description);
      setEditingPvStrategy(pv.team_strategy || "");
      setShowPromptEditor(true);
    }
  }

  async function savePromptVersion() {
    if (!editingPvLabel.trim()) return;
    setPvSaving(true);
    try {
      await api.promptVersions.create({
        version_label: editingPvLabel,
        description: editingPvDesc,
        agent_prompts: editingPrompts,
        team_strategy: editingPvStrategy,
      });
      await loadPromptVersions();
      setRegPromptVer(editingPvLabel);
      setShowPromptEditor(false);
    } catch (e: any) {
      alert(e.message || "Failed to save");
    }
    setPvSaving(false);
  }

  const selectedTeam = teams.find(t => t.id === teamId);
  const totalRuns = stats?.total_runs || traces.length || 0;
  const failures = stats?.failures || traces.filter((t: any) => t.status === "error").length;
  const successRate = totalRuns > 0 ? ((totalRuns - failures) / totalRuns) : 0;
  const avgLatency = stats?.avg_latency_ms || 0;
  const totalCost = stats?.total_cost || 0;

  const qualityData = evalRuns.slice(0, 15).reverse().map((r: any) => ({
    run: (r.prompt_version || r.id?.slice(0, 6)),
    success: +((r.task_success_rate || 0) * 100).toFixed(1),
    toolAcc: +((r.tool_accuracy || 0) * 100).toFixed(1),
    routing: +((r.routing_accuracy || 0) * 100).toFixed(1),
    reasoning: +((r.reasoning_quality || 0) * 100).toFixed(1),
    safety: +((r.safety_compliance || 0) * 100).toFixed(1),
    faithfulness: +((r.faithfulness || 0) * 100).toFixed(1),
  }));

  const latencyData = traces.slice(0, 25).reverse().map((t: any, i: number) => ({
    idx: i + 1,
    latency: +(t.total_latency_ms || 0).toFixed(0),
    tokens: t.total_tokens || 0,
    cost: +(t.total_cost || 0).toFixed(4),
  }));

  const tip = { contentStyle: { background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 } };

  // OTel data transforms
  const spanTypeData = otelStats?.by_type ? Object.entries(otelStats.by_type).map(([type, data]: [string, any]) => ({
    name: type.replace(/_/g, " "),
    value: data.count,
    tokens: data.tokens_in + data.tokens_out,
    cost: data.cost,
    avgLatency: data.avg_latency_ms,
    fill: SPAN_TYPE_COLORS[type] || SPAN_TYPE_COLORS.unknown,
  })) : [];

  const modelData = otelStats?.by_model ? Object.entries(otelStats.by_model).map(([model, data]: [string, any]) => ({
    model: model || "unknown",
    count: data.count,
    tokens_in: data.tokens_in,
    tokens_out: data.tokens_out,
    cost: data.cost,
  })) : [];

  const tokenFlow = otelStats?.token_flow || [];

  const regressionCount = comparison?.comparison
    ? Object.values(comparison.comparison).filter((v: any) => v.regression).length
    : 0;

  return (
    <div className="space-y-5 max-w-6xl">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Monitoring & Observability</h1>
        <div className="flex items-center gap-2">
          <select value={teamId} onChange={e => setTeamId(e.target.value)} className="input !w-auto">
            {teams.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
          </select>
          <button onClick={loadAll} className="btn-secondary">Refresh</button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-[var(--border)]">
        {([
          { id: "overview" as const, label: "Overview" },
          { id: "regression" as const, label: "Regression Testing" },
          { id: "otel" as const, label: "OTel Observability" },
        ]).map(tab => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-all ${
              activeTab === tab.id ? "border-[var(--accent)] text-[var(--accent)]" : "border-transparent text-[var(--text-muted)] hover:text-[var(--text)]"
            }`}>
            {tab.label}
          </button>
        ))}
      </div>

      {/* ═══════════════════ OVERVIEW TAB ═══════════════════ */}
      {activeTab === "overview" && (
        <div className="space-y-4">
          {/* App Info */}
          <div className="card !p-0 overflow-hidden">
            <table className="w-full text-sm">
              <thead><tr className="text-[var(--text-muted)] text-xs bg-[var(--bg)]">
                <th className="text-left px-4 py-2.5">Project</th>
                <th className="text-left px-4 py-2.5">Application</th>
                <th className="text-left px-4 py-2.5">Status</th>
                <th className="text-right px-4 py-2.5">Eval Runs</th>
                <th className="text-right px-4 py-2.5">OTel Spans</th>
              </tr></thead>
              <tbody><tr className="border-t border-[var(--border)]">
                <td className="px-4 py-2.5">SDLC Agent</td>
                <td className="px-4 py-2.5">{selectedTeam?.name || "—"}</td>
                <td className="px-4 py-2.5"><span className="badge bg-[var(--success-light)] text-[var(--success)]">Active</span></td>
                <td className="text-right px-4 py-2.5 font-medium">{evalRuns.length}</td>
                <td className="text-right px-4 py-2.5 font-medium">{otelStats?.total_spans || 0}</td>
              </tr></tbody>
            </table>
          </div>

          {/* Overview KPIs */}
          <div className="grid grid-cols-5 gap-3">
            <Metric label="Total Runs" value={String(totalRuns)} />
            <Metric label="Pass Rate" value={`${(successRate * 100).toFixed(1)}%`}
              accent={successRate >= 0.9 ? "text-emerald-600" : successRate >= 0.7 ? "text-amber-600" : "text-red-600"} />
            <Metric label="Avg Latency" value={`${avgLatency.toFixed(0)}ms`} sub={`P95: ${(stats?.p95_latency_ms || 0).toFixed(0)}ms`} />
            <Metric label="Total Cost" value={`$${totalCost.toFixed(4)}`} sub={`$${totalRuns > 0 ? (totalCost/totalRuns).toFixed(4) : "0"}/run`} />
            <Metric label="Failures" value={String(failures)}
              accent={failures > 0 ? "text-red-600" : "text-emerald-600"} />
          </div>

          {/* Quality Trends + Latency */}
          <div className="grid grid-cols-2 gap-4">
            <div className="card">
              <h2 className="text-sm font-medium mb-1">Quality Trends (Eval Runs)</h2>
              <p className="text-[11px] text-[var(--text-muted)] mb-3">CLASSic framework: Cost, Latency, Accuracy, Stability, Security</p>
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
              ) : <div className="text-sm text-[var(--text-muted)] text-center py-10">Run evaluations to see quality trends</div>}
            </div>

            <div className="card">
              <h2 className="text-sm font-medium mb-1">Latency Distribution</h2>
              <p className="text-[11px] text-[var(--text-muted)] mb-3">Per-request latency (last 25 requests)</p>
              {latencyData.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={latencyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="idx" stroke="var(--text-muted)" fontSize={10} />
                    <YAxis stroke="var(--text-muted)" fontSize={10} />
                    <Tooltip {...tip} />
                    <Bar dataKey="latency" name="Latency (ms)" fill="#2563eb" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : <div className="text-sm text-[var(--text-muted)] text-center py-10">No trace data yet</div>}
            </div>
          </div>

          {/* Percentile Table */}
          <div className="card">
            <h2 className="text-sm font-medium mb-2">Operational Efficiency</h2>
            <table className="w-full text-sm">
              <thead><tr className="text-[var(--text-muted)] text-xs">
                <th className="text-left py-2">Metric</th><th className="text-right">Avg</th><th className="text-right">P50</th><th className="text-right">P95</th><th className="text-right">P99</th>
              </tr></thead>
              <tbody>
                <tr className="border-t border-[var(--border)]">
                  <td className="py-2 font-medium">Latency</td>
                  <td className="text-right">{avgLatency.toFixed(0)}ms</td>
                  <td className="text-right">{(stats?.p50_latency_ms || 0).toFixed(0)}ms</td>
                  <td className="text-right">{(stats?.p95_latency_ms || 0).toFixed(0)}ms</td>
                  <td className="text-right">{(stats?.p99_latency_ms || 0).toFixed(0)}ms</td>
                </tr>
                <tr className="border-t border-[var(--border)]">
                  <td className="py-2 font-medium">Tokens/Run</td>
                  <td className="text-right">{stats?.avg_tokens || 0}</td>
                  <td className="text-right" colSpan={3}>Total: {stats?.total_tokens || 0}</td>
                </tr>
                <tr className="border-t border-[var(--border)]">
                  <td className="py-2 font-medium">Cost/Run</td>
                  <td className="text-right">${totalRuns > 0 ? (totalCost / totalRuns).toFixed(4) : "0.00"}</td>
                  <td className="text-right" colSpan={3}>Total: ${totalCost.toFixed(4)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ═══════════════════ REGRESSION TESTING TAB ═══════════════════ */}
      {activeTab === "regression" && (
        <div className="space-y-4">
          {/* Sub-tabs */}
          <div className="flex gap-1 border-b border-[var(--border)]">
            {([
              { id: "golden" as const, label: "Golden Dataset" },
              { id: "run" as const, label: "Run Tests" },
              { id: "results" as const, label: "Results & Detail" },
              { id: "compare" as const, label: "Trace Diff & RCA" },
            ]).map(tab => (
              <button key={tab.id} onClick={() => setRegSubTab(tab.id)}
                className={`px-3 py-1.5 text-xs font-medium border-b-2 transition-all ${
                  regSubTab === tab.id ? "border-[var(--accent)] text-[var(--accent)]" : "border-transparent text-[var(--text-muted)] hover:text-[var(--text)]"
                }`}>
                {tab.label}
              </button>
            ))}
          </div>

          {/* ── Golden Dataset Manager ── */}
          {regSubTab === "golden" && (
            <div className="space-y-3">
              <div className="card">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h2 className="text-sm font-medium">Golden Test Cases</h2>
                    <p className="text-[11px] text-[var(--text-muted)]">
                      Curated test cases with known-good outputs. Click a row to expand full details.
                    </p>
                  </div>
                  <div className="flex gap-2 items-center">
                    <span className="text-[10px] text-[var(--text-muted)]">{goldenCases.length} cases ({goldenCases.filter(c => c.is_active).length} active)</span>
                    <button onClick={() => api.golden.sync().then(loadGolden)} className="btn-secondary text-xs">
                      Sync from JSON
                    </button>
                  </div>
                </div>
                <div className="space-y-0 max-h-[700px] overflow-y-auto">
                  <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--bg)] text-[10px] text-[var(--text-muted)] uppercase tracking-wide font-medium border-b border-[var(--border)] sticky top-0 z-10">
                    <div className="w-5"></div>
                    <div className="flex-[2]">Name / ID</div>
                    <div className="flex-[3]">Prompt</div>
                    <div className="w-20 text-center">Agent(s)</div>
                    <div className="w-20 text-center">Complexity</div>
                    <div className="w-14 text-center">Ver.</div>
                    <div className="w-14 text-center">Active</div>
                  </div>
                  {goldenCases.map(c => (
                    <div key={c.id}>
                      <div
                        onClick={() => setExpandedCaseId(expandedCaseId === c.id ? null : c.id)}
                        className={`flex items-center gap-2 px-3 py-2 border-b border-[var(--border)] text-xs cursor-pointer transition-colors ${
                          expandedCaseId === c.id ? "bg-[var(--accent)]/5" : "hover:bg-[var(--bg-hover)]"
                        }`}>
                        <div className="w-5 text-[10px] text-[var(--text-muted)]">{expandedCaseId === c.id ? "▼" : "▶"}</div>
                        <div className="flex-[2]">
                          <div className="font-medium">{c.name}</div>
                          <div className="text-[10px] text-[var(--text-muted)] font-mono">{c.id}</div>
                        </div>
                        <div className="flex-[3] truncate text-[var(--text-muted)]">{c.prompt}</div>
                        <div className="w-20 text-center">
                          <span className="px-1.5 py-0.5 rounded bg-violet-50 text-violet-700 text-[10px] border border-violet-200">{c.expected_agent}</span>
                        </div>
                        <div className="w-20 text-center">
                          <span className={`px-1.5 py-0.5 rounded text-[10px] border ${
                            c.complexity === "quick" ? "bg-green-50 text-green-700 border-green-200"
                            : c.complexity === "medium" ? "bg-amber-50 text-amber-700 border-amber-200"
                            : "bg-red-50 text-red-700 border-red-200"
                          }`}>{c.complexity}</span>
                        </div>
                        <div className="w-14 text-center text-[10px]">{c.version}</div>
                        <div className="w-14 text-center">
                          {c.is_active ? (
                            <span className="w-2 h-2 bg-emerald-500 rounded-full inline-block" />
                          ) : (
                            <span className="w-2 h-2 bg-gray-300 rounded-full inline-block" />
                          )}
                        </div>
                      </div>
                      {expandedCaseId === c.id && (
                        <div className="px-4 py-3 border-b border-[var(--border)] bg-[var(--bg)]/50">
                          <div className="grid grid-cols-2 gap-3 text-xs">
                            <div>
                              <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1 font-medium">Prompt</div>
                              <div className="p-2 rounded bg-white/80 border border-[var(--border)] leading-relaxed">{c.prompt}</div>
                            </div>
                            <div>
                              <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1 font-medium">Reference Output</div>
                              <div className="p-2 rounded bg-emerald-50 border border-emerald-200 leading-relaxed">{c.reference_output || "—"}</div>
                            </div>
                            <div>
                              <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1 font-medium">Expected Behavior</div>
                              <div className="space-y-1.5 p-2 rounded bg-white/80 border border-[var(--border)]">
                                <div><span className="text-[var(--text-muted)]">Target Agent:</span> <span className="px-1.5 py-0.5 rounded bg-violet-50 text-violet-700 text-[10px] border border-violet-200">{c.expected_agent}</span></div>
                                <div><span className="text-[var(--text-muted)]">Expected Tools:</span> {(c.expected_tools || []).map((t: string) => (
                                  <span key={t} className="inline-block ml-1 px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 text-[10px] border border-blue-200">{t}</span>
                                ))}</div>
                                <div><span className="text-[var(--text-muted)]">Delegation Pattern:</span> <span className="font-mono text-[10px]">{(c.expected_delegation_pattern || []).join(" → ")}</span></div>
                                <div><span className="text-[var(--text-muted)]">Output Keywords:</span> {(c.expected_output_keywords || []).map((k: string) => (
                                  <span key={k} className="inline-block ml-1 px-1 py-0.5 rounded bg-gray-100 text-gray-600 text-[10px]">{k}</span>
                                ))}</div>
                              </div>
                            </div>
                            <div>
                              <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1 font-medium">Budgets & Thresholds</div>
                              <div className="space-y-1.5 p-2 rounded bg-white/80 border border-[var(--border)]">
                                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                                  <div><span className="text-[var(--text-muted)]">Max LLM Calls:</span> <span className="font-semibold">{c.max_llm_calls}</span></div>
                                  <div><span className="text-[var(--text-muted)]">Max Tool Calls:</span> <span className="font-semibold">{c.max_tool_calls}</span></div>
                                  <div><span className="text-[var(--text-muted)]">Max Tokens:</span> <span className="font-semibold">{c.max_tokens?.toLocaleString()}</span></div>
                                  <div><span className="text-[var(--text-muted)]">Max Latency:</span> <span className="font-semibold">{c.max_latency_ms ? `${(c.max_latency_ms / 1000).toFixed(0)}s` : "—"}</span></div>
                                </div>
                                <div className="border-t border-[var(--border)] pt-1.5 mt-1">
                                  <span className="text-[var(--text-muted)]">Quality Thresholds:</span>
                                  <div className="flex gap-2 mt-1">
                                    {c.quality_thresholds && Object.entries(c.quality_thresholds).map(([k, v]) => (
                                      <span key={k} className="px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 text-[10px] border border-amber-200">
                                        {k}: {((v as number) * 100).toFixed(0)}%
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── Run Configuration Panel ── */}
          {regSubTab === "run" && (
            <div className="space-y-3">
              <div className="card">
                <h2 className="text-sm font-medium mb-3">Run Regression Tests</h2>
                <div className="grid grid-cols-3 gap-3 mb-3">
                  <div>
                    <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">Model</label>
                    <select value={regModel} onChange={e => setRegModel(e.target.value)} className="input text-xs w-full">
                      <option value="">Default (from config)</option>
                      {availableModels.map(m => (
                        <option key={m.id} value={m.id}>
                          {m.name} ({m.provider})
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">Prompt Version</label>
                    <div className="flex gap-1">
                      <select value={regPromptVer} onChange={e => setRegPromptVer(e.target.value)} className="input text-xs flex-1">
                        <option value="v1">v1 (current)</option>
                        {promptVersions.map(pv => (
                          <option key={pv.id} value={pv.version_label}>
                            {pv.version_label}{pv.description ? ` — ${pv.description}` : ""}
                          </option>
                        ))}
                      </select>
                      <button onClick={openPromptEditor} className="btn-secondary !px-2 text-[10px]" title="Edit & create new version">
                        Edit
                      </button>
                    </div>
                  </div>
                  <div>
                    <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">Baseline Run (optional)</label>
                    <select value={regBaselineRunId} onChange={e => setRegBaselineRunId(e.target.value)} className="input text-xs w-full">
                      <option value="">No baseline</option>
                      {regRuns.map(r => (
                        <option key={r.id} value={r.id}>
                          {r.id.slice(0, 8)} — {r.model} ({r.num_cases} cases, {((r.pass_rate || 0) * 100).toFixed(0)}% pass)
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Test Case Selection */}
                <div className="mb-3">
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-[10px] text-[var(--text-muted)] font-medium uppercase tracking-wide">Select Test Cases</label>
                    <div className="flex gap-2 items-center">
                      <span className="text-[10px] text-[var(--text-muted)]">
                        {selectedCaseIds.size > 0 ? `${selectedCaseIds.size} selected` : "All active cases"}
                      </span>
                      <button onClick={selectAllCases} className="text-[10px] text-[var(--accent)] underline">
                        {selectedCaseIds.size === goldenCases.filter(c => c.is_active).length ? "Deselect All" : "Select All"}
                      </button>
                    </div>
                  </div>
                  <div className="border border-[var(--border)] rounded max-h-[240px] overflow-y-auto">
                    {goldenCases.filter(c => c.is_active).map(c => (
                      <label key={c.id}
                        className={`flex items-center gap-2 px-3 py-1.5 text-xs border-b border-[var(--border)] last:border-b-0 cursor-pointer transition-colors ${
                          selectedCaseIds.has(c.id) ? "bg-[var(--accent)]/5" : "hover:bg-[var(--bg-hover)]"
                        }`}>
                        <input type="checkbox" checked={selectedCaseIds.has(c.id)} onChange={() => toggleCase(c.id)}
                          className="h-3.5 w-3.5 rounded border-gray-300 accent-[var(--accent)]" />
                        <span className="font-mono text-[10px] text-[var(--text-muted)] w-24 shrink-0">{c.id}</span>
                        <span className="font-medium flex-1">{c.name}</span>
                        <span className="px-1.5 py-0.5 rounded bg-violet-50 text-violet-700 text-[10px] border border-violet-200">{c.expected_agent}</span>
                        <span className={`px-1.5 py-0.5 rounded text-[10px] border ${
                          c.complexity === "quick" ? "bg-green-50 text-green-700 border-green-200"
                          : c.complexity === "medium" ? "bg-amber-50 text-amber-700 border-amber-200"
                          : "bg-red-50 text-red-700 border-red-200"
                        }`}>{c.complexity}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <button onClick={runRegression} disabled={regRunning} className="btn-primary w-full">
                  {regRunning ? (
                    <span className="flex items-center justify-center gap-1.5">
                      <span className="h-3 w-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Running...
                    </span>
                  ) : `Run ${selectedCaseIds.size > 0 ? selectedCaseIds.size : "All Active"} Cases`}
                </button>
              </div>

              {/* Prompt Version Editor */}
              {showPromptEditor && (
                <div className="card space-y-3 border-2 border-[var(--accent)]">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-medium">Prompt Version Editor</h3>
                    <button onClick={() => setShowPromptEditor(false)} className="text-xs text-[var(--text-muted)] hover:text-[var(--text)]">Close</button>
                  </div>
                  <p className="text-[11px] text-[var(--text-muted)]">
                    Edit agent system prompts and save as a new versioned artifact. Each version is immutable after creation — create a new version for changes.
                  </p>
                  <div className="grid grid-cols-3 gap-3">
                    <div>
                      <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">Version Label</label>
                      <input value={editingPvLabel} onChange={e => setEditingPvLabel(e.target.value)}
                        placeholder="e.g. v2-improved-routing" className="input text-xs w-full" />
                    </div>
                    <div>
                      <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">Description</label>
                      <input value={editingPvDesc} onChange={e => setEditingPvDesc(e.target.value)}
                        placeholder="What changed in this version?" className="input text-xs w-full" />
                    </div>
                    <div>
                      <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">Team Strategy</label>
                      <select value={editingPvStrategy} onChange={e => setEditingPvStrategy(e.target.value)} className="input text-xs w-full">
                        <option value="router_decides">Router Decides</option>
                        <option value="sequential">Sequential</option>
                        <option value="parallel">Parallel</option>
                        <option value="supervisor">Supervisor</option>
                      </select>
                    </div>
                  </div>

                  {/* Load from existing version */}
                  {promptVersions.length > 0 && (
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-[var(--text-muted)]">Load from:</span>
                      {promptVersions.slice(0, 5).map(pv => (
                        <button key={pv.id} onClick={() => loadPromptFromVersion(pv.id)}
                          className="text-[10px] px-2 py-0.5 rounded border border-[var(--border)] hover:bg-[var(--bg-hover)]">
                          {pv.version_label}
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Per-agent prompt editors */}
                  <div className="space-y-2">
                    {Object.entries(editingPrompts).map(([role, prompt]) => (
                      <div key={role}>
                        <label className="text-[10px] text-[var(--text-muted)] block mb-0.5 uppercase tracking-wide">{role} agent prompt</label>
                        <textarea
                          value={prompt}
                          onChange={e => setEditingPrompts(prev => ({ ...prev, [role]: e.target.value }))}
                          rows={4}
                          className="input text-xs w-full font-mono !leading-relaxed resize-y"
                        />
                      </div>
                    ))}
                  </div>

                  <div className="flex items-center justify-between pt-2 border-t border-[var(--border)]">
                    <span className="text-[10px] text-[var(--text-muted)]">
                      {Object.keys(editingPrompts).length} agent prompts loaded
                    </span>
                    <div className="flex gap-2">
                      <button onClick={() => setShowPromptEditor(false)} className="btn-secondary text-xs">Cancel</button>
                      <button onClick={savePromptVersion} disabled={pvSaving || !editingPvLabel.trim()} className="btn-primary text-xs">
                        {pvSaving ? "Saving..." : "Save as New Version"}
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Saved Prompt Versions */}
              {promptVersions.length > 0 && !showPromptEditor && (
                <div className="card">
                  <h3 className="text-xs text-[var(--text-muted)] mb-2">Saved Prompt Versions ({promptVersions.length})</h3>
                  <div className="space-y-1 max-h-[200px] overflow-y-auto">
                    {promptVersions.map(pv => (
                      <div key={pv.id} className="flex items-center justify-between text-xs py-1.5 border-b border-[var(--border)]">
                        <div className="flex items-center gap-2">
                          <span className="font-medium font-mono">{pv.version_label}</span>
                          {pv.description && <span className="text-[var(--text-muted)]">{pv.description}</span>}
                          {pv.team_strategy && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200">{pv.team_strategy}</span>
                          )}
                        </div>
                        <div className="flex items-center gap-2 text-[var(--text-muted)]">
                          <span className="text-[10px]">{Object.keys(pv.agent_prompts || {}).length} prompts</span>
                          {pv.created_at && <span className="text-[10px]">{new Date(pv.created_at).toLocaleDateString()}</span>}
                          <button onClick={() => { setRegPromptVer(pv.version_label); }}
                            className="text-[10px] text-[var(--accent)] underline">Use</button>
                          <button onClick={() => loadPromptFromVersion(pv.id)}
                            className="text-[10px] text-[var(--accent)] underline">View</button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Quick inline result from the last run */}
              {regRunResult && !regRunResult.error && (
                <div className={`card !p-3 border-l-4 ${
                  regRunResult.summary?.failed > 0 ? "border-l-red-500 bg-red-50/30" : "border-l-emerald-500 bg-emerald-50/30"
                }`}>
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium">
                        {regRunResult.summary?.passed}/{regRunResult.summary?.total_cases} passed
                        {regRunResult.summary?.failed > 0 && ` — ${regRunResult.summary.failed} failed`}
                      </div>
                      <div className="text-[11px] text-[var(--text-muted)]">
                        Run: {regRunResult.run_id} | Model: {regRunResult.model} | Prompt: {regRunResult.prompt_version}
                        | Latency: {regRunResult.summary?.avg_latency_ms?.toFixed(0)}ms | Cost: ${regRunResult.summary?.total_cost?.toFixed(4)}
                      </div>
                    </div>
                    <PassFailBadge pass={regRunResult.summary?.failed === 0} />
                  </div>
                  {regRunResult.summary?.regressions && (
                    <div className="mt-2 pt-2 border-t border-[var(--border)] flex gap-4 text-[10px] text-[var(--text-muted)]">
                      <span>Cost regressions: {regRunResult.summary.regressions.cost}</span>
                      <span>Latency regressions: {regRunResult.summary.regressions.latency}</span>
                      <span>Quality regressions: {regRunResult.summary.regressions.quality}</span>
                      <span>Trace regressions: {regRunResult.summary.regressions.trace}</span>
                    </div>
                  )}
                </div>
              )}
              {regRunResult?.error && (
                <div className="card !p-3 border-l-4 border-l-red-500 bg-red-50/30 text-xs text-red-700">
                  Error: {regRunResult.error}
                </div>
              )}

              {/* Regression Run History */}
              <div className="card">
                <h3 className="text-xs text-[var(--text-muted)] mb-2">Regression Run History ({regRuns.length} runs)</h3>
                {regRuns.length > 0 ? (
                  <div className="space-y-1 max-h-[250px] overflow-y-auto">
                    {regRuns.map(r => (
                      <div key={r.id}
                        className="flex items-center justify-between text-xs py-2 border-b border-[var(--border)] cursor-pointer hover:bg-[var(--bg-hover)]"
                        onClick={() => loadRegResults(r.id)}>
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-[var(--text-muted)]">{r.id}</span>
                          <span className="badge bg-[var(--accent-light)] text-[var(--accent)]">{r.prompt_version || "v1"}</span>
                          <span className="text-[var(--text-muted)]">{r.model}</span>
                        </div>
                        <div className="flex items-center gap-3 text-[var(--text-muted)]">
                          <span>{r.num_cases} cases</span>
                          <span className={`font-medium ${r.pass_rate >= 0.7 ? "text-emerald-600" : "text-red-600"}`}>
                            {r.passed}/{r.num_cases} passed
                          </span>
                          <span>${(r.total_cost || 0).toFixed(4)}</span>
                          {r.created_at && <span>{new Date(r.created_at).toLocaleDateString()}</span>}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-sm text-[var(--text-muted)] text-center py-6">No regression runs yet.</div>
                )}
              </div>
            </div>
          )}

          {/* ── Results & Detail View ── */}
          {regSubTab === "results" && (
            <div className="space-y-3">
              {/* Run Selector */}
              <div className="card !p-3 flex items-center gap-3">
                <label className="text-xs text-[var(--text-muted)]">View run:</label>
                <select value={regSelectedRunId} onChange={e => loadRegResults(e.target.value)} className="input !py-1 text-xs flex-1">
                  <option value="">Select a run...</option>
                  {regRuns.map(r => (
                    <option key={r.id} value={r.id}>
                      {r.id} — {r.model} ({r.passed}/{r.num_cases} passed) {r.created_at ? new Date(r.created_at).toLocaleDateString() : ""}
                    </option>
                  ))}
                </select>
              </div>

              {regResults && (
                <>
                  {/* Summary banner */}
                  <div className="grid grid-cols-5 gap-3">
                    <Metric label="Pass Rate" value={`${((regResults.summary?.pass_rate || 0) * 100).toFixed(0)}%`}
                      accent={(regResults.summary?.pass_rate || 0) >= 0.7 ? "text-emerald-600" : "text-red-600"} />
                    <Metric label="Avg Similarity" value={`${((regResults.summary?.avg_semantic_similarity || 0) * 100).toFixed(1)}%`} />
                    <Metric label="Avg Latency" value={`${(regResults.summary?.avg_latency_ms || 0).toFixed(0)}ms`} />
                    <Metric label="Total Cost" value={`$${(regResults.summary?.total_cost || 0).toFixed(4)}`} />
                    <Metric label="Total Tokens" value={String(regResults.summary?.total_tokens || 0)} />
                  </div>

                  {/* Per-case results table */}
                  <div className="card !p-0 overflow-hidden">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-[var(--text-muted)] text-[10px] bg-[var(--bg)] uppercase">
                          <th className="text-left px-3 py-2">Case</th>
                          <th className="text-center px-2 py-2">Agent</th>
                          <th className="text-right px-2 py-2">Similarity</th>
                          <th className="text-right px-2 py-2">DeepEval</th>
                          <th className="text-right px-2 py-2">Latency</th>
                          <th className="text-right px-2 py-2">Cost</th>
                          <th className="text-center px-2 py-2">Trace</th>
                          <th className="text-center px-2 py-2">Quality</th>
                          <th className="text-center px-2 py-2">Status</th>
                          <th className="text-center px-2 py-2">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(regResults.results || []).map((r: any) => (
                          <tr key={r.golden_case_id} className={`border-t border-[var(--border)] ${!r.overall_pass ? "bg-red-50/30" : ""}`}>
                            <td className="px-3 py-2">
                              <div className="font-medium">{r.golden_case_name}</div>
                              <div className="text-[10px] text-[var(--text-muted)]">{r.golden_case_id}</div>
                            </td>
                            <td className="text-center px-2 py-2">
                              <span className="px-1.5 py-0.5 rounded bg-violet-50 text-violet-700 text-[10px] border border-violet-200">{r.actual_agent || "—"}</span>
                            </td>
                            <td className="text-right px-2 py-2 font-mono">
                              <span className={r.semantic_similarity >= 0.7 ? "text-emerald-600" : "text-amber-600"}>
                                {(r.semantic_similarity * 100).toFixed(1)}%
                              </span>
                            </td>
                            <td className="text-right px-2 py-2 font-mono">
                              {(() => {
                                const ds = r.deepeval_scores || {};
                                const vals = Object.values(ds).filter((v): v is number => typeof v === "number");
                                if (!vals.length) return <span className="text-[var(--text-muted)]">—</span>;
                                const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
                                return <span className={avg >= 0.7 ? "text-emerald-600" : avg >= 0.4 ? "text-amber-600" : "text-red-600"}>
                                  {(avg * 100).toFixed(0)}%
                                </span>;
                              })()}
                            </td>
                            <td className="text-right px-2 py-2">{(r.actual_latency_ms || 0).toFixed(0)}ms</td>
                            <td className="text-right px-2 py-2">${(r.actual_cost || 0).toFixed(4)}</td>
                            <td className="text-center px-2 py-2">
                              {r.trace_regression ? (
                                <span className="text-[10px] text-red-600 font-medium">FAIL</span>
                              ) : (
                                <span className="text-[10px] text-emerald-600 font-medium">OK</span>
                              )}
                            </td>
                            <td className="text-center px-2 py-2">
                              {r.quality_regression ? (
                                <span className="text-[10px] text-red-600 font-medium">FAIL</span>
                              ) : (
                                <span className="text-[10px] text-emerald-600 font-medium">OK</span>
                              )}
                            </td>
                            <td className="text-center px-2 py-2">
                              <PassFailBadge pass={r.overall_pass} />
                            </td>
                            <td className="text-center px-2 py-2">
                              <button onClick={() => loadCaseDetail(regSelectedRunId, r.golden_case_id)}
                                className="text-[10px] text-[var(--accent)] underline">Detail</button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}

              {/* Case Detail Drawer */}
              {regCaseDetail && (
                <div className="card space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-medium">
                      Detail: {regCaseDetail.golden_case?.name || regCaseDetail.result?.golden_case_name}
                    </h3>
                    <div className="flex gap-2">
                      <button onClick={() => runRCA(regSelectedRunId, regCaseDetail.result?.golden_case_id, regBaselineRunId)}
                        disabled={rcaLoading} className="btn-secondary text-xs">
                        {rcaLoading ? "Analyzing..." : "Run RCA"}
                      </button>
                      <button onClick={() => setRegCaseDetail(null)} className="text-xs text-[var(--text-muted)]">Close</button>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    {/* Expected vs Actual */}
                    <div>
                      <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1">Expected Output (Reference)</div>
                      <div className="p-2 rounded bg-emerald-50 border border-emerald-200 text-xs max-h-32 overflow-y-auto">
                        {regCaseDetail.golden_case?.reference_output || "—"}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1">Actual Output</div>
                      <div className="p-2 rounded bg-blue-50 border border-blue-200 text-xs max-h-32 overflow-y-auto">
                        {regCaseDetail.result?.actual_output || "—"}
                      </div>
                    </div>
                  </div>

                  {/* Evaluation Scores: G-Eval + DeepEval side by side */}
                  <div className="grid grid-cols-2 gap-3">
                    {/* G-Eval */}
                    {regCaseDetail.result?.quality_scores && Object.keys(regCaseDetail.result.quality_scores).length > 0 && (
                      <div>
                        <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1">G-Eval (LLM-as-Judge with CoT)</div>
                        <div className="space-y-1.5">
                          {Object.entries(regCaseDetail.result.quality_scores).map(([k, v]: [string, any]) => (
                            <div key={k} className="p-2 rounded bg-[var(--bg)] border border-[var(--border)]">
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-xs font-medium capitalize">{k.replace(/_/g, " ")}</span>
                                <span className={`text-xs font-mono font-bold ${Number(v) >= 0.7 ? "text-emerald-600" : Number(v) >= 0.4 ? "text-amber-600" : "text-red-600"}`}>
                                  {(Number(v) * 100).toFixed(0)}%
                                </span>
                              </div>
                              {regCaseDetail.result?.eval_reasoning?.[k] && (
                                <div className="text-[10px] text-[var(--text-muted)] italic">
                                  {regCaseDetail.result.eval_reasoning[k]}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* DeepEval Agentic Metrics */}
                    <div>
                      <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1">DeepEval Agentic Metrics</div>
                      {regCaseDetail.result?.deepeval_scores && Object.keys(regCaseDetail.result.deepeval_scores).length > 0 ? (
                        <div className="space-y-2">
                          {DEEPEVAL_METRICS.map(({ key, label, reasonKey, description }) => {
                            const ds = regCaseDetail.result.deepeval_scores;
                            const val = ds[key];
                            if (val === undefined && val === null) return null;
                            const numVal = typeof val === "number" ? val : typeof val === "string" ? parseFloat(val) : 0;
                            const reason = ds[reasonKey] || "";
                            return (
                              <div key={key} className="rounded bg-[var(--bg)] border border-[var(--border)] overflow-hidden">
                                <div className="flex items-center justify-between p-2">
                                  <div className="flex-1 min-w-0">
                                    <span className="text-xs font-medium">{label}</span>
                                    <span className="text-[10px] text-[var(--text-muted)] ml-2">{description}</span>
                                  </div>
                                  <div className="flex items-center gap-2 ml-2 shrink-0">
                                    <div className="w-20 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                                      <div className="h-full rounded-full"
                                        style={{
                                          width: `${Math.min(100, numVal * 100)}%`,
                                          backgroundColor: numVal >= 0.7 ? "#059669" : numVal >= 0.4 ? "#ca8a04" : "#dc2626",
                                        }} />
                                    </div>
                                    <span className={`text-xs font-mono font-bold min-w-[3rem] text-right ${
                                      numVal >= 0.7 ? "text-emerald-600" : numVal >= 0.4 ? "text-amber-600" : "text-red-600"
                                    }`}>
                                      {(numVal * 100).toFixed(0)}%
                                    </span>
                                  </div>
                                </div>
                                {reason && (
                                  <div className="px-2 pb-2">
                                    <div className="text-[10px] text-[var(--text-muted)] bg-[var(--bg-secondary)] rounded p-2 leading-relaxed whitespace-pre-wrap">
                                      <span className="font-semibold text-[var(--text-secondary)]">Reasoning: </span>{reason}
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <div className="text-[10px] text-[var(--text-muted)] p-3 bg-[var(--bg)] rounded border border-[var(--border)]">
                          DeepEval metrics not available for this run. Ensure DeepEval is configured.
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Trace Structural Assertions — with expected vs actual */}
                  {regCaseDetail.result?.trace_assertions && Object.keys(regCaseDetail.result.trace_assertions).length > 0 && (
                    <div>
                      <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1">Trace Structural Assertions</div>
                      <div className="card !p-0 overflow-hidden">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="text-[10px] text-[var(--text-muted)] bg-[var(--bg)] uppercase">
                              <th className="text-left px-3 py-1.5">Assertion</th>
                              <th className="text-left px-3 py-1.5">Expected</th>
                              <th className="text-left px-3 py-1.5">Actual</th>
                              <th className="text-left px-3 py-1.5">Reason</th>
                              <th className="text-center px-3 py-1.5">Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(regCaseDetail.result.trace_assertions).map(([k, v]: [string, any]) => (
                              <tr key={k} className={`border-t border-[var(--border)] ${!v.passed ? "bg-red-50/40" : ""}`}>
                                <td className="px-3 py-2 font-medium capitalize">{k.replace(/_/g, " ")}</td>
                                <td className="px-3 py-2 text-[10px] font-mono text-[var(--text-muted)]">
                                  {Array.isArray(v.expected) ? v.expected.join(", ") : String(v.expected ?? "—")}
                                </td>
                                <td className="px-3 py-2 text-[10px] font-mono">
                                  <span className={!v.passed ? "text-red-600" : "text-emerald-600"}>
                                    {Array.isArray(v.actual) ? v.actual.join(", ") : String(v.actual ?? "—")}
                                  </span>
                                </td>
                                <td className="px-3 py-2 text-[10px] text-[var(--text-muted)]">{v.reason}</td>
                                <td className="text-center px-3 py-2"><PassFailBadge pass={v.passed} /></td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Cost/Latency Detail */}
                  <div className="grid grid-cols-6 gap-2">
                    {[
                      { label: "LLM Calls", val: regCaseDetail.result?.actual_llm_calls, budget: regCaseDetail.golden_case?.max_llm_calls },
                      { label: "Tool Calls", val: regCaseDetail.result?.actual_tool_calls, budget: regCaseDetail.golden_case?.max_tool_calls },
                      { label: "Tokens In", val: regCaseDetail.result?.actual_tokens_in },
                      { label: "Tokens Out", val: regCaseDetail.result?.actual_tokens_out },
                      { label: "Latency", val: `${(regCaseDetail.result?.actual_latency_ms || 0).toFixed(0)}ms` },
                      { label: "Cost", val: `$${(regCaseDetail.result?.actual_cost || 0).toFixed(4)}` },
                    ].map(({ label, val, budget }) => (
                      <div key={label} className="p-2 rounded bg-[var(--bg)] border border-[var(--border)] text-center">
                        <div className="text-sm font-semibold">{val ?? 0}</div>
                        <div className="text-[10px] text-[var(--text-muted)]">{label}</div>
                        {budget !== undefined && <div className="text-[9px] text-[var(--text-muted)]">Budget: {budget}</div>}
                      </div>
                    ))}
                  </div>

                  {/* Execution Trace */}
                  {regCaseDetail.result?.full_trace && regCaseDetail.result.full_trace.length > 0 && (
                    <div>
                      <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1">Execution Trace</div>
                      <div className="space-y-1 max-h-48 overflow-y-auto">
                        {regCaseDetail.result.full_trace.map((step: any, i: number) => (
                          <div key={i} className="flex items-center gap-2 text-[10px] p-1.5 rounded bg-[var(--bg)] border border-[var(--border)]">
                            <span className="w-5 text-center font-mono text-[var(--text-muted)]">{i + 1}</span>
                            <span className="px-1.5 py-0.5 rounded text-[9px] font-medium"
                              style={{ background: `${SPAN_TYPE_COLORS[step.step] || SPAN_TYPE_COLORS.unknown}22`, color: SPAN_TYPE_COLORS[step.step] || SPAN_TYPE_COLORS.unknown }}>
                              {step.step}
                            </span>
                            {step.agent && <span className="font-medium">{step.agent}</span>}
                            {step.selected_agent && <span className="font-medium">{step.selected_agent}</span>}
                            {step.tool_calls?.length > 0 && (
                              <span className="text-[var(--text-muted)]">
                                Tools: {step.tool_calls.map((tc: any) => tc.tool).join(", ")}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* RCA Results */}
                  {regRCA && (
                    <div className="border-t border-[var(--border)] pt-3">
                      <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1">Root Cause Analysis</div>
                      <div className={`p-3 rounded border ${
                        regRCA.root_cause_category === "unknown" ? "bg-gray-50 border-gray-200"
                        : "bg-amber-50 border-amber-200"
                      }`}>
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-xs font-semibold px-2 py-0.5 rounded bg-amber-100 text-amber-800">
                            {(regRCA.root_cause_category || "unknown").replace(/_/g, " ").toUpperCase()}
                          </span>
                          {regRCA.divergence_point?.description && (
                            <span className="text-[10px] text-[var(--text-muted)]">
                              Step {regRCA.divergence_point.step_index ?? "?"}: {regRCA.divergence_point.description}
                            </span>
                          )}
                        </div>
                        <div className="text-xs mb-2">{regRCA.analysis}</div>
                        {regRCA.recommendations?.length > 0 && (
                          <div>
                            <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1">Recommendations</div>
                            <ul className="list-disc list-inside text-xs space-y-0.5">
                              {regRCA.recommendations.map((rec: string, i: number) => (
                                <li key={i}>{rec}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* ── Trace Diff & RCA ── */}
          {regSubTab === "compare" && (
            <div className="space-y-3">
              <div className="card">
                <h2 className="text-sm font-medium mb-3">Trace-Based Comparison</h2>
                <p className="text-[11px] text-[var(--text-muted)] mb-3">
                  Compare execution traces side-by-side for the same golden test case across two regression runs.
                  Only test cases present in both runs are shown for selection.
                </p>
                <div className="grid grid-cols-3 gap-3 mb-3">
                  <div>
                    <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">Baseline Run</label>
                    <select value={regDiffRunA} onChange={e => setRegDiffRunA(e.target.value)} className="input text-xs w-full">
                      <option value="">Select baseline...</option>
                      {regRuns.map(r => (
                        <option key={r.id} value={r.id}>
                          {r.id.slice(0, 8)} — {r.model} ({r.passed}/{r.num_cases}) {r.created_at ? new Date(r.created_at).toLocaleDateString() : ""}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">Candidate Run</label>
                    <select value={regDiffRunB} onChange={e => setRegDiffRunB(e.target.value)} className="input text-xs w-full">
                      <option value="">Select candidate...</option>
                      {regRuns.map(r => (
                        <option key={r.id} value={r.id}>
                          {r.id.slice(0, 8)} — {r.model} ({r.passed}/{r.num_cases}) {r.created_at ? new Date(r.created_at).toLocaleDateString() : ""}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">Test Case</label>
                    <select value={regDiffCaseId} onChange={e => setRegDiffCaseId(e.target.value)}
                      className="input text-xs w-full"
                      disabled={!regDiffRunA || !regDiffRunB}>
                      <option value="">
                        {!regDiffRunA || !regDiffRunB
                          ? "Select both runs first..."
                          : comparableCases.length === 0
                            ? "No common test cases"
                            : `Select case (${comparableCases.length} common)...`}
                      </option>
                      {comparableCases.map(c => (
                        <option key={c.id} value={c.id}>{c.id} — {c.name}</option>
                      ))}
                    </select>
                    {regDiffRunA && regDiffRunB && comparableCases.length === 0 && (
                      <p className="text-[10px] text-red-500 mt-0.5">These runs have no test cases in common.</p>
                    )}
                  </div>
                </div>
                <button onClick={loadTraceDiff}
                  disabled={!regDiffRunA || !regDiffRunB || !regDiffCaseId}
                  className="btn-primary text-xs">
                  Compare Traces
                </button>
              </div>

              {regDiff && (
                <div className="space-y-3">
                  {/* Cost/Latency delta summary */}
                  {regDiff.cost_diff && (
                    <div className="grid grid-cols-6 gap-2">
                      {[
                        { label: "Tokens In Delta", val: regDiff.cost_diff.tokens_in_delta },
                        { label: "Tokens Out Delta", val: regDiff.cost_diff.tokens_out_delta },
                        { label: "Cost Delta", val: `$${(regDiff.cost_diff.cost_delta || 0).toFixed(4)}` },
                        { label: "Latency Delta", val: `${(regDiff.cost_diff.latency_delta || 0).toFixed(0)}ms` },
                        { label: "LLM Calls Delta", val: regDiff.cost_diff.llm_calls_delta },
                        { label: "Tool Calls Delta", val: regDiff.cost_diff.tool_calls_delta },
                      ].map(({ label, val }) => (
                        <div key={label} className="p-2 rounded bg-[var(--bg)] border border-[var(--border)] text-center text-xs">
                          <div className={`font-semibold ${typeof val === "number" && val > 0 ? "text-red-600" : typeof val === "number" && val < 0 ? "text-emerald-600" : ""}`}>
                            {typeof val === "number" ? (val > 0 ? `+${val}` : val) : val}
                          </div>
                          <div className="text-[10px] text-[var(--text-muted)]">{label}</div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Side-by-side output comparison */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="card">
                      <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1">Baseline Output (Run A)</div>
                      <div className="p-2 bg-[var(--bg)] rounded text-xs max-h-32 overflow-y-auto">
                        {regDiff.run_a?.actual_output || "—"}
                      </div>
                      <div className="mt-2 text-[10px] text-[var(--text-muted)]">
                        Agent: {regDiff.run_a?.actual_agent} | Tools: {(regDiff.run_a?.actual_tools || []).join(", ")}
                        | Similarity: {((regDiff.run_a?.semantic_similarity || 0) * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="card">
                      <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1">Candidate Output (Run B)</div>
                      <div className="p-2 bg-[var(--bg)] rounded text-xs max-h-32 overflow-y-auto">
                        {regDiff.run_b?.actual_output || "—"}
                      </div>
                      <div className="mt-2 text-[10px] text-[var(--text-muted)]">
                        Agent: {regDiff.run_b?.actual_agent} | Tools: {(regDiff.run_b?.actual_tools || []).join(", ")}
                        | Similarity: {((regDiff.run_b?.semantic_similarity || 0) * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  {/* Combined Quality + DeepEval Radar Chart */}
                  {(regDiff.run_a?.quality_scores || regDiff.run_a?.deepeval_scores) && (
                    <div className="card">
                      <h3 className="text-xs text-[var(--text-muted)] mb-2">G-Eval + DeepEval Scores: Baseline vs Candidate</h3>
                      <ResponsiveContainer width="100%" height={280}>
                        <RadarChart data={[
                          ...Object.keys({...regDiff.run_a?.quality_scores, ...regDiff.run_b?.quality_scores}).map(k => ({
                            metric: k.replace(/_/g, " "),
                            Baseline: Number(regDiff.run_a?.quality_scores?.[k] || 0) * 100,
                            Candidate: Number(regDiff.run_b?.quality_scores?.[k] || 0) * 100,
                          })),
                          ...DEEPEVAL_METRICS.map(({ key, label }) => ({
                            metric: label,
                            Baseline: Number(regDiff.run_a?.deepeval_scores?.[key] || 0) * 100,
                            Candidate: Number(regDiff.run_b?.deepeval_scores?.[key] || 0) * 100,
                          })),
                        ]}>
                          <PolarGrid stroke="var(--border)" />
                          <PolarAngleAxis dataKey="metric" tick={{ fontSize: 8 }} />
                          <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 9 }} />
                          <Radar name="Baseline" dataKey="Baseline" stroke="#6b7280" fill="#6b7280" fillOpacity={0.2} />
                          <Radar name="Candidate" dataKey="Candidate" stroke="#2563eb" fill="#2563eb" fillOpacity={0.2} />
                          <Legend wrapperStyle={{ fontSize: 10 }} />
                          <Tooltip />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {/* Trace Diff Timeline */}
                  {regDiff.trace_diff && regDiff.trace_diff.length > 0 && (
                    <div className="card">
                      <h3 className="text-xs text-[var(--text-muted)] mb-2">Step-by-Step Trace Diff</h3>
                      <div className="space-y-1 max-h-[300px] overflow-y-auto">
                        {regDiff.trace_diff.map((d: any, i: number) => (
                          <div key={i} className={`p-2 rounded border text-[10px] ${
                            d.diverged ? "bg-amber-50 border-amber-200" : "bg-[var(--bg)] border-[var(--border)]"
                          }`}>
                            <div className="flex items-center gap-2 mb-1">
                              <span className="font-mono font-bold w-5">#{d.step}</span>
                              <span className={`px-1.5 py-0.5 rounded font-medium ${
                                d.status === "match" ? "bg-emerald-100 text-emerald-700"
                                : d.status === "diverged" ? "bg-amber-100 text-amber-700"
                                : "bg-red-100 text-red-700"
                              }`}>
                                {d.status.toUpperCase()}
                              </span>
                              {d.reasons?.map((r: string, ri: number) => (
                                <span key={ri} className="text-amber-700">{r}</span>
                              ))}
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                              <div className="text-[var(--text-muted)]">
                                <span className="font-medium text-[var(--text)]">Baseline:</span>{" "}
                                {d.baseline ? `${d.baseline.type}${d.baseline.agent ? ` (${d.baseline.agent})` : ""}${d.baseline.tools ? ` [${d.baseline.tools.join(", ")}]` : ""}` : "—"}
                              </div>
                              <div className="text-[var(--text-muted)]">
                                <span className="font-medium text-[var(--text)]">Candidate:</span>{" "}
                                {d.failing ? `${d.failing.type}${d.failing.agent ? ` (${d.failing.agent})` : ""}${d.failing.tools ? ` [${d.failing.tools.join(", ")}]` : ""}` : "—"}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Run RCA from diff view */}
                  <div className="card !p-3 flex items-center justify-between">
                    <span className="text-xs text-[var(--text-muted)]">Run LLM-powered root cause analysis on the candidate run</span>
                    <button onClick={() => runRCA(regDiffRunB, regDiffCaseId, regDiffRunA)}
                      disabled={rcaLoading} className="btn-primary text-xs">
                      {rcaLoading ? "Analyzing..." : "Run Root Cause Analysis"}
                    </button>
                  </div>

                  {regRCA && (
                    <div className={`card border-l-4 ${
                      regRCA.root_cause_category === "unknown" ? "border-l-gray-400" : "border-l-amber-400"
                    }`}>
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-xs font-semibold px-2 py-0.5 rounded bg-amber-100 text-amber-800 uppercase">
                          {(regRCA.root_cause_category || "unknown").replace(/_/g, " ")}
                        </span>
                        {regRCA.divergence_point?.step_index != null && (
                          <span className="text-[10px] text-[var(--text-muted)]">
                            First divergence at step {regRCA.divergence_point.step_index}
                          </span>
                        )}
                      </div>
                      <div className="text-xs mb-2">{regRCA.analysis}</div>
                      {regRCA.recommendations?.length > 0 && (
                        <div>
                          <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1">Recommendations</div>
                          <ul className="list-disc list-inside text-xs space-y-0.5">
                            {regRCA.recommendations.map((r: string, i: number) => (
                              <li key={i}>{r}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Detailed Baseline vs Candidate Comparison */}
              {regDiff && (
                <div className="space-y-3">
                  {/* DeepEval Scores Comparison Table */}
                  {(regDiff.run_a?.deepeval_scores || regDiff.run_b?.deepeval_scores) && (
                    <div className="card">
                      <h3 className="text-xs text-[var(--text-muted)] mb-2">DeepEval Agentic Metrics: Baseline vs Candidate</h3>
                      <div className="card !p-0 overflow-hidden">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="text-[10px] text-[var(--text-muted)] bg-[var(--bg)] uppercase">
                              <th className="text-left px-3 py-1.5">Metric</th>
                              <th className="text-right px-3 py-1.5">Baseline</th>
                              <th className="text-right px-3 py-1.5">Candidate</th>
                              <th className="text-right px-3 py-1.5">Delta</th>
                              <th className="text-center px-3 py-1.5">Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {DEEPEVAL_METRICS.map(({ key, label, reasonKey }) => {
                              const aVal = Number(regDiff.run_a?.deepeval_scores?.[key] || 0);
                              const bVal = Number(regDiff.run_b?.deepeval_scores?.[key] || 0);
                              const delta = bVal - aVal;
                              const aReason = regDiff.run_a?.deepeval_scores?.[reasonKey] || "";
                              const bReason = regDiff.run_b?.deepeval_scores?.[reasonKey] || "";
                              return (
                                <React.Fragment key={key}>
                                  <tr className={`border-t border-[var(--border)] ${delta < -0.1 ? "bg-red-50/40" : ""}`}>
                                    <td className="px-3 py-1.5 font-medium">{label}</td>
                                    <td className="text-right px-3 py-1.5 font-mono">{(aVal * 100).toFixed(0)}%</td>
                                    <td className="text-right px-3 py-1.5 font-mono">{(bVal * 100).toFixed(0)}%</td>
                                    <td className="text-right px-3 py-1.5"><DeltaBadge delta={delta} /></td>
                                    <td className="text-center px-3 py-1.5"><PassFailBadge pass={delta >= -0.1} /></td>
                                  </tr>
                                  {(aReason || bReason) && (
                                    <tr className="border-t border-[var(--border)] border-dashed">
                                      <td colSpan={5} className="px-3 py-1.5">
                                        <div className="grid grid-cols-2 gap-2 text-[10px]">
                                          {aReason && (
                                            <div className="bg-[var(--bg)] rounded p-1.5 leading-relaxed">
                                              <span className="font-semibold text-[var(--text-muted)]">Baseline: </span>
                                              <span className="text-[var(--text-secondary)]">{aReason}</span>
                                            </div>
                                          )}
                                          {bReason && (
                                            <div className="bg-[var(--bg)] rounded p-1.5 leading-relaxed">
                                              <span className="font-semibold text-[var(--text-muted)]">Candidate: </span>
                                              <span className="text-[var(--text-secondary)]">{bReason}</span>
                                            </div>
                                          )}
                                        </div>
                                      </td>
                                    </tr>
                                  )}
                                </React.Fragment>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* G-Eval Quality Scores Comparison */}
                  {(regDiff.run_a?.quality_scores || regDiff.run_b?.quality_scores) && (
                    <div className="card">
                      <h3 className="text-xs text-[var(--text-muted)] mb-2">G-Eval Quality Scores: Baseline vs Candidate</h3>
                      <div className="card !p-0 overflow-hidden">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="text-[10px] text-[var(--text-muted)] bg-[var(--bg)] uppercase">
                              <th className="text-left px-3 py-1.5">Criterion</th>
                              <th className="text-right px-3 py-1.5">Baseline</th>
                              <th className="text-right px-3 py-1.5">Candidate</th>
                              <th className="text-right px-3 py-1.5">Delta</th>
                              <th className="text-left px-3 py-1.5">Baseline Reasoning</th>
                              <th className="text-left px-3 py-1.5">Candidate Reasoning</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.keys({...regDiff.run_a?.quality_scores, ...regDiff.run_b?.quality_scores}).map(k => {
                              const aVal = Number(regDiff.run_a?.quality_scores?.[k] || 0);
                              const bVal = Number(regDiff.run_b?.quality_scores?.[k] || 0);
                              const delta = bVal - aVal;
                              return (
                                <tr key={k} className={`border-t border-[var(--border)] ${delta < -0.1 ? "bg-red-50/40" : ""}`}>
                                  <td className="px-3 py-1.5 font-medium capitalize">{k.replace(/_/g, " ")}</td>
                                  <td className="text-right px-3 py-1.5 font-mono">{(aVal * 100).toFixed(0)}%</td>
                                  <td className="text-right px-3 py-1.5 font-mono">{(bVal * 100).toFixed(0)}%</td>
                                  <td className="text-right px-3 py-1.5"><DeltaBadge delta={delta} /></td>
                                  <td className="px-3 py-1.5 text-[10px] text-[var(--text-muted)] max-w-[200px] truncate">
                                    {regDiff.run_a?.eval_reasoning?.[k] || "—"}
                                  </td>
                                  <td className="px-3 py-1.5 text-[10px] text-[var(--text-muted)] max-w-[200px] truncate">
                                    {regDiff.run_b?.eval_reasoning?.[k] || "—"}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Operational Metrics Comparison */}
                  <div className="card">
                    <h3 className="text-xs text-[var(--text-muted)] mb-2">Operational Metrics: Baseline vs Candidate</h3>
                    <div className="card !p-0 overflow-hidden">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="text-[10px] text-[var(--text-muted)] bg-[var(--bg)] uppercase">
                            <th className="text-left px-3 py-1.5">Metric</th>
                            <th className="text-right px-3 py-1.5">Baseline</th>
                            <th className="text-right px-3 py-1.5">Candidate</th>
                            <th className="text-right px-3 py-1.5">Delta</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[
                            { label: "Agent Used", a: regDiff.run_a?.actual_agent, b: regDiff.run_b?.actual_agent, isText: true },
                            { label: "Tools Used", a: (regDiff.run_a?.actual_tools || []).join(", "), b: (regDiff.run_b?.actual_tools || []).join(", "), isText: true },
                            { label: "Delegation", a: (regDiff.run_a?.actual_delegation_pattern || []).join(" → "), b: (regDiff.run_b?.actual_delegation_pattern || []).join(" → "), isText: true },
                            { label: "Model", a: regDiff.run_a?.model_used, b: regDiff.run_b?.model_used, isText: true },
                            { label: "Sem. Similarity", a: regDiff.run_a?.semantic_similarity, b: regDiff.run_b?.semantic_similarity, fmt: (v: number) => `${(v * 100).toFixed(1)}%` },
                            { label: "LLM Calls", a: regDiff.run_a?.actual_llm_calls, b: regDiff.run_b?.actual_llm_calls },
                            { label: "Tool Calls", a: regDiff.run_a?.actual_tool_calls, b: regDiff.run_b?.actual_tool_calls },
                            { label: "Tokens In", a: regDiff.run_a?.actual_tokens_in, b: regDiff.run_b?.actual_tokens_in },
                            { label: "Tokens Out", a: regDiff.run_a?.actual_tokens_out, b: regDiff.run_b?.actual_tokens_out },
                            { label: "Latency", a: regDiff.run_a?.actual_latency_ms, b: regDiff.run_b?.actual_latency_ms, fmt: (v: number) => `${v.toFixed(0)}ms` },
                            { label: "Cost", a: regDiff.run_a?.actual_cost, b: regDiff.run_b?.actual_cost, fmt: (v: number) => `$${v.toFixed(4)}` },
                          ].map(({ label, a, b, isText, fmt }) => {
                            const fmtFn = fmt || ((v: any) => String(v ?? "—"));
                            const delta = !isText && typeof a === "number" && typeof b === "number" ? b - a : null;
                            return (
                              <tr key={label} className="border-t border-[var(--border)]">
                                <td className="px-3 py-1.5 font-medium">{label}</td>
                                <td className="text-right px-3 py-1.5 font-mono text-[var(--text-muted)]">{isText ? (a || "—") : fmtFn(a || 0)}</td>
                                <td className="text-right px-3 py-1.5 font-mono">{isText ? (b || "—") : fmtFn(b || 0)}</td>
                                <td className="text-right px-3 py-1.5">
                                  {delta !== null ? (
                                    <span className={`text-[10px] font-mono ${delta > 0 ? "text-red-600" : delta < 0 ? "text-emerald-600" : ""}`}>
                                      {delta > 0 ? "+" : ""}{label === "Cost" ? `$${delta.toFixed(4)}` : label === "Latency" ? `${delta.toFixed(0)}ms` : label === "Sem. Similarity" ? `${(delta * 100).toFixed(1)}%` : delta}
                                    </span>
                                  ) : (
                                    isText && a !== b ? <span className="text-[10px] text-amber-600">Changed</span> : <span className="text-[10px] text-[var(--text-muted)]">—</span>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Trace Assertions Comparison */}
                  {(regDiff.run_a?.trace_assertions || regDiff.run_b?.trace_assertions) && (
                    <div className="card">
                      <h3 className="text-xs text-[var(--text-muted)] mb-2">Trace Assertions: Baseline vs Candidate</h3>
                      <div className="card !p-0 overflow-hidden">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="text-[10px] text-[var(--text-muted)] bg-[var(--bg)] uppercase">
                              <th className="text-left px-3 py-1.5">Assertion</th>
                              <th className="text-center px-3 py-1.5">Baseline</th>
                              <th className="text-left px-3 py-1.5">Baseline Detail</th>
                              <th className="text-center px-3 py-1.5">Candidate</th>
                              <th className="text-left px-3 py-1.5">Candidate Detail</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.keys({...regDiff.run_a?.trace_assertions, ...regDiff.run_b?.trace_assertions}).map(k => {
                              const aV = regDiff.run_a?.trace_assertions?.[k] || {};
                              const bV = regDiff.run_b?.trace_assertions?.[k] || {};
                              return (
                                <tr key={k} className={`border-t border-[var(--border)] ${!bV.passed ? "bg-red-50/40" : ""}`}>
                                  <td className="px-3 py-1.5 font-medium capitalize">{k.replace(/_/g, " ")}</td>
                                  <td className="text-center px-3 py-1.5">{aV.passed !== undefined ? <PassFailBadge pass={aV.passed} /> : "—"}</td>
                                  <td className="px-3 py-1.5 text-[10px] text-[var(--text-muted)]">
                                    {aV.reason || "—"}
                                  </td>
                                  <td className="text-center px-3 py-1.5">{bV.passed !== undefined ? <PassFailBadge pass={bV.passed} /> : "—"}</td>
                                  <td className="px-3 py-1.5 text-[10px] text-[var(--text-muted)]">
                                    {bV.reason || "—"}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ═══════════════════ OTEL OBSERVABILITY TAB ═══════════════════ */}
      {activeTab === "otel" && (
        <div className="space-y-4">
          {/* OTel KPIs */}
          <div className="grid grid-cols-5 gap-3">
            <Metric label="Total Spans" value={String(otelStats?.total_spans || 0)} sub="Across all traces" />
            <Metric label="Total Traces" value={String(otelStats?.total_traces || 0)} />
            <Metric label="Error Spans" value={String(otelStats?.error_spans || 0)}
              accent={otelStats?.error_spans > 0 ? "text-red-600" : "text-emerald-600"} />
            <Metric label="Span Types" value={String(spanTypeData.length)} sub="routing, tool, llm, etc." />
            <Metric label="Models Used" value={String(modelData.length)} sub={modelData.map(m => m.model).join(", ").slice(0, 30)} />
          </div>

          {/* Span Type Breakdown + Model Breakdown */}
          <div className="grid grid-cols-2 gap-4">
            <div className="card">
              <h3 className="text-xs text-[var(--text-muted)] mb-2">Span Type Distribution (GenAI Semantic Conventions)</h3>
              {spanTypeData.length > 0 ? (
                <div className="flex items-center gap-4">
                  <ResponsiveContainer width="50%" height={200}>
                    <PieChart>
                      <Pie data={spanTypeData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                        outerRadius={80} label={({ name, percent }: { name?: string; percent?: number }) => `${name ?? ""} ${((percent ?? 0) * 100).toFixed(0)}%`}
                        labelLine={false} fontSize={9}>
                        {spanTypeData.map((entry, i) => (
                          <Cell key={i} fill={entry.fill} />
                        ))}
                      </Pie>
                      <Tooltip {...tip} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="flex-1 space-y-1.5">
                    {spanTypeData.map(s => (
                      <div key={s.name} className="flex items-center justify-between text-xs py-1 border-b border-[var(--border)]">
                        <div className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded-sm" style={{ background: s.fill }} />
                          <span className="capitalize">{s.name}</span>
                        </div>
                        <div className="flex gap-3 text-[var(--text-muted)]">
                          <span>{s.value} spans</span>
                          <span>{s.tokens} tok</span>
                          <span>${s.cost.toFixed(4)}</span>
                          <span>{s.avgLatency.toFixed(0)}ms</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : <div className="text-sm text-[var(--text-muted)] text-center py-10">No span data yet</div>}
            </div>

            <div className="card">
              <h3 className="text-xs text-[var(--text-muted)] mb-2">Model Usage (gen_ai.request.model)</h3>
              {modelData.length > 0 ? (
                <div className="space-y-2">
                  <ResponsiveContainer width="100%" height={140}>
                    <BarChart data={modelData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" stroke="var(--text-muted)" fontSize={10} />
                      <YAxis type="category" dataKey="model" stroke="var(--text-muted)" fontSize={9} width={100} />
                      <Tooltip {...tip} />
                      <Bar dataKey="count" name="Spans" fill="#6366f1" radius={[0, 3, 3, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                  <div className="space-y-1">
                    {modelData.map(m => (
                      <div key={m.model} className="flex items-center justify-between text-xs border-b border-[var(--border)] py-1">
                        <span className="font-mono">{m.model}</span>
                        <div className="flex gap-3 text-[var(--text-muted)]">
                          <span>In: {m.tokens_in}</span>
                          <span>Out: {m.tokens_out}</span>
                          <span>${m.cost.toFixed(4)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : <div className="text-sm text-[var(--text-muted)] text-center py-10">No model data</div>}
            </div>
          </div>

          {/* Token Flow + Cost Over Time */}
          {tokenFlow.length > 0 && (
            <div className="grid grid-cols-2 gap-4">
              <div className="card">
                <h3 className="text-xs text-[var(--text-muted)] mb-2">Token Flow (gen_ai.usage.*)</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={tokenFlow}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="id" stroke="var(--text-muted)" fontSize={10} />
                    <YAxis stroke="var(--text-muted)" fontSize={10} />
                    <Tooltip {...tip} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Area type="monotone" dataKey="tokens_in" name="Input Tokens" stroke="#2563eb" fill="#2563eb" fillOpacity={0.15} />
                    <Area type="monotone" dataKey="tokens_out" name="Output Tokens" stroke="#7c3aed" fill="#7c3aed" fillOpacity={0.15} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="card">
                <h3 className="text-xs text-[var(--text-muted)] mb-2">Cost per Request (DBSpanProcessor Cost Estimation)</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={tokenFlow}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="id" stroke="var(--text-muted)" fontSize={10} />
                    <YAxis stroke="var(--text-muted)" fontSize={10} />
                    <Tooltip {...tip} formatter={(value) => `$${Number(value ?? 0).toFixed(4)}`} />
                    <Bar dataKey="cost" name="Cost ($)" fill="#059669" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Recent Traces with Span Details */}
          <div className="card">
            <h3 className="text-xs text-[var(--text-muted)] mb-2">Recent Traces (with OTel Span Breakdown)</h3>
            <div className="space-y-1 max-h-[350px] overflow-y-auto">
              {traces.slice(0, 15).map((t: any) => {
                const spanCounts: Record<string, number> = {};
                for (const s of t.spans || []) {
                  spanCounts[s.span_type] = (spanCounts[s.span_type] || 0) + 1;
                }
                return (
                  <div key={t.id} className="flex items-center justify-between text-xs py-2 border-b border-[var(--border)]">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${t.status === "completed" ? "bg-emerald-500" : "bg-red-500"}`} />
                      <span className="truncate">{t.user_prompt}</span>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0 ml-2 text-[var(--text-muted)]">
                      {Object.entries(spanCounts).map(([type, count]) => (
                        <span key={type} className="px-1.5 py-0.5 rounded text-[9px] border"
                          style={{ borderColor: SPAN_TYPE_COLORS[type] || "#6b7280", color: SPAN_TYPE_COLORS[type] || "#6b7280" }}>
                          {type.replace(/_/g, " ")} ×{count}
                        </span>
                      ))}
                      <span className="ml-1">{(t.total_latency_ms || 0).toFixed(0)}ms</span>
                      <span>{t.total_tokens || 0} tok</span>
                      <span>${(t.total_cost || 0).toFixed(4)}</span>
                    </div>
                  </div>
                );
              })}
              {traces.length === 0 && <div className="text-[var(--text-muted)] text-center py-6">Send chat messages to generate traces</div>}
            </div>
          </div>

          {/* OTel Architecture Info */}
          <div className="card">
            <h3 className="text-xs text-[var(--text-muted)] mb-2">OpenTelemetry Architecture</h3>
            <div className="grid grid-cols-4 gap-2 text-xs text-[var(--text-muted)]">
              <div className="p-2.5 rounded bg-indigo-50 border border-indigo-100">
                <div className="font-medium text-indigo-700 mb-1">GenAI Semantic Conventions</div>
                <div>Spans follow <code className="text-[10px]">gen_ai.*</code> namespace: <code className="text-[10px]">gen_ai.request.model</code>, <code className="text-[10px]">gen_ai.usage.input_tokens</code>, <code className="text-[10px]">gen_ai.operation.name</code></div>
              </div>
              <div className="p-2.5 rounded bg-indigo-50 border border-indigo-100">
                <div className="font-medium text-indigo-700 mb-1">OpenInference Auto-Instrumentation</div>
                <div>LangChainInstrumentor automatically captures all LLM calls, tool invocations, and chain executions as OTel spans.</div>
              </div>
              <div className="p-2.5 rounded bg-indigo-50 border border-indigo-100">
                <div className="font-medium text-indigo-700 mb-1">Custom DBSpanProcessor</div>
                <div>Persists every span to SQLite with cost estimation (tokens × model rates). Enriches spans with <code className="text-[10px]">span.type</code> classification.</div>
              </div>
              <div className="p-2.5 rounded bg-indigo-50 border border-indigo-100">
                <div className="font-medium text-indigo-700 mb-1">OTLP Export</div>
                <div>Dual export: local DB + optional OTLP HTTP to Phoenix, Langfuse, Jaeger, or Datadog via <code className="text-[10px]">BatchSpanProcessor</code>.</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
