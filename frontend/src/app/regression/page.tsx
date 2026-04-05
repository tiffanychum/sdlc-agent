"use client";
import React, { useEffect, useState, useCallback, useMemo } from "react";
import { api } from "@/lib/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Legend,
} from "recharts";

// ── Shared helpers ───────────────────────────────────────────────────────────

const mdComponents = {
  code({ className, children, ...props }: any) {
    const isInline = !className;
    return isInline ? (
      <code className="px-1 py-0.5 rounded bg-[var(--bg)] text-[var(--accent)] text-[11px] font-mono" {...props}>{children}</code>
    ) : (
      <pre className="bg-[var(--bg)] rounded p-2 my-1.5 overflow-x-auto border border-[var(--border)]">
        <code className="text-[11px] font-mono leading-relaxed" {...props}>{children}</code>
      </pre>
    );
  },
  p({ children }: any) { return <p className="mb-1.5 last:mb-0 leading-relaxed">{children}</p>; },
  ul({ children }: any) { return <ul className="list-disc pl-4 mb-1.5 space-y-0.5">{children}</ul>; },
  ol({ children }: any) { return <ol className="list-decimal pl-4 mb-1.5 space-y-0.5">{children}</ol>; },
  li({ children }: any) { return <li className="leading-relaxed">{children}</li>; },
  h1({ children }: any) { return <h1 className="text-sm font-semibold mt-2 mb-1">{children}</h1>; },
  h2({ children }: any) { return <h2 className="text-[13px] font-semibold mt-2 mb-0.5">{children}</h2>; },
  h3({ children }: any) { return <h3 className="text-[12px] font-semibold mt-1.5 mb-0.5">{children}</h3>; },
  strong({ children }: any) { return <strong className="font-semibold">{children}</strong>; },
  table({ children }: any) {
    return <div className="overflow-x-auto my-1.5"><table className="text-[11px] border-collapse w-full">{children}</table></div>;
  },
  th({ children }: any) { return <th className="border border-[var(--border)] px-2 py-0.5 bg-[var(--bg)] font-medium text-left">{children}</th>; },
  td({ children }: any) { return <td className="border border-[var(--border)] px-2 py-0.5">{children}</td>; },
  a({ href, children }: any) {
    return <a href={href} target="_blank" rel="noopener noreferrer" className="text-[var(--accent)] underline hover:opacity-80">{children}</a>;
  },
};

function Md({ content, className = "" }: { content: string; className?: string }) {
  if (!content || content === "—") return <span className="text-[var(--text-muted)] italic text-xs">—</span>;
  return (
    <div className={`text-xs leading-relaxed ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>{content}</ReactMarkdown>
    </div>
  );
}

function PassFailBadge({ pass }: { pass: boolean }) {
  return (
    <span className={`text-[10px] font-semibold px-2 py-0.5 rounded ${pass ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"}`}>
      {pass ? "PASS" : "FAIL"}
    </span>
  );
}

function Metric({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: string }) {
  return (
    <div className="card text-center">
      <div className={`text-2xl font-semibold ${accent || ""}`}>{value}</div>
      <div className="text-xs text-[var(--text-muted)] mt-1">{label}</div>
      {sub && <div className="text-[11px] text-[var(--text-muted)] mt-0.5">{sub}</div>}
    </div>
  );
}

const DEEPEVAL_METRICS = [
  { key: "deepeval_relevancy", label: "Relevancy", reasonKey: "deepeval_relevancy_reason", description: "Does the response directly address the user's query?" },
  { key: "deepeval_faithfulness", label: "Faithfulness", reasonKey: "deepeval_faithfulness_reason", description: "Is the response grounded in tool outputs?" },
  { key: "tool_correctness", label: "Tool Correct.", reasonKey: "tool_correctness_reason", description: "Were the correct tools selected?" },
  { key: "argument_correctness", label: "Arg Correct.", reasonKey: "argument_correctness_reason", description: "Were tool arguments well-formed?" },
  { key: "task_completion", label: "Task Complete", reasonKey: "task_completion_reason", description: "Did the agent fully accomplish the task?" },
  { key: "step_efficiency_de", label: "Step Effic.", reasonKey: "step_efficiency_de_reason", description: "Were steps minimal and necessary?" },
  { key: "plan_quality", label: "Plan Quality", reasonKey: "plan_quality_reason", description: "Was the plan logical and actionable?" },
  { key: "plan_adherence", label: "Plan Adhere.", reasonKey: "plan_adherence_reason", description: "Did the agent follow its own plan?" },
];

const STRAT_INFO: Record<string, { label: string; desc: string; cls: string }> = {
  router_decides: { label: "Router Decides", desc: "LLM routes to one agent per request", cls: "bg-blue-50 text-blue-700 border-blue-200" },
  sequential:     { label: "Sequential",     desc: "Agents execute in a fixed pipeline order", cls: "bg-emerald-50 text-emerald-700 border-emerald-200" },
  parallel:       { label: "Parallel",       desc: "All agents run simultaneously", cls: "bg-amber-50 text-amber-700 border-amber-200" },
  supervisor:     { label: "Supervisor",     desc: "Supervisor dynamically delegates between agents", cls: "bg-violet-50 text-violet-700 border-violet-200" },
  auto:           { label: "Auto",           desc: "Meta-router AI picks strategy per request", cls: "bg-orange-50 text-orange-700 border-orange-200" },
};

// ── Main Page ────────────────────────────────────────────────────────────────

type SubTab = "golden" | "run" | "results" | "compare";

export default function RegressionPage() {
  const [subTab, setSubTab] = useState<SubTab>("run");
  const [teams, setTeams] = useState<any[]>([]);
  const [teamId, setTeamId] = useState("");

  // golden dataset
  const [goldenCases, setGoldenCases] = useState<any[]>([]);
  const [expandedCaseId, setExpandedCaseId] = useState<string | null>(null);

  // run tests
  const [regModels, setRegModels] = useState<Set<string>>(new Set());
  const [regPromptVer, setRegPromptVer] = useState("v1");
  const [regBaselineRunId, setRegBaselineRunId] = useState("");
  const [selectedCaseIds, setSelectedCaseIds] = useState<Set<string>>(new Set());
  const [caseSearch, setCaseSearch] = useState("");
  const [regRunning, setRegRunning] = useState(false);
  const [regRunResult, setRegRunResult] = useState<any>(null);

  // results
  const [regRuns, setRegRuns] = useState<any[]>([]);
  const [runsSearch, setRunsSearch] = useState("");
  const [runsSort, setRunsSort] = useState<"date" | "pass" | "cost">("date");
  const [regSelectedRunId, setRegSelectedRunId] = useState("");
  const [regResults, setRegResults] = useState<any>(null);
  const [regCaseDetail, setRegCaseDetail] = useState<any>(null);
  const [resultsSearch, setResultsSearch] = useState("");
  const [resultsFilter, setResultsFilter] = useState<"all" | "pass" | "fail">("all");

  // compare
  const [regDiffRunA, setRegDiffRunA] = useState("");
  const [regDiffRunB, setRegDiffRunB] = useState("");
  const [regDiffCaseId, setRegDiffCaseId] = useState("");
  const [regDiff, setRegDiff] = useState<any>(null);
  const [regRCA, setRegRCA] = useState<any>(null);
  const [rcaLoading, setRcaLoading] = useState(false);

  // config
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [promptVersions, setPromptVersions] = useState<any[]>([]);
  const [showBaselineInfo, setShowBaselineInfo] = useState(false);

  const selectedTeam = teams.find(t => t.id === teamId);

  useEffect(() => {
    api.teams.list().then(t => { setTeams(t); if (t.length) setTeamId(t[0].id); });
    api.models.list().then(setAvailableModels).catch(() => {});
    api.promptVersions.list().then(setPromptVersions).catch(() => {});
  }, []);

  const loadGolden = useCallback(async () => {
    try { const c = await api.golden.list(); setGoldenCases(c); } catch { /* ignore */ }
  }, []);

  const loadRegRuns = useCallback(async () => {
    try { const r = await api.regression.runs(); setRegRuns(r); } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    loadGolden();
    loadRegRuns();
  }, [loadGolden, loadRegRuns]);

  // ── Run tests ──────────────────────────────────────────────────

  async function runRegression() {
    setRegRunning(true);
    setRegRunResult(null);
    const models = regModels.size > 0 ? Array.from(regModels) : [undefined];
    try {
      if (models.length > 1) {
        const runs: any[] = [];
        for (const model of models) {
          const result = await api.regression.run({
            team_id: teamId,
            case_ids: selectedCaseIds.size > 0 ? Array.from(selectedCaseIds) : undefined,
            model: model || undefined,
            prompt_version: regPromptVer !== "v1" ? regPromptVer : undefined,
            baseline_run_id: regBaselineRunId || undefined,
          });
          runs.push(result);
        }
        setRegRunResult({ multi_run: true, runs });
      } else {
        const result = await api.regression.run({
          team_id: teamId,
          case_ids: selectedCaseIds.size > 0 ? Array.from(selectedCaseIds) : undefined,
          model: models[0] || undefined,
          prompt_version: regPromptVer !== "v1" ? regPromptVer : undefined,
          baseline_run_id: regBaselineRunId || undefined,
        });
        setRegRunResult(result);
      }
      await loadRegRuns();
      setSubTab("results");
    } catch (e: any) {
      setRegRunResult({ error: e.message });
    }
    setRegRunning(false);
  }

  // ── Results ────────────────────────────────────────────────────

  async function loadRegResults(runId: string) {
    setRegSelectedRunId(runId);
    setRegCaseDetail(null);
    setResultsSearch("");
    setResultsFilter("all");
    try { const r = await api.regression.results(runId); setRegResults(r); } catch { /* ignore */ }
  }

  async function loadCaseDetail(runId: string, caseId: string) {
    try { const d = await api.regression.caseDetail(runId, caseId); setRegCaseDetail(d); } catch { /* ignore */ }
  }

  async function loadTraceDiff() {
    if (!regDiffRunA || !regDiffRunB || !regDiffCaseId) return;
    try { const d = await api.regression.diff(regDiffRunA, regDiffRunB, regDiffCaseId); setRegDiff(d); } catch { /* ignore */ }
  }

  async function runRCA(runId: string, caseId: string, baselineRunId?: string) {
    setRcaLoading(true);
    try { const r = await api.regression.rca(runId, caseId, baselineRunId); setRegRCA(r); } catch { /* ignore */ }
    setRcaLoading(false);
  }

  // ── Computed values ────────────────────────────────────────────

  const filteredCases = useMemo(() => {
    const q = caseSearch.toLowerCase();
    return goldenCases.filter(c => c.is_active && (
      !q || c.name.toLowerCase().includes(q) || c.id.toLowerCase().includes(q) ||
      (c.expected_agent || "").toLowerCase().includes(q) || c.prompt.toLowerCase().includes(q)
    ));
  }, [goldenCases, caseSearch]);

  const sortedRuns = useMemo(() => {
    const q = runsSearch.toLowerCase();
    const filtered = regRuns.filter(r =>
      !q || r.id.toLowerCase().includes(q) || (r.model || "").toLowerCase().includes(q) ||
      (r.prompt_version || "").toLowerCase().includes(q)
    );
    if (runsSort === "date") return [...filtered].sort((a, b) => new Date(b.created_at || 0).getTime() - new Date(a.created_at || 0).getTime());
    if (runsSort === "pass") return [...filtered].sort((a, b) => (b.pass_rate || 0) - (a.pass_rate || 0));
    if (runsSort === "cost") return [...filtered].sort((a, b) => (b.total_cost || 0) - (a.total_cost || 0));
    return filtered;
  }, [regRuns, runsSearch, runsSort]);

  const filteredResults = useMemo(() => {
    if (!regResults?.results) return [];
    const q = resultsSearch.toLowerCase();
    return regResults.results.filter((r: any) => {
      const matchSearch = !q || r.golden_case_name.toLowerCase().includes(q) || r.golden_case_id.toLowerCase().includes(q);
      const matchFilter = resultsFilter === "all" || (resultsFilter === "pass" ? r.overall_pass : !r.overall_pass);
      return matchSearch && matchFilter;
    });
  }, [regResults, resultsSearch, resultsFilter]);

  const comparableCases = useMemo(() => {
    const a = regRuns.find(r => r.id === regDiffRunA);
    const b = regRuns.find(r => r.id === regDiffRunB);
    if (!a?.case_ids || !b?.case_ids) return [];
    const setB = new Set(b.case_ids as string[]);
    const intersection = (a.case_ids as string[]).filter(id => setB.has(id));
    return goldenCases.filter(c => intersection.includes(c.id));
  }, [regDiffRunA, regDiffRunB, regRuns, goldenCases]);

  // ── Render ─────────────────────────────────────────────────────

  return (
    <div className="space-y-4 max-w-7xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Regression Testing</h1>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">Golden dataset driven quality gates, model comparison & RCA</p>
        </div>
        <div className="flex items-center gap-2">
          <select value={teamId} onChange={e => setTeamId(e.target.value)} className="input !w-auto">
            {teams.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
          </select>
          <button onClick={() => { loadGolden(); loadRegRuns(); }} className="btn-secondary">Refresh</button>
        </div>
      </div>

      {/* Sub-tabs */}
      <div className="flex gap-1 border-b border-[var(--border)]">
        {([
          { id: "run" as SubTab, label: "Run Tests" },
          { id: "results" as SubTab, label: `Results${regRuns.length ? ` (${regRuns.length})` : ""}` },
          { id: "golden" as SubTab, label: `Golden Dataset (${goldenCases.filter(c => c.is_active).length})` },
          { id: "compare" as SubTab, label: "Trace Diff & RCA" },
        ]).map(tab => (
          <button key={tab.id} onClick={() => setSubTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-all ${
              subTab === tab.id ? "border-zinc-900 text-zinc-900" : "border-transparent text-[var(--text-muted)] hover:text-[var(--text)]"
            }`}>
            {tab.label}
          </button>
        ))}
      </div>

      {/* ─────────── RUN TESTS ─────────── */}
      {subTab === "run" && (
        <div className="space-y-3">
          {/* Team strategy banner */}
          {selectedTeam && (
            <div className="flex items-center gap-3 px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg)] text-xs">
              <span className="text-[var(--text-muted)] shrink-0">Active Strategy:</span>
              {(() => {
                const strat = selectedTeam.decision_strategy || "router_decides";
                const info = STRAT_INFO[strat] || { label: strat, desc: "", cls: "bg-gray-100 text-gray-700 border-gray-200" };
                return (
                  <>
                    <span className={`px-2 py-0.5 rounded border text-[11px] font-semibold ${info.cls}`}>
                      {strat === "auto" ? "🤖 " : ""}{info.label}
                    </span>
                    <span className="text-[var(--text-muted)]">{info.desc}</span>
                  </>
                );
              })()}
              <a href="/" className="ml-auto text-[10px] underline text-zinc-500 hover:text-zinc-800 shrink-0">Configure in Studio →</a>
            </div>
          )}

          {/* Two-column: config | test cases */}
          <div className="grid grid-cols-[280px_1fr] gap-4">
            {/* Left: configuration */}
            <div className="space-y-3">
              {/* Models */}
              <div className="card !p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-[11px] font-medium uppercase text-[var(--text-muted)] tracking-wide">Models</label>
                  {regModels.size > 0 && (
                    <button onClick={() => setRegModels(new Set())} className="text-[10px] text-[var(--text-muted)] underline">Clear</button>
                  )}
                </div>
                <p className="text-[10px] text-[var(--text-muted)]">Select 1+ to compare models side-by-side</p>
                <div className="border border-[var(--border)] rounded overflow-hidden">
                  <label className={`flex items-center gap-2 px-2 py-1.5 text-[11px] border-b border-[var(--border)] cursor-pointer hover:bg-[var(--bg-hover)] ${regModels.size === 0 ? "bg-zinc-50 font-medium" : ""}`}>
                    <input type="checkbox" checked={regModels.size === 0} onChange={() => setRegModels(new Set())} className="h-3 w-3 rounded border-gray-300 accent-[var(--accent)]" />
                    <span className="flex-1">Default (from config)</span>
                  </label>
                  {availableModels.map(m => (
                    <label key={m.id} className={`flex items-center gap-2 px-2 py-1.5 text-[11px] border-b border-[var(--border)] last:border-b-0 cursor-pointer hover:bg-[var(--bg-hover)] ${regModels.has(m.id) ? "bg-zinc-50 font-medium" : ""}`}>
                      <input type="checkbox" checked={regModels.has(m.id)}
                        onChange={() => {
                          const next = new Set(regModels);
                          if (next.has(m.id)) next.delete(m.id); else next.add(m.id);
                          setRegModels(next);
                        }}
                        className="h-3 w-3 rounded border-gray-300 accent-[var(--accent)]" />
                      <span className="flex-1 truncate">{m.name}</span>
                      <span className="text-[10px] text-[var(--text-muted)] shrink-0">{m.provider}</span>
                    </label>
                  ))}
                </div>
                {regModels.size > 1 && (
                  <p className="text-[10px] text-zinc-700 font-medium">{regModels.size} models — will run sequentially and compare</p>
                )}
              </div>

              {/* Prompt Version */}
              <div className="card !p-3 space-y-1.5">
                <label className="text-[11px] font-medium uppercase text-[var(--text-muted)] tracking-wide">Prompt Version</label>
                <select value={regPromptVer} onChange={e => setRegPromptVer(e.target.value)} className="input text-xs w-full">
                  <option value="v1">v1 (current)</option>
                  {promptVersions.map(pv => (
                    <option key={pv.id} value={pv.version_label}>
                      {pv.version_label}{pv.description ? ` — ${pv.description}` : ""}
                    </option>
                  ))}
                </select>
              </div>

              {/* Baseline Run */}
              <div className="card !p-3 space-y-1.5">
                <div className="flex items-center gap-1.5">
                  <label className="text-[11px] font-medium uppercase text-[var(--text-muted)] tracking-wide">Baseline Run</label>
                  <button onClick={() => setShowBaselineInfo(!showBaselineInfo)} className="w-4 h-4 rounded-full bg-gray-200 text-gray-600 text-[10px] flex items-center justify-center font-bold hover:bg-gray-300">?</button>
                </div>
                {showBaselineInfo && (
                  <div className="text-[10px] text-[var(--text-muted)] bg-amber-50 border border-amber-200 rounded p-2 leading-relaxed">
                    <strong className="text-amber-700">What is a baseline?</strong> Pick a previous run to compare against. The system flags regressions — cases where the new run is worse than the baseline in cost, latency, quality, or trace structure. Essential for catching prompt/model regressions before deployment.
                  </div>
                )}
                <select value={regBaselineRunId} onChange={e => setRegBaselineRunId(e.target.value)} className="input text-xs w-full">
                  <option value="">No baseline (standalone run)</option>
                  {regRuns.map(r => (
                    <option key={r.id} value={r.id}>
                      {r.id.slice(0, 8)} · {r.model || "default"} · {r.passed}/{r.num_cases} pass · {r.created_at ? new Date(r.created_at).toLocaleDateString() : ""}
                    </option>
                  ))}
                </select>
              </div>

              {/* Run Button */}
              <button onClick={runRegression} disabled={regRunning} className="btn-primary w-full">
                {regRunning ? (
                  <span className="flex items-center justify-center gap-1.5">
                    <span className="h-3 w-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Running {selectedCaseIds.size > 0 ? selectedCaseIds.size : goldenCases.filter(c => c.is_active).length} cases…
                  </span>
                ) : (
                  `Run ${selectedCaseIds.size > 0 ? selectedCaseIds.size : goldenCases.filter(c => c.is_active).length} Test${(selectedCaseIds.size || goldenCases.filter(c => c.is_active).length) !== 1 ? "s" : ""}`
                )}
              </button>

              {/* Run result */}
              {regRunResult && !regRunning && (
                <div className={`text-xs p-3 rounded-lg border ${regRunResult.error ? "border-red-200 bg-red-50 text-red-700" : "border-emerald-200 bg-emerald-50 text-emerald-700"}`}>
                  {regRunResult.error ? (
                    <span>Error: {regRunResult.error}</span>
                  ) : regRunResult.multi_run ? (
                    <span>{regRunResult.runs?.length} model runs completed → see Results tab</span>
                  ) : (
                    <span>{regRunResult.summary?.num_passed}/{regRunResult.num_cases} passed · ${(regRunResult.summary?.total_cost || 0).toFixed(4)} · Results tab updated</span>
                  )}
                </div>
              )}
            </div>

            {/* Right: test case selection */}
            <div className="card !p-0 overflow-hidden flex flex-col">
              <div className="flex items-center justify-between px-3 py-2.5 border-b border-[var(--border)] bg-[var(--bg)]">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">Test Cases</span>
                  <span className="text-[10px] text-[var(--text-muted)]">
                    {selectedCaseIds.size > 0 ? `${selectedCaseIds.size} selected` : "All active cases"}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <input value={caseSearch} onChange={e => setCaseSearch(e.target.value)} placeholder="Search cases…"
                    className="input !py-1 !px-2 text-[11px] !w-44" />
                  <button onClick={() => {
                    const visible = filteredCases.map(c => c.id);
                    const allSelected = visible.every(id => selectedCaseIds.has(id));
                    if (allSelected) {
                      const next = new Set(selectedCaseIds);
                      visible.forEach(id => next.delete(id));
                      setSelectedCaseIds(next);
                    } else {
                      const next = new Set(selectedCaseIds);
                      visible.forEach(id => next.add(id));
                      setSelectedCaseIds(next);
                    }
                  }} className="text-[10px] text-zinc-500 underline shrink-0">
                    {filteredCases.every(c => selectedCaseIds.has(c.id)) && filteredCases.length > 0 ? "Deselect All" : "Select All"}
                  </button>
                </div>
              </div>

              {/* Column headers */}
              <div className="grid grid-cols-[24px_1fr_100px_80px_80px_64px] gap-2 px-3 py-1.5 bg-[var(--bg)] text-[10px] uppercase text-[var(--text-muted)] font-medium border-b border-[var(--border)]">
                <div />
                <div>Name / Prompt</div>
                <div className="text-center">Strategy</div>
                <div className="text-center">Agent</div>
                <div className="text-center">Complexity</div>
                <div className="text-center">Ver.</div>
              </div>

              <div className="overflow-y-auto flex-1" style={{ maxHeight: "calc(100vh - 320px)" }}>
                {filteredCases.map(c => (
                  <label key={c.id}
                    className={`grid grid-cols-[24px_1fr_100px_80px_80px_64px] gap-2 px-3 py-2 border-b border-[var(--border)] text-xs cursor-pointer transition-colors items-center ${
                      selectedCaseIds.has(c.id) ? "bg-zinc-50" : "hover:bg-[var(--bg-hover)]"
                    }`}>
                    <input type="checkbox" checked={selectedCaseIds.has(c.id)}
                      onChange={() => {
                        const next = new Set(selectedCaseIds);
                        if (next.has(c.id)) next.delete(c.id); else next.add(c.id);
                        setSelectedCaseIds(next);
                      }}
                      className="h-3.5 w-3.5 rounded border-gray-300 accent-[var(--accent)]" />
                    <div className="min-w-0">
                      <div className="font-medium truncate">{c.name}</div>
                      <div className="text-[10px] text-[var(--text-muted)] truncate">{c.prompt}</div>
                      <div className="text-[10px] text-[var(--text-muted)] font-mono">{c.id}</div>
                    </div>
                    <div className="text-center">
                      {c.strategy ? (
                        <span className={`px-1.5 py-0.5 rounded text-[10px] border font-medium ${
                          c.strategy === "auto" ? "bg-orange-50 text-orange-700 border-orange-200" : "bg-zinc-100 text-zinc-600 border-zinc-200"
                        }`} title={c.expected_strategy ? `Expects: ${c.expected_strategy}` : ""}>
                          {c.strategy === "auto" ? "🤖 auto" : c.strategy}
                        </span>
                      ) : (
                        <span className="text-[10px] text-[var(--text-muted)]">default</span>
                      )}
                    </div>
                    <div className="text-center">
                      <span className="px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-600 text-[10px] border border-zinc-200 font-medium">{c.expected_agent}</span>
                    </div>
                    <div className="text-center">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] border ${
                        c.complexity === "quick" ? "bg-green-50 text-green-700 border-green-200"
                        : c.complexity === "medium" ? "bg-amber-50 text-amber-700 border-amber-200"
                        : "bg-red-50 text-red-700 border-red-200"
                      }`}>{c.complexity}</span>
                    </div>
                    <div className="text-center text-[10px] text-[var(--text-muted)]">{c.version}</div>
                  </label>
                ))}
                {filteredCases.length === 0 && (
                  <div className="text-center py-8 text-sm text-[var(--text-muted)]">No cases match "{caseSearch}"</div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ─────────── RESULTS ─────────── */}
      {subTab === "results" && (
        <div className="grid grid-cols-[320px_1fr] gap-4">
          {/* Left: run list */}
          <div className="space-y-2">
            <div className="card !p-3 space-y-2">
              <div className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">Runs</div>
              <input value={runsSearch} onChange={e => setRunsSearch(e.target.value)} placeholder="Search by model, ID, version…"
                className="input !py-1 !px-2 text-[11px] w-full" />
              <div className="flex gap-1">
                {(["date", "pass", "cost"] as const).map(s => (
                  <button key={s} onClick={() => setRunsSort(s)}
                    className={`flex-1 py-1 text-[10px] rounded border transition-all ${runsSort === s ? "border-zinc-900 bg-zinc-900 text-white font-medium" : "border-[var(--border)] text-[var(--text-muted)] hover:border-zinc-400"}`}>
                    {s === "date" ? "Newest" : s === "pass" ? "Pass Rate" : "Cost"}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-1.5 overflow-y-auto" style={{ maxHeight: "calc(100vh - 280px)" }}>
              {sortedRuns.length === 0 ? (
                <div className="text-center py-8 text-sm text-[var(--text-muted)]">No runs yet. Go to Run Tests to start.</div>
              ) : sortedRuns.map(r => (
                <button key={r.id} onClick={() => loadRegResults(r.id)}
                  className={`w-full text-left p-3 rounded-lg border transition-all ${
                    regSelectedRunId === r.id ? "border-zinc-900 bg-zinc-50" : "border-[var(--border)] hover:border-zinc-400 bg-white"
                  }`}>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-[11px] font-semibold">{r.id.slice(0, 8)}</span>
                      <span className="px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-600 text-[10px] font-medium">{r.prompt_version || "v1"}</span>
                    </div>
                    <span className={`text-[11px] font-semibold ${(r.pass_rate || 0) >= 0.7 ? "text-emerald-600" : "text-red-600"}`}>
                      {r.passed}/{r.num_cases} pass
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-[var(--text-muted)]">
                    <span className="truncate max-w-[120px]">{r.model || "default model"}</span>
                    <div className="flex items-center gap-2 shrink-0">
                      <span>${(r.total_cost || 0).toFixed(4)}</span>
                      <span>{r.created_at ? new Date(r.created_at).toLocaleDateString() : "—"}</span>
                    </div>
                  </div>
                  {/* Pass rate bar */}
                  <div className="mt-1.5 h-1 bg-gray-200 rounded-full overflow-hidden">
                    <div className="h-full rounded-full bg-emerald-500 transition-all" style={{ width: `${(r.pass_rate || 0) * 100}%` }} />
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Right: run detail */}
          <div className="space-y-3">
            {!regSelectedRunId ? (
              <div className="card flex items-center justify-center py-16 text-sm text-[var(--text-muted)]">
                Select a run from the list to view results
              </div>
            ) : !regResults ? (
              <div className="card flex items-center justify-center py-16">
                <div className="h-5 w-5 border-2 border-zinc-200 border-t-zinc-700 rounded-full animate-spin" />
              </div>
            ) : (
              <>
                {/* Summary KPIs */}
                <div className="grid grid-cols-5 gap-3">
                  <Metric label="Pass Rate" value={`${((regResults.summary?.pass_rate || 0) * 100).toFixed(0)}%`}
                    accent={(regResults.summary?.pass_rate || 0) >= 0.7 ? "text-emerald-600" : "text-red-600"} />
                  <Metric label="Avg Similarity" value={`${((regResults.summary?.avg_semantic_similarity || 0) * 100).toFixed(1)}%`} />
                  <Metric label="Avg Latency" value={`${(regResults.summary?.avg_latency_ms || 0).toFixed(0)}ms`} />
                  <Metric label="Total Cost" value={`$${(regResults.summary?.total_cost || 0).toFixed(4)}`} />
                  <Metric label="Total Tokens" value={String(regResults.summary?.total_tokens || 0)} />
                </div>

                {/* Filter bar */}
                <div className="flex items-center gap-2">
                  <input value={resultsSearch} onChange={e => setResultsSearch(e.target.value)} placeholder="Search cases…"
                    className="input !py-1 !px-2 text-xs !w-56" />
                  {(["all", "pass", "fail"] as const).map(f => (
                    <button key={f} onClick={() => setResultsFilter(f)}
                      className={`px-3 py-1 text-[11px] rounded border transition-all capitalize ${resultsFilter === f ? "border-zinc-900 bg-zinc-900 text-white font-medium" : "border-[var(--border)] text-[var(--text-muted)] hover:border-zinc-400"}`}>
                      {f}
                    </button>
                  ))}
                  <span className="text-[10px] text-[var(--text-muted)] ml-auto">{filteredResults.length} cases</span>
                </div>

                {/* Cases table */}
                <div className="card !p-0 overflow-hidden">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-[var(--text-muted)] text-[10px] bg-[var(--bg)] uppercase">
                        <th className="text-left px-3 py-2">Case</th>
                        <th className="text-center px-2 py-2">Strategy</th>
                        <th className="text-center px-2 py-2">Agents</th>
                        <th className="text-right px-2 py-2">Sim.</th>
                        <th className="text-right px-2 py-2">DeepEval</th>
                        <th className="text-right px-2 py-2">Latency</th>
                        <th className="text-right px-2 py-2">Cost</th>
                        <th className="text-center px-2 py-2">Trace</th>
                        <th className="text-center px-2 py-2">Quality</th>
                        <th className="text-center px-2 py-2">Status</th>
                        <th className="text-center px-2 py-2">Detail</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredResults.map((r: any) => (
                        <tr key={r.golden_case_id} className={`border-t border-[var(--border)] ${!r.overall_pass ? "bg-red-50/30" : ""}`}>
                          <td className="px-3 py-2">
                            <div className="font-medium">{r.golden_case_name}</div>
                            <div className="text-[10px] text-[var(--text-muted)] font-mono">{r.golden_case_id}</div>
                          </td>
                          <td className="text-center px-2 py-2">
                            {r.actual_strategy ? (
                              <div className="flex flex-col items-center gap-0.5">
                                <span className="px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-600 text-[10px] border border-zinc-200 font-semibold whitespace-nowrap">{r.actual_strategy}</span>
                                {r.expected_strategy && r.expected_strategy !== r.actual_strategy && (
                                  <span className="text-[9px] text-red-500">≠ {r.expected_strategy}</span>
                                )}
                                {r.expected_strategy && r.expected_strategy === r.actual_strategy && (
                                  <span className="text-[9px] text-emerald-600">✓</span>
                                )}
                              </div>
                            ) : <span className="text-[10px] text-[var(--text-muted)]">default</span>}
                          </td>
                          <td className="text-center px-2 py-2">
                            <div className="flex flex-wrap justify-center items-center gap-0.5">
                              {(r.actual_agent || "—").split(" > ").map((a: string, idx: number) => (
                                <span key={idx} className="flex items-center gap-0.5">
                                  {idx > 0 && <span className="text-[8px] text-[var(--text-muted)]">→</span>}
                                  <span className="px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-600 text-[10px] border border-zinc-200 font-medium">{a}</span>
                                </span>
                              ))}
                            </div>
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
                              return <span className={avg >= 0.7 ? "text-emerald-600" : avg >= 0.4 ? "text-amber-600" : "text-red-600"}>{(avg * 100).toFixed(0)}%</span>;
                            })()}
                          </td>
                          <td className="text-right px-2 py-2">{(r.actual_latency_ms || 0).toFixed(0)}ms</td>
                          <td className="text-right px-2 py-2">${(r.actual_cost || 0).toFixed(4)}</td>
                          <td className="text-center px-2 py-2">
                            <span className={`text-[10px] font-medium ${r.trace_regression ? "text-red-600" : "text-emerald-600"}`}>{r.trace_regression ? "FAIL" : "OK"}</span>
                          </td>
                          <td className="text-center px-2 py-2">
                            <span className={`text-[10px] font-medium ${r.quality_regression ? "text-red-600" : "text-emerald-600"}`}>{r.quality_regression ? "FAIL" : "OK"}</span>
                          </td>
                          <td className="text-center px-2 py-2"><PassFailBadge pass={r.overall_pass} /></td>
                          <td className="text-center px-2 py-2">
                            <button onClick={() => loadCaseDetail(regSelectedRunId, r.golden_case_id)}
                              className="text-[10px] text-zinc-500 underline hover:text-zinc-800">Detail</button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Case detail drawer */}
                {regCaseDetail && <CaseDetailPanel detail={regCaseDetail} runId={regSelectedRunId} baselineRunId={regBaselineRunId} onClose={() => setRegCaseDetail(null)} onRCA={runRCA} rcaLoading={rcaLoading} regRCA={regRCA} />}
              </>
            )}
          </div>
        </div>
      )}

      {/* ─────────── GOLDEN DATASET ─────────── */}
      {subTab === "golden" && (
        <div className="card !p-0 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
            <div>
              <h2 className="text-sm font-medium">Golden Test Cases</h2>
              <p className="text-[11px] text-[var(--text-muted)]">Curated test cases with known-good outputs. Click a row to expand details.</p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-[var(--text-muted)]">{goldenCases.length} cases ({goldenCases.filter(c => c.is_active).length} active)</span>
              <button onClick={() => api.golden.sync().then(loadGolden)} className="btn-secondary text-xs">Sync from JSON</button>
            </div>
          </div>
          <div className="overflow-y-auto" style={{ maxHeight: "calc(100vh - 220px)" }}>
            {/* Header */}
            <div className="grid grid-cols-[24px_2fr_3fr_100px_80px_60px_60px] gap-2 px-3 py-1.5 bg-[var(--bg)] text-[10px] uppercase tracking-wide text-[var(--text-muted)] font-medium border-b border-[var(--border)] sticky top-0 z-10">
              <div />
              <div>Name / ID</div>
              <div>Prompt</div>
              <div className="text-center">Strategy</div>
              <div className="text-center">Complexity</div>
              <div className="text-center">Ver.</div>
              <div className="text-center">Active</div>
            </div>
            {goldenCases.map(c => (
              <div key={c.id}>
                <div onClick={() => setExpandedCaseId(expandedCaseId === c.id ? null : c.id)}
                  className={`grid grid-cols-[24px_2fr_3fr_100px_80px_60px_60px] gap-2 px-3 py-2 border-b border-[var(--border)] text-xs cursor-pointer transition-colors items-center ${
                      expandedCaseId === c.id ? "bg-zinc-50" : "hover:bg-[var(--bg-hover)]"
                  }`}>
                  <div className="text-[10px] text-[var(--text-muted)]">{expandedCaseId === c.id ? "▼" : "▶"}</div>
                  <div>
                    <div className="font-medium">{c.name}</div>
                    <div className="text-[10px] text-[var(--text-muted)] font-mono">{c.id}</div>
                  </div>
                  <div className="truncate text-[var(--text-muted)]">{c.prompt}</div>
                  <div className="text-center">
                    {c.strategy ? (
                      <span className={`px-1.5 py-0.5 rounded text-[10px] border font-medium ${c.strategy === "auto" ? "bg-orange-50 text-orange-700 border-orange-200" : "bg-zinc-100 text-zinc-600 border-zinc-200"}`}>
                        {c.strategy === "auto" ? "🤖 auto" : c.strategy}
                      </span>
                    ) : <span className="text-[10px] text-[var(--text-muted)]">default</span>}
                  </div>
                  <div className="text-center">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] border ${
                      c.complexity === "quick" ? "bg-green-50 text-green-700 border-green-200"
                      : c.complexity === "medium" ? "bg-amber-50 text-amber-700 border-amber-200"
                      : "bg-red-50 text-red-700 border-red-200"
                    }`}>{c.complexity}</span>
                  </div>
                  <div className="text-center text-[10px] text-[var(--text-muted)]">{c.version}</div>
                  <div className="text-center">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] ${c.is_active ? "bg-emerald-50 text-emerald-700" : "bg-gray-100 text-gray-500"}`}>
                      {c.is_active ? "Yes" : "No"}
                    </span>
                  </div>
                </div>
                {expandedCaseId === c.id && <GoldenCaseExpanded c={c} />}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ─────────── TRACE DIFF & RCA ─────────── */}
      {subTab === "compare" && (
        <div className="space-y-4">
          <div className="card">
            <h2 className="text-sm font-medium mb-2">Trace-Based Comparison</h2>
            <p className="text-[11px] text-[var(--text-muted)] mb-3">
              Compare execution traces side-by-side for the same golden test case across two regression runs.
            </p>
            <div className="grid grid-cols-3 gap-3 mb-3">
              <div>
                <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">Run A</label>
                <select value={regDiffRunA} onChange={e => setRegDiffRunA(e.target.value)} className="input text-xs w-full">
                  <option value="">Select run A…</option>
                  {regRuns.map(r => <option key={r.id} value={r.id}>{r.id.slice(0, 8)} — {r.model} ({r.passed}/{r.num_cases})</option>)}
                </select>
              </div>
              <div>
                <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">Run B</label>
                <select value={regDiffRunB} onChange={e => setRegDiffRunB(e.target.value)} className="input text-xs w-full">
                  <option value="">Select run B…</option>
                  {regRuns.filter(r => r.id !== regDiffRunA).map(r => <option key={r.id} value={r.id}>{r.id.slice(0, 8)} — {r.model} ({r.passed}/{r.num_cases})</option>)}
                </select>
              </div>
              <div>
                <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">Test Case</label>
                <select value={regDiffCaseId} onChange={e => setRegDiffCaseId(e.target.value)} className="input text-xs w-full" disabled={comparableCases.length === 0}>
                  <option value="">{comparableCases.length === 0 ? "Pick runs first" : "Select case…"}</option>
                  {comparableCases.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
              </div>
            </div>
            <button onClick={loadTraceDiff} disabled={!regDiffRunA || !regDiffRunB || !regDiffCaseId} className="btn-primary">Load Diff</button>
          </div>

          {regDiff && <TraceDiffPanel diff={regDiff} onRCA={() => runRCA(regDiffRunA, regDiffCaseId, regDiffRunB)} rcaLoading={rcaLoading} regRCA={regRCA} />}
        </div>
      )}
    </div>
  );
}

// ── Sub-components ───────────────────────────────────────────────────────────

function GoldenCaseExpanded({ c }: { c: any }) {
  return (
    <div className="px-4 py-3 bg-[var(--bg-hover)] border-b border-[var(--border)] grid grid-cols-2 gap-4 text-xs">
      <div className="space-y-2">
        <div>
          <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1 font-medium">Prompt</div>
          <div className="p-2 rounded bg-white border border-[var(--border)] text-xs">{c.prompt}</div>
        </div>
        <div>
          <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1 font-medium">Expected Behavior</div>
          <div className="space-y-1.5 p-2 rounded bg-white border border-[var(--border)]">
            <div className="flex items-center gap-1.5">
              <span className="text-[var(--text-muted)]">Team Strategy:</span>
              {c.strategy ? (
                <span className={`px-1.5 py-0.5 rounded text-[10px] border font-semibold ${c.strategy === "auto" ? "bg-orange-50 text-orange-700 border-orange-200" : "bg-zinc-100 text-zinc-600 border-zinc-200"}`}>
                  {c.strategy === "auto" ? "🤖 auto" : c.strategy}
                </span>
              ) : <span className="px-1.5 py-0.5 rounded bg-gray-100 text-gray-500 text-[10px] border border-gray-200">team default</span>}
              {c.expected_strategy && (
                <span className="text-[10px] text-[var(--text-muted)]">→ expects <span className="font-semibold text-zinc-700">{c.expected_strategy}</span></span>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-1">
              <span className="text-[var(--text-muted)]">Target Agents:</span>
              {(c.expected_delegation_pattern?.length > 1 ? c.expected_delegation_pattern : [c.expected_agent]).map((a: string, idx: number) => (
                <span key={idx} className="flex items-center gap-0.5">
                  {idx > 0 && <span className="text-[9px] text-[var(--text-muted)]">→</span>}
                  <span className="px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-600 text-[10px] border border-zinc-200 font-medium">{a}</span>
                </span>
              ))}
            </div>
            <div><span className="text-[var(--text-muted)]">Tools:</span> {(c.expected_tools || []).map((t: string) => (
              <span key={t} className="inline-block ml-1 px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-500 text-[10px] border border-zinc-200">{t}</span>
            ))}</div>
            <div><span className="text-[var(--text-muted)]">Keywords:</span> {(c.expected_output_keywords || []).map((k: string) => (
              <span key={k} className="inline-block ml-1 px-1 py-0.5 rounded bg-gray-100 text-gray-600 text-[10px]">{k}</span>
            ))}</div>
          </div>
        </div>
      </div>
      <div className="space-y-2">
        <div>
          <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1 font-medium">Budgets</div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 p-2 rounded bg-white border border-[var(--border)]">
            <div><span className="text-[var(--text-muted)]">Max LLM Calls:</span> <span className="font-semibold">{c.max_llm_calls}</span></div>
            <div><span className="text-[var(--text-muted)]">Max Tool Calls:</span> <span className="font-semibold">{c.max_tool_calls}</span></div>
            <div><span className="text-[var(--text-muted)]">Max Tokens:</span> <span className="font-semibold">{c.max_tokens?.toLocaleString()}</span></div>
            <div><span className="text-[var(--text-muted)]">Max Latency:</span> <span className="font-semibold">{c.max_latency_ms ? `${(c.max_latency_ms / 1000).toFixed(0)}s` : "—"}</span></div>
          </div>
        </div>
        <div>
          <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1 font-medium">Reference Output</div>
          <div className="p-2 rounded bg-white border border-[var(--border)] max-h-40 overflow-y-auto">
            <Md content={c.reference_output || "—"} />
          </div>
        </div>
      </div>
    </div>
  );
}

/** Renders the full agent trajectory from a full_trace array */
function TrajectoryView({ trace, tools }: { trace: any[]; tools: string[] }) {
  const steps = trace.filter(e => e.step === "execution" || e.step === "routing");
  if (!steps.length) {
    return (
      <div className="text-[10px] text-[var(--text-muted)] italic px-2">
        {tools.length > 0 ? `Tools used: ${tools.join(", ")}` : "No trace data available"}
      </div>
    );
  }
  return (
    <div className="space-y-1.5">
      {steps.map((entry, i) => {
        if (entry.step === "routing") {
          return (
            <div key={i} className="flex items-center gap-1.5 text-[10px] text-blue-600">
              <span className="w-4 h-4 flex items-center justify-center rounded bg-blue-50 border border-blue-200 font-mono text-[9px]">R</span>
              <span>Router → <strong>{entry.selected_agent || "?"}</strong></span>
            </div>
          );
        }
        const toolCalls: any[] = entry.tool_calls || [];
        return (
          <div key={i} className="rounded border border-[var(--border)] overflow-hidden">
            <div className="flex items-center gap-1.5 px-2 py-1 bg-zinc-50 border-b border-[var(--border)]">
              <span className="w-4 h-4 flex items-center justify-center rounded bg-zinc-200 text-zinc-600 font-mono text-[9px]">{i + 1}</span>
              <span className="text-[10px] font-semibold text-zinc-700">{entry.agent || "agent"}</span>
              {entry.model && <span className="text-[9px] text-[var(--text-muted)] ml-1 font-mono">{String(entry.model).split("/").pop()}</span>}
              <span className="ml-auto text-[9px] text-[var(--text-muted)]">{toolCalls.length} tool call{toolCalls.length !== 1 ? "s" : ""}</span>
            </div>
            {toolCalls.length > 0 && (
              <div className="px-2 py-1.5 space-y-1">
                {toolCalls.map((tc: any, ti: number) => (
                  <div key={ti} className="flex items-start gap-1.5 text-[10px]">
                    <span className="mt-0.5 w-3 h-3 flex items-center justify-center rounded bg-violet-50 border border-violet-200 text-violet-600 font-mono text-[8px] shrink-0">⚙</span>
                    <div className="min-w-0">
                      <span className="font-mono font-semibold text-violet-700">{tc.tool || tc.name || "?"}</span>
                      {tc.args && Object.keys(tc.args).length > 0 && (
                        <span className="text-[var(--text-muted)] ml-1 truncate">
                          ({Object.entries(tc.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ").slice(0, 80)})
                        </span>
                      )}
                      {tc.result !== undefined && tc.result !== null && (
                        <div className="text-[var(--text-muted)] truncate mt-0.5">→ {String(tc.result).slice(0, 120)}</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function CaseDetailPanel({ detail, runId, baselineRunId, onClose, onRCA, rcaLoading, regRCA }: any) {
  const res = detail.result || {};
  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">
          Detail: {detail.golden_case?.name || res.golden_case_name}
        </h3>
        <div className="flex gap-2">
          <button onClick={() => onRCA(runId, res.golden_case_id, baselineRunId)}
            disabled={rcaLoading} className="btn-secondary text-xs">
            {rcaLoading ? "Analyzing…" : "Run RCA"}
          </button>
          <button onClick={onClose} className="text-xs text-[var(--text-muted)]">✕ Close</button>
        </div>
      </div>

      {/* Trajectory */}
      <div>
        <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1.5 font-medium flex items-center gap-2">
          Agent Trajectory & Tool Calls
          {res.actual_delegation_pattern?.length > 0 && (
            <span className="flex items-center gap-0.5">
              {res.actual_delegation_pattern.map((a: string, i: number) => (
                <span key={i} className="flex items-center gap-0.5">
                  {i > 0 && <span className="text-[8px] text-[var(--text-muted)]">→</span>}
                  <span className="px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-600 text-[9px] border border-zinc-200 font-medium">{a}</span>
                </span>
              ))}
            </span>
          )}
        </div>
        <TrajectoryView trace={res.full_trace || []} tools={res.actual_tools || []} />
      </div>

      {/* Expected vs Actual */}
      <div className="rounded-lg border border-[var(--border)] overflow-hidden">
        <div className="grid grid-cols-2 divide-x divide-[var(--border)]">
          <div>
            <div className="flex items-center gap-1.5 px-3 py-2 bg-emerald-50 border-b border-[var(--border)]">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              <span className="text-[10px] font-semibold uppercase tracking-wide text-emerald-700">Expected Output</span>
            </div>
            <div className="p-3 max-h-60 overflow-y-auto bg-emerald-50/40">
              <Md content={detail.golden_case?.reference_output || "—"} />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-1.5 px-3 py-2 bg-zinc-50 border-b border-[var(--border)]">
              <span className="w-2 h-2 rounded-full bg-zinc-400" />
              <span className="text-[10px] font-semibold uppercase tracking-wide text-zinc-600">Actual Output</span>
              {res.semantic_similarity != null && (
                <span className={`ml-auto text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
                  res.semantic_similarity >= 0.75 ? "bg-emerald-100 text-emerald-700"
                  : res.semantic_similarity >= 0.5 ? "bg-amber-100 text-amber-700"
                  : "bg-red-100 text-red-700"
                }`}>
                  {(res.semantic_similarity * 100).toFixed(1)}% match
                </span>
              )}
            </div>
            <div className="p-3 max-h-60 overflow-y-auto bg-zinc-50">
              <Md content={res.actual_output || "—"} />
            </div>
          </div>
        </div>
      </div>

      {/* Radar chart for scores overview */}
      {(() => {
        const qMetrics = Object.keys(res.quality_scores || {}).map(k => ({ key: k, label: k.replace(/_/g, " ") }));
        const deMetrics = DEEPEVAL_METRICS.filter(m => res.deepeval_scores?.[m.key] != null);
        const allMetrics = [...qMetrics, ...deMetrics];
        const allScores = { ...(res.quality_scores || {}), ...(res.deepeval_scores || {}) };
        if (allMetrics.length < 3) return null;
        return (
          <div className="p-3 rounded border border-[var(--border)] bg-[var(--bg)]">
            <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1 font-medium">Score Overview</div>
            <SingleRadarChart scoresMap={allScores} metrics={allMetrics} color="#3b82f6" label="Score" />
          </div>
        );
      })()}

      {/* Scores */}
      <div className="grid grid-cols-2 gap-3">
        {res.quality_scores && Object.keys(res.quality_scores).length > 0 && (
          <div>
            <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1">G-Eval (LLM-as-Judge)</div>
            <div className="space-y-1.5">
              {Object.entries(res.quality_scores).map(([k, v]: [string, any]) => (
                <div key={k} className="p-2 rounded bg-[var(--bg)] border border-[var(--border)]">
                  <div className="flex items-center justify-between mb-0.5">
                    <span className="text-xs font-medium capitalize">{k.replace(/_/g, " ")}</span>
                    <span className={`text-xs font-mono font-bold ${Number(v) >= 0.7 ? "text-emerald-600" : Number(v) >= 0.4 ? "text-amber-600" : "text-red-600"}`}>
                      {(Number(v) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="h-1 bg-gray-200 rounded-full overflow-hidden">
                    <div className="h-full rounded-full" style={{ width: `${Math.min(100, Number(v) * 100)}%`, backgroundColor: Number(v) >= 0.7 ? "#059669" : Number(v) >= 0.4 ? "#ca8a04" : "#dc2626" }} />
                  </div>
                  {res.eval_reasoning?.[k] && (
                    <div className="text-[10px] text-[var(--text-muted)] italic mt-1">{res.eval_reasoning[k]}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        <div>
          <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1">DeepEval Agentic Metrics</div>
          {res.deepeval_scores && Object.keys(res.deepeval_scores).length > 0 ? (
            <div className="space-y-1.5">
              {DEEPEVAL_METRICS.map(({ key, label, reasonKey, description }) => {
                const ds = res.deepeval_scores;
                const val = ds[key];
                if (val === undefined || val === null) return null;
                const numVal = typeof val === "number" ? val : parseFloat(val) || 0;
                const reason = ds[reasonKey] || "";
                return (
                  <div key={key} className="rounded bg-[var(--bg)] border border-[var(--border)] overflow-hidden">
                    <div className="flex items-center justify-between p-2">
                      <div className="flex-1 min-w-0">
                        <span className="text-xs font-medium">{label}</span>
                        <span className="text-[10px] text-[var(--text-muted)] ml-2">{description}</span>
                      </div>
                      <div className="flex items-center gap-2 ml-2 shrink-0">
                        <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                          <div className="h-full rounded-full" style={{ width: `${Math.min(100, numVal * 100)}%`, backgroundColor: numVal >= 0.7 ? "#059669" : numVal >= 0.4 ? "#ca8a04" : "#dc2626" }} />
                        </div>
                        <span className={`text-xs font-mono font-bold min-w-[2.5rem] text-right ${numVal >= 0.7 ? "text-emerald-600" : numVal >= 0.4 ? "text-amber-600" : "text-red-600"}`}>
                          {(numVal * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    {reason && (
                      <div className="px-2 pb-2">
                        <div className="text-[10px] text-[var(--text-muted)] bg-[var(--bg-secondary,var(--bg))] rounded p-2 leading-relaxed">
                          <span className="font-semibold">Reasoning: </span>{reason}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-[10px] text-[var(--text-muted)] p-3 bg-[var(--bg)] rounded border border-[var(--border)]">
              DeepEval metrics not available.
            </div>
          )}
        </div>
      </div>

      {/* RCA */}
      {regRCA && (
        <div>
          <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1">Root Cause Analysis</div>
          <div className="p-3 rounded bg-amber-50 border border-amber-200 text-xs">
            <Md content={typeof regRCA === "string" ? regRCA : JSON.stringify(regRCA, null, 2)} />
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Radar chart for a single run's scores (used in CaseDetailPanel).
 * metrics: array of { key, label } — values pulled from scoresMap.
 */
function SingleRadarChart({
  scoresMap,
  metrics,
  color = "#3b82f6",
  label = "Score",
}: {
  scoresMap: Record<string, any>;
  metrics: { key: string; label: string }[];
  color?: string;
  label?: string;
}) {
  const data = metrics
    .filter(m => scoresMap[m.key] != null)
    .map(m => ({
      metric: m.label.length > 12 ? m.label.slice(0, 11) + "…" : m.label,
      fullLabel: m.label,
      value: Math.round((Number(scoresMap[m.key]) || 0) * 100),
    }));
  if (data.length < 3) return null;
  return (
    <ResponsiveContainer width="100%" height={220}>
      <RadarChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
        <PolarGrid stroke="#e5e7eb" />
        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: "#6b7280" }} />
        <Radar name={label} dataKey="value" stroke={color} fill={color} fillOpacity={0.25} strokeWidth={2} dot={{ r: 3, fill: color }} />
        <Legend iconSize={8} wrapperStyle={{ fontSize: 10 }} />
      </RadarChart>
    </ResponsiveContainer>
  );
}

/**
 * Radar chart overlapping two runs (used in TraceDiffPanel).
 * labelA / labelB are the legend labels.
 */
function DiffRadarChart({
  scoresA,
  scoresB,
  metrics,
  labelA = "Run A",
  labelB = "Run B",
}: {
  scoresA: Record<string, any>;
  scoresB: Record<string, any>;
  metrics: { key: string; label: string }[];
  labelA?: string;
  labelB?: string;
}) {
  const data = metrics
    .filter(m => scoresA[m.key] != null || scoresB[m.key] != null)
    .map(m => ({
      metric: m.label.length > 12 ? m.label.slice(0, 11) + "…" : m.label,
      fullLabel: m.label,
      A: scoresA[m.key] != null ? Math.round(Number(scoresA[m.key]) * 100) : 0,
      B: scoresB[m.key] != null ? Math.round(Number(scoresB[m.key]) * 100) : 0,
    }));
  if (data.length < 3) return null;

  // Small numeric legend under the chart
  const deltas = data.map(d => ({ label: d.fullLabel, A: d.A, B: d.B, delta: d.A - d.B }));

  return (
    <div>
      <ResponsiveContainer width="100%" height={240}>
        <RadarChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
          <PolarGrid stroke="#e5e7eb" />
          <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: "#6b7280" }} />
          <Radar name={labelA} dataKey="A" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} strokeWidth={2} dot={{ r: 3, fill: "#3b82f6" }} />
          <Radar name={labelB} dataKey="B" stroke="#f97316" fill="#f97316" fillOpacity={0.2} strokeWidth={2} dot={{ r: 3, fill: "#f97316" }} />
          <Legend iconSize={8} wrapperStyle={{ fontSize: 10 }} />
        </RadarChart>
      </ResponsiveContainer>
      {/* Compact score table under radar */}
      <div className="mt-1 grid grid-cols-2 gap-x-4 gap-y-0.5">
        {deltas.map(d => (
          <div key={d.label} className="flex items-center justify-between text-[10px]">
            <span className="text-[var(--text-muted)] truncate capitalize">{d.label.replace(/_/g, " ")}</span>
            <div className="flex items-center gap-1.5 shrink-0 ml-1">
              <span className="text-blue-600 font-mono">{d.A}%</span>
              <span className="text-[var(--text-muted)]">vs</span>
              <span className="text-orange-500 font-mono">{d.B}%</span>
              {d.delta !== 0 && (
                <span className={`text-[9px] px-1 py-0.5 rounded font-mono ${d.delta > 0 ? "bg-blue-50 text-blue-600" : "bg-orange-50 text-orange-600"}`}>
                  {d.delta > 0 ? "+" : ""}{d.delta}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TraceDiffPanel({ diff, onRCA, rcaLoading, regRCA }: any) {
  const rA = diff.run_a || {};
  const rB = diff.run_b || {};

  const fmtAgents = (r: any) => (r.actual_delegation_pattern?.length ? r.actual_delegation_pattern : [r.actual_agent]).filter(Boolean).join(" → ") || "—";

  // Collect all metric keys across both runs
  const allQualityKeys = Array.from(new Set([
    ...Object.keys(rA.quality_scores || {}),
    ...Object.keys(rB.quality_scores || {}),
  ]));
  const allDeepEvalKeys = DEEPEVAL_METRICS.filter(m =>
    (rA.deepeval_scores || {})[m.key] != null || (rB.deepeval_scores || {})[m.key] != null
  );

  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Trace Diff — {diff.case_id}</h3>
        <button onClick={onRCA} disabled={rcaLoading} className="btn-secondary text-xs">
          {rcaLoading ? "Analyzing…" : "Run RCA"}
        </button>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-[10px]">
        <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-blue-400 inline-block" /> Run A ({String(rA.id || "").slice(0, 8)})</div>
        <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-orange-400 inline-block" /> Run B ({String(rB.id || "").slice(0, 8)})</div>
      </div>

      {/* Side-by-side stats + trajectory */}
      <div className="grid grid-cols-2 gap-3">
        {[
          { label: "Run A", r: rA, color: "blue" },
          { label: "Run B", r: rB, color: "orange" },
        ].map(({ label, r, color }) => (
          <div key={label} className={`rounded-lg border overflow-hidden ${color === "blue" ? "border-blue-200" : "border-orange-200"}`}>
            <div className={`px-3 py-2 border-b flex items-center justify-between ${color === "blue" ? "bg-blue-50 border-blue-200" : "bg-orange-50 border-orange-200"}`}>
              <span className={`text-[11px] font-semibold ${color === "blue" ? "text-blue-700" : "text-orange-700"}`}>{label}</span>
              <span className="font-mono text-[10px] text-[var(--text-muted)]">{String(r.id || "").slice(0, 8)}</span>
              <PassFailBadge pass={r.overall_pass} />
            </div>
            <div className="p-3 space-y-2">
              {/* Agents */}
              <div className="text-[10px]">
                <span className="text-[var(--text-muted)]">Trajectory: </span>
                <span className="font-medium">{fmtAgents(r)}</span>
              </div>
              {/* Tools */}
              {r.actual_tools?.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {r.actual_tools.map((t: string, i: number) => (
                    <span key={i} className="px-1.5 py-0.5 rounded bg-violet-50 text-violet-600 text-[9px] border border-violet-200 font-mono">{t}</span>
                  ))}
                </div>
              )}
              {/* Stats row */}
              <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px]">
                <span><span className="text-[var(--text-muted)]">LLM calls:</span> <strong>{r.actual_llm_calls ?? "—"}</strong></span>
                <span><span className="text-[var(--text-muted)]">Tool calls:</span> <strong>{r.actual_tool_calls ?? "—"}</strong></span>
                <span><span className="text-[var(--text-muted)]">Latency:</span> <strong>{r.actual_latency_ms?.toFixed(0) ?? "—"}ms</strong></span>
                <span><span className="text-[var(--text-muted)]">Cost:</span> <strong>${r.actual_cost?.toFixed(4) ?? "—"}</strong></span>
                <span><span className="text-[var(--text-muted)]">Similarity:</span> <strong className={r.semantic_similarity >= 0.7 ? "text-emerald-600" : "text-amber-600"}>{r.semantic_similarity != null ? `${(r.semantic_similarity * 100).toFixed(1)}%` : "—"}</strong></span>
                {r.actual_strategy && <span><span className="text-[var(--text-muted)]">Strategy:</span> <strong>{r.actual_strategy}</strong></span>}
              </div>
              {/* Trajectory details */}
              {(r.full_trace?.length > 0) && (
                <details className="text-[10px]">
                  <summary className="cursor-pointer text-[var(--text-muted)] hover:text-[var(--text)] select-none">Tool call details ▸</summary>
                  <div className="mt-1.5">
                    <TrajectoryView trace={r.full_trace || []} tools={r.actual_tools || []} />
                  </div>
                </details>
              )}
              {/* Output */}
              <div>
                <div className="text-[10px] text-[var(--text-muted)] uppercase font-medium mb-0.5">Output</div>
                <div className="p-2 rounded bg-[var(--bg)] border border-[var(--border)] max-h-40 overflow-y-auto">
                  <Md content={r.actual_output || "—"} />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Radar score comparison chart */}
      {(allQualityKeys.length > 0 || allDeepEvalKeys.length > 0) && (() => {
        const qMetrics = allQualityKeys.map(k => ({ key: k, label: k.replace(/_/g, " ") }));
        const deMetrics = allDeepEvalKeys;
        const allMetrics = [...qMetrics, ...deMetrics];
        const scoresA = { ...(rA.quality_scores || {}), ...(rA.deepeval_scores || {}) };
        const scoresB = { ...(rB.quality_scores || {}), ...(rB.deepeval_scores || {}) };
        return (
          <div className="p-3 rounded border border-[var(--border)] bg-[var(--bg)]">
            <div className="text-[10px] text-[var(--text-muted)] uppercase mb-2 font-medium flex items-center gap-2">
              Score Comparison — Radar View
              <span className="text-[9px] font-normal normal-case text-blue-600">● Run A</span>
              <span className="text-[9px] font-normal normal-case text-orange-500">● Run B</span>
            </div>
            <DiffRadarChart
              scoresA={scoresA}
              scoresB={scoresB}
              metrics={allMetrics}
              labelA="Run A"
              labelB="Run B"
            />
          </div>
        );
      })()}

      {/* RCA */}
      {regRCA && (
        <div>
          <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1">Root Cause Analysis</div>
          <div className="p-3 rounded bg-amber-50 border border-amber-200 text-xs">
            <Md content={typeof regRCA === "string" ? regRCA : JSON.stringify(regRCA, null, 2)} />
          </div>
        </div>
      )}
    </div>
  );
}
