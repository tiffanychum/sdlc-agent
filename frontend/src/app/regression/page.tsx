"use client";
import React, { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { api } from "@/lib/api";
import { useRegressionRun } from "@/contexts/RegressionRunContext";
import { useTeam } from "@/contexts/TeamContext";
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

type SubTab = "golden" | "run" | "results" | "ab" | "perf" | "prompts";

export default function RegressionPage() {
  const [subTab, setSubTab] = useState<SubTab>("run");
  // Team selection is shared across pages via <TeamProvider> in layout.tsx so
  // switching teams here (or anywhere) persists and doesn't snap back on
  // navigation. Regression runs + A/B options are scoped by this teamId.
  const { teamId, setTeamId, teams, selectedTeam } = useTeam();

  // golden dataset
  const [goldenCases, setGoldenCases] = useState<any[]>([]);
  const [expandedCaseId, setExpandedCaseId] = useState<string | null>(null);

  // run tests — default model: claude-sonnet-4-6
  const [regModels, setRegModels] = useState<Set<string>>(new Set(["claude-sonnet-4-6"]));
  const [regPromptVer, setRegPromptVer] = useState("v1");
  const [regBaselineRunId, setRegBaselineRunId] = useState("");
  const [selectedCaseIds, setSelectedCaseIds] = useState<Set<string>>(new Set());
  const [caseSearch, setCaseSearch] = useState("");
  const [regRunning, setRegRunning] = useState(false);
  const [regRunResult, setRegRunResult] = useState<any>(null);

  // live streaming — delegated to global RegressionRunContext
  const regrCtx = useRegressionRun();
  const liveBottomRef = useRef<HTMLDivElement>(null);

  // results
  const [regRuns, setRegRuns] = useState<any[]>([]);
  const [runsSearch, setRunsSearch] = useState("");
  const [runsSort, setRunsSort] = useState<"date" | "pass" | "cost">("date");
  const [regSelectedRunId, setRegSelectedRunId] = useState("");
  const [regResults, setRegResults] = useState<any>(null);
  const [regCaseDetail, setRegCaseDetail] = useState<any>(null);
  const [resultsSearch, setResultsSearch] = useState("");
  const [resultsFilter, setResultsFilter] = useState<"all" | "pass" | "fail">("all");

  // A/B comparison
  const [abGoldenId, setAbGoldenId] = useState("");
  const [abOptions, setAbOptions] = useState<any[]>([]);   // available runs for chosen golden
  const [abOptionsLoading, setAbOptionsLoading] = useState(false);
  const [abRunIdA, setAbRunIdA] = useState("");
  const [abRunIdB, setAbRunIdB] = useState("");
  const [abResult, setAbResult] = useState<any>(null);
  const [abLoading, setAbLoading] = useState(false);
  const [abExpandAgent, setAbExpandAgent] = useState<string | null>(null);

  // A/B v2: cross-project workflow (pick runs first, then overlapping golden)
  const [abScope, setAbScope] = useState<"same" | "cross">("same");
  const [abV2Runs, setAbV2Runs] = useState<any[]>([]);
  const [abV2RunsLoading, setAbV2RunsLoading] = useState(false);
  const [abV2Overlap, setAbV2Overlap] = useState<any>(null);
  const [abV2OverlapLoading, setAbV2OverlapLoading] = useState(false);
  // Adaptive LLM judge panel
  const [abJudgeResult, setAbJudgeResult] = useState<any>(null);
  const [abJudgeLoading, setAbJudgeLoading] = useState(false);
  const [abJudgeError, setAbJudgeError] = useState<string>("");

  // Performance analysis (moved from monitoring)
  const [perfReport, setPerfReport] = useState<any>(null);
  const [perfDays, setPerfDays] = useState(7);
  const [regressionInsights, setRegressionInsights] = useState<any>(null);
  const [expandedAnomalies, setExpandedAnomalies] = useState<Set<number>>(new Set());

  // Prompt versions (moved from monitoring)
  const [promptVersionsReg, setPromptVersionsReg] = useState<any>(null);
  const [selectedRoleReg, setSelectedRoleReg] = useState("coder");
  const [promptDiffReg, setPromptDiffReg] = useState<any>(null);
  const [diffOldReg, setDiffOldReg] = useState("v1");
  const [diffNewReg, setDiffNewReg] = useState("v2");
  const [promptAbVersionAReg, setPromptAbVersionAReg] = useState("v1");
  const [promptAbVersionBReg, setPromptAbVersionBReg] = useState("v2");
  const [promptAbResultReg, setPromptAbResultReg] = useState<any>(null);
  const [promptAbLoadingReg, setPromptAbLoadingReg] = useState(false);
  const [optimizeLoading, setOptimizeLoading] = useState(false);
  const [optimizeResult, setOptimizeResult] = useState<any>(null);
  const [showOptimizeSetup, setShowOptimizeSetup] = useState(false);
  // Multi-role optimisation: Set keeps selection deterministic and unique.
  const [optimizeRoles, setOptimizeRoles] = useState<Set<string>>(new Set(["coder"]));
  // Per-role target version. When a role has no entry here we send "latest".
  const [optimizeVersions, setOptimizeVersions] = useState<Record<string, string>>({});
  const [optimizeMetric, setOptimizeMetric] = useState("step_efficiency");
  const [optimizeThreshold, setOptimizeThreshold] = useState("0.7");
  // Baseline mode: pick a past run ('run') or let the optimizer bootstrap
  // ('fresh'). We recommend 'run' because it gives deterministic baselines.
  const [optimizeBaselineMode, setOptimizeBaselineMode] = useState<"run" | "fresh">("run");
  const [optimizeBaselineRunId, setOptimizeBaselineRunId] = useState<string>("");
  // Explicit golden subset. Empty = let the optimizer derive from failures.
  const [optimizeGoldenIds, setOptimizeGoldenIds] = useState<Set<string>>(new Set());
  const [optimizeEarlyExit, setOptimizeEarlyExit] = useState(true);
  const [optimizeCommitOnPlateau, setOptimizeCommitOnPlateau] = useState(false);
  const [optimizeDryRun, setOptimizeDryRun] = useState(false);
  // One trajectory panel per role so parallel runs stay legible.
  const [optimizeTrajectory, setOptimizeTrajectory] = useState<Record<string, Array<{type: string; text: string; ts: number}>>>({});
  const [optimizeActiveRoleTab, setOptimizeActiveRoleTab] = useState<string>("");
  const optimizeAbortRef = useRef<AbortController | null>(null);
  const trajectoryEndRef = useRef<HTMLDivElement>(null);

  // config
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [promptVersions, setPromptVersions] = useState<any[]>([]);
  const [showBaselineInfo, setShowBaselineInfo] = useState(false);
  // Per-role versions loaded from PromptRegistry (keyed by role slug)
  const [roleVersionsMap, setRoleVersionsMap] = useState<Record<string, any[]>>({});
  // Per-role selected prompt version for the upcoming run
  const [rolePromptVerMap, setRolePromptVerMap] = useState<Record<string, string>>({});
  // Roles that belong to the *currently-selected* team (e.g. sdlc_2_0 →
  // ["builder", "planner_v2"]).  Used to scope every role-list in this page
  // (Run Tests, Prompt Versioning, Prompt Optimizer) so project A can't
  // see / edit / run project B's agents.  The three routing roles
  // (supervisor / meta_router / router) are considered global and always
  // included — they configure the orchestrator, not the team's own agents.
  const [teamAgentRoles, setTeamAgentRoles] = useState<string[]>([]);

  // The canonical "common" routing roles.  After Patch 5 these become
  // team-scoped rows in the registry, but for UI-filter purposes they are
  // always visible to every team.
  const COMMON_ROUTING_ROLES = ["supervisor", "meta_router", "router"] as const;

  // Helper: return a copy of a role→X map (e.g. registry versions) scoped to
  // the active team's agents + common routing roles.  Falls through to the
  // full map until the team data has loaded, so the UI doesn't flash empty.
  const scopeRolesToTeam = useCallback(<T,>(all: Record<string, T>): Record<string, T> => {
    if (!teamAgentRoles.length) return all;
    const allow = new Set<string>([...teamAgentRoles, ...COMMON_ROUTING_ROLES]);
    const out: Record<string, T> = {};
    for (const [role, v] of Object.entries(all)) {
      if (allow.has(role)) out[role] = v;
    }
    return out;
  }, [teamAgentRoles]);

  // `selectedTeam` is provided by useTeam() above.

  useEffect(() => {
    api.models.list().then(setAvailableModels).catch(() => {});
    api.promptVersions.list().then(setPromptVersions).catch(() => {});
  }, []);

  // Per-role prompt-version defaults depend on which team is selected — the
  // dev team has {coder, qa, devops, …} whereas sdlc_2_0 has {builder,
  // planner_v2}. Re-resolve whenever teamId changes so the Studio settings
  // for the active team drive the per-role defaults on the Run Tests tab.
  useEffect(() => {
    if (!teamId) return;
    Promise.all([
      // Patch 5: pass teamId so supervisor/router return team-scoped rows
      // (with automatic fallback to global rows for brand-new teams).
      api.prompts.versions(undefined, teamId).catch(() => ({ roles: {} })),
      api.teams.get(teamId).catch(() => ({ agents: [] })),
    ]).then(([versionsRes, teamData]: [any, any]) => {
      const allRoles = versionsRes?.roles || {};
      const agentRoles = (teamData?.agents || [])
        .map((a: any) => a.role)
        .filter((r: any): r is string => Boolean(r));
      setTeamAgentRoles(agentRoles);

      // Scope the Run Tests role list to this team's agents + common
      // routing roles only.  Without this, selecting sdlc_2_0 would still
      // show dev-team roles (coder, qa, devops, …) in the prompt-version
      // selector, which is confusing and lets users override prompts for
      // agents that aren't even in the active team.
      const allow = new Set<string>([...agentRoles, ...COMMON_ROUTING_ROLES]);
      const scopedRoles: Record<string, any[]> = {};
      for (const [role, versions] of Object.entries(allRoles) as [string, any[]][]) {
        if (allow.has(role)) scopedRoles[role] = versions;
      }
      setRoleVersionsMap(scopedRoles);

      const studioVersions: Record<string, string> = {};
      for (const agent of (teamData?.agents || [])) {
        if (agent.role && agent.prompt_version) studioVersions[agent.role] = agent.prompt_version;
      }
      // Default each role to the *latest* registry version.
      // list_versions is ordered newest-first (created_at.desc) so
      // versions[0] is always the latest, even when the only row is v1.
      // If Studio has pinned a non-v1 version for the role, respect that
      // pin (the user explicitly chose it); otherwise pick registry[0].
      const defaults: Record<string, string> = {};
      for (const [role, versions] of Object.entries(scopedRoles) as [string, any[]][]) {
        const allVersions = versions as any[];
        const latestVer = allVersions[0]?.version;
        const studioVer = studioVersions[role];
        if (studioVer && studioVer !== "v1") {
          defaults[role] = studioVer;
        } else if (latestVer) {
          defaults[role] = latestVer;
        }
      }
      setRolePromptVerMap(defaults);
    });
  }, [teamId]);

  // When the active team changes, reset the in-view run / A/B selection so
  // we don't keep displaying a dev-team run while the user is on sdlc_2_0.
  useEffect(() => {
    setRegSelectedRunId("");
    setRegResults(null);
    setRegCaseDetail(null);
    setAbGoldenId("");
    setAbOptions([]);
    setAbRunIdA("");
    setAbRunIdB("");
    setAbResult(null);
    setAbV2Runs([]);
    setAbV2Overlap(null);
    setAbJudgeResult(null);
    setAbJudgeError("");
  }, [teamId]);

  // If the currently-selected role in the Prompt Versioning tab, or the
  // roles chosen in the Prompt Optimizer, no longer belong to the active
  // team (e.g. user switched from dev team → sdlc_2_0 with {builder,
  // planner_v2}), snap them back to a valid role so the right-hand panels
  // don't show empty state / operate on invisible rows.
  useEffect(() => {
    if (!teamAgentRoles.length) return;
    const allow = new Set<string>([...teamAgentRoles, ...COMMON_ROUTING_ROLES]);
    if (!allow.has(selectedRoleReg)) {
      setSelectedRoleReg(teamAgentRoles[0]);
      setPromptDiffReg(null);
      setPromptAbResultReg(null);
    }
    setOptimizeRoles(prev => {
      const filtered = Array.from(prev).filter(r => allow.has(r));
      if (filtered.length === prev.size) return prev;
      return new Set<string>(filtered.length ? filtered : [teamAgentRoles[0]]);
    });
  }, [teamAgentRoles, selectedRoleReg]);

  const loadGolden = useCallback(async () => {
    try { const c = await api.golden.list(); setGoldenCases(c); } catch { /* ignore */ }
  }, []);

  const loadRegRuns = useCallback(async () => {
    try { const r = await api.regression.runs(teamId || undefined); setRegRuns(r); } catch { /* ignore */ }
  }, [teamId]);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const loadPerfReport = useCallback(async (days: number) => {
    try {
      const res = await fetch(`${API_BASE}/api/traces/performance-report?days=${days}`);
      setPerfReport(await res.json());
      setExpandedAnomalies(new Set());
    } catch { /* ignore */ }
  }, [API_BASE]);

  const loadRegressionInsights = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/traces/performance-report/regression-insights?days=90`);
      setRegressionInsights(await res.json());
    } catch { /* ignore */ }
  }, [API_BASE]);

  const loadPromptVersionsReg = useCallback(async () => {
    try {
      // Patch 5: scope to the active team so supervisor/router return the
      // team's own version history; agent roles are returned globally.
      const qs = teamId ? `?team_id=${encodeURIComponent(teamId)}` : "";
      const res = await fetch(`${API_BASE}/api/prompts/versions${qs}`);
      setPromptVersionsReg(await res.json());
    } catch { /* ignore */ }
  }, [API_BASE, teamId]);

  const loadPromptDiffReg = useCallback(async (role: string, vOld: string, vNew: string) => {
    try {
      // Thread team_id so the diff for team-scoped roles (supervisor, router)
      // compares the correct per-team rows, not the global fallback.
      const params = new URLSearchParams({ role, version_old: vOld, version_new: vNew });
      if (teamId) params.set("team_id", teamId);
      const res = await fetch(`${API_BASE}/api/prompts/diff?${params.toString()}`);
      setPromptDiffReg(await res.json());
    } catch { /* ignore */ }
  }, [API_BASE, teamId]);

  const runPromptAbCompareReg = useCallback(async (role: string, va: string, vb: string) => {
    setPromptAbLoadingReg(true);
    setPromptAbResultReg(null);
    try {
      const res = await fetch(`${API_BASE}/api/prompts/ab-compare?role=${role}&version_a=${va}&version_b=${vb}`);
      setPromptAbResultReg(await res.json());
    } catch { /* ignore */ }
    finally { setPromptAbLoadingReg(false); }
  }, [API_BASE]);

  const runOptimize = useCallback(async (opts: {
    roles: string[];
    versions: Record<string, string>;
    metric: string;
    threshold: string;
    baselineRunId: string | null;
    goldenIds: string[];
    earlyExit: boolean;
    commitOnPlateau: boolean;
    dryRun: boolean;
    teamId: string;
  }) => {
    setOptimizeLoading(true);
    setOptimizeResult(null);
    // Reset per-role trajectory buckets so parallel runs stay isolated.
    const initial: Record<string, Array<{type: string; text: string; ts: number}>> = {};
    for (const r of opts.roles) initial[r] = [];
    setOptimizeTrajectory(initial);
    setOptimizeActiveRoleTab(opts.roles[0] || "");

    const addEvent = (role: string, type: string, text: string) =>
      setOptimizeTrajectory(prev => ({
        ...prev,
        [role]: [...(prev[role] || []), { type, text, ts: Date.now() }],
      }));

    // Used by the "Stop" button to cancel the SSE stream mid-flight.
    const controller = new AbortController();
    optimizeAbortRef.current = controller;

    const finalByRole: Record<string, string> = {};

    try {
      const res = await fetch(`${API_BASE}/api/prompts/optimize/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          roles: opts.roles,
          versions: opts.versions,
          metric: opts.metric,
          threshold: parseFloat(opts.threshold) || 0.7,
          model: "claude-sonnet-4.6",
          team_id: opts.teamId,
          baseline_run_id: opts.baselineRunId || undefined,
          golden_ids: opts.goldenIds.length ? opts.goldenIds : undefined,
          early_exit: opts.earlyExit,
          commit_on_plateau: opts.commitOnPlateau,
          dry_run: opts.dryRun,
        }),
        signal: controller.signal,
      });
      if (!res.body) throw new Error("No SSE body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() || "";
        for (const part of parts) {
          if (!part.trim()) continue;
          const lines = part.split("\n");
          let eventType = ""; let dataStr = "";
          for (const line of lines) {
            if (line.startsWith("event: ")) eventType = line.slice(7);
            else if (line.startsWith("data: ")) dataStr = line.slice(6);
          }
          if (!eventType || !dataStr) continue;
          try {
            const d = JSON.parse(dataStr);
            // Global (non-per-role) events keyed under the special bucket "_all".
            const role: string = d.role || "_all";
            if (eventType === "fanout_start") {
              for (const r of (d.roles || [])) {
                addEvent(r, "start", `▶ Starting optimisation — team: ${d.team_id} | parallel cap: ${d.parallel_cap} | baseline: ${d.baseline_run_id || "fresh"} | early_exit: ${d.early_exit}${d.dry_run ? " | DRY-RUN" : ""}${d.commit_on_plateau ? " | commit_on_plateau" : ""}`);
              }
            } else if (eventType === "optimize_start") {
              addEvent(role, "start", `  model: ${d.model} | version: ${d.version} | metric: ${d.metric} | goldens: ${(d.golden_ids || []).length || "auto"}`);
            } else if (eventType === "phase_start") {
              addEvent(role, "phase", `── Phase: ${d.phase} ──`);
            } else if (eventType === "baseline_computed") {
              addEvent(role, "baseline", `📏 Baseline: pass_rate=${(d.pass_rate * 100).toFixed(1)}% | ${d.n_rows} rows`);
            } else if (eventType === "cycle_start") {
              addEvent(role, "cycle_start", `▶ Cycle ${d.cycle} (from ${d.from_version})`);
            } else if (eventType === "llm_drafting") {
              addEvent(role, "drafting", `  ✍  Drafting new prompt (cycle ${d.cycle})…`);
            } else if (eventType === "tool_start") {
              const argPreview = Object.entries(d.args || {}).map(([k,v]) => `${k}=${String(v).slice(0,60)}`).join(", ");
              addEvent(role, "tool", `  ⚙ ${d.tool}${argPreview ? `(${argPreview})` : ""}`);
            } else if (eventType === "tool_end") {
              const passInfo = typeof d.pass_rate === "number" ? ` pass=${(d.pass_rate * 100).toFixed(1)}%` : "";
              const metInfo = typeof d.metric_avg === "number" ? ` metric=${d.metric_avg.toFixed(3)}` : "";
              addEvent(role, "tool_result", `  ↳ ${d.tool}${passInfo}${metInfo}${d.run_id ? ` run_id=${d.run_id}` : ""}`);
            } else if (eventType === "cycle_end") {
              const dp = (d.delta_pass_pp * 100).toFixed(1);
              const dm = d.delta_metric.toFixed(3);
              const icon = d.classification === "crossed" ? "✅"
                        : d.classification === "improved" ? "📈"
                        : d.classification === "plateau"  ? "➖"
                        : d.classification === "marginal" ? "↗"
                        : "📉";
              addEvent(role, "cycle_end", `${icon} Cycle ${d.cycle}: ${d.classification} (ΔPass ${dp >= "0" ? "+" : ""}${dp}pp, Δmetric ${dm >= "0" ? "+" : ""}${dm}) → ${d.loop_decision}`);
            } else if (eventType === "report") {
              const c = d.cycles?.length || 0;
              const winLine = d.winner_cycle != null ? `winner cycle-${d.winner_cycle}` : "no winner";
              const commitLine = d.committed_version
                ? `✅ committed as ${d.committed_version}`
                : `ℹ️ no commit (${d.commit_status})`;
              addEvent(role, "report", `📋 Report — cycles=${c} | ${winLine} | ${commitLine}`);
              addEvent(role, "report", `   ${d.recommendation}`);
              finalByRole[role] = d.recommendation || "";
            } else if (eventType === "done") {
              addEvent(role, "done", `✓ Optimisation complete — ${role} / ${d.status}${d.committed_version ? ` → ${d.committed_version}` : ""}`);
            } else if (eventType === "fanout_done") {
              // swallow — the final "done" per role already logged.
            } else if (eventType === "error") {
              const ctx = d.phase ? ` [phase=${d.phase}${d.cycle ? ` cycle=${d.cycle}` : ""}]` : "";
              addEvent(role, "error", `✗ Error${ctx}: ${d.message}`);
            }
          } catch { /* skip */ }
        }
      }
      setOptimizeResult({ status: "completed", responses: finalByRole });
      await loadPromptVersionsReg();
    } catch (e: any) {
      if (e.name === "AbortError") {
        setOptimizeResult({ status: "stopped", error: "Stopped by user" });
      } else {
        setOptimizeResult({ status: "error", error: e.message });
      }
    } finally {
      optimizeAbortRef.current = null;
      setOptimizeLoading(false);
    }
  }, [API_BASE, loadPromptVersionsReg]);

  const stopOptimize = useCallback(() => {
    optimizeAbortRef.current?.abort();
  }, []);

  useEffect(() => {
    // Auto-scroll to bottom of the *active* role panel as new events stream in.
    const active = optimizeTrajectory[optimizeActiveRoleTab];
    if (active && active.length > 0) trajectoryEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [optimizeTrajectory, optimizeActiveRoleTab]);

  useEffect(() => {
    loadGolden();
    loadRegRuns();
  }, [loadGolden, loadRegRuns]);

  // Refresh run history when the active session finishes
  useEffect(() => {
    const active = regrCtx.sessions.find(s => s.sessionId === regrCtx.activeSessionId);
    if (active && (active.status === "done" || active.status === "error")) {
      loadRegRuns();
    }
  }, [regrCtx.sessions, regrCtx.activeSessionId, loadRegRuns]);

  // ── Run tests — delegates to global RegressionRunContext ───────────────────

  async function runRegression() {
    setRegRunning(true);
    setRegRunResult(null);

    // Send EVERY role's selected version (including v1) so the backend
    // uses exactly what the UI displayed, not the stale Studio-saved pin.
    // Previously v1 entries were filtered out, which caused the backend
    // to fall through to `agent.prompt_version` in the DB — so a user who
    // explicitly picked v1 could still silently get v3 from Studio.
    const pvByRole = Object.fromEntries(
      Object.entries(rolePromptVerMap).filter(([, v]) => Boolean(v))
    );
    const hasPvByRole = Object.keys(pvByRole).length > 0;
    const globalVersion = regPromptVer !== "v1" ? regPromptVer : undefined;
    const models = regModels.size > 0 ? Array.from(regModels) : [undefined as string | undefined];

    const baseParams = {
      team_id: teamId,
      case_ids: selectedCaseIds.size > 0 ? Array.from(selectedCaseIds) : undefined,
      prompt_version: !hasPvByRole ? (globalVersion || "v1") : "v1",
      prompt_versions_by_role: hasPvByRole ? pvByRole : undefined,
      baseline_run_id: regBaselineRunId || undefined,
    };

    try {
      // Each model gets its own session in the global context (shows as separate tab in widget)
      for (const model of models) {
        regrCtx.startRun(
          { ...baseParams, model: model || undefined },
          model ?? "default",
        );
      }
      setRegRunResult({ streamed: true });
      // Give runs a moment to start, then refresh the run history
      setTimeout(() => { loadRegRuns(); }, 3000);
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

  async function onAbGoldenChange(id: string) {
    setAbGoldenId(id);
    setAbOptions([]);
    setAbRunIdA("");
    setAbRunIdB("");
    setAbResult(null);
    if (!id) return;
    setAbOptionsLoading(true);
    try {
      const d = await api.regression.abOptions(id, teamId || undefined);
      const opts = d.options || [];
      setAbOptions(opts);
      // Auto-select: oldest as A, newest as B (if ≥2 distinct runs)
      if (opts.length >= 2) {
        setAbRunIdA(opts[opts.length - 1].run_id);
        setAbRunIdB(opts[0].run_id);
      } else if (opts.length === 1) {
        setAbRunIdA(opts[0].run_id);
        setAbRunIdB(opts[0].run_id);
      }
    } catch { /* ignore */ }
    setAbOptionsLoading(false);
  }

  async function loadAbCompare() {
    if (!abGoldenId) return;
    setAbLoading(true);
    setAbResult(null);
    // A new comparison invalidates the prior judgement — user must click again.
    setAbJudgeResult(null);
    setAbJudgeError("");
    try {
      const d = await api.regression.ab({
        golden_id: abGoldenId,
        run_id_a: abRunIdA || undefined,
        run_id_b: abRunIdB || undefined,
      });
      setAbResult(d);
    } catch { /* ignore */ }
    setAbLoading(false);
  }

  // v2: load ALL runs for the current scope (same-team or cross-team)
  async function loadAbV2Runs(scope: "same" | "cross") {
    setAbScope(scope);
    setAbV2RunsLoading(true);
    setAbV2Runs([]);
    setAbV2Overlap(null);
    setAbJudgeResult(null);
    setAbJudgeError("");
    try {
      const d = await api.regression.abV2Runs({
        scope,
        team_id: scope === "same" ? (teamId || undefined) : undefined,
        limit: 80,
      });
      setAbV2Runs(d.options || []);
    } catch { /* ignore */ }
    setAbV2RunsLoading(false);
  }

  // v2: when both run_ids are set, fetch overlap goldens
  async function loadAbV2Overlap() {
    if (!abRunIdA || !abRunIdB || abRunIdA === abRunIdB) {
      setAbV2Overlap(null);
      return;
    }
    setAbV2OverlapLoading(true);
    try {
      const d = await api.regression.abV2Overlap(abRunIdA, abRunIdB);
      setAbV2Overlap(d);
      // Auto-pick the first overlapping golden where results DIFFER (pass vs fail)
      const items: any[] = d.items || [];
      if (items.length && !abGoldenId) {
        const differ = items.find(it => it.side_a_pass !== it.side_b_pass);
        setAbGoldenId((differ || items[0]).golden_id);
      }
    } catch { /* ignore */ }
    setAbV2OverlapLoading(false);
  }

  // v2: trigger the adaptive LLM judge (cached; force_refresh re-runs Opus)
  async function runAbJudge(force: boolean = false) {
    if (!abRunIdA || !abRunIdB || !abGoldenId) return;
    setAbJudgeLoading(true);
    setAbJudgeError("");
    if (force) setAbJudgeResult(null);
    try {
      const d = await api.regression.abV2Judge({
        run_a: abRunIdA, run_b: abRunIdB, golden_id: abGoldenId,
        force_refresh: force,
      });
      setAbJudgeResult(d);
    } catch (err: any) {
      setAbJudgeError(err?.message || "Judge call failed");
    }
    setAbJudgeLoading(false);
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

  // Lazy-load perf + prompts when those tabs are first visited
  useEffect(() => {
    if (subTab === "perf") { loadPerfReport(perfDays); loadRegressionInsights(); }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [subTab]);

  // Load when the Prompt Versioning tab is first visited, AND refetch
  // whenever the active team changes while on that tab — supervisor/router
  // rows are team-scoped now, so switching team must re-query.
  useEffect(() => {
    if (subTab === "prompts") loadPromptVersionsReg();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [subTab, teamId]);

  // ── Render ─────────────────────────────────────────────────────

  return (
    <div className="space-y-4 max-w-7xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Regression Testing</h1>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">Golden dataset quality gates, A/B comparison, performance analysis & prompt versioning</p>
        </div>
        <div className="flex items-center gap-2">
          <select value={teamId} onChange={e => setTeamId(e.target.value)} className="input !w-auto">
            {teams.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
          </select>
          <button onClick={() => { loadGolden(); loadRegRuns(); }} className="btn-secondary">Refresh</button>
        </div>
      </div>

      {/* Sub-tabs */}
      <div className="flex gap-1 border-b border-[var(--border)] flex-wrap">
        {([
          { id: "run" as SubTab, label: "Run Tests" },
          { id: "results" as SubTab, label: `Results${regRuns.length ? ` (${regRuns.length})` : ""}` },
          { id: "golden" as SubTab, label: `Golden Dataset (${goldenCases.filter(c => c.is_active).length})` },
          { id: "ab" as SubTab, label: "A/B Comparison" },
          { id: "perf" as SubTab, label: "Performance Analysis" },
          { id: "prompts" as SubTab, label: "Prompt Versions" },
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
                <p className="text-[10px] text-[var(--text-muted)]">Defaults to <span className="font-medium text-zinc-600">claude-sonnet-4-6</span>. Select 1+ to compare models side-by-side.</p>
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

              {/* Prompt Version — per-role */}
              <div className="card !p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-[11px] font-medium uppercase text-[var(--text-muted)] tracking-wide">Prompt Versions</label>
                  <span className="text-[10px] text-zinc-400">per role</span>
                </div>
                <p className="text-[10px] text-[var(--text-muted)] leading-snug">
                  Defaults to the latest version per role (from Studio). Override per role if needed.
                </p>
                {Object.keys(roleVersionsMap).length > 0 ? (
                  <div className="space-y-1.5">
                    {Object.entries(roleVersionsMap).map(([role, versions]: [string, any[]]) => {
                      const allVersions = versions as any[];
                      // list_versions returns newest-first, so [0] is the latest
                      // registry version regardless of whether it's v1 or v42.
                      const latestVer = allVersions[0]?.version;
                      const selected = rolePromptVerMap[role] || latestVer || "v1";
                      const isLatest = latestVer && selected === latestVer;
                      return (
                        <div key={role} className={`flex items-center gap-2 px-2 py-1.5 rounded-md border ${selected !== "v1" ? "border-indigo-200 bg-indigo-50/50" : "border-transparent"}`}>
                          <span className="text-[11px] text-zinc-700 font-medium w-24 shrink-0">{role}</span>
                          <select
                            value={selected}
                            onChange={e => setRolePromptVerMap(prev => ({ ...prev, [role]: e.target.value }))}
                            className="input !py-0.5 text-[10px] flex-1">
                            {allVersions.map((v: any) => (
                              <option key={v.version} value={v.version}>
                                {v.version}{v.version === "v1" ? " (baseline)" : ""}
                                {v.cot_enhanced ? " [CoT]" : ""}{v.created_by === "optimizer" ? " [opt]" : ""}
                                {v.version === latestVer ? " ← latest" : ""}
                              </option>
                            ))}
                          </select>
                          {isLatest && <span className="text-[9px] font-medium text-emerald-600 shrink-0">latest</span>}
                        </div>
                      );
                    })}
                    {Object.values(rolePromptVerMap).some(v => v && v !== "v1") && (
                      <div className="text-[10px] text-indigo-600 font-medium mt-1">
                        {Object.entries(rolePromptVerMap).filter(([, v]) => v && v !== "v1").map(([r, v]) => `${r}→${v}`).join(", ")} will use non-baseline prompts
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-[10px] text-zinc-400 italic">Loading role versions…</div>
                )}
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
                  ) : (
                    <span>Run started — see live widget (bottom-left) for progress. Results tab updates when done.</span>
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
                        <span className="px-1.5 py-0.5 rounded text-[10px] border border-dashed border-zinc-300 text-zinc-500 font-medium" title="Inherits team active strategy">
                          {selectedTeam?.decision_strategy || "router_decides"}
                        </span>
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

      {/* Live run widget is shown globally via the floating RegressionRunWidget */}

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
                {/* Run ID header with copy button */}
                <div className="flex items-center gap-2 px-1">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase font-medium">Run</span>
                  <code className="text-xs font-mono font-semibold text-zinc-700 bg-zinc-100 px-2 py-0.5 rounded border border-zinc-200">{regSelectedRunId}</code>
                  <button
                    onClick={() => { navigator.clipboard.writeText(regSelectedRunId); }}
                    className="text-[10px] px-2 py-0.5 rounded border border-[var(--border)] text-[var(--text-muted)] hover:border-zinc-400 hover:text-zinc-700 transition-all"
                    title="Copy run ID"
                  >Copy</button>
                  {regResults?.model && (
                    <span className="text-[10px] text-[var(--text-muted)] ml-auto">{regResults.model}</span>
                  )}
                </div>

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
                {regCaseDetail && <CaseDetailPanel detail={regCaseDetail} onClose={() => setRegCaseDetail(null)} />}
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
                    ) : (
                      <span className="px-1.5 py-0.5 rounded text-[10px] border border-dashed border-zinc-300 text-zinc-500 font-medium" title="Inherits team active strategy">
                        {selectedTeam?.decision_strategy || "router_decides"}
                      </span>
                    )}
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
                {expandedCaseId === c.id && <GoldenCaseExpanded c={c} activeStrategy={selectedTeam?.decision_strategy || "router_decides"} />}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ─────────── A/B COMPARISON ─────────── */}
      {subTab === "ab" && (
        <div className="space-y-4">
          {/* v2 workflow: pick scope + runs → see overlapping goldens */}
          <div className="card space-y-3 border-l-4 border-indigo-400">
            <div className="flex items-center gap-2 flex-wrap">
              <h2 className="text-sm font-medium">A/B Comparison — runs-first workflow</h2>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200 font-medium">new</span>
            </div>
            <p className="text-[11px] text-[var(--text-muted)]">
              Pick two regression runs and see every golden they share. Supports <b>same-team</b> (within the current project) or <b>cross-team</b> (e.g. dev vs sdlc_2_0).
            </p>

            {/* 1) Scope toggle */}
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[10px] text-[var(--text-muted)] uppercase">Scope:</span>
              {(["same", "cross"] as const).map(s => (
                <button key={s}
                  onClick={() => loadAbV2Runs(s)}
                  className={`px-2.5 py-1 rounded-md text-[11px] border transition-colors ${
                    abScope === s
                      ? "bg-indigo-600 text-white border-indigo-600 font-medium"
                      : "bg-white text-zinc-600 border-[var(--border)] hover:bg-zinc-50"
                  }`}>
                  {s === "same" ? `Same-team (${teamId || "current"})` : "Cross-team (all projects)"}
                </button>
              ))}
              {abV2RunsLoading && <span className="text-[10px] text-[var(--text-muted)] animate-pulse">loading…</span>}
            </div>

            {/* 2) Two run pickers (shared list, disabled-if-other) */}
            {abV2Runs.length > 0 && (
              <div className="grid grid-cols-2 gap-3">
                {(["A", "B"] as const).map(side => {
                  const val = side === "A" ? abRunIdA : abRunIdB;
                  const setVal = side === "A" ? setAbRunIdA : setAbRunIdB;
                  const other = side === "A" ? abRunIdB : abRunIdA;
                  return (
                    <div key={side} className={`rounded-lg border p-2.5 ${side === "A" ? "border-blue-200 bg-blue-50/40" : "border-orange-200 bg-orange-50/40"}`}>
                      <div className={`text-[11px] font-bold mb-1 ${side === "A" ? "text-blue-700" : "text-orange-700"}`}>Run {side}</div>
                      <select
                        value={val}
                        onChange={e => {
                          setVal(e.target.value);
                          setAbV2Overlap(null);
                          setAbGoldenId("");
                          setAbResult(null);
                          setAbJudgeResult(null);
                        }}
                        className="input text-[11px] w-full">
                        <option value="">— pick run —</option>
                        {abV2Runs.map(r => (
                          <option key={r.run_id} value={r.run_id} disabled={r.run_id === other}>
                            {r.run_id.slice(0, 8)} · {abScope === "cross" ? `${r.team_id || "?"} · ` : ""}{r.model?.split("/").pop() || "?"} · {r.passed}/{r.total} pass · {r.started_at ? new Date(r.started_at).toLocaleDateString("en", { month: "short", day: "numeric" }) : ""}
                          </option>
                        ))}
                      </select>
                    </div>
                  );
                })}
              </div>
            )}
            {abV2Runs.length > 0 && abRunIdA && abRunIdB && abRunIdA !== abRunIdB && (
              <button onClick={loadAbV2Overlap} disabled={abV2OverlapLoading}
                      className="px-3 py-1 rounded-md bg-indigo-600 text-white text-[11px] hover:bg-indigo-700 disabled:opacity-50">
                {abV2OverlapLoading ? "Finding shared goldens…" : "Find overlapping golden tests →"}
              </button>
            )}

            {/* 3) Overlap dropdown */}
            {abV2Overlap && (
              <div className="space-y-1.5 pt-1">
                <div className="text-[11px] text-[var(--text-muted)]">
                  <b>{abV2Overlap.overlap_count}</b> shared golden{abV2Overlap.overlap_count === 1 ? "" : "s"} ·
                  only in A: <b>{(abV2Overlap.only_in_a || []).length}</b> ·
                  only in B: <b>{(abV2Overlap.only_in_b || []).length}</b>
                </div>
                {abV2Overlap.overlap_count === 0 ? (
                  <p className="text-[11px] text-amber-700">No shared golden between these runs. Pick a different pair.</p>
                ) : (
                  <select value={abGoldenId}
                          onChange={e => {
                            setAbGoldenId(e.target.value);
                            setAbResult(null);
                            setAbJudgeResult(null);
                          }}
                          className="input text-[11px] w-full max-w-lg">
                    <option value="">— pick overlapping golden —</option>
                    {abV2Overlap.items.map((it: any) => {
                      const n = it.golden_id.replace(/^golden_?/, "");
                      const tag =
                        it.side_a_pass === it.side_b_pass
                          ? (it.side_a_pass ? "both ✓" : "both ✗")
                          : `A:${it.side_a_pass ? "✓" : "✗"} B:${it.side_b_pass ? "✓" : "✗"}`;
                      return (
                        <option key={it.golden_id} value={it.golden_id}>
                          #{n} · {it.name} · {tag}
                        </option>
                      );
                    })}
                  </select>
                )}
                {abGoldenId && (
                  <button onClick={loadAbCompare} disabled={abLoading}
                          className="btn-primary text-[11px] mt-1">
                    {abLoading ? "Loading…" : "Compare these runs →"}
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Legacy workflow: pick golden first, then two runs */}
          <div className="card space-y-3">
            <h2 className="text-sm font-medium">Or: golden-first workflow</h2>
            <p className="text-[11px] text-[var(--text-muted)]">
              Compare two configurations (model, prompt version) for the same golden test. Fetches the most recent regression result matching each set of filters.
            </p>
            {/* Step 1: pick golden test */}
            <div>
              <label className="text-[10px] text-[var(--text-muted)] block mb-0.5">1 · Golden Test</label>
              <select value={abGoldenId} onChange={e => onAbGoldenChange(e.target.value)} className="input text-xs w-full max-w-lg">
                <option value="">— select test —</option>
                {goldenCases.filter(c => c.is_active).map(c => (
                  <option key={c.id} value={c.id}>{c.name || c.id}</option>
                ))}
              </select>
            </div>

            {/* Step 2: pick runs (populated after golden selection) */}
            {abGoldenId && (
              <div className="space-y-2">
                <div className="text-[10px] text-[var(--text-muted)]">
                  2 · Pick two runs to compare — each run shows its model and prompt-version set
                </div>
                {abOptionsLoading && (
                  <p className="text-xs text-[var(--text-muted)] animate-pulse">Loading available runs…</p>
                )}
                {!abOptionsLoading && abOptions.length === 0 && (
                  <p className="text-xs text-amber-600">No regression results for this test yet — run it first.</p>
                )}
                {!abOptionsLoading && abOptions.length > 0 && (
                  <div className="grid grid-cols-2 gap-4">
                    {(["A","B"] as const).map(side => {
                      const val = side === "A" ? abRunIdA : abRunIdB;
                      const setVal = side === "A" ? setAbRunIdA : setAbRunIdB;
                      const other = side === "A" ? abRunIdB : abRunIdA;
                      const color = side === "A" ? "border-blue-200 bg-blue-50" : "border-orange-200 bg-orange-50";
                      const label = side === "A" ? "text-blue-700" : "text-orange-700";
                      return (
                        <div key={side} className={`rounded-lg border p-3 ${color}`}>
                          <div className={`text-[11px] font-bold mb-2 ${label}`}>Side {side}</div>
                          <div className="space-y-1">
                            {abOptions.map(opt => {
                              const selected = val === opt.run_id;
                              const isOther = other === opt.run_id && opt.run_id !== val;
                              const date = opt.created_at ? new Date(opt.created_at).toLocaleDateString("en", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "?";
                              return (
                                <button key={opt.run_id}
                                  onClick={() => setVal(opt.run_id)}
                                  className={`w-full text-left px-2 py-1.5 rounded border text-[11px] transition-all ${
                                    selected
                                      ? side === "A" ? "border-blue-400 bg-blue-100 font-medium" : "border-orange-400 bg-orange-100 font-medium"
                                      : isOther
                                      ? "border-[var(--border)] opacity-40 cursor-not-allowed"
                                      : "border-[var(--border)] bg-white hover:bg-zinc-50"
                                  }`}
                                  disabled={isOther && !selected}>
                                  <div className="flex items-center gap-2 flex-wrap">
                                    <span className="font-mono text-[10px] text-[var(--text-muted)]">{opt.run_id.slice(0, 8)}</span>
                                    <span className="font-medium truncate max-w-[120px]">{opt.model?.split("/").pop() || opt.model}</span>
                                    <span className={`px-1.5 py-0.5 rounded text-[9px] font-mono border ${
                                      opt.prompt_version !== "v1" ? "bg-violet-50 text-violet-700 border-violet-200" : "bg-zinc-100 text-zinc-500 border-zinc-200"
                                    }`}>{opt.prompt_version}</span>
                                    {opt.actual_strategy && (
                                      <span className="px-1.5 py-0.5 rounded text-[9px] border bg-zinc-50 text-zinc-500 border-zinc-200">{opt.actual_strategy}</span>
                                    )}
                                    <span className={`ml-auto text-[10px] font-semibold ${opt.overall_pass ? "text-emerald-600" : "text-red-500"}`}>
                                      {opt.overall_pass ? "✓" : "✗"}
                                    </span>
                                  </div>
                                  {opt.router_prompt_version && (
                                    <div className="text-[9px] text-purple-500 mt-0.5 font-mono">{opt.router_prompt_version}</div>
                                  )}
                                  <div className="text-[10px] text-[var(--text-muted)] mt-0.5">{date}</div>
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            <button onClick={loadAbCompare} disabled={!abGoldenId || !abRunIdA || abLoading} className="btn-primary text-sm">
              {abLoading ? "Loading…" : "Compare →"}
            </button>
          </div>

          {abResult?.error && !abResult.side_a && !abResult.side_b && (
            <div className="card border-l-4 border-amber-400">
              <p className="text-xs text-amber-700">{abResult.error}</p>
              <p className="text-[11px] text-[var(--text-muted)] mt-1">Run regression tests for this golden case first, then compare.</p>
            </div>
          )}

          {abResult && (abResult.side_a || abResult.side_b) && (
            <div className="space-y-4">
              {/* Header */}
              <div className="card !py-3">
                <div className="flex items-center gap-3 flex-wrap">
                  <div>
                    <div className="text-[10px] text-[var(--text-muted)] uppercase font-medium">{abResult.golden_id}</div>
                    <div className="text-sm font-semibold">{abResult.golden_name}</div>
                  </div>
                  {abResult.golden_prompt && (
                    <div className="flex-1 min-w-0 text-xs text-[var(--text-muted)] italic truncate" title={abResult.golden_prompt}>
                      {abResult.golden_prompt.slice(0, 140)}…
                    </div>
                  )}
                </div>
              </div>

              {/* Expected Output */}
              {abResult.expected_output && (
                <div className="card">
                  <div className="text-[10px] text-[var(--text-muted)] uppercase font-medium mb-1.5 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" /> Expected Output
                  </div>
                  <div className="p-2 rounded bg-emerald-50 border border-emerald-100 max-h-36 overflow-y-auto">
                    <Md content={abResult.expected_output} />
                  </div>
                </div>
              )}

              {/* Side-by-side actual outputs */}
              <div className="grid grid-cols-2 gap-4">
                {(["a", "b"] as const).map(side => {
                  const s = abResult[`side_${side}`];
                  if (!s) return (
                    <div key={side} className="card flex items-center justify-center py-12 text-xs text-[var(--text-muted)]">
                      No data for Side {side.toUpperCase()} with these filters
                    </div>
                  );
                  const color = side === "a" ? "blue" : "orange";
                  return (
                    <div key={side} className={`rounded-lg border overflow-hidden ${side === "a" ? "border-blue-200" : "border-orange-200"}`}>
                      <div className={`px-3 py-2 border-b flex items-center gap-2 flex-wrap ${side === "a" ? "bg-blue-50 border-blue-200" : "bg-orange-50 border-orange-200"}`}>
                        <span className={`text-[11px] font-bold ${side === "a" ? "text-blue-700" : "text-orange-700"}`}>Side {side.toUpperCase()}</span>
                        <span className="text-[10px] text-[var(--text-muted)] font-mono">{s.model || "any"}</span>
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white border">{s.prompt_version || "v?"}</span>
                        {s.actual_strategy && <span className="text-[10px] text-[var(--text-muted)]">{s.actual_strategy}</span>}
                        <span className={`ml-auto text-[10px] font-semibold ${s.overall_pass ? "text-emerald-600" : "text-red-600"}`}>
                          {s.overall_pass ? "✓ PASS" : "✗ FAIL"}
                        </span>
                      </div>
                      <div className="p-3 space-y-3">
                        {/* Stats */}
                        <div className="grid grid-cols-3 gap-1.5 text-[10px]">
                          <span><span className="text-[var(--text-muted)]">LLM:</span> <b>{s.actual_llm_calls ?? "—"}</b></span>
                          <span><span className="text-[var(--text-muted)]">Tools:</span> <b>{s.actual_tool_calls ?? "—"}</b></span>
                          <span><span className="text-[var(--text-muted)]">Latency:</span> <b>{s.actual_latency_ms ? `${(s.actual_latency_ms/1000).toFixed(1)}s` : "—"}</b></span>
                          <span><span className="text-[var(--text-muted)]">Cost:</span> <b>${s.actual_cost?.toFixed(5) ?? "—"}</b></span>
                          <span><span className="text-[var(--text-muted)]">Sim:</span> <b className={s.semantic_similarity >= 0.7 ? "text-emerald-600" : "text-amber-600"}>{s.semantic_similarity != null ? `${(s.semantic_similarity*100).toFixed(1)}%` : "—"}</b></span>
                          {s.run_id && <span className="text-[var(--text-muted)] font-mono truncate" title={s.run_id}>{s.run_id.slice(0,8)}</span>}
                        </div>
                        {/* Trajectory path */}
                        <div className="text-[10px]">
                          <span className="text-[var(--text-muted)]">Trajectory: </span>
                          <span className="font-medium">{(s.actual_delegation_pattern?.length ? s.actual_delegation_pattern : [s.actual_agent]).filter(Boolean).join(" → ") || "—"}</span>
                        </div>
                        {/* Tools */}
                        {s.actual_tools?.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {s.actual_tools.map((t: string, i: number) => (
                              <span key={i} className="px-1.5 py-0.5 rounded bg-violet-50 text-violet-600 text-[9px] border border-violet-200 font-mono">{t}</span>
                            ))}
                          </div>
                        )}
                        {/* Per-agent breakdown */}
                        {Object.keys(s.per_agent || {}).length > 0 && (
                          <div className="space-y-1">
                            <div className="text-[10px] text-[var(--text-muted)] uppercase font-medium">Per-Agent</div>
                            {Object.entries(s.per_agent || {}).map(([agent, info]: [string, any]) => {
                              const isOpen = abExpandAgent === `${side}:${agent}`;
                              return (
                                <div key={agent} className="border border-[var(--border)] rounded overflow-hidden">
                                  <button
                                    onClick={() => setAbExpandAgent(isOpen ? null : `${side}:${agent}`)}
                                    className="w-full flex items-center gap-2 px-2 py-1 bg-zinc-50 hover:bg-zinc-100 text-left">
                                    <span className="text-[10px] font-semibold text-zinc-700">{agent}</span>
                                    <span className="text-[9px] text-[var(--text-muted)] ml-auto">{info.tool_calls} tools · {info.llm_calls} llm {isOpen ? "▾" : "▸"}</span>
                                  </button>
                                  {isOpen && info.tools?.length > 0 && (
                                    <div className="px-2 py-1.5 flex flex-wrap gap-1">
                                      {info.tools.map((t: string, i: number) => (
                                        <span key={i} className="px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 text-[9px] border border-amber-200 font-mono">{t}</span>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        )}
                        {/* Actual output */}
                        <div>
                          <div className="text-[10px] text-[var(--text-muted)] uppercase font-medium mb-0.5">Actual Output</div>
                          <div className="p-2 rounded bg-[var(--bg)] border border-[var(--border)] max-h-48 overflow-y-auto">
                            <Md content={s.actual_output || "—"} />
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* DeepEval metric diff table */}
              {Object.keys(abResult.metric_diff || {}).length > 0 && (
                <div className="card">
                  <h3 className="text-sm font-medium mb-3">DeepEval Metric Comparison</h3>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-left text-[var(--text-muted)] border-b border-[var(--border)]">
                        <th className="pb-2 font-medium">Metric</th>
                        <th className="pb-2 font-medium text-right">Side A</th>
                        <th className="pb-2 font-medium text-right">Side B</th>
                        <th className="pb-2 font-medium text-right">Δ</th>
                        <th className="pb-2 font-medium pr-2">A Reasoning</th>
                        <th className="pb-2 font-medium">B Reasoning</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(abResult.metric_diff || {}).map(([metric, d]: [string, any]) => (
                        <tr key={metric} className="border-b border-[var(--border)] last:border-0 align-top">
                          <td className="py-2 font-medium capitalize">{metric.replace(/_/g, " ")}</td>
                          <td className="py-2 text-right font-mono">
                            {d.a_score != null ? (
                              <span className={`${Number(d.a_score) >= 0.7 ? "text-emerald-600" : Number(d.a_score) >= 0.4 ? "text-amber-600" : "text-red-600"}`}>
                                {(Number(d.a_score)*100).toFixed(0)}%
                              </span>
                            ) : "—"}
                          </td>
                          <td className="py-2 text-right font-mono">
                            {d.b_score != null ? (
                              <span className={`${Number(d.b_score) >= 0.7 ? "text-emerald-600" : Number(d.b_score) >= 0.4 ? "text-amber-600" : "text-red-600"}`}>
                                {(Number(d.b_score)*100).toFixed(0)}%
                              </span>
                            ) : "—"}
                          </td>
                          <td className={`py-2 text-right font-mono font-medium ${d.improved ? "text-emerald-600" : d.regressed ? "text-red-600" : "text-[var(--text-muted)]"}`}>
                            {d.delta != null ? `${d.delta > 0 ? "+" : ""}${(d.delta*100).toFixed(0)}pp` : "—"}
                          </td>
                          <td className="py-2 pr-2 text-[10px] text-[var(--text-muted)] max-w-[180px]">
                            {d.a_reason ? <span className="line-clamp-3" title={d.a_reason}>{d.a_reason}</span> : "—"}
                          </td>
                          <td className="py-2 text-[10px] text-[var(--text-muted)] max-w-[180px]">
                            {d.b_reason ? <span className="line-clamp-3" title={d.b_reason}>{d.b_reason}</span> : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* ─── Adaptive LLM Judge (4th score family) ─── */}
              <div className="card border-l-4 border-purple-400 space-y-3">
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <div>
                    <h3 className="text-sm font-medium">Output-quality judgement <span className="text-[10px] text-purple-700 bg-purple-50 border border-purple-200 rounded px-1.5 py-0.5 ml-1">Opus 4.7</span></h3>
                    <p className="text-[11px] text-[var(--text-muted)]">
                      Task-adaptive pairwise rubric. Derives relevant quality dimensions from the golden, scores with quoted evidence, and contrasts the verdict against DeepEval to reveal what these metrics missed.
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    {abJudgeResult?.cached && (
                      <span className="text-[10px] text-[var(--text-muted)] italic">cached · {abJudgeResult.judge_model}</span>
                    )}
                    <button
                      onClick={() => runAbJudge(false)}
                      disabled={abJudgeLoading || !abGoldenId || !abRunIdA || !abRunIdB}
                      className="btn-primary text-[11px]">
                      {abJudgeLoading ? "Judging (30-90s)…" : abJudgeResult ? "View judgment" : "Run LLM judge →"}
                    </button>
                    {abJudgeResult && (
                      <button
                        onClick={() => runAbJudge(true)}
                        disabled={abJudgeLoading}
                        title="Re-run with fresh Opus call"
                        className="text-[11px] px-2 py-1 rounded border border-[var(--border)] hover:bg-zinc-50">
                        ↻
                      </button>
                    )}
                  </div>
                </div>

                {abJudgeError && (
                  <div className="text-[11px] text-red-700 bg-red-50 border border-red-200 rounded px-2 py-1.5">
                    {abJudgeError}
                  </div>
                )}

                {abJudgeResult?.payload && (() => {
                  const p = abJudgeResult.payload;
                  const rs = p.rubric_score || {};
                  const dims: any[] = p.task_dimensions || [];
                  const perDim: Record<string, any> = p.per_dimension || {};
                  const gap: any[] = p.deepeval_gap_analysis || [];
                  const defects: string[] = p.key_defects_in_loser || [];
                  const winClr = p.winner === "A" ? "text-blue-700 bg-blue-50 border-blue-300"
                              : p.winner === "B" ? "text-orange-700 bg-orange-50 border-orange-300"
                              : "text-zinc-700 bg-zinc-50 border-zinc-300";
                  return (
                    <div className="space-y-3">
                      {/* Verdict strip */}
                      <div className="flex items-center gap-3 flex-wrap">
                        <span className={`text-[11px] font-bold px-2 py-1 rounded border ${winClr}`}>
                          Winner: {p.winner?.toUpperCase() || "—"}
                        </span>
                        <span className="text-[11px] text-[var(--text-muted)]">
                          confidence <b>{((p.confidence ?? 0) * 100).toFixed(0)}%</b>
                        </span>
                        <span className="text-[11px]">
                          rubric A <b className="text-blue-700">{rs.side_a != null ? (rs.side_a * 100).toFixed(0) + "%" : "—"}</b>
                          {" "}·{" "}
                          rubric B <b className="text-orange-700">{rs.side_b != null ? (rs.side_b * 100).toFixed(0) + "%" : "—"}</b>
                          {rs.delta && <span className="text-[var(--text-muted)]"> ({rs.delta})</span>}
                        </span>
                        {abJudgeResult.cross_team && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200">
                            cross-team · {abJudgeResult.team_a} vs {abJudgeResult.team_b}
                          </span>
                        )}
                      </div>

                      {p.verdict && (
                        <div className="text-[12px] bg-purple-50/40 border border-purple-100 rounded p-2.5">
                          <Md content={p.verdict} />
                        </div>
                      )}

                      {/* Task-derived dimensions */}
                      {dims.length > 0 && (
                        <div className="overflow-x-auto">
                          <table className="w-full text-[11px]">
                            <thead>
                              <tr className="text-left text-[var(--text-muted)] border-b border-[var(--border)]">
                                <th className="pb-2 font-medium pr-2">Dimension</th>
                                <th className="pb-2 font-medium pr-2">Why it matters</th>
                                <th className="pb-2 font-medium text-right pr-2">A</th>
                                <th className="pb-2 font-medium text-right pr-2">B</th>
                                <th className="pb-2 font-medium text-right pr-2">Δ</th>
                                <th className="pb-2 font-medium pr-2">Evidence A</th>
                                <th className="pb-2 font-medium">Evidence B</th>
                              </tr>
                            </thead>
                            <tbody>
                              {dims.filter(d => d.applicable_here !== false).map(d => {
                                const pd = perDim[d.dim] || {};
                                const ea = pd.side_a || {};
                                const eb = pd.side_b || {};
                                const scA = typeof ea.score === "number" ? ea.score : null;
                                const scB = typeof eb.score === "number" ? eb.score : null;
                                return (
                                  <tr key={d.dim} className="border-b border-[var(--border)] last:border-0 align-top">
                                    <td className="py-1.5 font-medium pr-2">{d.dim}</td>
                                    <td className="py-1.5 text-[var(--text-muted)] pr-2 max-w-[180px]">{d.why_it_matters}</td>
                                    <td className="py-1.5 text-right font-mono pr-2">
                                      {scA != null ? <span className={scA >= 0.7 ? "text-emerald-600" : scA >= 0.4 ? "text-amber-600" : "text-red-600"}>{(scA*100).toFixed(0)}%</span> : "—"}
                                    </td>
                                    <td className="py-1.5 text-right font-mono pr-2">
                                      {scB != null ? <span className={scB >= 0.7 ? "text-emerald-600" : scB >= 0.4 ? "text-amber-600" : "text-red-600"}>{(scB*100).toFixed(0)}%</span> : "—"}
                                    </td>
                                    <td className="py-1.5 text-right font-mono pr-2 text-[var(--text-muted)]">{pd.delta || "—"}</td>
                                    <td className="py-1.5 text-[10px] text-[var(--text-muted)] max-w-[200px] pr-2">
                                      <span className="line-clamp-3" title={ea.evidence}>{ea.evidence || "—"}</span>
                                    </td>
                                    <td className="py-1.5 text-[10px] text-[var(--text-muted)] max-w-[200px]">
                                      <span className="line-clamp-3" title={eb.evidence}>{eb.evidence || "—"}</span>
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                          {dims.some(d => d.applicable_here === false) && (
                            <div className="text-[10px] text-[var(--text-muted)] mt-1">
                              Skipped (not applicable to this task): {dims.filter(d => d.applicable_here === false).map(d => d.dim).join(", ")}
                            </div>
                          )}
                        </div>
                      )}

                      {/* DeepEval gap analysis */}
                      {gap.length > 0 && (
                        <div className="rounded-md border border-amber-200 bg-amber-50/40 p-2.5">
                          <div className="text-[11px] font-semibold text-amber-800 mb-1.5">DeepEval gap analysis</div>
                          <div className="space-y-1.5">
                            {gap.map((g, i) => {
                              const clr = g.pointed === "right" ? "text-emerald-700 bg-emerald-50 border-emerald-200"
                                       : g.pointed === "wrong" ? "text-red-700 bg-red-50 border-red-200"
                                       : "text-zinc-600 bg-zinc-50 border-zinc-200";
                              return (
                                <div key={i} className="text-[11px]">
                                  <div className="flex items-center gap-2 flex-wrap">
                                    <span className="font-mono font-medium">{g.metric}</span>
                                    <span className={`text-[10px] px-1.5 py-0.5 rounded border ${clr}`}>pointed: {g.pointed}</span>
                                    <span className="text-[10px] text-[var(--text-muted)] font-mono">{g.deepeval_verdict}</span>
                                  </div>
                                  <div className="text-[11px] text-zinc-700 mt-0.5"><b>actual:</b> {g.actual_story}</div>
                                  {g.corrective_signal && (
                                    <div className="text-[10px] text-[var(--text-muted)] italic mt-0.5">↳ {g.corrective_signal}</div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Key defects in loser */}
                      {defects.length > 0 && (
                        <div>
                          <div className="text-[11px] font-semibold text-zinc-700 mb-1">Key defects in the losing side</div>
                          <ul className="list-disc list-inside text-[11px] text-zinc-700 space-y-0.5">
                            {defects.map((d, i) => <li key={i}>{d}</li>)}
                          </ul>
                        </div>
                      )}
                    </div>
                  );
                })()}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ─────────── PERFORMANCE ANALYSIS ─────────── */}
      {subTab === "perf" && (
        <div className="space-y-5">
          <div className="flex items-center gap-3">
            <span className="text-sm text-[var(--text-muted)]">Look-back window:</span>
            {[1, 7, 14, 30].map(d => (
              <button key={d} onClick={() => { setPerfDays(d); loadPerfReport(d); }}
                className={`px-3 py-1 rounded text-xs font-medium border transition-all ${perfDays === d ? "bg-zinc-900 text-white border-zinc-900" : "border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)]"}`}>
                {d}d
              </button>
            ))}
            <button onClick={() => { loadPerfReport(perfDays); loadRegressionInsights(); }} className="btn-secondary ml-2">Refresh</button>
          </div>

          {!perfReport && <div className="card text-center py-8 text-[var(--text-muted)] text-sm">Loading performance report…</div>}

          {perfReport && (
            <>
              <div className="grid grid-cols-4 gap-3">
                <Metric label="Total cost" value={`$${(perfReport.cost_breakdown?.summary?.total_cost_usd || 0).toFixed(4)}`} sub={`${perfDays}d window`} />
                <Metric label="Total agent calls" value={String(perfReport.cost_breakdown?.summary?.total_calls || 0)} />
                <Metric label="Avg cost/call" value={`$${(perfReport.cost_breakdown?.summary?.avg_cost_per_call_usd || 0).toFixed(5)}`} />
                <Metric label="Anomalies detected" value={String((perfReport.performance_anomalies || []).length)} accent={(perfReport.performance_anomalies || []).length > 0 ? "text-amber-600" : "text-emerald-600"} />
              </div>

              <div className="card">
                <h3 className="text-sm font-medium mb-3">Agent Call Latency Percentiles (ms)</h3>
                {Object.keys(perfReport.agent_latency_percentiles || {}).length === 0 ? (
                  <p className="text-xs text-[var(--text-muted)]">No agent execution spans recorded yet.</p>
                ) : (
                  <table className="w-full text-xs">
                    <thead><tr className="text-left text-[var(--text-muted)] border-b border-[var(--border)]">
                      <th className="pb-2 font-medium">Agent</th>
                      <th className="pb-2 font-medium text-right">Count</th>
                      <th className="pb-2 font-medium text-right">Avg</th>
                      <th className="pb-2 font-medium text-right">p50</th>
                      <th className="pb-2 font-medium text-right">p95</th>
                      <th className="pb-2 font-medium text-right text-amber-600">p99</th>
                      <th className="pb-2 font-medium text-right">Max</th>
                    </tr></thead>
                    <tbody>
                      {Object.entries(perfReport.agent_latency_percentiles || {}).map(([agent, d]: [string, any]) => (
                        <tr key={agent} className="border-b border-[var(--border)] last:border-0">
                          <td className="py-1.5 font-medium">{agent === "_overall" ? "▶ Overall" : agent}</td>
                          <td className="py-1.5 text-right text-[var(--text-muted)]">{d.count}</td>
                          <td className="py-1.5 text-right">{d.avg}ms</td>
                          <td className="py-1.5 text-right">{d.p50}ms</td>
                          <td className="py-1.5 text-right">{d.p95}ms</td>
                          <td className={`py-1.5 text-right font-medium ${d.p99 > 5000 ? "text-red-600" : d.p99 > 2000 ? "text-amber-600" : "text-emerald-600"}`}>{d.p99}ms</td>
                          <td className="py-1.5 text-right text-[var(--text-muted)]">{d.max}ms</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>

              <div className="card">
                <h3 className="text-sm font-medium mb-3">Tool Failure Rates</h3>
                {Object.keys(perfReport.tool_failure_rates || {}).length === 0 ? (
                  <p className="text-xs text-[var(--text-muted)]">No tool call spans recorded yet.</p>
                ) : (
                  <table className="w-full text-xs">
                    <thead><tr className="text-left text-[var(--text-muted)] border-b border-[var(--border)]">
                      <th className="pb-2 font-medium">Tool</th>
                      <th className="pb-2 font-medium text-right">Total</th>
                      <th className="pb-2 font-medium text-right">Failed</th>
                      <th className="pb-2 font-medium text-right">Failure %</th>
                    </tr></thead>
                    <tbody>
                      {Object.entries(perfReport.tool_failure_rates || {})
                        .sort(([, a]: any, [, b]: any) => (b.failure_rate_pct || 0) - (a.failure_rate_pct || 0))
                        .map(([tool, d]: [string, any]) => (
                        <tr key={tool} className="border-b border-[var(--border)] last:border-0">
                          <td className="py-1.5 font-medium">{tool === "_overall" ? "▶ Overall" : tool}</td>
                          <td className="py-1.5 text-right">{d.total}</td>
                          <td className="py-1.5 text-right">{d.failed}</td>
                          <td className={`py-1.5 text-right font-medium ${(d.failure_rate_pct || 0) > 10 ? "text-red-600" : (d.failure_rate_pct || 0) > 3 ? "text-amber-600" : "text-emerald-600"}`}>
                            {d.failure_rate_pct}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="card">
                  <h3 className="text-sm font-medium mb-3">Cost by Agent Role</h3>
                  {Object.keys(perfReport.cost_breakdown?.by_agent || {}).length === 0 ? (
                    <p className="text-xs text-[var(--text-muted)]">No cost data recorded yet.</p>
                  ) : (
                    <div className="space-y-2">
                      {Object.entries(perfReport.cost_breakdown?.by_agent || {}).map(([agent, d]: [string, any]) => {
                        const total = perfReport.cost_breakdown?.summary?.total_cost_usd || 1;
                        const pct = total > 0 ? ((d.total_cost_usd / total) * 100) : 0;
                        return (
                          <div key={agent}>
                            <div className="flex justify-between text-xs mb-0.5">
                              <span className="font-medium">{agent}</span>
                              <span className="text-[var(--text-muted)]">${d.total_cost_usd.toFixed(5)} ({pct.toFixed(1)}%)</span>
                            </div>
                            <div className="h-1.5 rounded bg-zinc-100">
                              <div className="h-1.5 rounded bg-zinc-800" style={{ width: `${Math.min(100, pct)}%` }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
                <div className="card">
                  <h3 className="text-sm font-medium mb-3">Context Window Utilisation</h3>
                  {Object.keys(perfReport.context_window_utilization || {}).length === 0 ? (
                    <p className="text-xs text-[var(--text-muted)]">No LLM call spans recorded yet.</p>
                  ) : (
                    <table className="w-full text-xs">
                      <thead><tr className="text-left text-[var(--text-muted)] border-b border-[var(--border)]">
                        <th className="pb-2 font-medium">Model</th>
                        <th className="pb-2 font-medium text-right">Avg util</th>
                        <th className="pb-2 font-medium text-right">p99 util</th>
                        <th className="pb-2 font-medium text-right">⚠ &gt;80%</th>
                      </tr></thead>
                      <tbody>
                        {Object.entries(perfReport.context_window_utilization || {}).map(([model, d]: [string, any]) => (
                          <tr key={model} className="border-b border-[var(--border)] last:border-0">
                            <td className="py-1.5 font-medium truncate max-w-[120px]" title={model}>{model.split("/").pop() || model}</td>
                            <td className="py-1.5 text-right">{d.avg_utilization_pct}%</td>
                            <td className={`py-1.5 text-right font-medium ${d.p99_utilization_pct > 80 ? "text-red-600" : d.p99_utilization_pct > 50 ? "text-amber-600" : "text-emerald-600"}`}>{d.p99_utilization_pct}%</td>
                            <td className={`py-1.5 text-right ${d.at_risk_count > 0 ? "text-amber-600" : "text-[var(--text-muted)]"}`}>{d.at_risk_count}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>

              {/* Latency Anomalies — expandable */}
              {(perfReport.performance_anomalies || []).length > 0 && (
                <div className="card border-l-4 border-amber-400">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-medium">⚠ Latency Anomalies (z-score ≥ 2.5)</h3>
                    <span className="text-[10px] text-[var(--text-muted)]">
                      {(perfReport.performance_anomalies || []).length} total · click a row to expand
                    </span>
                  </div>
                  <div className="space-y-1.5">
                    {(perfReport.performance_anomalies || []).slice(0, 15).map((a: any, i: number) => {
                      const isOpen = expandedAnomalies.has(i);
                      const critCls = a.severity === "CRITICAL" ? "border-red-300 bg-red-50" : "border-amber-200 bg-amber-50";
                      const toggleAnom = () => setExpandedAnomalies(prev => {
                        const next = new Set(prev);
                        if (next.has(i)) next.delete(i); else next.add(i);
                        return next;
                      });
                      return (
                        <div key={i} className={`rounded border text-xs overflow-hidden ${critCls}`}>
                          {/* Summary row — always visible */}
                          <button
                            onClick={toggleAnom}
                            className="w-full flex items-center gap-2 px-2.5 py-2 hover:brightness-95 transition-all text-left">
                            <span className={`text-[10px] font-bold shrink-0 ${a.severity === "CRITICAL" ? "text-red-600" : "text-amber-600"}`}>
                              {a.severity}
                            </span>
                            <span className="font-medium truncate max-w-[180px]">{a.name || "—"}</span>
                            <span className="text-[var(--text-muted)] text-[10px] shrink-0">({a.span_type})</span>
                            <span className="font-mono shrink-0">{a.duration_ms?.toFixed(0)}ms</span>
                            <span className="text-[var(--text-muted)] text-[10px] shrink-0">vs μ {a.mean_ms?.toFixed(0)}ms</span>
                            <span className="font-mono text-amber-700 text-[10px] shrink-0">z={a.z_score}</span>
                            {a.agent_role && (
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-white border border-[var(--border)] text-zinc-600 shrink-0">{a.agent_role}</span>
                            )}
                            {a.prompt_version && a.prompt_version !== "v1" && (
                              <span className="text-[9px] px-1.5 py-0.5 rounded bg-violet-100 text-violet-700 border border-violet-200 font-mono shrink-0">{a.prompt_version}</span>
                            )}
                            <span className="ml-auto text-[10px] text-[var(--text-muted)] shrink-0">{isOpen ? "▾" : "▸"}</span>
                          </button>

                          {/* Expanded detail */}
                          {isOpen && (
                            <div className="px-3 pb-3 pt-0 space-y-2 border-t border-[var(--border)]">
                              <div className="grid grid-cols-2 gap-x-6 gap-y-1 mt-2 text-[10px]">
                                <div className="flex gap-1.5">
                                  <span className="text-[var(--text-muted)] shrink-0">Span type:</span>
                                  <span className="font-medium">{a.span_type}</span>
                                </div>
                                <div className="flex gap-1.5">
                                  <span className="text-[var(--text-muted)] shrink-0">Duration:</span>
                                  <span className="font-mono font-medium">{a.duration_ms?.toFixed(1)}ms</span>
                                </div>
                                <div className="flex gap-1.5">
                                  <span className="text-[var(--text-muted)] shrink-0">Mean:</span>
                                  <span className="font-mono">{a.mean_ms?.toFixed(1)}ms</span>
                                </div>
                                <div className="flex gap-1.5">
                                  <span className="text-[var(--text-muted)] shrink-0">Std Dev:</span>
                                  <span className="font-mono">{a.stdev_ms?.toFixed(1)}ms</span>
                                </div>
                                <div className="flex gap-1.5">
                                  <span className="text-[var(--text-muted)] shrink-0">Z-score:</span>
                                  <span className={`font-mono font-bold ${a.z_score >= 4 ? "text-red-600" : "text-amber-600"}`}>{a.z_score}</span>
                                </div>
                                {a.model && a.model !== "unknown" && (
                                  <div className="flex gap-1.5">
                                    <span className="text-[var(--text-muted)] shrink-0">Model:</span>
                                    <span className="font-mono truncate">{a.model?.split("/").pop() || a.model}</span>
                                  </div>
                                )}
                                {a.agent_role && (
                                  <div className="flex gap-1.5">
                                    <span className="text-[var(--text-muted)] shrink-0">Agent:</span>
                                    <span className="font-medium">{a.agent_role}</span>
                                  </div>
                                )}
                                {a.prompt_version && (
                                  <div className="flex gap-1.5">
                                    <span className="text-[var(--text-muted)] shrink-0">Prompt ver.:</span>
                                    <span className={`font-mono ${a.prompt_version !== "v1" ? "text-violet-700" : ""}`}>{a.prompt_version}</span>
                                  </div>
                                )}
                                {a.trace_id && (
                                  <div className="flex gap-1.5">
                                    <span className="text-[var(--text-muted)] shrink-0">Trace:</span>
                                    <span className="font-mono text-zinc-500">{String(a.trace_id).slice(0, 16)}</span>
                                  </div>
                                )}
                              </div>
                              {a.task && (
                                <div className="text-[10px] mt-1">
                                  <span className="text-[var(--text-muted)]">Task: </span>
                                  <span className="text-zinc-700 italic">{a.task}</span>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                  {(perfReport.performance_anomalies || []).length > 15 && (
                    <p className="text-[10px] text-[var(--text-muted)] mt-2">
                      Showing 15 of {(perfReport.performance_anomalies || []).length} anomalies. Narrow the look-back window to see fewer.
                    </p>
                  )}
                </div>
              )}
              {(perfReport.performance_anomalies || []).length === 0 && (
                <div className="card"><p className="text-xs text-emerald-600">✓ No latency anomalies in the last {perfDays} days.</p></div>
              )}
            </>
          )}

          {/* Regression Quality Insights */}
          <div className="mt-2">
            <h2 className="text-sm font-semibold mb-3 text-zinc-700 uppercase tracking-wide">Regression Quality Insights</h2>
            {!regressionInsights && <div className="card text-center py-6 text-[var(--text-muted)] text-xs">Loading…</div>}
            {regressionInsights && regressionInsights.summary?.total_runs > 0 && (
              <>
                <div className="grid grid-cols-4 gap-3 mb-4">
                  <Metric label="Total regression runs" value={String(regressionInsights.summary.total_runs)} />
                  <Metric label="Overall pass rate" value={`${((regressionInsights.summary.pass_rate || 0)*100).toFixed(1)}%`} accent={(regressionInsights.summary.pass_rate || 0) >= 0.8 ? "text-emerald-600" : "text-amber-600"} />
                  <Metric label="Worst metric" value={regressionInsights.summary.worst_metric || "—"} accent="text-red-600" />
                  <Metric label="Most failed agent" value={regressionInsights.summary.most_failed_agent || "—"} accent="text-amber-600" />
                </div>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="card">
                    <h3 className="text-sm font-medium mb-3">DeepEval Metric Averages (lowest first)</h3>
                    <table className="w-full text-xs">
                      <thead><tr className="text-left text-[var(--text-muted)] border-b border-[var(--border)]">
                        <th className="pb-2 font-medium">Metric</th>
                        <th className="pb-2 font-medium text-right">Avg</th>
                        <th className="pb-2 font-medium text-right">Threshold</th>
                        <th className="pb-2 font-medium text-right text-red-500">Below</th>
                      </tr></thead>
                      <tbody>
                        {(regressionInsights.worst_metrics || []).map((m: any) => (
                          <tr key={m.metric} className="border-b border-[var(--border)] last:border-0">
                            <td className="py-1.5 font-medium">{m.metric}</td>
                            <td className={`py-1.5 text-right font-medium ${m.avg < m.threshold ? "text-red-600" : "text-emerald-600"}`}>{m.avg}</td>
                            <td className="py-1.5 text-right text-[var(--text-muted)]">{m.threshold}</td>
                            <td className={`py-1.5 text-right ${m.below_threshold > 0 ? "text-red-600" : "text-[var(--text-muted)]"}`}>{m.below_threshold}/{m.count}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="card">
                    <h3 className="text-sm font-medium mb-3">Pass Rate by Golden Test</h3>
                    <table className="w-full text-xs">
                      <thead><tr className="text-left text-[var(--text-muted)] border-b border-[var(--border)]">
                        <th className="pb-2 font-medium">Test</th>
                        <th className="pb-2 font-medium text-right">Runs</th>
                        <th className="pb-2 font-medium text-right">Pass rate</th>
                      </tr></thead>
                      <tbody>
                        {Object.entries(regressionInsights.pass_rate_by_test || {})
                          .sort(([, a]: any, [, b]: any) => (a.pass_rate || 0) - (b.pass_rate || 0))
                          .slice(0, 10)
                          .map(([tid, d]: [string, any]) => (
                            <tr key={tid} className="border-b border-[var(--border)] last:border-0">
                              <td className="py-1.5 font-mono text-[10px]">{tid}</td>
                              <td className="py-1.5 text-right text-[var(--text-muted)]">{d.total}</td>
                              <td className={`py-1.5 text-right font-medium ${(d.pass_rate||0) >= 0.8 ? "text-emerald-600" : (d.pass_rate||0) >= 0.5 ? "text-amber-600" : "text-red-600"}`}>
                                {((d.pass_rate||0)*100).toFixed(0)}%
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* ─────────── PROMPT VERSIONS ─────────── */}
      {subTab === "prompts" && (
        <div className="space-y-6">
          <div className="card flex items-start justify-between gap-4">
            <div>
              <h2 className="text-sm font-semibold text-zinc-700 uppercase tracking-wide mb-1">Prompt Version Registry</h2>
              <p className="text-xs text-[var(--text-muted)]">Versioned system prompts per agent role. v1 = seed, v2 = CoT-enhanced, v3+ = optimizer-generated.</p>
            </div>
            <button onClick={loadPromptVersionsReg} className="btn-secondary text-xs shrink-0">Refresh</button>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="card lg:col-span-1 space-y-4">
              <h3 className="text-xs font-semibold text-zinc-600 uppercase tracking-wide">Agent Roles</h3>
              <div className="flex flex-col gap-1">
                {promptVersionsReg && Object.keys(scopeRolesToTeam(promptVersionsReg.roles || {})).map((role: string) => {
                  const versions = promptVersionsReg.roles[role] || [];
                  const latest = versions[0];
                  return (
                    <button key={role} onClick={() => { setSelectedRoleReg(role); setPromptDiffReg(null); setPromptAbResultReg(null); }}
                      className={`text-left px-3 py-2 rounded text-xs transition-all ${selectedRoleReg === role ? "bg-[var(--accent)] text-white" : "hover:bg-[var(--bg-hover)] text-zinc-700"}`}>
                      <div className="font-medium">{role}</div>
                      <div className={`text-[11px] mt-0.5 ${selectedRoleReg === role ? "text-indigo-200" : "text-[var(--text-muted)]"}`}>
                        {versions.length} version{versions.length !== 1 ? "s" : ""} · latest: {latest?.version}
                        {latest?.cot_enhanced ? " (CoT)" : ""}
                      </div>
                    </button>
                  );
                })}
                {!promptVersionsReg && <div className="text-xs text-[var(--text-muted)]">Loading…</div>}
              </div>
            </div>
            <div className="lg:col-span-2 space-y-4">
              {/* Version timeline */}
              {promptVersionsReg?.roles?.[selectedRoleReg] && (
                <div className="card">
                  <h3 className="text-xs font-semibold text-zinc-600 uppercase tracking-wide mb-3">Version History — {selectedRoleReg}</h3>
                  <div className="space-y-2">
                    {(promptVersionsReg.roles[selectedRoleReg] as any[]).map((v: any, i: number) => (
                      <div key={v.version} className="flex items-start gap-3 pb-2 border-b border-[var(--border)] last:border-0">
                        <div className={`text-xs font-mono font-semibold px-2 py-0.5 rounded ${i === 0 ? "bg-emerald-100 text-emerald-700" : "bg-zinc-100 text-zinc-600"}`}>{v.version}</div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            {v.cot_enhanced && <span className="text-[10px] bg-violet-100 text-violet-700 px-1.5 py-0.5 rounded font-medium">CoT</span>}
                            <span className="text-[10px] bg-zinc-100 text-zinc-600 px-1.5 py-0.5 rounded">{v.created_by}</span>
                            {i === 0 && <span className="text-[10px] bg-emerald-100 text-emerald-700 px-1.5 py-0.5 rounded">active</span>}
                          </div>
                          {v.rationale && <p className="text-[11px] text-zinc-600 mt-1 line-clamp-2">{v.rationale}</p>}
                          {v.metric_scores && Object.keys(v.metric_scores).length > 0 && (
                            <div className="flex gap-2 mt-1 flex-wrap">
                              {Object.entries(v.metric_scores).map(([m, s]: [string, any]) => (
                                <span key={m} className="text-[10px] text-zinc-500">{m}={typeof s === "number" ? s.toFixed(2) : s}</span>
                              ))}
                            </div>
                          )}
                          <div className="text-[10px] text-[var(--text-muted)] mt-0.5">{v.created_at ? new Date(v.created_at).toLocaleString() : ""}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {/* Prompt Diff */}
              <div className="card space-y-3">
                <h3 className="text-xs font-semibold text-zinc-600 uppercase tracking-wide">Prompt Diff</h3>
                <div className="flex items-center gap-2 flex-wrap">
                  {(() => {
                    const vers: any[] = promptVersionsReg?.roles?.[selectedRoleReg] || [];
                    return vers.length > 0 ? (
                      <>
                        <select className="input text-xs" value={diffOldReg} onChange={e => setDiffOldReg(e.target.value)}>
                          {[...vers].reverse().map((v: any) => <option key={v.version} value={v.version}>{v.version}{v.cot_enhanced ? " [CoT]" : ""}{v.created_by === "optimizer" ? " [opt]" : ""}</option>)}
                        </select>
                        <span className="text-xs text-[var(--text-muted)]">→</span>
                        <select className="input text-xs" value={diffNewReg} onChange={e => setDiffNewReg(e.target.value)}>
                          {[...vers].reverse().map((v: any) => <option key={v.version} value={v.version}>{v.version}{v.cot_enhanced ? " [CoT]" : ""}{v.created_by === "optimizer" ? " [opt]" : ""}</option>)}
                        </select>
                      </>
                    ) : (
                      <>
                        <input className="input !w-20 text-xs" value={diffOldReg} onChange={e => setDiffOldReg(e.target.value)} placeholder="v1" />
                        <span className="text-xs text-[var(--text-muted)]">→</span>
                        <input className="input !w-20 text-xs" value={diffNewReg} onChange={e => setDiffNewReg(e.target.value)} placeholder="v2" />
                      </>
                    );
                  })()}
                  <button className="btn-secondary text-xs" onClick={() => loadPromptDiffReg(selectedRoleReg, diffOldReg, diffNewReg)}>Show Diff</button>
                </div>
                {promptDiffReg?.diff && (
                  <div>
                    <div className="flex gap-4 text-xs text-[var(--text-muted)] mb-2">
                      <span className="text-emerald-600">+{promptDiffReg.lines_added} added</span>
                      <span className="text-red-500">−{promptDiffReg.lines_removed} removed</span>
                    </div>
                    <pre className="text-[11px] bg-zinc-50 rounded p-3 overflow-x-auto max-h-80 border border-[var(--border)] whitespace-pre-wrap leading-relaxed">
                      {(promptDiffReg.diff as string).split("\n").map((line: string, i: number) => (
                        <span key={i} className={line.startsWith("+") && !line.startsWith("+++") ? "text-emerald-700 block" : line.startsWith("-") && !line.startsWith("---") ? "text-red-600 block" : line.startsWith("@@") ? "text-blue-600 block" : "text-zinc-600 block"}>{line}</span>
                      ))}
                    </pre>
                  </div>
                )}
              </div>
              {/* Prompt Version A/B */}
              <div className="card space-y-3">
                <h3 className="text-xs font-semibold text-zinc-600 uppercase tracking-wide">Prompt Version A/B Comparison</h3>
                <p className="text-xs text-[var(--text-muted)]">Compare two prompt versions for <span className="font-mono font-medium text-zinc-700">{selectedRoleReg}</span> across all golden tests.</p>
                <div className="flex items-center gap-2 flex-wrap">
                  {(() => {
                    const vers: any[] = promptVersionsReg?.roles?.[selectedRoleReg] || [];
                    return vers.length > 0 ? (
                      <>
                        <span className="text-xs text-[var(--text-muted)]">Version A:</span>
                        <select className="input text-xs" value={promptAbVersionAReg} onChange={e => setPromptAbVersionAReg(e.target.value)}>
                          {[...vers].reverse().map((v: any) => <option key={v.version} value={v.version}>{v.version}{v.cot_enhanced ? " [CoT]" : ""}{v.created_by === "optimizer" ? " [opt]" : ""}</option>)}
                        </select>
                        <span className="text-xs text-[var(--text-muted)]">vs</span>
                        <span className="text-xs text-[var(--text-muted)]">Version B:</span>
                        <select className="input text-xs" value={promptAbVersionBReg} onChange={e => setPromptAbVersionBReg(e.target.value)}>
                          {[...vers].reverse().map((v: any) => <option key={v.version} value={v.version}>{v.version}{v.cot_enhanced ? " [CoT]" : ""}{v.created_by === "optimizer" ? " [opt]" : ""}</option>)}
                        </select>
                      </>
                    ) : null;
                  })()}
                  <button className="btn-secondary text-xs" onClick={() => runPromptAbCompareReg(selectedRoleReg, promptAbVersionAReg, promptAbVersionBReg)} disabled={promptAbLoadingReg}>
                    {promptAbLoadingReg ? "Comparing…" : "Compare →"}
                  </button>
                </div>
                {promptAbResultReg?.error && <p className="text-xs text-red-500">{promptAbResultReg.error}</p>}
                {promptAbResultReg && !promptAbResultReg.error && (
                  <div className="space-y-4 mt-2">
                    <div className={`px-3 py-2 rounded text-xs font-medium border-l-4 ${promptAbResultReg.recommendation?.startsWith("✓") ? "bg-emerald-50 border-emerald-500 text-emerald-800" : promptAbResultReg.recommendation?.startsWith("✗") ? "bg-red-50 border-red-500 text-red-800" : "bg-zinc-50 border-zinc-300 text-zinc-700"}`}>
                      {promptAbResultReg.recommendation}
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      {[
                        { label: "Pass Rate", aVal: `${((promptAbResultReg.summary?.a_pass_rate||0)*100).toFixed(1)}%`, bVal: `${((promptAbResultReg.summary?.b_pass_rate||0)*100).toFixed(1)}%`, better: (promptAbResultReg.summary?.pass_rate_delta_pp||0) > 0 },
                        { label: "Avg Cost", aVal: `$${(promptAbResultReg.summary?.a_avg_cost||0).toFixed(5)}`, bVal: `$${(promptAbResultReg.summary?.b_avg_cost||0).toFixed(5)}`, better: (promptAbResultReg.summary?.b_avg_cost||0) <= (promptAbResultReg.summary?.a_avg_cost||0) },
                        { label: "Avg Latency", aVal: `${((promptAbResultReg.summary?.a_avg_latency_ms||0)/1000).toFixed(1)}s`, bVal: `${((promptAbResultReg.summary?.b_avg_latency_ms||0)/1000).toFixed(1)}s`, better: (promptAbResultReg.summary?.b_avg_latency_ms||0) <= (promptAbResultReg.summary?.a_avg_latency_ms||0) },
                      ].map(c => (
                        <div key={c.label} className="card text-center !p-2">
                          <div className="text-[10px] text-[var(--text-muted)] mb-1">{c.label}</div>
                          <div className="text-xs"><span className="text-[var(--text-muted)]">A:</span> {c.aVal}</div>
                          <div className="text-xs mt-0.5"><span className="text-[var(--text-muted)]">B:</span> {c.bVal}</div>
                        </div>
                      ))}
                    </div>
                    {Object.keys(promptAbResultReg.metrics || {}).length > 0 && (
                      <table className="w-full text-xs">
                        <thead><tr className="text-left text-[var(--text-muted)] border-b border-[var(--border)]">
                          <th className="pb-1.5 font-medium">Metric</th>
                          <th className="pb-1.5 font-medium text-right">A ({promptAbVersionAReg})</th>
                          <th className="pb-1.5 font-medium text-right">B ({promptAbVersionBReg})</th>
                          <th className="pb-1.5 font-medium text-right">Δ</th>
                        </tr></thead>
                        <tbody>
                          {Object.entries(promptAbResultReg.metrics || {})
                            .sort(([, a]: any, [, b]: any) => Math.abs(parseFloat(b.delta)||0) - Math.abs(parseFloat(a.delta)||0))
                            .map(([metric, d]: [string, any]) => {
                              const delta = parseFloat(d.delta);
                              return (
                                <tr key={metric} className="border-b border-[var(--border)] last:border-0">
                                  <td className="py-1.5 font-medium">{metric}</td>
                                  <td className="py-1.5 text-right text-[var(--text-muted)]">{parseFloat(d.a_avg).toFixed(3)}</td>
                                  <td className="py-1.5 text-right text-[var(--text-muted)]">{parseFloat(d.b_avg).toFixed(3)}</td>
                                  <td className={`py-1.5 text-right font-medium ${d.improved ? "text-emerald-600" : d.regressed ? "text-red-600" : "text-[var(--text-muted)]"}`}>
                                    {!isNaN(delta) ? `${delta>0?"+":""}${delta.toFixed(3)}` : d.delta}
                                  </td>
                                </tr>
                              );
                            })}
                        </tbody>
                      </table>
                    )}
                  </div>
                )}
              </div>
              {/* Prompt Optimizer */}
              <div className="card space-y-3">
                <h3 className="text-xs font-semibold text-zinc-600 uppercase tracking-wide">Run Optimization Loop</h3>
                <p className="text-xs text-[var(--text-muted)]">
                  Triggers the PromptOptimizer agent on one or more roles in parallel (cap 2).
                  You can pin the baseline to a past regression run, hand-pick which goldens to
                  evaluate on, and let it early-exit once the threshold is crossed.
                </p>
                {showOptimizeSetup ? (
                  <div className="p-3 border border-indigo-200 bg-indigo-50/40 rounded-lg space-y-3">
                    <div className="text-[11px] font-medium text-indigo-700 uppercase tracking-wide">Optimization Setup</div>

                    {/* Row 1: multi-role picker + metric + threshold + early-exit */}
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-[10px] text-zinc-500 block mb-1">Agent Roles (pick one or more)</label>
                        <div className="max-h-28 overflow-y-auto border border-[var(--border)] rounded bg-white p-1.5 space-y-0.5">
                          {(promptVersionsReg ? Object.keys(scopeRolesToTeam(promptVersionsReg.roles || {})) : [selectedRoleReg]).map((r: string) => {
                            const checked = optimizeRoles.has(r);
                            const versions = promptVersionsReg?.roles?.[r] || [];
                            return (
                              <label key={r} className="flex items-center gap-2 text-xs px-1 py-0.5 rounded hover:bg-zinc-50 cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={checked}
                                  onChange={e => setOptimizeRoles(prev => {
                                    const next = new Set(prev);
                                    if (e.target.checked) next.add(r); else next.delete(r);
                                    return next;
                                  })}
                                />
                                <span className="font-medium">{r}</span>
                                {checked && (
                                  <select
                                    value={optimizeVersions[r] || "latest"}
                                    onChange={e => setOptimizeVersions(prev => ({ ...prev, [r]: e.target.value }))}
                                    className="ml-auto input text-[10px] !py-0.5"
                                  >
                                    <option value="latest">latest</option>
                                    {versions.map((v: any) => (
                                      <option key={v.version} value={v.version}>{v.version}{v.cot_enhanced ? " [CoT]" : ""}</option>
                                    ))}
                                  </select>
                                )}
                              </label>
                            );
                          })}
                        </div>
                      </div>

                      <div className="space-y-2">
                        <div>
                          <label className="text-[10px] text-zinc-500 block mb-0.5">Metric to Improve</label>
                          <select value={optimizeMetric} onChange={e => setOptimizeMetric(e.target.value)} className="input text-xs w-full">
                            {["step_efficiency","tool_usage","completeness","faithfulness","coherence","correctness","relevance"].map(m => <option key={m} value={m}>{m}</option>)}
                          </select>
                        </div>
                        <div>
                          <label className="text-[10px] text-zinc-500 block mb-0.5">Threshold (0–1)</label>
                          <input type="number" min="0" max="1" step="0.05" value={optimizeThreshold} onChange={e => setOptimizeThreshold(e.target.value)} className="input text-xs w-full" />
                        </div>
                        <label className="flex items-center gap-2 text-xs">
                          <input type="checkbox" checked={optimizeEarlyExit} onChange={e => setOptimizeEarlyExit(e.target.checked)} />
                          <span>Early exit once threshold crossed (recommended)</span>
                        </label>
                        <label className="flex items-center gap-2 text-xs" title="Force-commit the winner even when no significant improvement was achieved. Defaults to OFF — the optimizer will only commit when a cycle crosses the threshold or improves ≥5pp pass-rate / ≥0.05 metric.">
                          <input type="checkbox" checked={optimizeCommitOnPlateau} onChange={e => setOptimizeCommitOnPlateau(e.target.checked)} />
                          <span>Force-commit even if no improvement</span>
                        </label>
                        <label className="flex items-center gap-2 text-xs" title="Run all cycles and report findings but register nothing in the prompt registry. Useful for previewing drafts.">
                          <input type="checkbox" checked={optimizeDryRun} onChange={e => setOptimizeDryRun(e.target.checked)} />
                          <span>Dry run (preview only, never commit)</span>
                        </label>
                      </div>
                    </div>

                    {/* Row 2: baseline source */}
                    <div>
                      <label className="text-[10px] text-zinc-500 block mb-1">Baseline</label>
                      <div className="flex gap-2 items-center">
                        <label className="flex items-center gap-1.5 text-xs">
                          <input type="radio" checked={optimizeBaselineMode === "run"} onChange={() => setOptimizeBaselineMode("run")} />
                          Past regression run
                        </label>
                        <label className="flex items-center gap-1.5 text-xs">
                          <input type="radio" checked={optimizeBaselineMode === "fresh"} onChange={() => setOptimizeBaselineMode("fresh")} />
                          Fresh bootstrap
                        </label>
                      </div>
                      {optimizeBaselineMode === "run" && (
                        <select
                          value={optimizeBaselineRunId}
                          onChange={e => setOptimizeBaselineRunId(e.target.value)}
                          className="input text-xs w-full mt-1"
                        >
                          <option value="">— pick a run —</option>
                          {regRuns.slice(0, 40).map((r: any) => (
                            <option key={r.id} value={r.id}>
                              {r.id.slice(0, 10)} · {r.model} · pass {r.passed}/{r.num_cases} · {r.created_at?.slice(0, 16).replace("T", " ")}
                            </option>
                          ))}
                        </select>
                      )}
                    </div>

                    {/* Row 3: golden subset */}
                    <div>
                      <label className="text-[10px] text-zinc-500 block mb-1">
                        Golden tests ({optimizeGoldenIds.size} selected — empty = derive from baseline failures)
                      </label>
                      <div className="max-h-28 overflow-y-auto border border-[var(--border)] rounded bg-white p-1.5 grid grid-cols-2 gap-x-2">
                        {goldenCases.filter(c => c.is_active).map(c => (
                          <label key={c.id} className="flex items-center gap-1.5 text-[11px] px-1 py-0.5 rounded hover:bg-zinc-50 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={optimizeGoldenIds.has(c.id)}
                              onChange={e => setOptimizeGoldenIds(prev => {
                                const next = new Set(prev);
                                if (e.target.checked) next.add(c.id); else next.delete(c.id);
                                return next;
                              })}
                            />
                            <span className="font-mono text-[10px] text-zinc-500">{c.id}</span>
                            <span className="truncate">{c.name}</span>
                          </label>
                        ))}
                      </div>
                    </div>

                    <div className="flex gap-2 pt-1">
                      <button
                        className="btn-primary text-xs"
                        disabled={optimizeLoading || optimizeRoles.size === 0 || (optimizeBaselineMode === "run" && !optimizeBaselineRunId)}
                        onClick={() => {
                          setShowOptimizeSetup(false);
                          const versionsMap: Record<string, string> = {};
                          Array.from(optimizeRoles).forEach(r => {
                            versionsMap[r] = optimizeVersions[r] || "latest";
                          });
                          runOptimize({
                            roles: Array.from(optimizeRoles),
                            versions: versionsMap,
                            metric: optimizeMetric,
                            threshold: optimizeThreshold,
                            baselineRunId: optimizeBaselineMode === "run" ? optimizeBaselineRunId : null,
                            goldenIds: Array.from(optimizeGoldenIds),
                            earlyExit: optimizeEarlyExit,
                            commitOnPlateau: optimizeCommitOnPlateau,
                            dryRun: optimizeDryRun,
                            teamId: teamId || "default",
                          });
                        }}
                      >
                        {optimizeLoading ? "Running…" : `▶ Run Optimization (${optimizeRoles.size} role${optimizeRoles.size === 1 ? "" : "s"})`}
                      </button>
                      <button className="btn-ghost text-xs" onClick={() => setShowOptimizeSetup(false)}>Cancel</button>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center gap-3">
                    <button
                      className="btn-primary text-xs"
                      onClick={() => {
                        setOptimizeRoles(new Set([selectedRoleReg]));
                        setOptimizeVersions({ [selectedRoleReg]: "latest" });
                        setOptimizeBaselineMode("run");
                        setOptimizeBaselineRunId(regRuns[0]?.id || "");
                        setOptimizeGoldenIds(new Set());
                        setShowOptimizeSetup(true);
                        setOptimizeTrajectory({});
                        setOptimizeResult(null);
                      }}
                      disabled={optimizeLoading}
                    >
                      {optimizeLoading ? "Running optimizer…" : "⚙ Configure & Run"}
                    </button>
                    {optimizeLoading && (
                      <>
                        <span className="text-xs text-[var(--text-muted)] animate-pulse">Running — may take 2–10 min…</span>
                        <button className="btn-secondary text-xs" onClick={stopOptimize}>■ Stop</button>
                      </>
                    )}
                  </div>
                )}

                {/* Per-role trajectory tabs */}
                {Object.keys(optimizeTrajectory).length > 0 && (
                  <div className="mt-2">
                    <div className="flex items-center gap-2 mb-1.5">
                      <div className="text-[10px] font-medium text-zinc-500 uppercase tracking-wide">Live Trajectory</div>
                      <div className="flex gap-1 flex-wrap">
                        {Object.keys(optimizeTrajectory).map(r => {
                          const active = r === optimizeActiveRoleTab;
                          const count = optimizeTrajectory[r]?.length || 0;
                          return (
                            <button
                              key={r}
                              onClick={() => setOptimizeActiveRoleTab(r)}
                              className={`text-[10px] px-2 py-0.5 rounded border ${active ? "bg-indigo-600 text-white border-indigo-700" : "bg-white text-zinc-600 border-[var(--border)] hover:bg-zinc-50"}`}
                            >
                              {r} <span className={active ? "text-indigo-100" : "text-zinc-400"}>({count})</span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                    <div className="bg-zinc-950 rounded-lg p-3 max-h-72 overflow-y-auto font-mono text-[10px] space-y-0.5">
                      {(optimizeTrajectory[optimizeActiveRoleTab] || []).map((ev, i) => (
                        <div key={i} className={ev.type==="agent"?"text-cyan-400":ev.type==="tool"?"text-yellow-300":ev.type==="tool_result"?"text-zinc-400":ev.type==="version"||ev.type==="version_done"?"text-green-400":ev.type==="regression"?"text-purple-300":ev.type==="thinking"?"text-zinc-300":ev.type==="start"?"text-blue-300 font-semibold":ev.type==="done"?"text-emerald-400 font-semibold":ev.type==="error"?"text-red-400":"text-zinc-500"}>
                          {ev.type !== "thinking" && <span className="text-zinc-600 mr-1">{new Date(ev.ts).toLocaleTimeString("en",{hour12:false,hour:"2-digit",minute:"2-digit",second:"2-digit"})}</span>}
                          {ev.text}
                        </div>
                      ))}
                      <div ref={trajectoryEndRef} />
                    </div>
                  </div>
                )}

                {optimizeResult && (
                  <div className="mt-2">
                    <div className={`text-xs font-medium mb-1 ${optimizeResult.status==="completed"?"text-emerald-600":optimizeResult.status==="stopped"?"text-amber-600":"text-red-500"}`}>
                      {optimizeResult.status==="completed" ? "✓ Optimization complete" : optimizeResult.status==="stopped" ? "■ Stopped" : "✗ Error"}
                    </div>
                    {optimizeResult.responses ? (
                      <div className="space-y-2">
                        {Object.entries(optimizeResult.responses as Record<string, string>).map(([role, text]) => (
                          <div key={role}>
                            <div className="text-[10px] font-medium text-indigo-700 uppercase tracking-wide mb-0.5">{role}</div>
                            <pre className="text-[11px] bg-zinc-50 rounded p-3 max-h-56 overflow-y-auto border border-[var(--border)] whitespace-pre-wrap leading-relaxed text-zinc-700">{text}</pre>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <pre className="text-[11px] bg-zinc-50 rounded p-3 max-h-64 overflow-y-auto border border-[var(--border)] whitespace-pre-wrap leading-relaxed text-zinc-700">
                        {optimizeResult.error}
                      </pre>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


function GoldenCaseExpanded({ c, activeStrategy }: { c: any; activeStrategy?: string }) {
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
              ) : (
                <span className="px-1.5 py-0.5 rounded text-[10px] border border-dashed border-zinc-300 text-zinc-500 font-semibold" title="Inherits team active strategy">
                  {activeStrategy || "router_decides"}
                </span>
              )}
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

function CaseDetailPanel({ detail, onClose }: any) {
  const res = detail.result || {};
  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">
          Detail: {detail.golden_case?.name || res.golden_case_name}
        </h3>
        <button onClick={onClose} className="text-xs text-[var(--text-muted)]">✕ Close</button>
      </div>

      {/* Meta row: strategy, prompt versions, router versions */}
      <div className="flex flex-wrap items-center gap-2 text-[10px] text-[var(--text-muted)]">
        {res.actual_strategy && (
          <span className="px-1.5 py-0.5 rounded bg-zinc-100 border border-zinc-200 text-zinc-600 font-mono">
            strategy: {res.actual_strategy}
          </span>
        )}
        {res.prompt_version && (
          <span className="px-1.5 py-0.5 rounded bg-blue-50 border border-blue-200 text-blue-700 font-mono">
            agent prompts: {res.prompt_version}
          </span>
        )}
        {res.router_prompt_version && (
          <span className="px-1.5 py-0.5 rounded bg-purple-50 border border-purple-200 text-purple-700 font-mono">
            routing: {res.router_prompt_version}
          </span>
        )}
        {res.model_used && (
          <span className="ml-auto text-[10px] text-zinc-400">{res.model_used}</span>
        )}
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

      {/* Verdict Breakdown — which criteria caused pass/fail */}
      {(() => {
        const assertions = res.trace_assertions || {};
        const hasAssertions = Object.keys(assertions).length > 0;
        if (!hasAssertions) return null;

        const flagRows = [
          { key: "trace_regression", label: "Trace Assertions", fail: res.trace_regression },
          { key: "quality_regression", label: "Quality (Similarity + DeepEval)", fail: res.quality_regression },
          { key: "cost_regression", label: "Cost Budget", fail: res.cost_regression },
          { key: "latency_regression", label: "Latency Budget", fail: res.latency_regression },
        ];

        const assertionLabels: Record<string, string> = {
          required_tools_called: "Required Tools Called",
          delegation_pattern: "Agent Delegation Pattern",
          llm_call_budget: "LLM Call Budget",
          tool_call_budget: "Tool Call Budget",
          token_budget: "Token Budget",
          latency_budget: "Latency Budget",
        };

        return (
          <div className="rounded-lg border border-[var(--border)] overflow-hidden">
            <div className="px-3 py-2 bg-[var(--bg)] border-b border-[var(--border)] flex items-center justify-between">
              <span className="text-[10px] font-medium uppercase tracking-wide text-[var(--text-muted)]">Verdict Breakdown</span>
              <span className={`text-[11px] font-bold px-2 py-0.5 rounded ${res.overall_pass ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"}`}>
                {res.overall_pass ? "PASS" : "FAIL"}
              </span>
            </div>
            <div className="p-2 space-y-1">
              {/* High-level flags */}
              <div className="grid grid-cols-2 gap-1">
                {flagRows.map(f => (
                  <div key={f.key} className={`flex items-center justify-between text-[10px] px-2 py-1 rounded ${f.fail ? "bg-red-50 border border-red-200" : "bg-emerald-50 border border-emerald-200"}`}>
                    <span className={f.fail ? "text-red-700" : "text-emerald-700"}>{f.label}</span>
                    <span className={`font-bold ${f.fail ? "text-red-600" : "text-emerald-600"}`}>{f.fail ? "✗ FAIL" : "✓ OK"}</span>
                  </div>
                ))}
              </div>
              {/* Per-assertion details */}
              <div className="mt-1 space-y-0.5">
                {Object.entries(assertions).map(([key, val]: [string, any]) => (
                  <div key={key} className={`flex items-start gap-2 text-[10px] px-2 py-1.5 rounded ${val.passed ? "bg-emerald-50/60" : "bg-red-50/60"}`}>
                    <span className={`shrink-0 font-bold mt-0.5 ${val.passed ? "text-emerald-600" : "text-red-600"}`}>
                      {val.passed ? "✓" : "✗"}
                    </span>
                    <div className="flex-1 min-w-0">
                      <span className="font-semibold text-zinc-700">{assertionLabels[key] || key}</span>
                      {val.reason && (
                        <div className="text-zinc-500 mt-0.5">{val.reason}</div>
                      )}
                      {val.expected !== undefined && val.actual !== undefined && (
                        <div className="text-zinc-400 mt-0.5 font-mono">
                          expected: {JSON.stringify(val.expected)} · actual: {JSON.stringify(val.actual)}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );
      })()}

      {/* Radar chart for scores overview */}
      {(() => {
        const deMetrics = DEEPEVAL_METRICS.filter(m => res.deepeval_scores?.[m.key] != null);
        if (deMetrics.length < 3) return null;
        return (
          <div className="p-3 rounded border border-[var(--border)] bg-[var(--bg)]">
            <div className="text-[10px] text-[var(--text-muted)] uppercase mb-1 font-medium">DeepEval Score Overview</div>
            <SingleRadarChart scoresMap={res.deepeval_scores || {}} metrics={deMetrics} color="#3b82f6" label="Score" />
          </div>
        );
      })()}

      {/* DeepEval Scores only */}
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

