"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

const TEAM_STRATEGIES = [
  { v: "router_decides", l: "Router", d: "LLM picks the best agent" },
  { v: "sequential", l: "Sequential", d: "Agents run in order" },
  { v: "parallel", l: "Parallel", d: "All agents run at once" },
  { v: "supervisor", l: "Supervisor", d: "Supervisor delegates and reviews" },
  { v: "auto", l: "Auto", d: "AI picks the best strategy per request" },
];

const AGENT_STRATEGIES = [
  { value: "react", label: "ReAct", desc: "Reason + Act loop" },
  { value: "plan_execute", label: "Plan & Execute", desc: "Plan first, then execute" },
  { value: "reflexion", label: "Reflexion", desc: "Self-reflection after each step" },
  { value: "cot", label: "Chain-of-Thought", desc: "Think thoroughly before acting" },
];

// ── Prompt Templates ──────────────────────────────────────────────────────────
// Generic, structure-only templates — user fills in the bracketed sections.
// Each template notes the prompting technique(s) it demonstrates.
const PROMPT_TEMPLATES = [
  {
    id: "react",
    label: "ReAct Agent",
    technique: "ReAct (Reason + Act)",
    desc: "Interleaves reasoning traces with tool actions. Best for tasks requiring multi-step tool use with clear justification.",
    icon: "⚙️",
    system_prompt: `You are [FILL IN: agent name and one-sentence role description].

## Reasoning Protocol — ReAct Pattern
Before EVERY tool call or response, output:
**THOUGHT:** What am I trying to achieve? What information do I have so far?
**ACTION:** Which tool will I call next, and with which exact parameters?
After each tool result:
**OBSERVATION:** What did the tool return? Does it confirm my hypothesis?
**NEXT:** Do I have enough information to respond, or do I need another action?

## Responsibilities
[FILL IN: 2-4 bullet points describing what this agent is responsible for]

## Tools Available
[FILL IN: list the tool groups this agent will use and what each is for]

## Constraints
- [FILL IN: any hard limits, e.g. "Never modify production files without approval"]
- Always explain your reasoning before acting
- If uncertain, ask for clarification rather than guessing`,
  },
  {
    id: "cot",
    label: "Chain-of-Thought (CoT) Agent",
    technique: "Chain-of-Thought (CoT) + Self-Critique",
    desc: "Forces explicit step-by-step reasoning before any action. Ideal for analysis, planning, and code review tasks.",
    icon: "🧠",
    system_prompt: `You are [FILL IN: agent name and one-sentence role description].

## Chain-of-Thought Protocol
For every request, work through these sections in order:

**SITUATION ANALYSIS**
- What exactly is being asked?
- What context and constraints apply?
- What would a successful outcome look like?

**STEP-BY-STEP PLAN**
1. [Think through the steps here before acting]
2. ...

**SELF-CRITIQUE**
- Is my plan complete and correct?
- What could go wrong?
- Have I considered edge cases?

**EXECUTION**
[Now carry out the plan, one step at a time]

## Responsibilities
[FILL IN: what this agent owns]

## Output Format
[FILL IN: how the final answer should be structured — e.g. markdown, JSON, prose]`,
  },
  {
    id: "plan_execute",
    label: "Plan & Execute Agent",
    technique: "Plan-and-Execute / Decomposition",
    desc: "Separates planning from execution. Agent writes a complete plan first, gets implicit approval, then executes step by step.",
    icon: "📋",
    system_prompt: `You are [FILL IN: agent name and one-sentence role description].

## Plan-and-Execute Protocol

### Phase 1 — Planning
Before taking any action, produce a numbered execution plan:
\`\`\`
PLAN:
1. [Step description] — [which tool or reasoning]
2. ...
n. [Final step: deliver result]
\`\`\`

### Phase 2 — Execution
Execute each step in order, logging:
\`\`\`
STEP 1: [description]
  → Tool: [tool name]({args})
  → Result: [summary of outcome]
  → Status: ✓ complete / ✗ failed (reason)
\`\`\`

### Phase 3 — Summary
After all steps, deliver:
- Final result / artifact
- Execution summary (steps completed, any skipped)
- Recommendations for next steps

## Responsibilities
[FILL IN: what this agent owns]

## Constraints
[FILL IN: limits and guardrails]`,
  },
  {
    id: "reflexion",
    label: "Reflexion Agent",
    technique: "Reflexion (Self-Reflection Loop)",
    desc: "After each action, the agent reflects on quality and self-corrects before proceeding. Best for iterative quality improvement tasks.",
    icon: "🔄",
    system_prompt: `You are [FILL IN: agent name and one-sentence role description].

## Reflexion Protocol
After EVERY tool call or intermediate output, apply this loop:

**ACT:** [Perform the action / call the tool]
**EVALUATE:** On a scale of 1-5, how good was the result?
  - What is missing or incorrect?
  - What would make this better?
**REFLECT:** If score < 4, identify the root cause of the shortcoming.
**REFINE:** Generate an improved action / revised output addressing the reflection.
**REPEAT** until quality score ≥ 4 or 3 iterations reached.

## Success Criteria
[FILL IN: define what "good enough" looks like for this agent's outputs]

## Responsibilities
[FILL IN: what this agent owns]

## Constraints
- Maximum 3 refinement iterations per step
- [FILL IN: domain-specific constraints]`,
  },
  {
    id: "tool_specialist",
    label: "Tool-Specialist Agent",
    technique: "Tool-Use + Output Contract",
    desc: "Focused agent that owns a specific toolset and always returns structured outputs. Good for narrow, high-reliability tasks.",
    icon: "🔧",
    system_prompt: `You are [FILL IN: agent name], a specialist responsible for [FILL IN: narrow domain].
You ONLY perform tasks within your domain. For anything outside it, hand off to the appropriate agent.

## Primary Tools
[FILL IN: list the 2-4 tools you use most]

## Mandatory Output Contract
Every response MUST include:
\`\`\`json
{
  "status": "success | partial | failed",
  "result": "[FILL IN: describe what you return]",
  "artifacts": [],
  "errors": [],
  "next_suggested_agent": "optional — who should act next"
}
\`\`\`

## Domain Scope
IN SCOPE:
- [FILL IN: list what you handle]

OUT OF SCOPE (hand off):
- [FILL IN: list what you don't handle and who does]

## Error Handling
On tool failure: retry once with adjusted parameters, then report the error with full context.`,
  },
  {
    id: "analyst",
    label: "Analyst / Data Agent",
    technique: "Structured Output + Few-Shot Exemplar",
    desc: "For agents that query data, run SQL, or produce analytical summaries. Uses structured output and exemplar-based guidance.",
    icon: "📊",
    system_prompt: `You are [FILL IN: agent name], an analytical agent specialising in [FILL IN: data domain].

## Analysis Protocol
1. **Clarify** — restate the question in measurable terms
2. **Query** — retrieve data using available tools
3. **Validate** — check data quality (nulls, outliers, time range)
4. **Analyse** — apply the appropriate technique (aggregation / trend / comparison)
5. **Conclude** — state the finding in plain language with confidence

## Output Format
\`\`\`
## Finding
[1-3 sentence answer to the question]

## Supporting Data
| Metric | Value | Period |
|--------|-------|--------|
| [FILL IN] | ... | ... |

## Caveats
- [FILL IN: data limitations, confidence level]
\`\`\`

## Example (Few-Shot)
Q: "What was the average latency last week?"
→ Query: SELECT AVG(latency_ms) FROM spans WHERE created_at > NOW()-7d
→ Result: 342ms average, p99=1.2s
→ Finding: "Average agent latency was 342ms (p99: 1.2s) over the past 7 days."

## Responsibilities
[FILL IN: what analytical questions you own]`,
  },
  {
    id: "blank",
    label: "Custom (Blank)",
    technique: "Write your own",
    desc: "Start from scratch. Recommended: include a role statement, reasoning protocol, responsibilities, and constraints.",
    icon: "📝",
    system_prompt: `You are [FILL IN: agent name and role].

## Responsibilities
[FILL IN]

## Reasoning Protocol
[FILL IN: how the agent should think before acting]

## Constraints
[FILL IN]`,
  },
];

// ── Helpers ───────────────────────────────────────────────────────────────────
type AgentDraft = {
  name: string; role: string; description: string;
  system_prompt: string; tool_groups: string[]; model: string;
  decision_strategy: string;
};
type PendingEdit = Partial<AgentDraft & { prompt_version: string }>;

export default function StudioPage() {
  const [teams, setTeams] = useState<any[]>([]);
  const [team, setTeam] = useState<any>(null);
  const [skills, setSkills] = useState<any[]>([]);
  const [tools, setTools] = useState<any>({});
  const [newTeam, setNewTeam] = useState("");
  const [editingName, setEditingName] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [defaultModelName, setDefaultModelName] = useState("from .env");

  // ── Add-Agent modal ──────────────────────────────────────────────
  const [showAddAgent, setShowAddAgent] = useState(false);
  const [addStep, setAddStep] = useState<"template" | "form">("template");
  const [selectedTemplate, setSelectedTemplate] = useState<(typeof PROMPT_TEMPLATES)[0] | null>(null);
  const emptyDraft = (): AgentDraft => ({
    name: "", role: "", description: "", system_prompt: "",
    tool_groups: [], model: "", decision_strategy: "react",
  });
  const [draft, setDraft] = useState<AgentDraft>(emptyDraft());

  // ── Skill form ───────────────────────────────────────────────────
  const [showSkill, setShowSkill] = useState(false);
  const [skill, setSkill] = useState({ name: "", description: "", instructions: "", trigger_pattern: "" });

  // ── Per-agent pending edits (dirty tracking) ─────────────────────
  const [pendingEdits, setPendingEdits] = useState<Record<string, PendingEdit>>({});
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null);
  const [savingAgent, setSavingAgent] = useState<string | null>(null);

  // ── Prompt version viewer (per agent) ───────────────────────────
  const [agentVersions, setAgentVersions] = useState<Record<string, any[]>>({});
  const [viewingPrompt, setViewingPrompt] = useState<Record<string, string>>({});
  const [promptTextCache, setPromptTextCache] = useState<Record<string, string>>({});
  const promptTextCacheRef = useRef(promptTextCache);
  promptTextCacheRef.current = promptTextCache;

  useEffect(() => { load(); }, []);

  async function load() {
    const [t, s, tl, models, llmCfg, allVersions] = await Promise.all([
      api.teams.list(), api.skills.list(), api.tools.list(),
      api.models.list(), api.config.llm(),
      api.prompts.versions().catch(() => ({ roles: {} })),
    ]);
    setTeams(t); setSkills(s); setTools(tl);
    setAvailableModels(models);
    setDefaultModelName(llmCfg.default_model_name || llmCfg.default_model || "from .env");
    // Eagerly populate all role versions so dropdowns work without clicking Edit
    if (allVersions?.roles) {
      setAgentVersions(allVersions.roles);
    }
    if (t.length > 0 && !team) setTeam(await api.teams.get(t[0].id));
  }

  function modelOptions() {
    const byProvider: Record<string, any[]> = {};
    for (const m of availableModels) {
      const g = m.provider || "Other";
      (byProvider[g] = byProvider[g] || []).push(m);
    }
    return { byProvider };
  }

  // ── Prompt versions per agent role ──────────────────────────────
  const loadVersionsForRole = useCallback(async (role: string, force = false) => {
    if (!role || (agentVersions[role] && !force)) return;
    try {
      const res = await api.prompts.versions(role);
      const versions = res?.versions || [];
      setAgentVersions(prev => ({ ...prev, [role]: versions }));
    } catch {
      setAgentVersions(prev => ({ ...prev, [role]: [] }));
    }
  }, [agentVersions]);

  const loadPromptText = useCallback(async (role: string, version: string) => {
    const key = `${role}::${version}`;
    if (promptTextCacheRef.current[key]) {
      setViewingPrompt(prev => ({ ...prev, [role]: promptTextCacheRef.current[key] }));
      return;
    }
    try {
      const res = await api.prompts.text(role, version);
      const txt = res?.text || "(empty)";
      setPromptTextCache(prev => ({ ...prev, [key]: txt }));
      setViewingPrompt(prev => ({ ...prev, [role]: txt }));
    } catch {
      setViewingPrompt(prev => ({ ...prev, [role]: "(failed to load)" }));
    }
  }, []);

  // ── Team operations ──────────────────────────────────────────────
  async function selectTeam(id: string) { setTeam(await api.teams.get(id)); }

  async function createTeam() {
    if (!newTeam.trim()) return;
    const r = await api.teams.create({ name: newTeam });
    setNewTeam("");
    const t = await api.teams.list(); setTeams(t);
    setTeam(await api.teams.get(r.id));
  }

  async function renameTeam(id: string) {
    if (!editName.trim()) { setEditingName(null); return; }
    await api.teams.update(id, { name: editName });
    setEditingName(null);
    const t = await api.teams.list(); setTeams(t);
    if (team?.id === id) setTeam(await api.teams.get(id));
  }

  async function deleteTeam(id: string) {
    if (!confirm("Delete this team and all its agents?")) return;
    await api.teams.delete(id);
    const t = await api.teams.list(); setTeams(t);
    if (t.length > 0) setTeam(await api.teams.get(t[0].id));
    else setTeam(null);
  }

  async function updateStrategy(s: string) {
    if (!team) return;
    await api.teams.update(team.id, { decision_strategy: s });
    setTeam({ ...team, decision_strategy: s });
  }

  // ── Agent operations ─────────────────────────────────────────────
  async function addAgent() {
    if (!team) return;
    await api.teams.addAgent(team.id, draft);
    setShowAddAgent(false);
    setDraft(emptyDraft());
    setSelectedTemplate(null);
    setAddStep("template");
    setTeam(await api.teams.get(team.id));
  }

  async function removeAgent(id: string) {
    await api.agents.delete(id);
    if (team) setTeam(await api.teams.get(team.id));
  }

  function setPendingField(agentId: string, field: string, value: any) {
    setPendingEdits(prev => ({
      ...prev,
      [agentId]: { ...(prev[agentId] || {}), [field]: value },
    }));
  }

  function isDirty(agentId: string) {
    return Object.keys(pendingEdits[agentId] || {}).length > 0;
  }

  async function saveAgent(a: any) {
    const pending = pendingEdits[a.id];
    if (!pending || Object.keys(pending).length === 0) return;
    setSavingAgent(a.id);
    try {
      await api.agents.update(a.id, pending);
      setPendingEdits(prev => { const n = { ...prev }; delete n[a.id]; return n; });
      if (team) setTeam(await api.teams.get(team.id));
    } finally {
      setSavingAgent(null);
    }
  }

  function cancelEdit(agentId: string) {
    setPendingEdits(prev => { const n = { ...prev }; delete n[agentId]; return n; });
    setExpandedAgent(null);
  }

  async function toggleSkill(agentId: string, skillId: string, on: boolean) {
    if (!team) return;
    const a = team.agents.find((x: any) => x.id === agentId);
    if (!a) return;
    const ids = a.skills.map((s: any) => s.id);
    const next = on ? ids.filter((i: string) => i !== skillId) : [...ids, skillId];
    await api.agents.assignSkills(agentId, next);
    setTeam(await api.teams.get(team.id));
  }

  async function rebuild() { if (team) await api.teams.rebuild(team.id); }

  // ── Skill operations ─────────────────────────────────────────────
  async function createSkill() {
    await api.skills.create(skill);
    setShowSkill(false); setSkill({ name: "", description: "", instructions: "", trigger_pattern: "" });
    setSkills(await api.skills.list());
  }

  // ── Helpers ───────────────────────────────────────────────────────
  function effectiveVal(a: any, field: string) {
    const pending = pendingEdits[a.id];
    return pending && field in pending ? pending[field as keyof PendingEdit] : a[field];
  }

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-xl font-semibold">Agent Studio</h1>
        <p className="text-[13px] text-[var(--text-muted)] mt-1">Configure teams, agents, skills, and tools</p>
      </div>

      {/* Team Tabs */}
      <div className="flex items-center gap-1.5 border-b border-[var(--border)] pb-2.5">
        {teams.map(t => (
          <div key={t.id} className="flex items-center group">
            {editingName === t.id ? (
              <input value={editName} onChange={e => setEditName(e.target.value)}
                onBlur={() => renameTeam(t.id)}
                onKeyDown={e => e.key === "Enter" && renameTeam(t.id)}
                autoFocus className="input !w-28 !py-1 !text-xs" />
            ) : (
              <button onClick={() => selectTeam(t.id)}
                onDoubleClick={() => { setEditingName(t.id); setEditName(t.name); }}
                className={`px-3 py-1.5 rounded-md text-[13px] transition-all ${team?.id === t.id ? "bg-zinc-900 text-white font-medium" : "text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-hover)]"}`}>
                {t.name}
              </button>
            )}
            {team?.id === t.id && t.id !== "default" && (
              <button onClick={() => deleteTeam(t.id)}
                className="opacity-0 group-hover:opacity-100 text-[10px] text-[var(--error)] ml-0.5 hover:bg-[var(--error-light)] rounded px-1 transition-all"
                title="Delete team">×</button>
            )}
          </div>
        ))}
        <div className="flex items-center gap-1.5 ml-auto">
          <input value={newTeam} onChange={e => setNewTeam(e.target.value)}
            onKeyDown={e => e.key === "Enter" && createTeam()}
            placeholder="New team..." className="input !w-32 !text-xs !py-1" />
          <button onClick={createTeam} className="btn-primary !py-1 !px-2.5 !text-xs">+</button>
        </div>
      </div>

      {team && (
        <div className="space-y-5">
          {/* Team Strategy */}
          <section className="card">
            <div className="flex items-center justify-between mb-2">
              <div>
                <h2 className="text-sm font-medium">Team Strategy</h2>
                <p className="text-[11px] text-[var(--text-muted)]">Controls which agents run, when, and in what order</p>
              </div>
            </div>
            <div className="grid grid-cols-5 gap-2">
              {TEAM_STRATEGIES.map(s => (
                <button key={s.v} onClick={() => updateStrategy(s.v)}
                  className={`p-3 rounded-lg border text-left transition-all ${
                    team.decision_strategy === s.v
                      ? "border-zinc-900 bg-zinc-900"
                      : "border-[var(--border)] hover:border-zinc-400"
                  }`}>
                  <div className={`text-sm font-medium ${team.decision_strategy === s.v ? "text-white" : ""}`}>{s.l}</div>
                  <div className={`text-[10px] mt-0.5 ${team.decision_strategy === s.v ? "text-zinc-400" : "text-[var(--text-muted)]"}`}>{s.d}</div>
                </button>
              ))}
            </div>
            <div className="mt-2 text-[10px] text-[var(--text-muted)] flex items-center gap-1.5">
              <span className="font-medium">Active:</span>
              <span className="font-medium text-zinc-700">{TEAM_STRATEGIES.find(s => s.v === team.decision_strategy)?.l || team.decision_strategy}</span>
              <span className="mx-1">|</span>
              <span>Each agent below has its own reasoning strategy (ReAct, Plan-Execute, etc.)</span>
            </div>
          </section>

          {/* Agents */}
          <section className="card">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-medium">Agents ({team.agents?.length || 0})</h2>
              <div className="flex gap-2">
                <button onClick={rebuild} className="btn-secondary !text-xs">Rebuild</button>
                <button onClick={() => { setShowAddAgent(true); setAddStep("template"); setSelectedTemplate(null); setDraft(emptyDraft()); }} className="btn-primary !text-xs">+ Add Agent</button>
              </div>
            </div>

            <div className="space-y-2.5">
              {team.agents?.map((a: any) => {
                const dirty = isDirty(a.id);
                const isExpanded = expandedAgent === a.id;
                const roleVersions = agentVersions[a.role] || [];
                const currentPv = effectiveVal(a, "prompt_version") as string || "v1";

                return (
                  <div key={a.id} className={`rounded-lg border transition-all ${dirty ? "border-amber-300 bg-amber-50/30" : "border-[var(--border)] bg-[var(--bg)]"}`}>
                    {/* Header row */}
                    <div className="p-3.5">
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-sm">{a.name}</span>
                          <span className="badge bg-zinc-100 text-zinc-600 border border-zinc-200">{a.role}</span>
                          {dirty && <span className="text-[10px] text-amber-600 font-medium">● unsaved</span>}
                        </div>
                        <div className="flex items-center gap-1.5">
                          {dirty && (
                            <>
                              <button
                                onClick={() => saveAgent(a)}
                                disabled={savingAgent === a.id}
                                className="btn-primary !text-[11px] !py-0.5 !px-2.5">
                                {savingAgent === a.id ? "Saving…" : "Save"}
                              </button>
                              <button onClick={() => cancelEdit(a.id)} className="btn-ghost !text-[11px] !py-0.5 !px-2">Discard</button>
                            </>
                          )}
                          <button
                            onClick={() => {
                              if (isExpanded) { setExpandedAgent(null); }
                              else { setExpandedAgent(a.id); loadVersionsForRole(a.role); }
                            }}
                            className="btn-secondary !text-[11px] !py-0.5 !px-2">
                            {isExpanded ? "Collapse" : "Edit"}
                          </button>
                          <button onClick={() => removeAgent(a.id)} className="btn-danger !text-[11px] !py-0 !px-1.5">Remove</button>
                        </div>
                      </div>
                      <p className="text-xs text-[var(--text-muted)] mt-1">{effectiveVal(a, "description") as string || a.description}</p>

                      {/* Quick selectors — Model + Strategy + Prompt Version (immediate save) */}
                      <div className="flex flex-wrap items-center gap-4 mt-2.5">
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide">Model:</span>
                          <select
                            value={effectiveVal(a, "model") as string || ""}
                            onChange={e => setPendingField(a.id, "model", e.target.value)}
                            className="input !w-auto !py-1 !text-xs">
                            <option value="">Default ({defaultModelName})</option>
                            {Object.entries(modelOptions().byProvider).map(([provider, models]) => (
                              <optgroup key={provider} label={provider}>
                                {(models as any[]).map((m: any) => (
                                  <option key={m.id} value={m.id}>{m.name}</option>
                                ))}
                              </optgroup>
                            ))}
                          </select>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide">Strategy:</span>
                          <select
                            value={effectiveVal(a, "decision_strategy") as string || "react"}
                            onChange={e => setPendingField(a.id, "decision_strategy", e.target.value)}
                            className="input !w-auto !py-1 !text-xs">
                            {AGENT_STRATEGIES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
                          </select>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide">Prompt Ver:</span>
                          <select
                            value={currentPv}
                            onChange={async e => {
                              const ver = e.target.value;
                              setPendingField(a.id, "prompt_version", ver);
                              await loadVersionsForRole(a.role);
                              loadPromptText(a.role, ver);
                            }}
                            className="input !w-auto !py-1 !text-xs"
                            onClick={() => loadVersionsForRole(a.role)}>
                            <option value="v1">v1 (baseline)</option>
                            {roleVersions.filter((v: any) => v.version !== "v1").map((v: any) => (
                              <option key={v.version} value={v.version}>
                                {v.version}{v.cot_enhanced ? " [CoT]" : ""}{v.created_by === "optimizer" ? " [opt]" : ""}
                              </option>
                            ))}
                          </select>
                          <button
                            onClick={() => loadPromptText(a.role, currentPv)}
                            className="text-[10px] text-indigo-600 hover:underline"
                            title="View prompt text for selected version">
                            View
                          </button>
                        </div>
                      </div>

                      {/* Prompt text viewer */}
                      {viewingPrompt[a.role] && (
                        <div className="mt-2.5 bg-zinc-50 border border-zinc-200 rounded p-2.5 relative">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wide">
                              Prompt — {a.role} @ {currentPv}
                            </span>
                            <button
                              onClick={() => setViewingPrompt(prev => { const n = { ...prev }; delete n[a.role]; return n; })}
                              className="text-[10px] text-zinc-400 hover:text-zinc-700">✕ close</button>
                          </div>
                          <pre className="text-[11px] text-zinc-700 whitespace-pre-wrap leading-relaxed max-h-60 overflow-y-auto">
                            {viewingPrompt[a.role]}
                          </pre>
                        </div>
                      )}

                      <div className="flex flex-wrap gap-1 mt-2">
                        <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mr-1 self-center">Tools:</span>
                        {a.tool_groups?.map((g: string) => (
                          <span key={g} className="badge bg-zinc-100 text-zinc-500 border border-zinc-200">{g}</span>
                        ))}
                      </div>

                      <div className="flex flex-wrap gap-1 mt-1.5">
                        <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mr-1 self-center">Skills:</span>
                        {skills.map(s => {
                          const on = a.skills?.some((x: any) => x.id === s.id);
                          return (
                            <button key={s.id} onClick={() => toggleSkill(a.id, s.id, on)}
                              className={`badge border cursor-pointer transition-all ${on ? "bg-[var(--success-light)] text-[var(--success)] border-[var(--success)]/30" : "bg-transparent text-[var(--text-muted)] border-[var(--border)] hover:text-[var(--text)]"}`}>
                              {s.name}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    {/* Expanded edit panel */}
                    {isExpanded && (
                      <div className="border-t border-[var(--border)] p-3.5 space-y-3 bg-zinc-50/50">
                        <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wide">Edit Agent</div>
                        <div className="grid grid-cols-2 gap-2.5">
                          <div>
                            <label className="text-[10px] text-[var(--text-muted)] mb-0.5 block">Name</label>
                            <input
                              value={effectiveVal(a, "name") as string}
                              onChange={e => setPendingField(a.id, "name", e.target.value)}
                              className="input text-xs w-full" />
                          </div>
                          <div>
                            <label className="text-[10px] text-[var(--text-muted)] mb-0.5 block">Role (slug)</label>
                            <input
                              value={effectiveVal(a, "role") as string}
                              onChange={e => setPendingField(a.id, "role", e.target.value)}
                              className="input text-xs w-full" />
                          </div>
                        </div>
                        <div>
                          <label className="text-[10px] text-[var(--text-muted)] mb-0.5 block">Description</label>
                          <input
                            value={effectiveVal(a, "description") as string || ""}
                            onChange={e => setPendingField(a.id, "description", e.target.value)}
                            className="input text-xs w-full" />
                        </div>
                        <div>
                          <label className="text-[10px] text-[var(--text-muted)] mb-0.5 block">System Prompt</label>
                          <textarea
                            value={effectiveVal(a, "system_prompt") as string || ""}
                            onChange={e => setPendingField(a.id, "system_prompt", e.target.value)}
                            rows={8}
                            className="input text-xs w-full font-mono leading-relaxed" />
                          <p className="text-[10px] text-[var(--text-muted)] mt-0.5">
                            Note: editing the prompt here creates a custom override stored on the agent. Use the Prompt Version selector above to switch to registry-managed versions.
                          </p>
                        </div>
                        <div>
                          <label className="text-[10px] text-[var(--text-muted)] mb-1 block">Tool Groups</label>
                          <div className="flex gap-1.5 flex-wrap">
                            {Object.keys(tools).map(g => {
                              const currentGroups = (effectiveVal(a, "tool_groups") as string[] | undefined) || a.tool_groups || [];
                              const active = currentGroups.includes(g);
                              return (
                                <button key={g}
                                  onClick={() => setPendingField(a.id, "tool_groups", active ? currentGroups.filter((x: string) => x !== g) : [...currentGroups, g])}
                                  className={`badge border cursor-pointer transition-all ${active ? "bg-zinc-900 text-white border-zinc-900" : "text-[var(--text-muted)] border-[var(--border)] hover:border-zinc-400"}`}>
                                  {g}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                        <div className="flex gap-2 pt-1">
                          <button onClick={() => saveAgent(a)} disabled={!dirty || savingAgent === a.id} className="btn-primary !text-xs">
                            {savingAgent === a.id ? "Saving…" : "Save Changes"}
                          </button>
                          <button onClick={() => cancelEdit(a.id)} className="btn-ghost !text-xs">Discard</button>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>

          {/* Skills */}
          <section className="card">
            <div className="flex items-center justify-between mb-2">
              <div>
                <h2 className="text-sm font-medium">Skills Library</h2>
                <p className="text-[11px] text-[var(--text-muted)]">Skills define HOW agents behave (instructions only, no tools)</p>
              </div>
              <button onClick={() => setShowSkill(true)} className="btn-primary !text-xs">+ New Skill</button>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {skills.map(s => (
                <div key={s.id} className="p-2.5 rounded-lg border border-[var(--border)] bg-[var(--bg)]">
                  <div className="text-sm font-medium">{s.name}</div>
                  <p className="text-[11px] text-[var(--text-muted)] mt-0.5 line-clamp-2">{s.description}</p>
                  {s.trigger_pattern && <div className="text-[10px] text-[var(--warning)] mt-0.5">trigger: &quot;{s.trigger_pattern}&quot;</div>}
                </div>
              ))}
            </div>
            {showSkill && (
              <div className="mt-2 p-3.5 rounded-lg border border-[var(--border)] bg-[var(--bg)] space-y-2.5">
                <div className="grid grid-cols-2 gap-2.5">
                  <input value={skill.name} onChange={e => setSkill({ ...skill, name: e.target.value })} placeholder="Skill name" className="input" />
                  <input value={skill.trigger_pattern} onChange={e => setSkill({ ...skill, trigger_pattern: e.target.value })} placeholder="Trigger pattern" className="input" />
                </div>
                <input value={skill.description} onChange={e => setSkill({ ...skill, description: e.target.value })} placeholder="Description" className="input" />
                <textarea value={skill.instructions} onChange={e => setSkill({ ...skill, instructions: e.target.value })} placeholder="Instructions" rows={3} className="input" />
                <div className="flex gap-2">
                  <button onClick={createSkill} className="btn-primary">Create</button>
                  <button onClick={() => setShowSkill(false)} className="btn-ghost">Cancel</button>
                </div>
              </div>
            )}
          </section>

          {/* Tools */}
          <section className="card">
            <h2 className="text-sm font-medium mb-1">MCP Tool Registry</h2>
            <p className="text-[11px] text-[var(--text-muted)] mb-3">MCP tools define WHAT agents can do. Assign tool groups above.</p>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(tools).map(([group, groupTools]) => (
                <div key={group} className="p-2.5 rounded-lg border border-[var(--border)] bg-[var(--bg)]">
                  <div className="text-sm font-medium text-zinc-700 mb-1.5">{group}</div>
                  {(groupTools as any[]).map((t: any) => (
                    <div key={t.name} className="text-[11px] text-[var(--text-muted)] py-0.5">
                      <span className="text-[var(--text)] font-mono">{t.name}</span> {t.description.slice(0, 45)}…
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </section>
        </div>
      )}

      {/* ── Add Agent Modal ───────────────────────────────────────── */}
      {showAddAgent && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl max-h-[90vh] flex flex-col overflow-hidden">
            {/* Modal header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-100">
              <div>
                <h2 className="text-base font-semibold">Add Agent</h2>
                <p className="text-[11px] text-zinc-500 mt-0.5">
                  {addStep === "template" ? "Choose a prompt template to start from" : "Customize your agent"}
                </p>
              </div>
              <button onClick={() => setShowAddAgent(false)} className="text-zinc-400 hover:text-zinc-700 text-xl leading-none">×</button>
            </div>

            {/* Step indicator */}
            <div className="flex items-center gap-2 px-5 py-2.5 border-b border-zinc-100 bg-zinc-50">
              {(["template", "form"] as const).map((step, i) => (
                <div key={step} className="flex items-center gap-2">
                  {i > 0 && <div className="w-6 h-px bg-zinc-300" />}
                  <div className={`flex items-center gap-1.5 text-xs font-medium ${addStep === step ? "text-zinc-900" : addStep === "form" && step === "template" ? "text-zinc-400" : "text-zinc-400"}`}>
                    <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${addStep === step ? "bg-zinc-900 text-white" : addStep === "form" && step === "template" ? "bg-zinc-300 text-white" : "bg-zinc-200 text-zinc-500"}`}>
                      {addStep === "form" && step === "template" ? "✓" : i + 1}
                    </div>
                    {step === "template" ? "Choose Template" : "Configure Agent"}
                  </div>
                </div>
              ))}
            </div>

            {/* Step content */}
            <div className="flex-1 overflow-y-auto p-5">
              {addStep === "template" ? (
                <div className="grid grid-cols-2 gap-3">
                  {PROMPT_TEMPLATES.map(tmpl => (
                    <button
                      key={tmpl.id}
                      onClick={() => {
                        setSelectedTemplate(tmpl);
                        setDraft(prev => ({ ...prev, system_prompt: tmpl.system_prompt, decision_strategy: tmpl.id === "react" ? "react" : tmpl.id === "cot" ? "cot" : tmpl.id === "plan_execute" ? "plan_execute" : tmpl.id === "reflexion" ? "reflexion" : "react" }));
                      }}
                      className={`p-3.5 rounded-xl border-2 text-left transition-all hover:border-zinc-400 ${selectedTemplate?.id === tmpl.id ? "border-zinc-900 bg-zinc-50" : "border-zinc-200"}`}>
                      <div className="flex items-center gap-2 mb-1.5">
                        <span className="text-lg">{tmpl.icon}</span>
                        <span className="text-sm font-semibold text-zinc-800">{tmpl.label}</span>
                      </div>
                      <div className="text-[10px] font-mono text-indigo-600 mb-1">{tmpl.technique}</div>
                      <p className="text-[11px] text-zinc-500 leading-relaxed">{tmpl.desc}</p>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="space-y-3.5">
                  {selectedTemplate && (
                    <div className="flex items-center gap-2 p-2.5 bg-indigo-50 border border-indigo-100 rounded-lg">
                      <span className="text-base">{selectedTemplate.icon}</span>
                      <div>
                        <div className="text-xs font-semibold text-indigo-800">{selectedTemplate.label}</div>
                        <div className="text-[10px] text-indigo-600 font-mono">{selectedTemplate.technique}</div>
                      </div>
                      <button onClick={() => setAddStep("template")} className="ml-auto text-[10px] text-indigo-500 hover:underline">change</button>
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-2.5">
                    <div>
                      <label className="text-[10px] text-zinc-500 mb-0.5 block">Agent Name *</label>
                      <input value={draft.name} onChange={e => setDraft({ ...draft, name: e.target.value })} placeholder="e.g. Code Writer" className="input text-xs w-full" />
                    </div>
                    <div>
                      <label className="text-[10px] text-zinc-500 mb-0.5 block">Role slug *</label>
                      <input value={draft.role} onChange={e => setDraft({ ...draft, role: e.target.value })} placeholder="e.g. coder" className="input text-xs w-full" />
                    </div>
                  </div>
                  <div>
                    <label className="text-[10px] text-zinc-500 mb-0.5 block">Description</label>
                    <input value={draft.description} onChange={e => setDraft({ ...draft, description: e.target.value })} placeholder="What does this agent do?" className="input text-xs w-full" />
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-1">
                      <label className="text-[10px] text-zinc-500 mb-0.5 block">Model</label>
                      <select value={draft.model} onChange={e => setDraft({ ...draft, model: e.target.value })} className="input text-xs w-full">
                        <option value="">Default ({defaultModelName})</option>
                        {Object.entries(modelOptions().byProvider).map(([provider, models]) => (
                          <optgroup key={provider} label={provider}>
                            {(models as any[]).map((m: any) => (
                              <option key={m.id} value={m.id}>{m.name}</option>
                            ))}
                          </optgroup>
                        ))}
                      </select>
                    </div>
                    <div className="flex-1">
                      <label className="text-[10px] text-zinc-500 mb-0.5 block">Reasoning Strategy</label>
                      <select value={draft.decision_strategy} onChange={e => setDraft({ ...draft, decision_strategy: e.target.value })} className="input text-xs w-full">
                        {AGENT_STRATEGIES.map(s => <option key={s.value} value={s.value}>{s.label} — {s.desc}</option>)}
                      </select>
                    </div>
                  </div>
                  <div>
                    <label className="text-[10px] text-zinc-500 mb-0.5 block">
                      System Prompt — fill in the <span className="font-mono text-amber-600">[FILL IN: ...]</span> placeholders
                    </label>
                    <textarea
                      value={draft.system_prompt}
                      onChange={e => setDraft({ ...draft, system_prompt: e.target.value })}
                      rows={12}
                      className="input text-xs w-full font-mono leading-relaxed" />
                  </div>
                  <div>
                    <label className="text-[10px] text-zinc-500 mb-1 block">Tool Groups</label>
                    <div className="flex gap-1.5 flex-wrap">
                      {Object.keys(tools).map(g => (
                        <button key={g}
                          onClick={() => setDraft({ ...draft, tool_groups: draft.tool_groups.includes(g) ? draft.tool_groups.filter(x => x !== g) : [...draft.tool_groups, g] })}
                          className={`badge border cursor-pointer transition-all ${draft.tool_groups.includes(g) ? "bg-zinc-900 text-white border-zinc-900" : "text-[var(--text-muted)] border-[var(--border)] hover:border-zinc-400"}`}>
                          {g}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Modal footer */}
            <div className="flex items-center justify-between px-5 py-3.5 border-t border-zinc-100 bg-zinc-50">
              <button onClick={() => setShowAddAgent(false)} className="btn-ghost !text-xs">Cancel</button>
              <div className="flex gap-2">
                {addStep === "template" ? (
                  <button
                    onClick={() => setAddStep("form")}
                    disabled={!selectedTemplate}
                    className="btn-primary !text-xs">
                    Next →
                  </button>
                ) : (
                  <>
                    <button onClick={() => setAddStep("template")} className="btn-secondary !text-xs">← Back</button>
                    <button
                      onClick={addAgent}
                      disabled={!draft.name || !draft.role}
                      className="btn-primary !text-xs">
                      Create Agent
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
