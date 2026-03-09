"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

const LLM_MODELS = [
  { value: "", label: "Default (from .env)" },
  { value: "gemini-2.5-flash-lite", label: "Gemini 2.5 Flash Lite" },
  { value: "llama-3.1-8b-cs", label: "Llama 3.1 8B" },
  { value: "mistral-small-3", label: "Mistral Small 3" },
  { value: "grok-4.1-fast-reasoning", label: "Grok 4.1 Fast" },
  { value: "claude-haiku-3", label: "Claude Haiku 3" },
  { value: "gpt-4o-mini-search", label: "GPT-4o Mini (Search)" },
];

const TEAM_STRATEGIES = [
  { v: "router_decides", l: "Router", d: "LLM picks the best agent" },
  { v: "sequential", l: "Sequential", d: "Agents run in order" },
  { v: "parallel", l: "Parallel", d: "All agents run at once" },
  { v: "supervisor", l: "Supervisor", d: "Supervisor delegates and reviews" },
];

const AGENT_STRATEGIES = [
  { value: "react", label: "ReAct", desc: "Reason + Act loop" },
  { value: "plan_execute", label: "Plan & Execute", desc: "Plan first, then execute" },
  { value: "reflexion", label: "Reflexion", desc: "Self-reflection after each step" },
  { value: "cot", label: "Chain-of-Thought", desc: "Think thoroughly before acting" },
];

export default function StudioPage() {
  const [teams, setTeams] = useState<any[]>([]);
  const [team, setTeam] = useState<any>(null);
  const [skills, setSkills] = useState<any[]>([]);
  const [tools, setTools] = useState<any>({});
  const [newTeam, setNewTeam] = useState("");
  const [editingName, setEditingName] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [showAgent, setShowAgent] = useState(false);
  const [showSkill, setShowSkill] = useState(false);
  const [agent, setAgent] = useState({ name: "", role: "", description: "", system_prompt: "", tool_groups: [] as string[], model: "" });
  const [skill, setSkill] = useState({ name: "", description: "", instructions: "", trigger_pattern: "" });

  useEffect(() => { load(); }, []);

  async function load() {
    const [t, s, tl] = await Promise.all([api.teams.list(), api.skills.list(), api.tools.list()]);
    setTeams(t); setSkills(s); setTools(tl);
    if (t.length > 0 && !team) setTeam(await api.teams.get(t[0].id));
  }

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

  async function addAgent() {
    if (!team) return;
    await api.teams.addAgent(team.id, agent);
    setShowAgent(false); setAgent({ name: "", role: "", description: "", system_prompt: "", tool_groups: [], model: "" });
    setTeam(await api.teams.get(team.id));
  }

  async function removeAgent(id: string) {
    await api.agents.delete(id);
    if (team) setTeam(await api.teams.get(team.id));
  }

  async function updateAgentModel(agentId: string, model: string) {
    await api.agents.update(agentId, { model });
    if (team) setTeam(await api.teams.get(team.id));
  }

  async function createSkill() {
    await api.skills.create(skill);
    setShowSkill(false); setSkill({ name: "", description: "", instructions: "", trigger_pattern: "" });
    setSkills(await api.skills.list());
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
                className={`px-3 py-1.5 rounded-md text-[13px] transition-all ${team?.id === t.id ? "bg-[var(--accent-light)] text-[var(--accent)] font-medium" : "text-[var(--text-muted)] hover:text-[var(--text)]"}`}>
                {t.name}
              </button>
            )}
            {team?.id === t.id && t.id !== "default" && (
              <button onClick={() => deleteTeam(t.id)}
                className="opacity-0 group-hover:opacity-100 text-[10px] text-[var(--error)] ml-0.5 hover:bg-[var(--error-light)] rounded px-1 transition-all"
                title="Delete team">x</button>
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
          {/* Agents */}
          <section className="card">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-medium">Agents ({team.agents?.length || 0})</h2>
              <div className="flex gap-2">
                <button onClick={rebuild} className="btn-secondary !text-xs">Rebuild</button>
                <button onClick={() => setShowAgent(true)} className="btn-primary !text-xs">+ Add Agent</button>
              </div>
            </div>

            <div className="space-y-2.5">
              {team.agents?.map((a: any) => (
                <div key={a.id} className="p-3.5 rounded-lg border border-[var(--border)] bg-[var(--bg)]">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{a.name}</span>
                      <span className="badge bg-[var(--accent-light)] text-[var(--accent)]">{a.role}</span>
                    </div>
                    <button onClick={() => removeAgent(a.id)} className="btn-danger !text-[11px] !py-0 !px-1.5">Remove</button>
                  </div>
                  <p className="text-xs text-[var(--text-muted)] mt-1">{a.description}</p>

                  {/* Model + Strategy Selectors */}
                  <div className="flex items-center gap-4 mt-2.5">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide">Model:</span>
                      <select value={a.model || ""} onChange={e => updateAgentModel(a.id, e.target.value)}
                        className="input !w-auto !py-1 !text-xs">
                        {LLM_MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                      </select>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide">Strategy:</span>
                      <select value={a.decision_strategy || "react"}
                        onChange={async e => { await api.agents.update(a.id, { decision_strategy: e.target.value }); if (team) setTeam(await api.teams.get(team.id)); }}
                        className="input !w-auto !py-1 !text-xs">
                        {AGENT_STRATEGIES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
                      </select>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-1 mt-2">
                    <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mr-1 self-center">Tools:</span>
                    {a.tool_groups?.map((g: string) => (
                      <span key={g} className="badge bg-[var(--purple-light)] text-[var(--purple)]">{g}</span>
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
              ))}
            </div>

            {showAgent && (
              <div className="mt-3 p-3.5 rounded-lg border border-[var(--border)] bg-[var(--bg)] space-y-2.5">
                <div className="grid grid-cols-2 gap-2.5">
                  <input value={agent.name} onChange={e => setAgent({ ...agent, name: e.target.value })} placeholder="Name" className="input" />
                  <input value={agent.role} onChange={e => setAgent({ ...agent, role: e.target.value })} placeholder="Role (e.g. coder)" className="input" />
                </div>
                <input value={agent.description} onChange={e => setAgent({ ...agent, description: e.target.value })} placeholder="Description" className="input" />
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[var(--text-muted)]">Model:</span>
                  <select value={agent.model} onChange={e => setAgent({ ...agent, model: e.target.value })} className="input !w-auto">
                    {LLM_MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                  </select>
                </div>
                <textarea value={agent.system_prompt} onChange={e => setAgent({ ...agent, system_prompt: e.target.value })} placeholder="System prompt" rows={3} className="input" />
                <div className="flex gap-1.5 flex-wrap">
                  <span className="text-xs text-[var(--text-muted)] self-center">Tool groups:</span>
                  {Object.keys(tools).map(g => (
                    <button key={g} onClick={() => setAgent({ ...agent, tool_groups: agent.tool_groups.includes(g) ? agent.tool_groups.filter(x => x !== g) : [...agent.tool_groups, g] })}
                      className={`badge border cursor-pointer ${agent.tool_groups.includes(g) ? "bg-[var(--purple-light)] text-[var(--purple)] border-[var(--purple)]/30" : "text-[var(--text-muted)] border-[var(--border)]"}`}>
                      {g}
                    </button>
                  ))}
                </div>
                <div className="flex gap-2 pt-1">
                  <button onClick={addAgent} className="btn-primary">Create</button>
                  <button onClick={() => setShowAgent(false)} className="btn-ghost">Cancel</button>
                </div>
              </div>
            )}
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
                  <div className="text-sm font-medium text-[var(--purple)] mb-1.5">{group}</div>
                  {(groupTools as any[]).map((t: any) => (
                    <div key={t.name} className="text-[11px] text-[var(--text-muted)] py-0.5">
                      <span className="text-[var(--text)] font-mono">{t.name}</span> {t.description.slice(0, 45)}...
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
