"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

export default function StudioPage() {
  const [teams, setTeams] = useState<any[]>([]);
  const [team, setTeam] = useState<any>(null);
  const [skills, setSkills] = useState<any[]>([]);
  const [tools, setTools] = useState<any>({});
  const [newTeam, setNewTeam] = useState("");
  const [showAgent, setShowAgent] = useState(false);
  const [showSkill, setShowSkill] = useState(false);
  const [agent, setAgent] = useState({ name: "", role: "", description: "", system_prompt: "", tool_groups: [] as string[] });
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
    await api.teams.create({ name: newTeam });
    setNewTeam("");
    const t = await api.teams.list(); setTeams(t);
    setTeam(await api.teams.get(t[t.length - 1].id));
  }

  async function updateStrategy(s: string) {
    if (!team) return;
    await api.teams.update(team.id, { decision_strategy: s });
    setTeam({ ...team, decision_strategy: s });
  }

  async function addAgent() {
    if (!team) return;
    await api.teams.addAgent(team.id, agent);
    setShowAgent(false); setAgent({ name: "", role: "", description: "", system_prompt: "", tool_groups: [] });
    setTeam(await api.teams.get(team.id));
  }

  async function removeAgent(id: string) {
    await api.agents.delete(id);
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

  const STRATEGIES = [
    { v: "router_decides", l: "Router", d: "LLM picks the best agent" },
    { v: "sequential", l: "Sequential", d: "Agents run in order" },
    { v: "parallel", l: "Parallel", d: "All agents run at once" },
    { v: "supervisor", l: "Supervisor", d: "Supervisor delegates and reviews" },
  ];

  return (
    <div className="space-y-8 max-w-5xl">
      <div>
        <h1 className="text-xl font-semibold">Agent Studio</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">Configure teams, agents, skills, and tools</p>
      </div>

      {/* Team Tabs */}
      <div className="flex items-center gap-2 border-b border-[var(--border)] pb-3">
        {teams.map(t => (
          <button key={t.id} onClick={() => selectTeam(t.id)}
            className={`px-3 py-1.5 rounded-md text-sm transition-all ${team?.id === t.id ? "bg-[var(--accent)]/10 text-[var(--accent)] font-medium" : "text-[var(--text-muted)] hover:text-[var(--text)]"}`}>
            {t.name}
          </button>
        ))}
        <div className="flex items-center gap-1.5 ml-auto">
          <input value={newTeam} onChange={e => setNewTeam(e.target.value)}
            onKeyDown={e => e.key === "Enter" && createTeam()}
            placeholder="New team..." className="input !w-36 !text-xs !py-1.5" />
          <button onClick={createTeam} className="btn-primary !py-1.5 !px-3">+</button>
        </div>
      </div>

      {team && (
        <div className="space-y-6">
          {/* Strategy */}
          <section className="card">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-medium">Decision Strategy</h2>
              <button onClick={rebuild} className="text-xs text-[var(--success)] hover:underline">Rebuild Orchestrator</button>
            </div>
            <div className="grid grid-cols-4 gap-2">
              {STRATEGIES.map(s => (
                <button key={s.v} onClick={() => updateStrategy(s.v)}
                  className={`text-left p-3 rounded-lg text-xs transition-all border ${
                    team.decision_strategy === s.v
                      ? "border-[var(--accent)] bg-[var(--accent)]/5 text-[var(--accent)]"
                      : "border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--text-muted)]"
                  }`}>
                  <div className="font-medium text-sm">{s.l}</div>
                  <div className="mt-0.5 opacity-70">{s.d}</div>
                </button>
              ))}
            </div>
          </section>

          {/* Agents */}
          <section className="card">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-medium">Agents ({team.agents?.length || 0})</h2>
              <button onClick={() => setShowAgent(true)} className="btn-primary !text-xs">Add Agent</button>
            </div>

            <div className="space-y-3">
              {team.agents?.map((a: any) => (
                <div key={a.id} className="p-4 rounded-lg bg-[var(--bg)] border border-[var(--border)]">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-sm">{a.name}</span>
                        <span className="badge bg-[var(--accent)]/10 text-[var(--accent)]">{a.role}</span>
                      </div>
                      <p className="text-xs text-[var(--text-muted)] mt-1">{a.description}</p>
                    </div>
                    <button onClick={() => removeAgent(a.id)} className="text-xs text-[var(--error)] hover:underline">Remove</button>
                  </div>

                  <div className="flex flex-wrap gap-1.5 mt-3">
                    <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mr-1 self-center">Tools:</span>
                    {a.tool_groups?.map((g: string) => (
                      <span key={g} className="badge bg-purple-500/10 text-purple-400">{g}</span>
                    ))}
                  </div>

                  <div className="flex flex-wrap gap-1.5 mt-2">
                    <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mr-1 self-center">Skills:</span>
                    {skills.map(s => {
                      const on = a.skills?.some((x: any) => x.id === s.id);
                      return (
                        <button key={s.id} onClick={() => toggleSkill(a.id, s.id, on)}
                          className={`badge border transition-all ${on ? "bg-[var(--success)]/10 text-[var(--success)] border-[var(--success)]/30" : "bg-transparent text-[var(--text-muted)] border-[var(--border)] hover:text-[var(--text)]"}`}>
                          {s.name}
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>

            {showAgent && (
              <div className="mt-4 p-4 rounded-lg bg-[var(--bg)] border border-[var(--border)] space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <input value={agent.name} onChange={e => setAgent({ ...agent, name: e.target.value })} placeholder="Name" className="input" />
                  <input value={agent.role} onChange={e => setAgent({ ...agent, role: e.target.value })} placeholder="Role (e.g. coder)" className="input" />
                </div>
                <input value={agent.description} onChange={e => setAgent({ ...agent, description: e.target.value })} placeholder="Description" className="input" />
                <textarea value={agent.system_prompt} onChange={e => setAgent({ ...agent, system_prompt: e.target.value })} placeholder="System prompt" rows={3} className="input" />
                <div className="flex gap-2">
                  {Object.keys(tools).map(g => (
                    <button key={g} onClick={() => setAgent({ ...agent, tool_groups: agent.tool_groups.includes(g) ? agent.tool_groups.filter(x => x !== g) : [...agent.tool_groups, g] })}
                      className={`badge border transition-all cursor-pointer ${agent.tool_groups.includes(g) ? "bg-purple-500/10 text-purple-400 border-purple-500/30" : "text-[var(--text-muted)] border-[var(--border)]"}`}>
                      {g}
                    </button>
                  ))}
                </div>
                <div className="flex gap-2">
                  <button onClick={addAgent} className="btn-primary">Create</button>
                  <button onClick={() => setShowAgent(false)} className="btn-ghost">Cancel</button>
                </div>
              </div>
            )}
          </section>

          {/* Skills */}
          <section className="card">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h2 className="text-sm font-medium">Skills Library</h2>
                <p className="text-[11px] text-[var(--text-muted)] mt-0.5">Skills define HOW agents behave (instructions only, no tools)</p>
              </div>
              <button onClick={() => setShowSkill(true)} className="btn-primary !text-xs">New Skill</button>
            </div>

            <div className="grid grid-cols-3 gap-2">
              {skills.map(s => (
                <div key={s.id} className="p-3 rounded-lg bg-[var(--bg)] border border-[var(--border)]">
                  <div className="text-sm font-medium">{s.name}</div>
                  <p className="text-xs text-[var(--text-muted)] mt-1 line-clamp-2">{s.description}</p>
                  {s.trigger_pattern && <div className="text-[10px] text-[var(--warning)] mt-1">trigger: &quot;{s.trigger_pattern}&quot;</div>}
                </div>
              ))}
            </div>

            {showSkill && (
              <div className="mt-3 p-4 rounded-lg bg-[var(--bg)] border border-[var(--border)] space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <input value={skill.name} onChange={e => setSkill({ ...skill, name: e.target.value })} placeholder="Skill name" className="input" />
                  <input value={skill.trigger_pattern} onChange={e => setSkill({ ...skill, trigger_pattern: e.target.value })} placeholder="Trigger pattern (optional)" className="input" />
                </div>
                <input value={skill.description} onChange={e => setSkill({ ...skill, description: e.target.value })} placeholder="Description" className="input" />
                <textarea value={skill.instructions} onChange={e => setSkill({ ...skill, instructions: e.target.value })} placeholder="Instructions (injected into agent prompt)" rows={3} className="input" />
                <div className="flex gap-2">
                  <button onClick={createSkill} className="btn-primary">Create</button>
                  <button onClick={() => setShowSkill(false)} className="btn-ghost">Cancel</button>
                </div>
              </div>
            )}
          </section>

          {/* Tools */}
          <section className="card">
            <h2 className="text-sm font-medium mb-3">MCP Tool Registry</h2>
            <p className="text-[11px] text-[var(--text-muted)] mb-4">MCP tools define WHAT agents can do. Assign tool groups to agents above.</p>
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(tools).map(([group, groupTools]) => (
                <div key={group} className="p-3 rounded-lg bg-[var(--bg)] border border-[var(--border)]">
                  <div className="text-sm font-medium text-purple-400 mb-2">{group}</div>
                  {(groupTools as any[]).map((t: any) => (
                    <div key={t.name} className="text-xs text-[var(--text-muted)] py-0.5 flex gap-1">
                      <span className="text-[var(--text)] font-mono">{t.name}</span>
                      <span className="truncate">{t.description.slice(0, 50)}</span>
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
