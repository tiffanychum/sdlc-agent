"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

interface Agent {
  id: string; name: string; role: string; description: string;
  system_prompt: string; tool_groups: string[]; skills: { id: string; name: string }[];
}
interface Team {
  id: string; name: string; description: string; decision_strategy: string; agents: Agent[];
}
interface Skill { id: string; name: string; description: string; instructions: string; trigger_pattern: string; }
interface ToolGroup { [group: string]: { name: string; description: string }[] }

export default function StudioPage() {
  const [teams, setTeams] = useState<any[]>([]);
  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
  const [skills, setSkills] = useState<Skill[]>([]);
  const [tools, setTools] = useState<ToolGroup>({});
  const [newTeamName, setNewTeamName] = useState("");
  const [showNewAgent, setShowNewAgent] = useState(false);
  const [showNewSkill, setShowNewSkill] = useState(false);
  const [newAgent, setNewAgent] = useState({ name: "", role: "", description: "", system_prompt: "", tool_groups: [] as string[] });
  const [newSkill, setNewSkill] = useState({ name: "", description: "", instructions: "", trigger_pattern: "" });

  useEffect(() => { load(); }, []);

  async function load() {
    const [t, s, tl] = await Promise.all([api.teams.list(), api.skills.list(), api.tools.list()]);
    setTeams(t); setSkills(s); setTools(tl);
    if (t.length > 0) { const full = await api.teams.get(t[0].id); setSelectedTeam(full); }
  }

  async function selectTeam(id: string) { setSelectedTeam(await api.teams.get(id)); }

  async function createTeam() {
    if (!newTeamName) return;
    await api.teams.create({ name: newTeamName, description: "", decision_strategy: "router_decides" });
    setNewTeamName(""); await load();
  }

  async function updateStrategy(strategy: string) {
    if (!selectedTeam) return;
    await api.teams.update(selectedTeam.id, { decision_strategy: strategy });
    setSelectedTeam({ ...selectedTeam, decision_strategy: strategy });
  }

  async function addAgent() {
    if (!selectedTeam) return;
    await api.teams.addAgent(selectedTeam.id, newAgent);
    setShowNewAgent(false); setNewAgent({ name: "", role: "", description: "", system_prompt: "", tool_groups: [] });
    setSelectedTeam(await api.teams.get(selectedTeam.id));
  }

  async function deleteAgent(id: string) {
    await api.agents.delete(id);
    if (selectedTeam) setSelectedTeam(await api.teams.get(selectedTeam.id));
  }

  async function createSkill() {
    await api.skills.create(newSkill);
    setShowNewSkill(false); setNewSkill({ name: "", description: "", instructions: "", trigger_pattern: "" });
    setSkills(await api.skills.list());
  }

  async function assignSkill(agentId: string, skillId: string, assigned: boolean) {
    if (!selectedTeam) return;
    const agent = selectedTeam.agents.find(a => a.id === agentId);
    if (!agent) return;
    const currentIds = agent.skills.map(s => s.id);
    const newIds = assigned ? currentIds.filter(id => id !== skillId) : [...currentIds, skillId];
    await api.agents.assignSkills(agentId, newIds);
    setSelectedTeam(await api.teams.get(selectedTeam.id));
  }

  async function rebuildTeam() {
    if (!selectedTeam) return;
    await api.teams.rebuild(selectedTeam.id);
  }

  const STRATEGIES = [
    { value: "router_decides", label: "Router Decides", desc: "LLM classifies request and routes to one agent" },
    { value: "sequential", label: "Sequential", desc: "Agents run in order, passing context" },
    { value: "parallel", label: "Parallel", desc: "All agents run simultaneously" },
    { value: "supervisor", label: "Supervisor", desc: "Supervisor reviews and can re-delegate" },
  ];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Agent Studio</h1>

      {/* Teams Sidebar + Main */}
      <div className="grid grid-cols-4 gap-6">
        {/* Team List */}
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">Teams</h2>
          {teams.map(t => (
            <button key={t.id} onClick={() => selectTeam(t.id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${selectedTeam?.id === t.id ? "bg-blue-600/20 text-blue-400 border border-blue-500/30" : "bg-gray-800/50 hover:bg-gray-800 text-gray-300"}`}>
              <div className="font-medium">{t.name}</div>
              <div className="text-xs text-gray-500">{t.agents_count} agent(s) · {t.decision_strategy}</div>
            </button>
          ))}
          <div className="flex gap-2">
            <input value={newTeamName} onChange={e => setNewTeamName(e.target.value)} placeholder="New team name"
              className="flex-1 px-2 py-1.5 text-sm bg-gray-800 border border-gray-700 rounded-md" />
            <button onClick={createTeam} className="px-3 py-1.5 text-sm bg-blue-600 rounded-md hover:bg-blue-500">+</button>
          </div>
        </div>

        {/* Team Detail */}
        <div className="col-span-3 space-y-6">
          {selectedTeam ? (
            <>
              {/* Decision Strategy */}
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h2 className="font-semibold">Decision Strategy</h2>
                  <button onClick={rebuildTeam} className="px-3 py-1 text-xs bg-green-600/20 text-green-400 border border-green-500/30 rounded-md hover:bg-green-600/30">Rebuild Orchestrator</button>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {STRATEGIES.map(s => (
                    <button key={s.value} onClick={() => updateStrategy(s.value)}
                      className={`text-left px-3 py-2 rounded-lg text-sm border transition-colors ${selectedTeam.decision_strategy === s.value ? "bg-blue-600/20 border-blue-500/30 text-blue-400" : "bg-gray-800/50 border-gray-700 hover:bg-gray-800 text-gray-400"}`}>
                      <div className="font-medium">{s.label}</div>
                      <div className="text-xs opacity-70">{s.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Agents */}
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h2 className="font-semibold">Team Members (Agents)</h2>
                  <button onClick={() => setShowNewAgent(true)} className="px-3 py-1 text-xs bg-blue-600 rounded-md hover:bg-blue-500">+ Add Agent</button>
                </div>

                {selectedTeam.agents.map(agent => (
                  <div key={agent.id} className="bg-gray-800/50 border border-gray-700 rounded-lg p-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="font-medium">{agent.name}</span>
                        <span className="ml-2 text-xs px-2 py-0.5 bg-gray-700 rounded-full">{agent.role}</span>
                      </div>
                      <button onClick={() => deleteAgent(agent.id)} className="text-xs text-red-400 hover:text-red-300">Delete</button>
                    </div>
                    <p className="text-sm text-gray-400">{agent.description}</p>

                    {/* Tool Groups */}
                    <div className="flex gap-1 flex-wrap">
                      <span className="text-xs text-gray-500">MCP Tools:</span>
                      {agent.tool_groups.map(tg => (
                        <span key={tg} className="text-xs px-2 py-0.5 bg-purple-600/20 text-purple-400 border border-purple-500/30 rounded-full">{tg}</span>
                      ))}
                    </div>

                    {/* Skills */}
                    <div className="flex gap-1 flex-wrap">
                      <span className="text-xs text-gray-500">Skills:</span>
                      {skills.map(skill => {
                        const assigned = agent.skills.some(s => s.id === skill.id);
                        return (
                          <button key={skill.id} onClick={() => assignSkill(agent.id, skill.id, assigned)}
                            className={`text-xs px-2 py-0.5 rounded-full border transition-colors ${assigned ? "bg-green-600/20 text-green-400 border-green-500/30" : "bg-gray-700/50 text-gray-500 border-gray-600 hover:text-gray-300"}`}>
                            {skill.name}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                ))}

                {/* New Agent Form */}
                {showNewAgent && (
                  <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
                    <input value={newAgent.name} onChange={e => setNewAgent({ ...newAgent, name: e.target.value })} placeholder="Agent name" className="w-full px-3 py-2 text-sm bg-gray-900 border border-gray-700 rounded-md" />
                    <input value={newAgent.role} onChange={e => setNewAgent({ ...newAgent, role: e.target.value })} placeholder="Role (e.g., coder)" className="w-full px-3 py-2 text-sm bg-gray-900 border border-gray-700 rounded-md" />
                    <input value={newAgent.description} onChange={e => setNewAgent({ ...newAgent, description: e.target.value })} placeholder="Description" className="w-full px-3 py-2 text-sm bg-gray-900 border border-gray-700 rounded-md" />
                    <textarea value={newAgent.system_prompt} onChange={e => setNewAgent({ ...newAgent, system_prompt: e.target.value })} placeholder="System prompt" rows={3} className="w-full px-3 py-2 text-sm bg-gray-900 border border-gray-700 rounded-md" />
                    <div className="flex gap-2 flex-wrap">
                      {Object.keys(tools).map(group => (
                        <button key={group} onClick={() => {
                          const tgs = newAgent.tool_groups.includes(group)
                            ? newAgent.tool_groups.filter(g => g !== group)
                            : [...newAgent.tool_groups, group];
                          setNewAgent({ ...newAgent, tool_groups: tgs });
                        }} className={`text-xs px-2 py-1 rounded-md border ${newAgent.tool_groups.includes(group) ? "bg-purple-600/20 text-purple-400 border-purple-500/30" : "bg-gray-700 text-gray-400 border-gray-600"}`}>
                          {group}
                        </button>
                      ))}
                    </div>
                    <div className="flex gap-2">
                      <button onClick={addAgent} className="px-4 py-1.5 text-sm bg-blue-600 rounded-md hover:bg-blue-500">Create</button>
                      <button onClick={() => setShowNewAgent(false)} className="px-4 py-1.5 text-sm bg-gray-700 rounded-md hover:bg-gray-600">Cancel</button>
                    </div>
                  </div>
                )}
              </div>

              {/* Skills Library */}
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h2 className="font-semibold">Skills Library</h2>
                  <button onClick={() => setShowNewSkill(true)} className="px-3 py-1 text-xs bg-blue-600 rounded-md hover:bg-blue-500">+ New Skill</button>
                </div>
                <p className="text-xs text-gray-500">Skills define HOW agents should behave. They inject instructions only — no tools.</p>

                <div className="grid grid-cols-2 gap-2">
                  {skills.map(skill => (
                    <div key={skill.id} className="bg-gray-800/50 border border-gray-700 rounded-lg p-3">
                      <div className="font-medium text-sm">{skill.name}</div>
                      <p className="text-xs text-gray-400 mt-1">{skill.description}</p>
                      {skill.trigger_pattern && <span className="text-xs text-yellow-400 mt-1 inline-block">Trigger: &quot;{skill.trigger_pattern}&quot;</span>}
                    </div>
                  ))}
                </div>

                {showNewSkill && (
                  <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
                    <input value={newSkill.name} onChange={e => setNewSkill({ ...newSkill, name: e.target.value })} placeholder="Skill name" className="w-full px-3 py-2 text-sm bg-gray-900 border border-gray-700 rounded-md" />
                    <input value={newSkill.description} onChange={e => setNewSkill({ ...newSkill, description: e.target.value })} placeholder="Description" className="w-full px-3 py-2 text-sm bg-gray-900 border border-gray-700 rounded-md" />
                    <textarea value={newSkill.instructions} onChange={e => setNewSkill({ ...newSkill, instructions: e.target.value })} placeholder="Instructions (injected into agent prompt)" rows={3} className="w-full px-3 py-2 text-sm bg-gray-900 border border-gray-700 rounded-md" />
                    <input value={newSkill.trigger_pattern} onChange={e => setNewSkill({ ...newSkill, trigger_pattern: e.target.value })} placeholder="Trigger pattern (optional)" className="w-full px-3 py-2 text-sm bg-gray-900 border border-gray-700 rounded-md" />
                    <div className="flex gap-2">
                      <button onClick={createSkill} className="px-4 py-1.5 text-sm bg-blue-600 rounded-md hover:bg-blue-500">Create</button>
                      <button onClick={() => setShowNewSkill(false)} className="px-4 py-1.5 text-sm bg-gray-700 rounded-md hover:bg-gray-600">Cancel</button>
                    </div>
                  </div>
                )}
              </div>

              {/* Tool Registry */}
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
                <h2 className="font-semibold">MCP Tool Registry</h2>
                <p className="text-xs text-gray-500">MCP tools define WHAT agents can do. Assign tool groups to agents above.</p>
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(tools).map(([group, groupTools]) => (
                    <div key={group} className="bg-gray-800/50 border border-gray-700 rounded-lg p-3">
                      <div className="font-medium text-sm text-purple-400 mb-2">{group}</div>
                      {(groupTools as any[]).map((t: any) => (
                        <div key={t.name} className="text-xs text-gray-400 py-0.5">
                          <span className="text-gray-300">{t.name}</span> — {t.description.slice(0, 60)}...
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="text-gray-500 text-center py-20">Select a team to configure</div>
          )}
        </div>
      </div>
    </div>
  );
}
