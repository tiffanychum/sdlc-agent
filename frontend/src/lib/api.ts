const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJSON(path: string, options?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  teams: {
    list: () => fetchJSON("/api/teams"),
    get: (id: string) => fetchJSON(`/api/teams/${id}`),
    create: (data: any) => fetchJSON("/api/teams", { method: "POST", body: JSON.stringify(data) }),
    update: (id: string, data: any) => fetchJSON(`/api/teams/${id}`, { method: "PUT", body: JSON.stringify(data) }),
    delete: (id: string) => fetchJSON(`/api/teams/${id}`, { method: "DELETE" }),
    chat: (id: string, message: string) =>
      fetchJSON(`/api/teams/${id}/chat`, { method: "POST", body: JSON.stringify({ message }) }),
    rebuild: (id: string) => fetchJSON(`/api/teams/${id}/rebuild`, { method: "POST" }),
    addAgent: (teamId: string, data: any) =>
      fetchJSON(`/api/teams/${teamId}/agents`, { method: "POST", body: JSON.stringify(data) }),
  },
  agents: {
    update: (id: string, data: any) => fetchJSON(`/api/agents/${id}`, { method: "PUT", body: JSON.stringify(data) }),
    delete: (id: string) => fetchJSON(`/api/agents/${id}`, { method: "DELETE" }),
    assignSkills: (id: string, skillIds: string[]) =>
      fetchJSON(`/api/agents/${id}/skills`, { method: "PUT", body: JSON.stringify(skillIds) }),
  },
  skills: {
    list: () => fetchJSON("/api/skills"),
    create: (data: any) => fetchJSON("/api/skills", { method: "POST", body: JSON.stringify(data) }),
    update: (id: string, data: any) => fetchJSON(`/api/skills/${id}`, { method: "PUT", body: JSON.stringify(data) }),
    delete: (id: string) => fetchJSON(`/api/skills/${id}`, { method: "DELETE" }),
  },
  tools: {
    list: () => fetchJSON("/api/tools"),
  },
  traces: {
    list: (limit = 50) => fetchJSON(`/api/traces?limit=${limit}`),
    get: (id: string) => fetchJSON(`/api/traces/${id}`),
    stats: (days = 30) => fetchJSON(`/api/traces/stats?days=${days}`),
    evaluate: () => fetchJSON("/api/traces/evaluate", { method: "POST" }),
  },
  eval: {
    runs: () => fetchJSON("/api/eval/runs"),
    get: (id: string) => fetchJSON(`/api/eval/runs/${id}`),
    run: (teamId = "default") => fetchJSON("/api/eval/run", { method: "POST", body: JSON.stringify({ team_id: teamId }) }),
    compare: (teamId: string, configs: any[]) =>
      fetchJSON("/api/eval/compare", { method: "POST", body: JSON.stringify({ team_id: teamId, model_configs: configs }) }),
    compareRuns: (a: string, b: string) => fetchJSON(`/api/eval/compare/${a}/${b}`),
  },
  otel: {
    spanStats: (days = 30) => fetchJSON(`/api/otel/spans/stats?days=${days}`),
  },
};
