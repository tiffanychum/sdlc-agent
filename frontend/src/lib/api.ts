const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJSON(path: string, options?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

// ── SSE Event Types ──────────────────────────────────────────

export interface SSEEvent {
  type: string;
  data: Record<string, any>;
}

export type SSECallback = (event: SSEEvent) => void;

async function consumeSSE(
  path: string,
  body: Record<string, any>,
  onEvent: SSECallback,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw new Error(`SSE error: ${res.status}`);
  if (!res.body) throw new Error("No response body for SSE");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";

    for (const part of parts) {
      if (!part.trim()) continue;
      const lines = part.split("\n");
      let eventType = "";
      let dataStr = "";
      for (const line of lines) {
        if (line.startsWith("event: ")) eventType = line.slice(7);
        else if (line.startsWith("data: ")) dataStr = line.slice(6);
      }
      if (eventType && dataStr) {
        try {
          onEvent({ type: eventType, data: JSON.parse(dataStr) });
        } catch {
          onEvent({ type: eventType, data: { raw: dataStr } });
        }
      }
    }
  }
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
    chatStream: (id: string, message: string, onEvent: SSECallback, threadId?: string, signal?: AbortSignal, model?: string) =>
      consumeSSE(`/api/teams/${id}/chat/stream`, { message, thread_id: threadId, ...(model ? { model } : {}) }, onEvent, signal),
    chatResume: (id: string, threadId: string, hitlResponse: Record<string, any>, onEvent: SSECallback, signal?: AbortSignal) =>
      consumeSSE(`/api/teams/${id}/chat/resume`, { thread_id: threadId, hitl_response: hitlResponse }, onEvent, signal),
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
  rag: {
    configs: () => fetchJSON("/api/rag/configs"),
    stats: (days = 30) => fetchJSON(`/api/rag/stats?days=${days}`),
    queries: (days = 30, limit = 100) => fetchJSON(`/api/rag/queries?days=${days}&limit=${limit}`),
    history: (configId: string) => fetchJSON(`/api/rag/configs/${configId}/history`),
    query: (data: any) => fetchJSON("/api/rag/query", { method: "POST", body: JSON.stringify(data) }),
    evaluate: (queryId: string) => fetchJSON(`/api/rag/queries/${queryId}/evaluate`, { method: "POST" }),
  },
  golden: {
    list: () => fetchJSON("/api/golden"),
    create: (data: any) => fetchJSON("/api/golden", { method: "POST", body: JSON.stringify(data) }),
    update: (id: string, data: any) => fetchJSON(`/api/golden/${id}`, { method: "PUT", body: JSON.stringify(data) }),
    delete: (id: string) => fetchJSON(`/api/golden/${id}`, { method: "DELETE" }),
    sync: () => fetchJSON("/api/golden/sync", { method: "POST" }),
  },
  regression: {
    run: (params: { team_id?: string; case_ids?: string[]; model?: string; prompt_version?: string; prompt_versions_by_role?: Record<string, string>; baseline_run_id?: string }) =>
      fetchJSON("/api/regression/run", { method: "POST", body: JSON.stringify(params) }),
    stream: (
      params: { team_id?: string; case_ids?: string[]; model?: string; prompt_version?: string; prompt_versions_by_role?: Record<string, string>; baseline_run_id?: string },
      onEvent: SSECallback,
      signal?: AbortSignal,
    ) => consumeSSE("/api/regression/run/stream", params, onEvent, signal),
    runs: (team_id?: string) => fetchJSON(`/api/regression/runs${team_id ? `?team_id=${encodeURIComponent(team_id)}` : ""}`),
    results: (runId: string) => fetchJSON(`/api/regression/results/${runId}`),
    caseDetail: (runId: string, caseId: string) => fetchJSON(`/api/regression/results/${runId}/${caseId}`),
    abOptions: (golden_id: string, team_id?: string) => {
      const q = new URLSearchParams({ golden_id });
      if (team_id) q.append("team_id", team_id);
      return fetchJSON(`/api/regression/ab/options?${q.toString()}`);
    },
    ab: (params: { golden_id: string; run_id_a?: string; run_id_b?: string; model_a?: string; model_b?: string; version_a?: string; version_b?: string }) => {
      const q = new URLSearchParams();
      q.set("golden_id", params.golden_id);
      if (params.run_id_a) q.set("run_id_a", params.run_id_a);
      if (params.run_id_b) q.set("run_id_b", params.run_id_b);
      if (params.model_a) q.set("model_a", params.model_a);
      if (params.model_b) q.set("model_b", params.model_b);
      if (params.version_a) q.set("version_a", params.version_a);
      if (params.version_b) q.set("version_b", params.version_b);
      return fetchJSON(`/api/regression/ab?${q}`);
    },
    // v2 A/B picker: cross-project scope + overlap + adaptive LLM judge.
    abV2Runs: (params: { scope: "same" | "cross"; team_id?: string; limit?: number }) => {
      const q = new URLSearchParams({ scope: params.scope });
      if (params.team_id) q.set("team_id", params.team_id);
      if (params.limit) q.set("limit", String(params.limit));
      return fetchJSON(`/api/regression/ab/runs?${q.toString()}`);
    },
    abV2Overlap: (run_a: string, run_b: string) => {
      const q = new URLSearchParams({ run_a, run_b });
      return fetchJSON(`/api/regression/ab/overlap?${q.toString()}`);
    },
    abV2Judge: (body: { run_a: string; run_b: string; golden_id: string; force_refresh?: boolean }) =>
      fetchJSON("/api/regression/ab/judge", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  },
  models: {
    list: () => fetchJSON("/api/models"),
  },
  config: {
    llm: () => fetchJSON("/api/config/llm"),
  },
  promptVersions: {
    list: () => fetchJSON("/api/prompt-versions"),
    create: (data: any) => fetchJSON("/api/prompt-versions", { method: "POST", body: JSON.stringify(data) }),
    update: (id: string, data: any) => fetchJSON(`/api/prompt-versions/${id}`, { method: "PUT", body: JSON.stringify(data) }),
    delete: (id: string) => fetchJSON(`/api/prompt-versions/${id}`, { method: "DELETE" }),
    current: () => fetchJSON("/api/prompt-versions/current"),
  },
  prompts: {
    // Every prompts.* helper accepts an optional teamId so team-scoped
    // routing-prompt rows (supervisor, router) resolve correctly.  Agent-role
    // prompts are still global; the backend silently ignores team_id for
    // those rows so passing it is always safe.
    versions: (role?: string, teamId?: string) => {
      const params = new URLSearchParams();
      if (role) params.set("role", role);
      if (teamId) params.set("team_id", teamId);
      const qs = params.toString();
      return fetchJSON(`/api/prompts/versions${qs ? `?${qs}` : ""}`);
    },
    text: (role: string, version: string, teamId?: string) => {
      const params = new URLSearchParams({ role, version });
      if (teamId) params.set("team_id", teamId);
      return fetchJSON(`/api/prompts/text?${params.toString()}`);
    },
    diff: (role: string, vOld: string, vNew: string, teamId?: string) => {
      const params = new URLSearchParams({
        role, version_old: vOld, version_new: vNew,
      });
      if (teamId) params.set("team_id", teamId);
      return fetchJSON(`/api/prompts/diff?${params.toString()}`);
    },
    abCompare: (role: string, va: string, vb: string) => fetchJSON(`/api/prompts/ab-compare?role=${encodeURIComponent(role)}&version_a=${encodeURIComponent(va)}&version_b=${encodeURIComponent(vb)}`),
    // Combined routing-prompt view (supervisor / meta_router / router).
    // Pass teamId so supervisor and router rows resolve to the team's scope
    // (meta_router is global regardless).
    routing: (teamId?: string) => {
      const qs = teamId ? `?team_id=${encodeURIComponent(teamId)}` : "";
      return fetchJSON(`/api/prompts/routing${qs}`);
    },
    setRoutingVersion: (role: string, version: string, teamId?: string) =>
      fetchJSON(`/api/prompts/routing/${encodeURIComponent(role)}`, {
        method: "PUT",
        body: JSON.stringify({ version, team_id: teamId }),
      }),
  },
  workflows: {
    // Node-type catalogue + supported models / stores.  Drives the
    // palette on the left and the inspector's dropdowns on the right.
    catalog: () => fetchJSON("/api/workflows/catalog"),
    list: (teamId?: string) => {
      const qs = teamId ? `?team_id=${encodeURIComponent(teamId)}` : "";
      return fetchJSON(`/api/workflows${qs}`);
    },
    get: (id: string) => fetchJSON(`/api/workflows/${id}`),
    create: (data: { name: string; description?: string; graph?: any; team_id?: string | null }) =>
      fetchJSON("/api/workflows", { method: "POST", body: JSON.stringify(data) }),
    update: (id: string, data: { name?: string; description?: string; graph?: any }) =>
      fetchJSON(`/api/workflows/${id}`, { method: "PUT", body: JSON.stringify(data) }),
    delete: (id: string) =>
      fetch(`${API_BASE}/api/workflows/${id}`, { method: "DELETE" }).then((r) => {
        if (!r.ok) throw new Error(`API error: ${r.status}`);
      }),
    run: (id: string, mode: "ingest" | "query", input: Record<string, any>) =>
      fetchJSON(`/api/workflows/${id}/run`, {
        method: "POST",
        body: JSON.stringify({ mode, input }),
      }),
    // Live-trajectory run: emits run_start / node_start / node_end / run_end
    // SSE events so the UI can light up nodes + animate edges as they fire.
    runStream: (
      id: string,
      mode: "ingest" | "query",
      input: Record<string, any>,
      onEvent: SSECallback,
      signal?: AbortSignal,
    ) => consumeSSE(`/api/workflows/${id}/run/stream`, { mode, input }, onEvent, signal),
    runs: (id: string, limit = 50) => fetchJSON(`/api/workflows/${id}/runs?limit=${limit}`),
  },
};
