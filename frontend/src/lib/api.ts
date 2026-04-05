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
    run: (params: { team_id?: string; case_ids?: string[]; model?: string; prompt_version?: string; baseline_run_id?: string }) =>
      fetchJSON("/api/regression/run", { method: "POST", body: JSON.stringify(params) }),
    runs: () => fetchJSON("/api/regression/runs"),
    results: (runId: string) => fetchJSON(`/api/regression/results/${runId}`),
    caseDetail: (runId: string, caseId: string) => fetchJSON(`/api/regression/results/${runId}/${caseId}`),
    diff: (runA: string, runB: string, caseId: string) => fetchJSON(`/api/regression/diff/${runA}/${runB}/${caseId}`),
    rca: (runId: string, caseId: string, baselineRunId?: string) =>
      fetchJSON("/api/regression/rca", { method: "POST", body: JSON.stringify({ run_id: runId, case_id: caseId, baseline_run_id: baselineRunId }) }),
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
};
