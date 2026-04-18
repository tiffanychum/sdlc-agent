"use client";
import React, {
  createContext, useContext, useRef, useState, useCallback, useEffect,
} from "react";
import { api, SSEEvent } from "@/lib/api";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface LiveThinkingSegment { agent: string; text: string; }

export interface LiveTrajectoryStep {
  agent: string;
  action: string;     // "thinking" | "tool:<name>"
  status: "active" | "completed";
  timestamp: number;
}

export interface LiveCaseSession {
  caseId: string;
  caseLabel: string;
  prompt: string;
  index: number;
  total: number;
  status: "queued" | "running" | "pass" | "fail" | "error";
  // chat-style data
  trajectory: LiveTrajectoryStep[];
  thinkingSegments: LiveThinkingSegment[];  // live, cleared on agent_end
  thinkingHistory: { agent: string; text: string }[][];  // per agent turn, archived
  toolCalls: { tool: string; args: Record<string, any>; output?: string; agent?: string }[];
  // result
  result?: {
    passed: boolean;
    latency_ms: number;
    actual_output: string;
    trace_assertions: Record<string, any>;
    deepeval_scores: Record<string, any>;
    semantic_similarity: number;
    actual_cost: number;
    trajectory_pattern: string[];
    error?: string;
  };
}

export interface RunSession {
  sessionId: string;   // unique per startRun call
  runId: string | null;
  label: string;       // e.g. "claude-sonnet-4-6" or "Run #1"
  model?: string;
  status: "running" | "done" | "error" | "stopped";
  cases: Record<string, LiveCaseSession>;
  summary?: {
    total_cases: number;
    num_passed: number;
    total_cost: number;
    avg_latency_ms: number;
  };
  startedAt: number;
}

export interface RegressionRunParams {
  team_id?: string;
  case_ids?: string[];
  model?: string;
  prompt_version?: string;
  prompt_versions_by_role?: Record<string, string>;
  baseline_run_id?: string;
}

interface RegressionRunContextValue {
  sessions: RunSession[];
  activeSessionId: string | null;
  activeCaseId: string | null;
  isAnyRunning: boolean;
  setActiveSessionId: (id: string | null) => void;
  setActiveCaseId: (id: string | null) => void;
  startRun: (params: RegressionRunParams, label?: string) => string;   // returns sessionId
  stopRun: (sessionId: string) => void;
  clearSessions: () => void;
}

const RegressionRunContext = createContext<RegressionRunContextValue | null>(null);

export function useRegressionRun() {
  const ctx = useContext(RegressionRunContext);
  if (!ctx) throw new Error("useRegressionRun must be used inside RegressionRunProvider");
  return ctx;
}

// ── Provider ──────────────────────────────────────────────────────────────────

export function RegressionRunProvider({ children }: { children: React.ReactNode }) {
  const [sessions, setSessions] = useState<RunSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [activeCaseId, setActiveCaseId] = useState<string | null>(null);

  // Abort controllers keyed by sessionId
  const abortRefs = useRef<Record<string, AbortController>>({});

  const isAnyRunning = sessions.some(s => s.status === "running");

  // ── Update helpers ──────────────────────────────────────────────────────────

  const updateSession = useCallback((sessionId: string, updater: (s: RunSession) => RunSession) => {
    setSessions(prev => prev.map(s => s.sessionId === sessionId ? updater(s) : s));
  }, []);

  const updateCase = useCallback((sessionId: string, caseId: string, updater: (c: LiveCaseSession) => LiveCaseSession) => {
    setSessions(prev => prev.map(s => {
      if (s.sessionId !== sessionId) return s;
      const c = s.cases[caseId];
      if (!c) return s;
      return { ...s, cases: { ...s.cases, [caseId]: updater(c) } };
    }));
  }, []);

  // ── SSE event handler ───────────────────────────────────────────────────────

  const handleEvent = useCallback((sessionId: string, event: SSEEvent) => {
    const d = event.data;
    const caseId: string = d.case_id;

    switch (event.type) {
      case "run_start":
        updateSession(sessionId, s => ({ ...s, runId: d.run_id }));
        break;

      case "case_start":
        setSessions(prev => prev.map(s => {
          if (s.sessionId !== sessionId) return s;
          const newCase: LiveCaseSession = {
            caseId: d.case_id, caseLabel: d.case_label, prompt: d.prompt,
            index: d.index, total: d.total,
            status: "running",
            trajectory: [], thinkingSegments: [], thinkingHistory: [], toolCalls: [],
          };
          // Auto-select the first case that starts
          return { ...s, cases: { ...s.cases, [d.case_id]: newCase } };
        }));
        // Auto-focus first case of first session
        setActiveCaseId(prev => prev ?? d.case_id);
        break;

      case "agent_start":
        updateCase(sessionId, caseId, c => ({
          ...c,
          trajectory: [...c.trajectory, { agent: d.agent, action: "thinking", status: "active", timestamp: Date.now() }],
          thinkingSegments: [...c.thinkingSegments, { agent: d.agent, text: "" }],
        }));
        break;

      case "agent_end":
        updateCase(sessionId, caseId, c => {
          // Archive non-empty thinking segments as a history entry
          const nonEmpty = c.thinkingSegments.filter(s => s.text.trim());
          return {
            ...c,
            trajectory: c.trajectory.map(t =>
              t.agent === d.agent && t.status === "active" ? { ...t, status: "completed" as const } : t
            ),
            thinkingSegments: [],
            thinkingHistory: nonEmpty.length > 0 ? [...c.thinkingHistory, nonEmpty] : c.thinkingHistory,
          };
        });
        break;

      case "tool_start":
        updateCase(sessionId, caseId, c => ({
          ...c,
          trajectory: [...c.trajectory, { agent: d.agent, action: `tool:${d.tool}`, status: "active", timestamp: Date.now() }],
          toolCalls: [...c.toolCalls, { tool: d.tool, args: d.args || {}, agent: d.agent }],
        }));
        break;

      case "tool_end":
        updateCase(sessionId, caseId, c => ({
          ...c,
          trajectory: c.trajectory.map(t =>
            t.action === `tool:${d.tool}` && t.status === "active" ? { ...t, status: "completed" as const } : t
          ),
          toolCalls: c.toolCalls.map((tc, i) =>
            i === c.toolCalls.length - 1 && tc.tool === d.tool
              ? { ...tc, output: d.output_preview }
              : tc
          ),
        }));
        break;

      case "llm_token":
        updateCase(sessionId, caseId, c => {
          const segs = [...c.thinkingSegments];
          const last = segs[segs.length - 1];
          if (last && last.agent === (d.agent || "")) {
            segs[segs.length - 1] = { ...last, text: last.text + d.token };
          } else {
            segs.push({ agent: d.agent || "", text: d.token });
          }
          return { ...c, thinkingSegments: segs };
        });
        break;

      case "case_done":
        updateCase(sessionId, caseId, c => ({
          ...c,
          status: d.error ? "error" : d.passed ? "pass" : "fail",
          thinkingSegments: [],
          result: {
            passed: d.passed, latency_ms: d.latency_ms,
            actual_output: d.actual_output || "",
            trace_assertions: d.trace_assertions || {},
            deepeval_scores: d.deepeval_scores || {},
            semantic_similarity: d.semantic_similarity || 0,
            actual_cost: d.actual_cost || 0,
            trajectory_pattern: d.trajectory || [],
            error: d.error,
          },
        }));
        break;

      case "run_done":
        updateSession(sessionId, s => ({
          ...s, status: "done", summary: d.summary,
        }));
        break;

      case "error":
        if (caseId) {
          updateCase(sessionId, caseId, c => ({ ...c, status: "error" }));
        }
        break;
    }
  }, [updateSession, updateCase]);

  // ── Public API ──────────────────────────────────────────────────────────────

  const startRun = useCallback((params: RegressionRunParams, label?: string): string => {
    const sessionId = `session-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const ctrl = new AbortController();
    abortRefs.current[sessionId] = ctrl;

    const session: RunSession = {
      sessionId,
      runId: null,
      label: label ?? (params.model ?? "Run"),
      model: params.model,
      status: "running",
      cases: {},
      startedAt: Date.now(),
    };

    setSessions(prev => [...prev, session]);
    setActiveSessionId(sessionId);
    setActiveCaseId(null);

    // Start streaming in background
    api.regression.stream(params, (evt) => handleEvent(sessionId, evt), ctrl.signal)
      .then(() => {
        setSessions(prev => prev.map(s =>
          s.sessionId === sessionId && s.status === "running"
            ? { ...s, status: "done" }
            : s
        ));
      })
      .catch((e: Error) => {
        if (e.name === "AbortError") {
          setSessions(prev => prev.map(s =>
            s.sessionId === sessionId ? { ...s, status: "stopped" } : s
          ));
        } else {
          setSessions(prev => prev.map(s =>
            s.sessionId === sessionId ? { ...s, status: "error" } : s
          ));
        }
      });

    return sessionId;
  }, [handleEvent]);

  const stopRun = useCallback((sessionId: string) => {
    abortRefs.current[sessionId]?.abort();
    delete abortRefs.current[sessionId];
    setSessions(prev => prev.map(s =>
      s.sessionId === sessionId ? { ...s, status: "stopped" } : s
    ));
  }, []);

  const clearSessions = useCallback(() => {
    // Abort all running
    Object.values(abortRefs.current).forEach(c => c.abort());
    abortRefs.current = {};
    setSessions([]);
    setActiveSessionId(null);
    setActiveCaseId(null);
  }, []);

  return (
    <RegressionRunContext.Provider value={{
      sessions, activeSessionId, activeCaseId, isAnyRunning,
      setActiveSessionId, setActiveCaseId,
      startRun, stopRun, clearSessions,
    }}>
      {children}
    </RegressionRunContext.Provider>
  );
}
