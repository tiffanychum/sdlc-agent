"use client";

/**
 * ChatSessionContext — per-team persistent chat sessions.
 *
 * Problem this solves (C1-C5):
 *   Before, `chat/page.tsx` owned all live streaming state in React `useState`
 *   hooks.  As soon as the user navigated away (e.g. to /regression) or
 *   switched teams, the component unmounted and its `AbortController`
 *   aborted — the live chat was gone.  The user had no way to keep a slow
 *   multi-agent run going in the background while reviewing something else.
 *
 * What this provides:
 *   - One live chat session PER TEAM (so two teams can stream in parallel).
 *   - Provider lives in `layout.tsx`, so it survives page navigation and
 *     only dies on full reload.
 *   - Live state (liveSpans / trajectory / thinking / etc.) is keyed by
 *     teamId; the chat page reads the session for the currently-selected
 *     team and renders accordingly.
 *   - `sendMessage` / `resumeHITL` / `stopSession` run the `fetch`-based SSE
 *     loop *inside the provider* so it does not stop when the consumer
 *     component unmounts.
 *   - Conversation history (messages per conversation) is also owned here
 *     so incoming `response` events keep appending to the right conversation
 *     even while the chat page isn't mounted.
 *   - On `beforeunload` we snapshot every live session to localStorage
 *     (C5) so a page reload can display the last-known state.  We do NOT
 *     attempt backend reattach — the snapshot is display-only.
 *
 * Non-goals:
 *   - Queuing multiple parallel chats on the SAME team.  One chat per team,
 *     the same UX as today.
 *   - Surviving a full page reload.  Fetch streams cannot resume; the
 *     snapshot is for visual continuity only.
 */

import React, {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
} from "react";
import { api, SSEEvent } from "@/lib/api";

// ── Types (mirrored from chat/page.tsx so the page can import from here) ─────

export interface ToolCall { tool: string; args: Record<string, any>; agent?: string }
export interface SpanInfo {
  id: string; name: string; span_type: string;
  tokens_in: number; tokens_out: number; cost: number;
  model: string; status: string; error?: string | null;
  agent?: string;
}
export interface TraceInfo {
  trace_id: string; total_latency_ms: number; total_tokens: number;
  total_cost: number; spans: SpanInfo[];
}
export interface ThinkingSegment { agent: string; text: string }
export interface HITLData {
  type: "clarification" | "plan_review" | "action_confirmation" | "tool_review";
  thread_id: string;
  question?: string;
  options?: string[];
  plan?: PlanStep[];
  tool_name?: string;
  args?: Record<string, any>;
  risk_level?: string;
  reason?: string;
  output?: string;
  message?: string;
  agent?: string;
}
export interface PlanStep { step: number; description: string; status: string }
export interface TrajectoryStep {
  agent: string;
  action: string;
  status: "active" | "completed" | "error";
  timestamp: number;
}
export interface LiveSpan extends SpanInfo { agent?: string }

export interface Message {
  role: "user" | "assistant" | "hitl";
  content: string;
  agent?: string;
  toolCalls?: ToolCall[];
  trace?: TraceInfo;
  hitl?: HITLData;
  thinkingContent?: string;
}

export interface Conversation {
  id: string;
  title: string;
  teamId: string;
  threadId: string | null;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

/** Ephemeral live-stream state for one team's currently-running (or most
 * recently completed) chat turn. */
export interface LiveChatSession {
  teamId: string;
  conversationId: string | null;
  threadId: string | null;
  queryActive: boolean;
  inputLocked: boolean;
  statusText: string;
  resolvedStrategy: string | null;
  liveSpans: LiveSpan[];
  trajectory: TrajectoryStep[];
  thinkingSegments: ThinkingSegment[];
  activeAgent: string | null;
  activeTool: string | null;
  liveTokens: number;
  liveCost: number;
  liveStartTime: number;
  elapsedFinal: number;
  pendingHITL: HITLData | null;
  activeTrace: TraceInfo | null;
  error?: string;
}

const emptySession = (teamId: string): LiveChatSession => ({
  teamId,
  conversationId: null,
  threadId: null,
  queryActive: false,
  inputLocked: false,
  statusText: "Working...",
  resolvedStrategy: null,
  liveSpans: [],
  trajectory: [],
  thinkingSegments: [],
  activeAgent: null,
  activeTool: null,
  liveTokens: 0,
  liveCost: 0,
  liveStartTime: 0,
  elapsedFinal: 0,
  pendingHITL: null,
  activeTrace: null,
});

// ── Persistence keys (existing values, reused so we don't lose history) ──────

const CONVOS_KEY = "sdlc_conversations";
const LAST_TRACE_KEY = "sdlc_last_trace";
const LIVE_SNAPSHOT_KEY = "sdlc_live_chat_snapshot";   // new (C5)

function loadConversations(): Conversation[] {
  if (typeof window === "undefined") return [];
  try { return JSON.parse(localStorage.getItem(CONVOS_KEY) || "[]"); }
  catch { return []; }
}

function saveConversations(convos: Conversation[]) {
  try { localStorage.setItem(CONVOS_KEY, JSON.stringify(convos.slice(0, 50))); }
  catch { /* quota */ }
}

function genId() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 7); }

// ── Context shape ────────────────────────────────────────────────────────────

interface ChatSessionContextValue {
  // Conversations (persistent history)
  conversations: Conversation[];
  createConversation: (teamId: string, title?: string) => Conversation;
  deleteConversation: (id: string) => void;
  setConversationMessages: (id: string, mutate: (msgs: Message[]) => Message[]) => void;
  renameConversationIfUntitled: (id: string, from: string) => void;

  // Live sessions keyed by teamId
  getSession: (teamId: string) => LiveChatSession;
  setThinkingCollapsed: (teamId: string, collapsed: boolean) => void;
  clearLiveStateForTeam: (teamId: string) => void;

  // Streaming API
  sendMessage: (
    teamId: string,
    conversationId: string,
    message: string,
    opts?: { threadId?: string | null; model?: string },
  ) => Promise<void>;
  resumeHITL: (
    teamId: string,
    threadId: string,
    response: Record<string, any>,
  ) => Promise<void>;
  stopSession: (teamId: string) => void;

  // "Is any team streaming right now?" — powers cross-team badges (C4).
  runningTeamIds: string[];
  isTeamRunning: (teamId: string) => boolean;
}

const ChatSessionContext = createContext<ChatSessionContextValue | null>(null);

export function useChatSession() {
  const ctx = useContext(ChatSessionContext);
  if (!ctx) throw new Error("useChatSession must be used inside <ChatSessionProvider>");
  return ctx;
}

// ── Provider ─────────────────────────────────────────────────────────────────

export function ChatSessionProvider({ children }: { children: React.ReactNode }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [sessions, setSessions] = useState<Record<string, LiveChatSession>>({});

  // AbortControllers keyed by teamId (only one in-flight per team).
  const abortRefs = useRef<Record<string, AbortController>>({});

  // thinkingSegments mirrors per-team for stable-closure reads inside the
  // async stream callback (same pattern the old chat/page used).
  const thinkingSegmentsRefs = useRef<Record<string, ThinkingSegment[]>>({});
  const agentRefs = useRef<Record<string, string | null>>({});

  // ── Hydration: pull conversations out of localStorage once. ─────────────────
  useEffect(() => {
    setConversations(loadConversations());
  }, []);

  // Persist conversations on every change (cheap — capped at 50).
  useEffect(() => { saveConversations(conversations); }, [conversations]);

  // ── C5: snapshot live sessions on navigation / reload. ──────────────────────
  useEffect(() => {
    const snapshot = () => {
      try {
        // Only snapshot running sessions; completed ones are redundant
        // (their final Message already lives in `conversations`).
        const live = Object.values(sessions).filter(s => s.queryActive);
        if (live.length === 0) {
          localStorage.removeItem(LIVE_SNAPSHOT_KEY);
          return;
        }
        localStorage.setItem(LIVE_SNAPSHOT_KEY, JSON.stringify({
          at: Date.now(),
          sessions: live,
        }));
      } catch { /* quota / private mode — non-fatal */ }
    };
    window.addEventListener("beforeunload", snapshot);
    return () => window.removeEventListener("beforeunload", snapshot);
  }, [sessions]);

  // ── Helpers ─────────────────────────────────────────────────────────────────
  const updateSession = useCallback(
    (teamId: string, updater: (s: LiveChatSession) => LiveChatSession) => {
      setSessions(prev => ({
        ...prev,
        [teamId]: updater(prev[teamId] || emptySession(teamId)),
      }));
    }, []);

  const getSession = useCallback((teamId: string): LiveChatSession => {
    return sessions[teamId] || emptySession(teamId);
  }, [sessions]);

  const clearLiveStateForTeam = useCallback((teamId: string) => {
    thinkingSegmentsRefs.current[teamId] = [];
    agentRefs.current[teamId] = null;
    updateSession(teamId, s => ({
      ...emptySession(teamId),
      // Preserve the last activeTrace + conversationId + threadId so the UI
      // can keep showing the previous turn's trace when we start fresh.
      conversationId: s.conversationId,
      threadId: s.threadId,
      activeTrace: s.activeTrace,
    }));
  }, [updateSession]);

  const setThinkingCollapsed = useCallback((_teamId: string, _collapsed: boolean) => {
    // Thinking-collapsed is a pure UI-preference with no streaming tie-in;
    // the page still owns that local state.  Method kept for API symmetry.
  }, []);

  // ── Conversations ───────────────────────────────────────────────────────────
  const createConversation = useCallback((teamId: string, title?: string): Conversation => {
    const c: Conversation = {
      id: genId(),
      title: title || "New Chat",
      teamId,
      threadId: null,
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setConversations(prev => [c, ...prev]);
    return c;
  }, []);

  const deleteConversation = useCallback((id: string) => {
    setConversations(prev => prev.filter(c => c.id !== id));
  }, []);

  const setConversationMessages = useCallback(
    (id: string, mutate: (msgs: Message[]) => Message[]) => {
      setConversations(prev => prev.map(c =>
        c.id === id
          ? { ...c, messages: mutate(c.messages), updatedAt: Date.now() }
          : c
      ));
    }, []);

  const renameConversationIfUntitled = useCallback((id: string, from: string) => {
    setConversations(prev => prev.map(c => {
      if (c.id !== id) return c;
      if (c.title && c.title !== "New Chat") return c;
      const t = from.slice(0, 40);
      return { ...c, title: t.length < from.length ? t + "..." : t };
    }));
  }, []);

  // ── SSE handler — one per team, bound via teamId ────────────────────────────
  const handleEvent = useCallback((teamId: string, conversationId: string, event: SSEEvent) => {
    const { type, data } = event;

    // Make sure the per-team refs exist
    if (!thinkingSegmentsRefs.current[teamId]) thinkingSegmentsRefs.current[teamId] = [];
    if (agentRefs.current[teamId] === undefined) agentRefs.current[teamId] = null;

    const segs = thinkingSegmentsRefs.current[teamId];

    switch (type) {
      case "thread_id":
        updateSession(teamId, s => ({ ...s, threadId: data.thread_id }));
        setConversations(prev => prev.map(c =>
          c.id === conversationId ? { ...c, threadId: data.thread_id } : c
        ));
        break;

      case "strategy_selected":
        updateSession(teamId, s => ({
          ...s,
          resolvedStrategy: data.strategy,
          statusText: `Strategy: ${data.strategy} (auto-selected)`,
        }));
        break;

      case "agent_start":
        agentRefs.current[teamId] = data.agent;
        segs.push({ agent: data.agent, text: "" });
        updateSession(teamId, s => ({
          ...s,
          activeAgent: data.agent,
          statusText: `${data.agent} is working...`,
          trajectory: [...s.trajectory, {
            agent: data.agent, action: "thinking",
            status: "active", timestamp: Date.now(),
          }],
          thinkingSegments: [...segs],
        }));
        break;

      case "agent_end":
        updateSession(teamId, s => ({
          ...s,
          activeAgent: null,
          trajectory: s.trajectory.map(t =>
            t.agent === data.agent && t.status === "active"
              ? { ...t, status: "completed" as const } : t),
        }));
        agentRefs.current[teamId] = null;
        break;

      case "tool_start":
        updateSession(teamId, s => ({
          ...s,
          activeTool: data.tool,
          statusText: `${data.agent || "Agent"}: ${data.tool}...`,
          trajectory: [...s.trajectory, {
            agent: data.agent || agentRefs.current[teamId] || "",
            action: `tool:${data.tool}`,
            status: "active", timestamp: Date.now(),
          }],
        }));
        break;

      case "tool_end":
        updateSession(teamId, s => ({
          ...s,
          activeTool: null,
          trajectory: s.trajectory.map(t =>
            t.action === `tool:${data.tool}` && t.status === "active"
              ? { ...t, status: "completed" as const } : t),
        }));
        break;

      case "llm_token": {
        const tokenAgent = data.agent || agentRefs.current[teamId] || "unknown";
        const last = segs[segs.length - 1];
        if (last && last.agent === tokenAgent) last.text += data.token;
        else segs.push({ agent: tokenAgent, text: data.token });
        updateSession(teamId, s => ({ ...s, thinkingSegments: [...segs] }));
        break;
      }

      case "trace_span": {
        const spanEvent = data.event;
        const spanData: LiveSpan = {
          ...data.span,
          agent: data.span?.agent || agentRefs.current[teamId] || undefined,
        };
        updateSession(teamId, s => {
          let nextSpans: LiveSpan[];
          let addTokens = 0;
          let addCost = 0;
          if (spanEvent === "span_start") {
            nextSpans = [...s.liveSpans.filter(x => x.id !== spanData.id), spanData];
          } else {
            nextSpans = s.liveSpans.map(x =>
              x.id === spanData.id ? { ...spanData, agent: x.agent || spanData.agent } : x
            );
            addTokens = (spanData.tokens_in || 0) + (spanData.tokens_out || 0);
            addCost = spanData.cost || 0;
          }
          return {
            ...s,
            liveSpans: nextSpans,
            liveTokens: s.liveTokens + addTokens,
            liveCost: s.liveCost + addCost,
          };
        });
        break;
      }

      case "hitl_request": {
        const capturedSegsHITL = segs.filter(s => s.text.trim());
        updateSession(teamId, s => ({
          ...s,
          inputLocked: false,
          statusText: "Waiting for your input...",
          pendingHITL: data as HITLData,
          trajectory: [...s.trajectory, {
            agent: data.agent || "", action: `hitl:${data.type}`,
            status: "active", timestamp: Date.now(),
          }],
        }));
        setConversations(prev => prev.map(c => {
          if (c.id !== conversationId) return c;
          const addSeg: Message[] = capturedSegsHITL.length > 0 ? [{
            role: "assistant",
            content: "",
            thinkingContent: JSON.stringify(capturedSegsHITL),
          }] : [];
          return {
            ...c,
            messages: [
              ...c.messages,
              ...addSeg,
              {
                role: "hitl",
                content: data.message || "Agent needs your input",
                hitl: data as HITLData,
              },
            ],
            updatedAt: Date.now(),
          };
        }));
        thinkingSegmentsRefs.current[teamId] = [];
        break;
      }

      case "response": {
        const capturedSegs = segs.filter(s => s.text.trim());
        setConversations(prev => prev.map(c => {
          if (c.id !== conversationId) return c;
          return {
            ...c,
            messages: [
              ...c.messages,
              {
                role: "assistant",
                content: data.content,
                agent: data.agent_used,
                toolCalls: data.tool_calls,
                trace: data.trace,
                thinkingContent: capturedSegs.length > 0
                  ? JSON.stringify(capturedSegs) : undefined,
              },
            ],
            updatedAt: Date.now(),
          };
        }));
        updateSession(teamId, s => ({
          ...s,
          activeTrace: data.trace || null,
          thinkingSegments: [],
        }));
        try {
          if (data.trace) localStorage.setItem(LAST_TRACE_KEY, JSON.stringify(data.trace));
        } catch { /* ignore */ }
        thinkingSegmentsRefs.current[teamId] = [];
        break;
      }

      case "error":
        setConversations(prev => prev.map(c => {
          if (c.id !== conversationId) return c;
          return {
            ...c,
            messages: [...c.messages, {
              role: "assistant", content: `Error: ${data.message}`,
            }],
            updatedAt: Date.now(),
          };
        }));
        updateSession(teamId, s => ({ ...s, thinkingSegments: [], error: data.message }));
        thinkingSegmentsRefs.current[teamId] = [];
        break;

      case "done":
        updateSession(teamId, s => ({
          ...s,
          queryActive: false,
          inputLocked: false,
          statusText: "Working...",
          elapsedFinal: Date.now(),
        }));
        break;

      case "resumed":
        updateSession(teamId, s => ({
          ...s,
          inputLocked: true,
          statusText: "Resuming...",
          trajectory: s.trajectory.map(t =>
            t.action.startsWith("hitl:") && t.status === "active"
              ? { ...t, status: "completed" as const } : t),
        }));
        break;
    }
  }, [updateSession]);

  // ── Public: sendMessage ────────────────────────────────────────────────────
  const sendMessage = useCallback(async (
    teamId: string,
    conversationId: string,
    message: string,
    opts?: { threadId?: string | null; model?: string },
  ) => {
    // Abort any existing stream for this team (new turn supersedes the old).
    abortRefs.current[teamId]?.abort();

    const ctrl = new AbortController();
    abortRefs.current[teamId] = ctrl;

    // Reset live state + append user message to the conversation.
    thinkingSegmentsRefs.current[teamId] = [];
    agentRefs.current[teamId] = null;
    updateSession(teamId, _ => ({
      ...emptySession(teamId),
      teamId,
      conversationId,
      threadId: opts?.threadId ?? null,
      queryActive: true,
      inputLocked: true,
      liveStartTime: Date.now(),
    }));

    setConversations(prev => prev.map(c =>
      c.id === conversationId
        ? {
            ...c,
            messages: [...c.messages, { role: "user", content: message }],
            updatedAt: Date.now(),
          }
        : c
    ));

    try {
      await api.teams.chatStream(
        teamId, message,
        (evt) => handleEvent(teamId, conversationId, evt),
        opts?.threadId || undefined,
        ctrl.signal,
        opts?.model,
      );
    } catch (e: any) {
      if (e.name !== "AbortError") {
        setConversations(prev => prev.map(c =>
          c.id === conversationId
            ? {
                ...c,
                messages: [...c.messages, {
                  role: "assistant", content: `Error: ${e.message}`,
                }],
                updatedAt: Date.now(),
              }
            : c
        ));
      }
      updateSession(teamId, s => ({
        ...s, queryActive: false, inputLocked: false,
      }));
    }
  }, [handleEvent, updateSession]);

  // ── Public: resumeHITL ─────────────────────────────────────────────────────
  const resumeHITL = useCallback(async (
    teamId: string,
    threadId: string,
    response: Record<string, any>,
  ) => {
    const session = sessions[teamId];
    const conversationId = session?.conversationId;
    if (!conversationId) return;

    abortRefs.current[teamId]?.abort();
    const ctrl = new AbortController();
    abortRefs.current[teamId] = ctrl;

    updateSession(teamId, s => ({
      ...s,
      pendingHITL: null,
      inputLocked: true,
      statusText: "Resuming after your input...",
      thinkingSegments: [],
    }));
    thinkingSegmentsRefs.current[teamId] = [];

    try {
      await api.teams.chatResume(
        teamId, threadId, response,
        (evt) => handleEvent(teamId, conversationId, evt),
        ctrl.signal,
      );
    } catch (e: any) {
      if (e.name !== "AbortError") {
        setConversations(prev => prev.map(c =>
          c.id === conversationId
            ? {
                ...c,
                messages: [...c.messages, {
                  role: "assistant", content: `Error: ${e.message}`,
                }],
                updatedAt: Date.now(),
              }
            : c
        ));
      }
      updateSession(teamId, s => ({
        ...s, queryActive: false, inputLocked: false,
      }));
    }
  }, [sessions, handleEvent, updateSession]);

  // ── Public: stopSession ────────────────────────────────────────────────────
  const stopSession = useCallback((teamId: string) => {
    abortRefs.current[teamId]?.abort();
    delete abortRefs.current[teamId];
    thinkingSegmentsRefs.current[teamId] = [];
    updateSession(teamId, s => ({
      ...s,
      queryActive: false,
      inputLocked: false,
      statusText: "Working...",
      thinkingSegments: [],
    }));
  }, [updateSession]);

  // ── Derived: which teams are live RIGHT NOW (C4) ───────────────────────────
  const runningTeamIds = useMemo(
    () => Object.values(sessions).filter(s => s.queryActive).map(s => s.teamId),
    [sessions],
  );

  const isTeamRunning = useCallback(
    (teamId: string) => sessions[teamId]?.queryActive === true,
    [sessions],
  );

  const value: ChatSessionContextValue = {
    conversations,
    createConversation,
    deleteConversation,
    setConversationMessages,
    renameConversationIfUntitled,
    getSession,
    setThinkingCollapsed,
    clearLiveStateForTeam,
    sendMessage,
    resumeHITL,
    stopSession,
    runningTeamIds,
    isTeamRunning,
  };

  return (
    <ChatSessionContext.Provider value={value}>
      {children}
    </ChatSessionContext.Provider>
  );
}
