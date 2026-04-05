"use client";
import { useEffect, useState, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, SSEEvent } from "@/lib/api";

// ── Types ────────────────────────────────────────────────────────

interface ToolCall { tool: string; args: Record<string, any>; agent?: string }
interface SpanInfo {
  id: string; name: string; span_type: string;
  tokens_in: number; tokens_out: number; cost: number;
  model: string; status: string; error?: string | null;
  agent?: string;
}
interface TraceInfo { trace_id: string; total_latency_ms: number; total_tokens: number; total_cost: number; spans: SpanInfo[] }

interface Message {
  role: "user" | "assistant" | "hitl";
  content: string;
  agent?: string;
  toolCalls?: ToolCall[];
  trace?: TraceInfo;
  hitl?: HITLData;
  thinkingContent?: string;
}

interface HITLData {
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

interface PlanStep { step: number; description: string; status: string }

interface TrajectoryStep {
  agent: string;
  action: string;
  status: "active" | "completed" | "error";
  timestamp: number;
}

interface LiveSpan extends SpanInfo { agent?: string }

interface Conversation {
  id: string;
  title: string;
  teamId: string;
  threadId: string | null;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

// ── Constants & helpers ──────────────────────────────────────────

const CONVOS_KEY = "sdlc_conversations";
const LAST_TRACE_KEY = "sdlc_last_trace";
const CHAT_MODEL_KEY = "sdlc_chat_model";
const TYPE_DOT: Record<string, string> = {
  routing: "bg-blue-400", llm_call: "bg-violet-400", tool_call: "bg-emerald-400",
  agent_execution: "bg-sky-400", hitl: "bg-orange-400", supervisor: "bg-amber-400",
};

function genId() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 7); }

function loadConversations(): Conversation[] {
  if (typeof window === "undefined") return [];
  try { return JSON.parse(localStorage.getItem(CONVOS_KEY) || "[]"); }
  catch { return []; }
}

function saveConversations(convos: Conversation[]) {
  try { localStorage.setItem(CONVOS_KEY, JSON.stringify(convos.slice(0, 50))); }
  catch { /* quota */ }
}

function titleFromMessages(msgs: Message[]): string {
  const first = msgs.find(m => m.role === "user");
  if (!first) return "New Chat";
  const t = first.content.slice(0, 40);
  return t.length < first.content.length ? t + "..." : t;
}

/**
 * Human-readable label for a span row.
 * - llm_call  → model shortname (e.g. "claude-sonnet-4-6")
 * - tool_call → tool name (suffix after first ":")
 * - others    → name suffix after first ":" or bare name
 */
function spanDisplayLabel(s: SpanInfo): { primary: string; badge?: string } {
  if (s.span_type === "llm_call") {
    const model = s.model ? (s.model.split("/").pop() || s.model) : (s.name.split(":").slice(1).join(":") || "llm");
    return { primary: model, badge: "llm" };
  }
  if (s.span_type === "tool_call") {
    const tool = s.name.split(":").slice(1).join(":") || s.name;
    return { primary: tool, badge: "tool" };
  }
  const label = s.name.split(":").slice(1).join(":") || s.name;
  return { primary: label };
}

/** Model shortname helper. */
function shortModel(model: string): string {
  return model ? (model.split("/").pop() || model) : "";
}

/**
 * Group completed trace spans by agent for sequential display.
 *
 * The backend emits spans in end-order:
 *   routing → llm_call:model → tool:name → agent:agent_name
 *
 * The "agent_execution" span always ends last and wraps all its children.
 * We scan the flat list: collect pending llm/tool spans until we hit an
 * agent_execution span, then bundle them as that agent's group.
 * Routing/supervisor spans are placed in a "system" group first.
 */
interface AgentGroup { agent: string; agentSpan?: SpanInfo; spans: SpanInfo[] }

function groupSpansByAgent(spans: SpanInfo[]): AgentGroup[] {
  const systemSpans: SpanInfo[] = [];
  const groups: AgentGroup[] = [];
  let pending: SpanInfo[] = [];

  for (const s of spans) {
    if (s.span_type === "routing" || s.span_type === "supervisor") {
      systemSpans.push(s);
    } else if (s.span_type === "agent_execution") {
      // Suffix of "agent:{name}" is the real agent name
      const agentName = s.name.includes(":") ? s.name.split(":").slice(1).join(":") : s.name;
      groups.push({ agent: agentName, agentSpan: s, spans: [...pending] });
      pending = [];
    } else {
      pending.push(s);
    }
  }

  // Orphaned spans (no wrapping agent_execution) — show under their type
  if (pending.length > 0) {
    groups.push({ agent: "agent", spans: pending });
  }

  if (systemSpans.length > 0) {
    groups.unshift({ agent: "system", spans: systemSpans });
  }

  return groups;
}

// ── Markdown ─────────────────────────────────────────────────────

const mdComponents = {
  code({ className, children, ...props }: any) {
    const isInline = !className;
    return isInline ? (
      <code className="px-1 py-0.5 rounded bg-[var(--bg)] text-[var(--accent)] text-[12px] font-mono" {...props}>{children}</code>
    ) : (
      <pre className="bg-[var(--bg)] rounded-lg p-3 my-2 overflow-x-auto border border-[var(--border)]">
        <code className="text-[12px] font-mono leading-relaxed" {...props}>{children}</code>
      </pre>
    );
  },
  p({ children }: any) { return <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>; },
  ul({ children }: any) { return <ul className="list-disc pl-5 mb-2 space-y-0.5">{children}</ul>; },
  ol({ children }: any) { return <ol className="list-decimal pl-5 mb-2 space-y-0.5">{children}</ol>; },
  li({ children }: any) { return <li className="leading-relaxed">{children}</li>; },
  h1({ children }: any) { return <h1 className="text-base font-semibold mt-3 mb-1">{children}</h1>; },
  h2({ children }: any) { return <h2 className="text-[15px] font-semibold mt-2.5 mb-1">{children}</h2>; },
  h3({ children }: any) { return <h3 className="text-[14px] font-semibold mt-2 mb-0.5">{children}</h3>; },
  blockquote({ children }: any) {
    return <blockquote className="border-l-2 border-[var(--accent)] pl-3 my-2 text-[var(--text-secondary)] italic">{children}</blockquote>;
  },
  table({ children }: any) {
    return <div className="overflow-x-auto my-2"><table className="text-[12px] border-collapse w-full">{children}</table></div>;
  },
  th({ children }: any) { return <th className="border border-[var(--border)] px-2 py-1 bg-[var(--bg)] font-medium text-left">{children}</th>; },
  td({ children }: any) { return <td className="border border-[var(--border)] px-2 py-1">{children}</td>; },
  strong({ children }: any) { return <strong className="font-semibold">{children}</strong>; },
  a({ href, children }: any) {
    return <a href={href} target="_blank" rel="noopener noreferrer" className="text-[var(--accent)] underline hover:opacity-80">{children}</a>;
  },
};

function Md({ content }: { content: string }) {
  return <div className="text-[13px]"><ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>{content}</ReactMarkdown></div>;
}

// ── Main Component ───────────────────────────────────────────────

export default function ChatPage() {
  const [teams, setTeams] = useState<any[]>([]);
  const [teamId, setTeamId] = useState("default");
  const [hydrated, setHydrated] = useState(false);
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [defaultModelName, setDefaultModelName] = useState("from .env");
  const [chatModel, setChatModel] = useState("");

  // Conversation list
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvoId, setActiveConvoId] = useState<string | null>(null);

  // Current chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);

  // Which assistant message is selected (its trace shows in inspector)
  const [selectedMsgIndex, setSelectedMsgIndex] = useState<number | null>(null);

  const [queryActive, setQueryActive] = useState(false);
  const [inputLocked, setInputLocked] = useState(false);
  const [statusText, setStatusText] = useState("Working...");

  // Trace state
  const [liveSpans, setLiveSpans] = useState<LiveSpan[]>([]);
  const [trajectory, setTrajectory] = useState<TrajectoryStep[]>([]);
  const [activeAgent, setActiveAgent] = useState<string | null>(null);
  const [activeTool, setActiveTool] = useState<string | null>(null);
  const [liveTokens, setLiveTokens] = useState(0);
  const [liveCost, setLiveCost] = useState(0);
  const [liveStartTime, setLiveStartTime] = useState(0);
  const [elapsedFinal, setElapsedFinal] = useState(0);
  const [thinkingText, setThinkingText] = useState("");
  const [thinkingCollapsed, setThinkingCollapsed] = useState(false);
  const [pendingHITL, setPendingHITL] = useState<HITLData | null>(null);
  const [activeTrace, setActiveTrace] = useState<TraceInfo | null>(null);

  // Ref mirrors for stale-closure-safe reads inside the SSE callback
  const agentRef = useRef<string | null>(null);
  const thinkingTextRef = useRef("");

  const endRef = useRef<HTMLDivElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const userNearBottom = useRef(true);

  // ── Hydration ──────────────────────────────────────────────
  useEffect(() => {
    api.teams.list().then(setTeams);
    Promise.all([api.models.list(), api.config.llm()]).then(([models, llmCfg]) => {
      setAvailableModels(models);
      setDefaultModelName(llmCfg.default_model_name || llmCfg.default_model || "from .env");
    });
  }, []);

  useEffect(() => {
    const convos = loadConversations();
    setConversations(convos);
    if (convos.length > 0) {
      const latest = convos[0];
      setActiveConvoId(latest.id);
      setMessages(latest.messages);
      setThreadId(latest.threadId);
      setTeamId(latest.teamId);
    }
    try {
      const saved = localStorage.getItem(LAST_TRACE_KEY);
      if (saved) setActiveTrace(JSON.parse(saved));
    } catch { /* ignore */ }
    try {
      const savedModel = localStorage.getItem(CHAT_MODEL_KEY);
      if (savedModel !== null) setChatModel(savedModel);
    } catch { /* ignore */ }
    setHydrated(true);
  }, []);

  // Persist last trace
  useEffect(() => {
    if (!hydrated) return;
    try {
      if (activeTrace) localStorage.setItem(LAST_TRACE_KEY, JSON.stringify(activeTrace));
    } catch { /* quota */ }
  }, [activeTrace, hydrated]);

  // Persist model preference
  useEffect(() => {
    if (!hydrated) return;
    try { localStorage.setItem(CHAT_MODEL_KEY, chatModel); } catch { /* quota */ }
  }, [chatModel, hydrated]);

  // Persist conversations when messages change
  useEffect(() => {
    if (!hydrated || !activeConvoId) return;
    setConversations(prev => {
      const updated = prev.map(c =>
        c.id === activeConvoId
          ? { ...c, messages, threadId, title: titleFromMessages(messages), updatedAt: Date.now() }
          : c
      );
      saveConversations(updated);
      return updated;
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, threadId, hydrated, activeConvoId]);

  useEffect(() => {
    if (userNearBottom.current) endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleChatScroll() {
    const el = chatScrollRef.current;
    if (!el) return;
    userNearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  }

  // ── Reset helpers ──────────────────────────────────────────
  const resetLiveState = useCallback(() => {
    setLiveSpans([]); setTrajectory([]); setActiveAgent(null); setActiveTool(null);
    setLiveTokens(0); setLiveCost(0);
    setThinkingText(""); thinkingTextRef.current = "";
    setThinkingCollapsed(false);
    setPendingHITL(null); setElapsedFinal(0); setActiveTrace(null);
    setSelectedMsgIndex(null);
    agentRef.current = null;
  }, []);

  // ── SSE handler ────────────────────────────────────────────
  // Uses refs for values that need to be current inside the stable callback
  const handleSSEEvent = useCallback((event: SSEEvent) => {
    const { type, data } = event;
    switch (type) {
      case "thread_id":
        setThreadId(data.thread_id);
        break;

      case "agent_start":
        agentRef.current = data.agent;
        setActiveAgent(data.agent);
        setStatusText(`${data.agent} is working...`);
        setTrajectory(prev => [...prev, {
          agent: data.agent, action: "thinking",
          status: "active", timestamp: Date.now(),
        }]);
        break;

      case "agent_end":
        setTrajectory(prev => prev.map(t =>
          t.agent === data.agent && t.status === "active"
            ? { ...t, status: "completed" as const } : t
        ));
        setActiveAgent(null);
        agentRef.current = null;
        break;

      case "tool_start":
        setActiveTool(data.tool);
        setStatusText(`${data.agent || "Agent"}: ${data.tool}...`);
        setTrajectory(prev => [...prev, {
          agent: data.agent || agentRef.current || "", action: `tool:${data.tool}`,
          status: "active", timestamp: Date.now(),
        }]);
        break;

      case "tool_end":
        setActiveTool(null);
        setTrajectory(prev => prev.map(t =>
          t.action === `tool:${data.tool}` && t.status === "active"
            ? { ...t, status: "completed" as const } : t
        ));
        break;

      case "llm_token":
        thinkingTextRef.current += data.token;
        setThinkingText(thinkingTextRef.current);
        break;

      case "trace_span": {
        const spanEvent = data.event;
        const spanData: LiveSpan = { ...data.span, agent: data.span?.agent || agentRef.current || undefined };
        if (spanEvent === "span_start") {
          setLiveSpans(prev => [...prev.filter(s => s.id !== spanData.id), spanData]);
        } else {
          setLiveSpans(prev => prev.map(s => s.id === spanData.id ? { ...spanData, agent: s.agent || spanData.agent } : s));
          if (spanData.tokens_in || spanData.tokens_out)
            setLiveTokens(prev => prev + (spanData.tokens_in || 0) + (spanData.tokens_out || 0));
          if (spanData.cost)
            setLiveCost(prev => prev + spanData.cost);
        }
        break;
      }

      case "hitl_request": {
        // Read from ref to avoid stale closure
        const capturedThinkingHITL = thinkingTextRef.current;
        setInputLocked(false);
        setStatusText("Waiting for your input...");
        setPendingHITL(data as HITLData);
        setTrajectory(prev => [...prev, {
          agent: data.agent || "", action: `hitl:${data.type}`,
          status: "active", timestamp: Date.now(),
        }]);
        setMessages(prev => [
          ...prev,
          ...(capturedThinkingHITL ? [{
            role: "assistant" as const,
            content: "",
            thinkingContent: capturedThinkingHITL,
          }] : []),
          {
            role: "hitl" as const, content: data.message || "Agent needs your input",
            hitl: data as HITLData,
          },
        ]);
        thinkingTextRef.current = "";
        setThinkingText("");
        break;
      }

      case "response": {
        // Read from ref to avoid stale closure
        const capturedThinking = thinkingTextRef.current;
        setMessages(prev => {
          const next = [...prev, {
            role: "assistant" as const,
            content: data.content,
            agent: data.agent_used,
            toolCalls: data.tool_calls,
            trace: data.trace,
            thinkingContent: capturedThinking || undefined,
          }];
          // Auto-select this new message so its trace shows immediately
          setSelectedMsgIndex(next.length - 1);
          return next;
        });
        setActiveTrace(data.trace || null);
        thinkingTextRef.current = "";
        setThinkingText("");
        break;
      }

      case "error":
        setMessages(prev => [...prev, { role: "assistant", content: `Error: ${data.message}` }]);
        thinkingTextRef.current = "";
        setThinkingText("");
        break;

      case "done":
        setQueryActive(false); setInputLocked(false);
        setStatusText("Working..."); setElapsedFinal(Date.now());
        break;

      case "resumed":
        setInputLocked(true); setStatusText("Resuming...");
        setTrajectory(prev => prev.map(t =>
          t.action.startsWith("hitl:") && t.status === "active"
            ? { ...t, status: "completed" as const } : t
        ));
        break;
    }
  // Stable callback — all mutable values read through refs
  }, []);

  // ── Conversation management ────────────────────────────────
  function createConversation() {
    stopProcessing();
    const c: Conversation = { id: genId(), title: "New Chat", teamId, threadId: null, messages: [], createdAt: Date.now(), updatedAt: Date.now() };
    setConversations(prev => { const next = [c, ...prev]; saveConversations(next); return next; });
    switchToConversation(c);
  }

  function switchToConversation(c: Conversation) {
    stopProcessing();
    resetLiveState();
    setActiveConvoId(c.id);
    setMessages(c.messages);
    setThreadId(c.threadId);
    setTeamId(c.teamId);
    setQueryActive(false); setInputLocked(false);
    setSelectedMsgIndex(null);
    // Restore the last assistant message's trace for this conversation
    const lastAssistant = [...c.messages].reverse().find(m => m.role === "assistant" && m.trace);
    if (lastAssistant?.trace) setActiveTrace(lastAssistant.trace);
  }

  function deleteConversation(id: string) {
    setConversations(prev => {
      const next = prev.filter(c => c.id !== id);
      saveConversations(next);
      if (id === activeConvoId) {
        if (next.length > 0) { switchToConversation(next[0]); }
        else { setActiveConvoId(null); setMessages([]); setThreadId(null); resetLiveState(); }
      }
      return next;
    });
  }

  function clearCurrentChat() {
    setMessages([]); setThreadId(null); resetLiveState();
  }

  // ── Send / resume / stop ───────────────────────────────────
  function stopProcessing() {
    abortRef.current?.abort();
    setQueryActive(false); setInputLocked(false);
    thinkingTextRef.current = "";
    setThinkingText(""); setStatusText("Working...");
  }

  async function send() {
    if (!input.trim() || inputLocked || queryActive) return;
    const msg = input.trim();
    setInput("");

    if (!activeConvoId) {
      const c: Conversation = { id: genId(), title: msg.slice(0, 40), teamId, threadId: null, messages: [], createdAt: Date.now(), updatedAt: Date.now() };
      setConversations(prev => { const next = [c, ...prev]; saveConversations(next); return next; });
      setActiveConvoId(c.id);
    }

    setMessages(prev => [...prev, { role: "user", content: msg }]);
    setQueryActive(true); setInputLocked(true);
    resetLiveState(); setLiveStartTime(Date.now());
    userNearBottom.current = true;

    abortRef.current = new AbortController();
    try {
      await api.teams.chatStream(teamId, msg, handleSSEEvent, threadId || undefined, abortRef.current.signal, chatModel || undefined);
    } catch (e: any) {
      if (e.name !== "AbortError")
        setMessages(prev => [...prev, { role: "assistant", content: `Error: ${e.message}` }]);
      setQueryActive(false); setInputLocked(false);
    }
  }

  async function resumeHITL(response: Record<string, any>) {
    if (!threadId) return;
    setPendingHITL(null); setInputLocked(true);
    setStatusText("Resuming after your input..."); thinkingTextRef.current = ""; setThinkingText("");
    userNearBottom.current = true;
    abortRef.current = new AbortController();
    try {
      await api.teams.chatResume(teamId, threadId, response, handleSSEEvent, abortRef.current.signal);
    } catch (e: any) {
      if (e.name !== "AbortError")
        setMessages(prev => [...prev, { role: "assistant", content: `Error: ${e.message}` }]);
      setQueryActive(false); setInputLocked(false);
    }
  }

  // ── Trace panel selection ──────────────────────────────────
  function selectMessage(i: number, m: Message) {
    if (m.role !== "assistant" || !m.trace) return;
    setSelectedMsgIndex(i);
    setActiveTrace(m.trace);
    // Clear live overlay so the historical trace renders
    setLiveSpans([]); setTrajectory([]);
  }

  const hasLiveData = liveSpans.length > 0 || trajectory.length > 0;
  const showTrace = queryActive || hasLiveData;
  const elapsedMs = liveStartTime
    ? (queryActive ? Date.now() - liveStartTime : (elapsedFinal || Date.now()) - liveStartTime)
    : 0;

  return (
    <div className="flex h-[calc(100vh-4rem)]">
      {/* ══════ Sidebar ══════ */}
        <div className="w-56 flex-shrink-0 border-r border-[var(--border)] flex flex-col bg-white">
        <div className="p-3 border-b border-[var(--border)]">
          <button onClick={createConversation}
            className="w-full btn-primary !text-[12px] flex items-center justify-center gap-1.5">
            <span className="text-sm">+</span> New Chat
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {conversations.map(c => (
            <div key={c.id}
              className={`group flex items-center gap-1.5 px-3 py-2 cursor-pointer border-b border-[var(--border)] transition-colors ${
                c.id === activeConvoId
                  ? "bg-zinc-900 border-l-2 border-l-zinc-900"
                  : "hover:bg-[var(--bg-hover)] border-l-2 border-l-transparent"
              }`}
              onClick={() => c.id !== activeConvoId && switchToConversation(c)}
            >
              <div className="flex-1 min-w-0">
                <div className={`text-[12px] font-medium truncate ${c.id === activeConvoId ? "text-white" : ""}`}>{c.title}</div>
                <div className={`text-[10px] ${c.id === activeConvoId ? "text-zinc-400" : "text-[var(--text-muted)]"}`}>
                  {c.messages.length} msgs &middot; {new Date(c.updatedAt).toLocaleDateString()}
                </div>
              </div>
              <button
                onClick={e => { e.stopPropagation(); deleteConversation(c.id); }}
                className="opacity-0 group-hover:opacity-100 p-0.5 text-[var(--text-muted)] hover:text-red-500 transition-all flex-shrink-0"
                title="Delete"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14" />
                </svg>
              </button>
            </div>
          ))}
          {conversations.length === 0 && hydrated && (
            <div className="text-[11px] text-[var(--text-muted)] text-center py-6 px-3">
              No conversations yet
            </div>
          )}
        </div>
      </div>

      {/* ══════ Chat Panel ══════ */}
      <div className="flex flex-col flex-[3] min-w-0">
        <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--border)]">
          <div className="flex items-center gap-2 min-w-0">
            <h1 className="text-base font-semibold truncate">
              {activeConvoId ? (conversations.find(c => c.id === activeConvoId)?.title || "Chat") : "Chat"}
            </h1>
            {threadId && (
              <span className="text-[10px] text-[var(--text-muted)] font-mono flex-shrink-0">
                {threadId}
              </span>
            )}
          </div>
          <div className="flex items-center gap-1.5 flex-shrink-0">
            {messages.length > 0 && (
              <button onClick={clearCurrentChat} className="btn-ghost !text-[11px]"
                title="Clear messages in this chat">Clear</button>
            )}
            <select
              value={chatModel}
              onChange={e => setChatModel(e.target.value)}
              className="input !w-auto !text-[12px]"
              title="Override model for all agents in this chat"
            >
              <option value="">Default ({defaultModelName})</option>
              {(() => {
                const byProvider: Record<string, any[]> = {};
                for (const m of availableModels) {
                  const g = m.provider || "Other";
                  (byProvider[g] = byProvider[g] || []).push(m);
                }
                return Object.entries(byProvider).map(([provider, models]) => (
                  <optgroup key={provider} label={provider}>
                    {models.map((m: any) => (
                      <option key={m.id} value={m.id}>{m.name}</option>
                    ))}
                  </optgroup>
                ));
              })()}
            </select>
            <select value={teamId} onChange={e => setTeamId(e.target.value)} className="input !w-auto !text-[12px]">
              {teams.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
            </select>
          </div>
        </div>

        <div ref={chatScrollRef} onScroll={handleChatScroll}
          className="flex-1 overflow-y-auto space-y-2.5 p-4 pb-2">
          {messages.length === 0 && (
            <div className="text-center text-[var(--text-muted)] py-16">
              <p className="text-base">Ask the agent to do something</p>
              <p className="text-sm mt-1.5 opacity-70">Try: &quot;Read main.py&quot; or &quot;Run the tests&quot;</p>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i}>
              {m.role === "hitl" && m.hitl ? (
                <HITLWidget hitl={m.hitl} onSubmit={resumeHITL}
                  disabled={inputLocked || pendingHITL?.thread_id !== m.hitl.thread_id} />
              ) : (
                <div className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                  {/* Assistant messages are clickable to load their trace */}
                  <div
                    onClick={() => selectMessage(i, m)}
                    className={`max-w-[85%] rounded-xl px-4 py-2.5 transition-all ${
                      m.role === "user"
                        ? "bg-zinc-900 text-white"
                        : `card !p-3 ${m.trace ? "cursor-pointer" : ""} ${
                            selectedMsgIndex === i
                              ? "ring-2 ring-zinc-300 ring-offset-1"
                              : m.trace ? "hover:ring-1 hover:ring-zinc-200" : ""
                          }`
                    }`}
                  >
                    {m.agent && (
                      <div className="flex flex-wrap items-center gap-1 mb-1.5">
                        {m.agent.split(" > ").map((a: string, idx: number) => (
                          <span key={idx} className="flex items-center gap-1">
                            {idx > 0 && <span className="text-[10px] text-[var(--text-muted)]">→</span>}
                            <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-zinc-100 text-zinc-600 border border-zinc-200">{a}</span>
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Thinking content — expanded by default so users can read it */}
                    {m.thinkingContent && <ThinkingHistory content={m.thinkingContent} />}

                    {m.role === "user"
                      ? <div className="text-[13px] whitespace-pre-wrap leading-relaxed">{m.content}</div>
                      : m.content ? <Md content={m.content} /> : null
                    }

                    {/* Trace stats — shown below response, clicking also selects the message */}
                    {m.trace && (
                      <div className="flex gap-3 mt-1.5 text-[10px] text-[var(--text-muted)]">
                        <span>{m.trace.total_latency_ms.toFixed(0)}ms</span>
                        <span>{m.trace.total_tokens} tokens</span>
                        <span>${m.trace.total_cost.toFixed(4)}</span>
                        <span className="text-zinc-500">{m.trace.spans.length} spans ↗</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Live thinking box — stays visible during streaming */}
          {inputLocked && thinkingText && (
            <div className="flex justify-start">
              <div className="max-w-[85%] w-full">
                <button onClick={() => setThinkingCollapsed(c => !c)}
                  className="flex items-center gap-1.5 mb-1 text-[11px] text-[var(--text-muted)] hover:text-[var(--text)] transition-colors">
                  <span className={`transition-transform text-[9px] ${thinkingCollapsed ? "" : "rotate-90"}`}>&#9654;</span>
                  <span className="h-2 w-2 rounded-full bg-zinc-400 animate-pulse" />
                  {activeAgent ? `${activeAgent} is thinking...` : "Thinking..."}
                </button>
                {!thinkingCollapsed && (
                  <div className="bg-[#f4f4f5] dark:bg-[#27272a] rounded-xl px-4 py-3 border border-[#e4e4e7] dark:border-[#3f3f46]">
                    <div className="text-[12px] text-[#71717a] dark:text-[#a1a1aa] leading-relaxed whitespace-pre-wrap font-mono max-h-[200px] overflow-y-auto">
                      {thinkingText.slice(-2000)}<span className="animate-pulse">|</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Processing dots */}
          {inputLocked && !thinkingText && (
            <div className="flex justify-start">
              <div className="card !p-3 flex items-center gap-2.5">
                <div className="flex gap-1">
                  <div className="h-2 w-2 rounded-full bg-zinc-400 animate-bounce [animation-delay:0ms]" />
                  <div className="h-2 w-2 rounded-full bg-zinc-400 animate-bounce [animation-delay:150ms]" />
                  <div className="h-2 w-2 rounded-full bg-zinc-400 animate-bounce [animation-delay:300ms]" />
                </div>
                <span className="text-[13px] text-[var(--text-muted)]">{statusText}</span>
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        {/* Input bar */}
        <div className="flex gap-2 px-4 py-2.5 border-t border-[var(--border)]">
          <input value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && !e.shiftKey && send()}
            disabled={inputLocked}
            placeholder={queryActive && !inputLocked ? "Respond to the agent above..." : "Ask the agent..."}
            className="input flex-1" />
          {queryActive ? (
            <button onClick={stopProcessing}
              className="px-4 py-2 rounded-lg bg-red-600 text-white text-[13px] font-medium hover:bg-red-700 transition-colors flex items-center gap-1.5">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>
              Stop
            </button>
          ) : (
            <button onClick={send} disabled={inputLocked || !input.trim()} className="btn-primary">Send</button>
          )}
        </div>
      </div>

      {/* ══════ Trace Inspector ══════ */}
      <div className="w-72 flex-shrink-0 border-l border-[var(--border)] pl-4 pr-3 py-3 overflow-y-auto">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-medium text-[var(--text-muted)]">Trace Inspector</h2>
          {selectedMsgIndex !== null && !hasLiveData && (
            <span className="text-[10px] text-[var(--accent)] font-medium">msg #{selectedMsgIndex + 1}</span>
          )}
        </div>

        {/* Trajectory pills */}
        {trajectory.length > 0 && (
          <div className="mb-4">
            <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1.5">
              {queryActive ? "Live Trajectory" : "Trajectory"}
            </div>
            <div className="flex flex-wrap gap-1">
              {trajectory.map((t, i) => (
                <div key={i} className="flex items-center gap-0.5">
                  {i > 0 && <span className="text-[var(--text-muted)] text-[9px]">&rarr;</span>}
                  <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-medium leading-tight ${
                    t.status === "active"
                      ? t.action.startsWith("hitl:")
                        ? "bg-orange-50 text-orange-600 ring-1 ring-orange-300"
                        : "bg-zinc-900 text-white ring-1 ring-zinc-700"
                      : t.status === "error"
                      ? "bg-red-50 text-red-600"
                      : "bg-zinc-100 text-zinc-500"
                  }`}>
                    {t.status === "active" && (
                      <span className={`h-1.5 w-1.5 rounded-full animate-pulse ${
                        t.action.startsWith("hitl:") ? "bg-orange-500" : "bg-white"
                      }`} />
                    )}
                    {t.status === "completed" && <span className="text-green-500 text-[9px]">&#10003;</span>}
                    {t.action.startsWith("tool:")
                      ? <><span className="opacity-60">{t.agent}/</span>{t.action.replace("tool:", "")}</>
                      : t.action.startsWith("hitl:")
                      ? `${t.agent || "wait"}: ${t.action.replace("hitl:", "")}`
                      : t.agent || t.action}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Summary cards */}
        {showTrace && (
          <div className="grid grid-cols-3 gap-1.5 text-center mb-3">
            <div className="p-1.5 rounded-lg bg-[var(--bg)]">
              <div className="text-base font-semibold">{elapsedMs > 0 ? Math.round(elapsedMs) : "..."}<span className="text-[10px] text-[var(--text-muted)]">ms</span></div>
              <div className="text-[9px] text-[var(--text-muted)]">{queryActive ? "Elapsed" : "Latency"}</div>
            </div>
            <div className="p-1.5 rounded-lg bg-[var(--bg)]">
              <div className="text-base font-semibold">{liveTokens}</div>
              <div className="text-[9px] text-[var(--text-muted)]">Tokens</div>
            </div>
            <div className="p-1.5 rounded-lg bg-[var(--bg)]">
              <div className="text-base font-semibold">${liveCost.toFixed(4)}</div>
              <div className="text-[9px] text-[var(--text-muted)]">Cost</div>
            </div>
          </div>
        )}

        {/* HITL pause indicator */}
        {queryActive && pendingHITL && !inputLocked && (
          <div className="flex items-center gap-2 p-2 rounded-lg bg-orange-50 dark:bg-orange-900/15 border border-orange-200 dark:border-orange-800/30 text-[11px] text-orange-700 dark:text-orange-400 mb-3">
            <span className="h-2 w-2 rounded-full bg-orange-500 animate-pulse flex-shrink-0" />
            Paused — waiting for your response
          </div>
        )}

        {/* Live spans grouped by agent */}
        {liveSpans.length > 0 && (
          <SpanTree spans={liveSpans} activeAgent={queryActive ? activeAgent : null} activeTool={queryActive ? activeTool : null} isLive={queryActive} />
        )}

        {/* Historical trace — clicked message or last response */}
        {!hasLiveData && activeTrace && (
          <CompletedTrace trace={activeTrace} />
        )}

        {!showTrace && !activeTrace && (
          <div className="text-[12px] text-[var(--text-muted)] text-center py-10">
            Send a message — or click any response to inspect its trace
          </div>
        )}
      </div>
    </div>
  );
}


// ── Thinking History ─────────────────────────────────────────────
// Starts expanded so users can immediately read the model's reasoning.

function ThinkingHistory({ content }: { content: string }) {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <div className="mb-2">
      <button
        onClick={e => { e.stopPropagation(); setCollapsed(c => !c); }}
        className="flex items-center gap-1.5 text-[11px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors mb-1"
      >
        <span className={`transition-transform text-[9px] ${collapsed ? "" : "rotate-90"}`}>&#9654;</span>
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--text-muted)]" />
        {collapsed ? "Show thinking" : "Hide thinking"}
      </button>
      {!collapsed && (
        <div className="bg-[#f4f4f5] dark:bg-[#27272a] rounded-xl px-4 py-3 border border-[#e4e4e7] dark:border-[#3f3f46] mb-2">
          <div className="text-[12px] text-[#71717a] dark:text-[#a1a1aa] leading-relaxed whitespace-pre-wrap font-mono max-h-[240px] overflow-y-auto">
            {content}
          </div>
        </div>
      )}
    </div>
  );
}


// ── Span Detail Panel ────────────────────────────────────────────

function SpanDetail({ span }: { span: SpanInfo }) {
  return (
    <div className="mt-1 mb-1 ml-4 p-2 rounded-lg bg-[var(--bg-card)] border border-[var(--border)] text-[10px] space-y-1">
      <div className="grid grid-cols-2 gap-x-2 gap-y-0.5">
        <span className="text-[var(--text-muted)]">Type</span>
        <span className="font-mono">{span.span_type}</span>
        {span.model && <>
          <span className="text-[var(--text-muted)]">Model</span>
          <span className="font-mono truncate" title={span.model}>{span.model}</span>
        </>}
        {(span.tokens_in > 0 || span.tokens_out > 0) && <>
          <span className="text-[var(--text-muted)]">Tokens in</span>
          <span>{span.tokens_in}</span>
          <span className="text-[var(--text-muted)]">Tokens out</span>
          <span>{span.tokens_out}</span>
        </>}
        {span.cost > 0 && <>
          <span className="text-[var(--text-muted)]">Cost</span>
          <span>${span.cost.toFixed(6)}</span>
        </>}
        <span className="text-[var(--text-muted)]">Status</span>
        <span className={span.status === "completed" ? "text-green-500" : span.status === "running" ? "text-[var(--accent)]" : "text-red-500"}>
          {span.status}
        </span>
      </div>
      {span.error && (
        <div className="mt-1 pt-1 border-t border-[var(--border)]">
          <span className="text-red-500 font-medium">Error: </span>
          <span className="text-red-400 whitespace-pre-wrap break-words">{span.error}</span>
        </div>
      )}
    </div>
  );
}


// ── Span Tree (live) ─────────────────────────────────────────────

function SpanTree({ spans, activeAgent, activeTool, isLive }: {
  spans: LiveSpan[]; activeAgent: string | null; activeTool: string | null; isLive: boolean;
}) {
  // Group by s.agent (set from agentRef in SSE handler), preserving arrival order
  const grouped = new Map<string, LiveSpan[]>();
  for (const s of spans) {
    // Use the agent field directly (set from agentRef during streaming)
    // Fall back to name prefix only for system spans
    let agent: string;
    if (s.agent) {
      agent = s.agent;
    } else if (s.span_type === "routing" || s.span_type === "supervisor") {
      agent = "system";
    } else if (s.span_type === "agent_execution") {
      // Name is "agent:{name}" — extract the real agent name
      agent = s.name.includes(":") ? s.name.split(":").slice(1).join(":") : s.name;
    } else {
      agent = "system";
    }
    const group = grouped.get(agent) || [];
    group.push(s);
    grouped.set(agent, group);
  }

  // Extract model from agent_execution span (preferred) or first llm_call span
  const agentModels = new Map<string, string>();
  for (const [agent, agentSpans] of grouped) {
    const execSpan = agentSpans.find(s => s.span_type === "agent_execution" && s.model);
    const llmSpan = agentSpans.find(s => s.span_type === "llm_call" && s.model);
    const m = execSpan?.model || llmSpan?.model || "";
    if (m) agentModels.set(agent, m);
  }

  return (
    <div className="space-y-1.5">
      <div className="text-[11px] text-[var(--text-muted)] uppercase tracking-wide">
        {isLive ? "Live Spans" : "Spans"}
      </div>
      {Array.from(grouped.entries()).map(([agent, agentSpans]) => (
        <AgentSpanGroup key={agent} agent={agent} model={agentModels.get(agent)}
          // Exclude the agent_execution span from rows — it IS the group header
          spans={agentSpans.filter(s => s.span_type !== "agent_execution")}
          isActive={!!activeAgent && agent.toLowerCase().includes(activeAgent.toLowerCase())}
          activeTool={activeTool} isLive={isLive} />
      ))}
    </div>
  );
}

function AgentSpanGroup({ agent, model, spans, isActive, activeTool, isLive }: {
  agent: string; model?: string; spans: LiveSpan[];
  isActive: boolean; activeTool: string | null; isLive: boolean;
}) {
  const [expanded, setExpanded] = useState(true);
  const [expandedSpanId, setExpandedSpanId] = useState<string | null>(null);
  const completedCount = spans.filter(s => s.status === "completed").length;
  const totalTokens = spans.reduce((a, s) => a + (s.tokens_in || 0) + (s.tokens_out || 0), 0);
  const totalCost = spans.reduce((a, s) => a + (s.cost || 0), 0);

  return (
    <div className="border border-[var(--border)] rounded-lg overflow-hidden">
      <button onClick={() => setExpanded(e => !e)}
        className={`flex items-center gap-2 px-2 py-1.5 w-full text-xs font-medium text-left transition-colors ${
          isActive ? "bg-zinc-900 text-white" : "bg-[var(--bg)] text-[var(--text-secondary)] hover:bg-[var(--bg-card)]"
        }`}>
        <span className={`text-[9px] transition-transform ${expanded ? "rotate-90" : ""}`}>&#9654;</span>
        {isActive && isLive && <span className="h-2 w-2 rounded-full bg-white animate-pulse" />}
        <span className="flex-1 min-w-0 truncate">
          {agent}
          {model && <span className={`text-[9px] ml-1 font-normal ${isActive ? "text-zinc-300" : "text-[var(--text-muted)]"}`}>· {shortModel(model)}</span>}
        </span>
        <span className="text-[10px] text-[var(--text-muted)] font-normal whitespace-nowrap flex-shrink-0">
          {completedCount}/{spans.length}
          {totalTokens > 0 && <>&middot;{totalTokens}t</>}
          {totalCost > 0 && <>&middot;${totalCost.toFixed(4)}</>}
        </span>
      </button>
      {expanded && (
        <div className="divide-y divide-[var(--border)]">
          {spans.map(s => {
            const isToolActive = isLive && activeTool && s.name.includes(activeTool);
            const { primary, badge } = spanDisplayLabel(s);
            const isOpen = expandedSpanId === s.id;
            return (
              <div key={s.id}>
                <button
                  onClick={() => setExpandedSpanId(isOpen ? null : s.id)}
                  className={`flex items-center gap-1.5 px-2 py-1 text-[11px] w-full text-left transition-colors ${
                    isOpen ? "bg-zinc-100" : isToolActive ? "bg-zinc-50" : "hover:bg-[var(--bg-card)]"
                  }`}
                >
                  <div className={`h-1.5 w-1.5 rounded-full flex-shrink-0 ${TYPE_DOT[s.span_type] || "bg-gray-400"}`} />
                  <div className="flex-1 min-w-0 truncate text-left" title={s.name}>{primary}</div>
                  <div className="flex items-center gap-1 text-[10px] flex-shrink-0">
                    {badge === "llm" && (
                      <span className="px-1 py-0.5 rounded bg-zinc-100 text-zinc-500 text-[9px] font-medium">llm</span>
                    )}
                    {badge === "tool" && (
                      <span className="px-1 py-0.5 rounded bg-emerald-50 text-emerald-600 text-[9px] font-medium">tool</span>
                    )}
                    {(s.tokens_in + s.tokens_out) > 0 && (
                      <span className="text-[var(--text-muted)]">{s.tokens_in + s.tokens_out}t</span>
                    )}
                  </div>
                  {s.status === "running" ? (
                    <div className="h-2.5 w-2.5 border border-zinc-200 border-t-zinc-600 rounded-full animate-spin flex-shrink-0" />
                  ) : (
                    <div className={`h-1.5 w-1.5 rounded-full flex-shrink-0 ${s.status === "completed" ? "bg-green-500" : "bg-red-500"}`} />
                  )}
                </button>
                {isOpen && <SpanDetail span={s} />}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}


// ── Completed Trace ──────────────────────────────────────────────
// Groups spans by agent using end-order heuristic (see groupSpansByAgent).

function CompletedTrace({ trace }: { trace: TraceInfo }) {
  const groups = groupSpansByAgent(trace.spans);

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-1.5 text-center">
        <div className="p-1.5 rounded-lg bg-[var(--bg)]">
          <div className="text-base font-semibold">{trace.total_latency_ms.toFixed(0)}<span className="text-[10px] text-[var(--text-muted)]">ms</span></div>
          <div className="text-[9px] text-[var(--text-muted)]">Latency</div>
        </div>
        <div className="p-1.5 rounded-lg bg-[var(--bg)]">
          <div className="text-base font-semibold">{trace.total_tokens}</div>
          <div className="text-[9px] text-[var(--text-muted)]">Tokens</div>
        </div>
        <div className="p-1.5 rounded-lg bg-[var(--bg)]">
          <div className="text-base font-semibold">${trace.total_cost.toFixed(4)}</div>
          <div className="text-[9px] text-[var(--text-muted)]">Cost</div>
        </div>
      </div>

      <div className="space-y-1.5">
        <div className="text-[11px] text-[var(--text-muted)] uppercase tracking-wide">Spans</div>
        {groups.map((g, idx) => (
          <CompletedAgentGroup key={`${g.agent}-${idx}`}
            agent={g.agent} agentSpan={g.agentSpan} spans={g.spans} />
        ))}
      </div>
    </div>
  );
}

function CompletedAgentGroup({ agent, agentSpan, spans }: {
  agent: string; agentSpan?: SpanInfo; spans: SpanInfo[];
}) {
  const [expanded, setExpanded] = useState(true);
  const [expandedSpanId, setExpandedSpanId] = useState<string | null>(null);

  // Model: prefer agent_execution span's model, then first llm_call span's model
  const model = agentSpan?.model
    || spans.find(s => s.span_type === "llm_call" && s.model)?.model
    || "";

  const totalTokens = spans.reduce((a, s) => a + (s.tokens_in || 0) + (s.tokens_out || 0), 0);
  const totalCost = spans.reduce((a, s) => a + (s.cost || 0), 0);
  const errCount = spans.filter(s => s.status !== "completed").length;

  return (
    <div className="border border-[var(--border)] rounded-lg overflow-hidden">
      <button onClick={() => setExpanded(e => !e)}
        className="flex items-center gap-2 px-2 py-1.5 w-full text-xs font-medium text-left bg-[var(--bg)] text-[var(--text-secondary)] hover:bg-[var(--bg-card)] transition-colors">
        <span className={`text-[9px] transition-transform ${expanded ? "rotate-90" : ""}`}>&#9654;</span>
        <span className="flex-1 min-w-0 truncate">
          {agent}
          {model && <span className="text-[9px] text-[var(--text-muted)] ml-1 font-normal">· {shortModel(model)}</span>}
        </span>
        <span className="text-[10px] text-[var(--text-muted)] font-normal whitespace-nowrap flex-shrink-0">
          {spans.length > 0 && <>{spans.length} spans</>}
          {totalTokens > 0 && <>&middot;{totalTokens}t</>}
          {totalCost > 0 && <>&middot;${totalCost.toFixed(4)}</>}
          {errCount > 0 && <span className="text-red-400 ml-1">·{errCount}err</span>}
        </span>
      </button>
      {expanded && (
        <div className="divide-y divide-[var(--border)]">
          {spans.map(s => {
            const { primary, badge } = spanDisplayLabel(s);
            const isOpen = expandedSpanId === s.id;
            return (
              <div key={s.id}>
                <button
                  onClick={() => setExpandedSpanId(isOpen ? null : s.id)}
                  className={`flex items-center gap-1.5 p-1.5 w-full text-left text-[11px] transition-colors ${
                    isOpen ? "bg-zinc-100" : "hover:bg-[var(--bg-card)]"
                  }`}
                >
                  <div className={`h-1.5 w-1.5 rounded-full flex-shrink-0 ${TYPE_DOT[s.span_type] || "bg-gray-400"}`} />
                  <div className="flex-1 min-w-0 truncate font-medium" title={s.name}>{primary}</div>
                  <div className="flex items-center gap-1 text-[10px] flex-shrink-0">
                    {badge === "llm" && (
                      <span className="px-1 py-0.5 rounded bg-zinc-100 text-zinc-500 text-[9px] font-medium">llm</span>
                    )}
                    {badge === "tool" && (
                      <span className="px-1 py-0.5 rounded bg-emerald-50 text-emerald-600 text-[9px] font-medium">tool</span>
                    )}
                    {s.tokens_in + s.tokens_out > 0 && (
                      <span className="text-[var(--text-muted)]">{s.tokens_in + s.tokens_out}t</span>
                    )}
                    {s.cost > 0 && (
                      <span className="text-[var(--text-muted)]">${s.cost.toFixed(4)}</span>
                    )}
                  </div>
                  <div className={`h-1.5 w-1.5 rounded-full flex-shrink-0 ${s.status === "completed" ? "bg-green-500" : "bg-red-500"}`} />
                </button>
                {isOpen && <SpanDetail span={s} />}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}


// ── HITL Widgets ─────────────────────────────────────────────────

function HITLWidget({ hitl, onSubmit, disabled }: {
  hitl: HITLData; onSubmit: (response: Record<string, any>) => void; disabled: boolean;
}) {
  switch (hitl.type) {
    case "clarification": return <ClarificationWidget hitl={hitl} onSubmit={onSubmit} disabled={disabled} />;
    case "plan_review": return <PlanReviewWidget hitl={hitl} onSubmit={onSubmit} disabled={disabled} />;
    case "action_confirmation": return <ActionConfirmWidget hitl={hitl} onSubmit={onSubmit} disabled={disabled} />;
    case "tool_review": return <ToolReviewWidget hitl={hitl} onSubmit={onSubmit} disabled={disabled} />;
    default:
      return (
        <div className="card !p-3 border-l-4 border-l-orange-500 mx-4">
          <div className="text-[11px] text-orange-600 font-medium mb-1">Unknown: {hitl.type}</div>
          <div className="text-[13px]">{hitl.message}</div>
        </div>
      );
  }
}

function ClarificationWidget({ hitl, onSubmit, disabled }: {
  hitl: HITLData; onSubmit: (r: Record<string, any>) => void; disabled: boolean;
}) {
  const [answer, setAnswer] = useState("");
  const [submitted, setSubmitted] = useState(false);
  function submit(v: string) { if (!v.trim() || submitted) return; setSubmitted(true); onSubmit({ type: "clarification", answer: v.trim() }); }

  return (
    <div className="card !p-4 border-l-4 border-l-blue-500 mx-4 my-1">
      <div className="flex items-center gap-2 mb-2">
        <span className="h-5 w-5 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center text-blue-600 text-[11px] font-bold">?</span>
        <span className="text-[11px] text-blue-600 font-medium">{hitl.agent || "Agent"} needs your input</span>
      </div>
      <p className="text-[13px] mb-3">{hitl.question || hitl.message}</p>
      {hitl.options && hitl.options.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {hitl.options.map((opt, i) => (
            <button key={i} onClick={() => submit(opt)} disabled={disabled || submitted}
              className="px-3 py-1.5 rounded-lg text-[12px] border border-[var(--border)] bg-[var(--bg)] hover:bg-zinc-900 hover:text-white hover:border-zinc-900 transition-all disabled:opacity-50">{opt}</button>
          ))}
        </div>
      )}
      <div className="flex gap-2">
        <input value={answer} onChange={e => setAnswer(e.target.value)} onKeyDown={e => e.key === "Enter" && submit(answer)}
          disabled={disabled || submitted} placeholder="Type your answer..." className="input flex-1 !text-[12px]" />
        <button onClick={() => submit(answer)} disabled={disabled || submitted || !answer.trim()} className="btn-primary !text-[12px] !px-3">
          {submitted ? "Sent" : "Submit"}
        </button>
      </div>
    </div>
  );
}

function PlanReviewWidget({ hitl, onSubmit, disabled }: {
  hitl: HITLData; onSubmit: (r: Record<string, any>) => void; disabled: boolean;
}) {
  const [steps, setSteps] = useState<PlanStep[]>(hitl.plan || []);
  const [feedback, setFeedback] = useState("");
  const [showFeedback, setShowFeedback] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  return (
    <div className="card !p-4 border-l-4 border-l-purple-500 mx-4 my-1">
      <div className="flex items-center gap-2 mb-2">
        <span className="h-5 w-5 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center text-purple-600 text-[11px] font-bold">P</span>
        <span className="text-[11px] text-purple-600 font-medium">Plan Review</span>
        {hitl.agent && <span className="text-[10px] text-[var(--text-muted)]">from {hitl.agent}</span>}
      </div>
      <p className="text-[12px] text-[var(--text-muted)] mb-3">Review and edit the plan before the agent proceeds.</p>
      <div className="space-y-1.5 mb-3">
        {steps.map((s, i) => (
          <div key={i} className="flex items-center gap-2 group">
            <span className="text-[11px] text-[var(--text-muted)] w-5 text-right flex-shrink-0">{s.step}.</span>
            <input value={s.description}
              onChange={e => setSteps(prev => prev.map((x, j) => j === i ? { ...x, description: e.target.value } : x))}
              disabled={disabled || submitted} className="input flex-1 !text-[12px] !py-1" />
            <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
              <button onClick={() => { const a=[...steps]; if(i>0){[a[i],a[i-1]]=[a[i-1],a[i]]; setSteps(a.map((x,j)=>({...x,step:j+1})));} }}
                disabled={i === 0 || disabled || submitted} className="p-0.5 text-[10px] text-[var(--text-muted)] hover:text-[var(--accent)] disabled:opacity-30">&uarr;</button>
              <button onClick={() => { const a=[...steps]; if(i<a.length-1){[a[i],a[i+1]]=[a[i+1],a[i]]; setSteps(a.map((x,j)=>({...x,step:j+1})));} }}
                disabled={i === steps.length - 1 || disabled || submitted} className="p-0.5 text-[10px] text-[var(--text-muted)] hover:text-[var(--accent)] disabled:opacity-30">&darr;</button>
              <button onClick={() => setSteps(prev => prev.filter((_,j) => j !== i).map((x,j) => ({...x, step:j+1})))}
                disabled={disabled || submitted} className="p-0.5 text-[10px] text-red-400 hover:text-red-600 disabled:opacity-30">&times;</button>
            </div>
          </div>
        ))}
      </div>
      <button onClick={() => setSteps(prev => [...prev, { step: prev.length + 1, description: "", status: "pending" }])}
        disabled={disabled || submitted} className="text-[11px] text-[var(--accent)] hover:underline mb-3 disabled:opacity-50">+ Add step</button>
      {showFeedback && (
        <div className="mb-3"><textarea value={feedback} onChange={e => setFeedback(e.target.value)}
          disabled={disabled || submitted} placeholder="What changes would you like?" rows={2} className="input w-full !text-[12px]" /></div>
      )}
      <div className="flex gap-2">
        <button onClick={() => { setSubmitted(true); onSubmit({ type: "plan_review", approved: true, edited_plan: steps }); }}
          disabled={disabled || submitted} className="btn-primary !text-[12px] !px-4">{submitted ? "Approved" : "Approve Plan"}</button>
        <button onClick={() => { if(!feedback.trim()){setShowFeedback(true);return;} setSubmitted(true); onSubmit({ type: "plan_review", approved: false, feedback: feedback.trim() }); }}
          disabled={disabled || submitted} className="btn-ghost !text-[12px]">Request Changes</button>
      </div>
    </div>
  );
}

function ActionConfirmWidget({ hitl, onSubmit, disabled }: {
  hitl: HITLData; onSubmit: (r: Record<string, any>) => void; disabled: boolean;
}) {
  const [submitted, setSubmitted] = useState(false);
  const isHigh = hitl.risk_level === "high";
  function respond(approved: boolean) { setSubmitted(true); onSubmit({ type: "action_confirmation", approved }); }

  return (
    <div className={`card !p-4 border-l-4 mx-4 my-1 ${isHigh ? "border-l-red-500" : "border-l-amber-500"}`}>
      <div className="flex items-center gap-2 mb-2">
        <span className={`h-5 w-5 rounded-full flex items-center justify-center text-[11px] font-bold ${
          isHigh ? "bg-red-100 dark:bg-red-900/30 text-red-600" : "bg-amber-100 dark:bg-amber-900/30 text-amber-600"
        }`}>!</span>
        <span className={`text-[11px] font-medium ${isHigh ? "text-red-600" : "text-amber-600"}`}>Action Confirmation</span>
        {hitl.agent && <span className="text-[10px] text-[var(--text-muted)]">from {hitl.agent}</span>}
      </div>
      <p className="text-[12px] text-[var(--text-muted)] mb-2">{hitl.reason || hitl.message}</p>
      <div className="bg-[var(--bg)] rounded-lg p-2.5 mb-3 border border-[var(--border)]">
        <div className="text-[11px] text-[var(--text-muted)] mb-1">Tool: <span className="font-mono font-medium text-[var(--text)]">{hitl.tool_name}</span></div>
        {hitl.args && Object.keys(hitl.args).length > 0 && (
          <pre className="text-[11px] text-[var(--text-secondary)] overflow-x-auto whitespace-pre-wrap mt-1">{JSON.stringify(hitl.args, null, 2)}</pre>
        )}
      </div>
      <div className="flex gap-2">
        <button onClick={() => respond(true)} disabled={disabled || submitted}
          className={`!text-[12px] !px-4 py-1.5 rounded-lg font-medium transition-colors disabled:opacity-50 ${
            isHigh ? "bg-red-600 text-white hover:bg-red-700" : "bg-amber-500 text-white hover:bg-amber-600"
          }`}>{submitted ? "Allowed" : "Allow"}</button>
        <button onClick={() => respond(false)} disabled={disabled || submitted} className="btn-ghost !text-[12px]">{submitted ? "Denied" : "Deny"}</button>
      </div>
    </div>
  );
}

function ToolReviewWidget({ hitl, onSubmit, disabled }: {
  hitl: HITLData; onSubmit: (r: Record<string, any>) => void; disabled: boolean;
}) {
  const [editMode, setEditMode] = useState(false);
  const [modifiedOutput, setModifiedOutput] = useState(hitl.output || "");
  const [submitted, setSubmitted] = useState(false);
  function respond(action: "continue" | "modify" | "stop") {
    setSubmitted(true);
    const p: Record<string, any> = { type: "tool_review", action };
    if (action === "modify") p.modified_output = modifiedOutput;
    onSubmit(p);
  }

  return (
    <div className="card !p-4 border-l-4 border-l-teal-500 mx-4 my-1">
      <div className="flex items-center gap-2 mb-2">
        <span className="h-5 w-5 rounded-full bg-teal-100 dark:bg-teal-900/30 flex items-center justify-center text-teal-600 text-[11px] font-bold">R</span>
        <span className="text-[11px] text-teal-600 font-medium">Review Tool Output</span>
        {hitl.agent && <span className="text-[10px] text-[var(--text-muted)]">from {hitl.agent}</span>}
      </div>
      <div className="text-[11px] text-[var(--text-muted)] mb-2">Tool: <span className="font-mono font-medium text-[var(--text)]">{hitl.tool_name}</span></div>
      {editMode ? (
        <textarea value={modifiedOutput} onChange={e => setModifiedOutput(e.target.value)}
          disabled={disabled || submitted} rows={6} className="input w-full !text-[11px] font-mono mb-3" />
      ) : (
        <pre className="bg-[var(--bg)] rounded-lg p-2.5 mb-3 border border-[var(--border)] text-[11px] text-[var(--text-secondary)] overflow-auto max-h-48 whitespace-pre-wrap">
          {hitl.output || "(no output)"}
        </pre>
      )}
      <div className="flex gap-2 flex-wrap">
        <button onClick={() => respond("continue")} disabled={disabled || submitted} className="btn-primary !text-[12px] !px-3">{submitted ? "Continued" : "Continue"}</button>
        {!editMode
          ? <button onClick={() => setEditMode(true)} disabled={disabled || submitted} className="btn-ghost !text-[12px]">Edit Output</button>
          : <button onClick={() => respond("modify")} disabled={disabled || submitted} className="btn-ghost !text-[12px] !text-teal-600">Save & Continue</button>
        }
        <button onClick={() => respond("stop")} disabled={disabled || submitted} className="btn-ghost !text-[12px] !text-red-500">Stop</button>
      </div>
    </div>
  );
}
