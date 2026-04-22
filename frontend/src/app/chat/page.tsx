"use client";
import { useEffect, useState, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "@/lib/api";
import { useTeam } from "@/contexts/TeamContext";
import {
  useChatSession,
  type Message,
  type ToolCall,
  type SpanInfo,
  type TraceInfo,
  type ThinkingSegment,
  type HITLData,
  type PlanStep,
  type TrajectoryStep,
  type LiveSpan,
  type Conversation,
} from "@/contexts/ChatSessionContext";

// ── Constants & helpers ──────────────────────────────────────────

const CHAT_MODEL_KEY = "sdlc_chat_model";
const TYPE_DOT: Record<string, string> = {
  routing: "bg-blue-400", llm_call: "bg-violet-400", tool_call: "bg-emerald-400",
  agent_execution: "bg-sky-400", hitl: "bg-orange-400", supervisor: "bg-amber-400",
};

// Conversation persistence + id generation live in ChatSessionContext now.

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
  // Global team selection shared across pages. We keep a local `teamIdState`
  // representing the CURRENT conversation's team (which can differ from the
  // globally selected team while the user is viewing an older conversation),
  // and sync it to globalTeamId on new chats.
  const { teamId: globalTeamId, setTeamId: setGlobalTeamId, teams } = useTeam();
  const [teamIdState, setTeamIdState] = useState<string>(globalTeamId || "default");
  const teamId = teamIdState;
  const setTeamId = (id: string) => { setTeamIdState(id); setGlobalTeamId(id); };
  const [hydrated, setHydrated] = useState(false);
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [defaultModelName, setDefaultModelName] = useState("from .env");
  const [chatModel, setChatModel] = useState("");

  // Persistent chat state + live-stream state (C1-C3): everything that used to
  // be `useState` inside this component now lives in <ChatSessionProvider>,
  // keyed by teamId.  Navigating away and back no longer resets the chat, and
  // switching teams lets both streams run in parallel.
  const {
    conversations,
    createConversation: providerCreateConversation,
    deleteConversation: providerDeleteConversation,
    getSession,
    sendMessage: providerSendMessage,
    resumeHITL: providerResumeHITL,
    stopSession: providerStopSession,
    clearLiveStateForTeam,
    runningTeamIds,
  } = useChatSession();

  const [activeConvoId, setActiveConvoId] = useState<string | null>(null);
  const [input, setInput] = useState("");

  // Which assistant message is selected (its trace shows in inspector)
  const [selectedMsgIndex, setSelectedMsgIndex] = useState<number | null>(null);

  // UI-only preference (not streaming state — stays local).
  const [thinkingCollapsed, setThinkingCollapsed] = useState(false);

  // When the user clicks a past assistant message in the chat log, show that
  // message's trace in the right panel instead of the live stream's trace.
  // This overrides `session.activeTrace` until they click elsewhere.
  const [historicalTrace, setHistoricalTrace] = useState<TraceInfo | null>(null);

  // Derive everything stream-related from the provider so that the chat keeps
  // updating in the background when the user is on another page.
  const session = getSession(teamId);
  const queryActive = session.queryActive;
  const inputLocked = session.inputLocked;
  const statusText = session.statusText;
  const resolvedStrategy = session.resolvedStrategy;
  const liveSpans = session.liveSpans;
  const trajectory = session.trajectory;
  const activeAgent = session.activeAgent;
  const activeTool = session.activeTool;
  const liveTokens = session.liveTokens;
  const liveCost = session.liveCost;
  const liveStartTime = session.liveStartTime;
  const elapsedFinal = session.elapsedFinal;
  const thinkingSegments = session.thinkingSegments;
  const pendingHITL = session.pendingHITL;
  const activeTrace = session.activeTrace;

  // Current conversation's messages + threadId are derived from the provider's
  // conversation list (the provider appends incoming `response` events while we
  // may be unmounted, so deriving here means "always up to date").
  const activeConversation = conversations.find(c => c.id === activeConvoId) || null;
  const messages: Message[] = activeConversation?.messages || [];
  const threadId: string | null = activeConversation?.threadId ?? session.threadId;

  const endRef = useRef<HTMLDivElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const userNearBottom = useRef(true);

  // ── Hydration ──────────────────────────────────────────────
  useEffect(() => {
    Promise.all([api.models.list(), api.config.llm()]).then(([models, llmCfg]) => {
      setAvailableModels(models);
      setDefaultModelName(llmCfg.default_model_name || llmCfg.default_model || "from .env");
    });
  }, []);

  // Wait until the provider has finished loading conversations from
  // localStorage, then pick an initial active conversation.  We can't read
  // conversations immediately on first render — they start empty until the
  // provider's hydration effect fires.
  useEffect(() => {
    if (hydrated) return;
    if (conversations.length > 0) {
      // Try to honour the globally-selected team: prefer the most recent
      // conversation on that team; fall back to the absolute latest.
      const onTeam = globalTeamId ? conversations.find(c => c.teamId === globalTeamId) : undefined;
      const latest = onTeam || conversations[0];
      setActiveConvoId(latest.id);
      setTeamIdState(latest.teamId);
    } else if (globalTeamId) {
      // No conversations yet — start new chats against the currently-selected team.
      setTeamIdState(globalTeamId);
    }
    try {
      const savedModel = localStorage.getItem(CHAT_MODEL_KEY);
      if (savedModel !== null) setChatModel(savedModel);
    } catch { /* ignore */ }
    setHydrated(true);
  }, []);

  // `activeTrace` persistence to localStorage now happens inside the
  // ChatSessionProvider on every `response` event — no effect needed here.

  // Persist model preference
  useEffect(() => {
    if (!hydrated) return;
    try { localStorage.setItem(CHAT_MODEL_KEY, chatModel); } catch { /* quota */ }
  }, [chatModel, hydrated]);

  // Persistence of conversations + messages lives inside ChatSessionProvider
  // now (it auto-saves on every mutation), so there is no local "persist
  // messages" effect here anymore.

  useEffect(() => {
    if (userNearBottom.current) endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleChatScroll() {
    const el = chatScrollRef.current;
    if (!el) return;
    userNearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  }

  // Dropped: local `resetLiveState` / `handleSSEEvent`.
  // Both live in ChatSessionProvider so they survive page navigation.

  // ── SSE handler ────────────────────────────────────────────
  // Uses refs for values that need to be current inside the stable callback
  // ── Conversation management ────────────────────────────────
  // All mutations go through the provider so conversations + live state stay
  // in sync across tabs/pages.
  function createConversation() {
    providerStopSession(teamId);
    clearLiveStateForTeam(teamId);
    const c = providerCreateConversation(teamId);
    switchToConversation(c);
  }

  function switchToConversation(c: Conversation) {
    // Stopping an in-flight stream on the OLD team?  No — we deliberately
    // don't: the user wants background streams to keep running (C1 promise).
    // We just switch the "viewed" conversation + team here.
    setActiveConvoId(c.id);
    setTeamId(c.teamId);
    setSelectedMsgIndex(null);
  }

  function deleteConversation(id: string) {
    providerDeleteConversation(id);
    if (id === activeConvoId) {
      // Pick a new active convo from what's left, or clear selection.
      const remaining = conversations.filter(c => c.id !== id);
      if (remaining.length > 0) switchToConversation(remaining[0]);
      else {
        setActiveConvoId(null);
        clearLiveStateForTeam(teamId);
      }
    }
  }

  function clearCurrentChat() {
    if (activeConvoId) {
      // Replace messages with empty via the provider API (no direct setState).
      // We don't expose a helper for this — the simplest equivalent is to
      // stop the session + create a fresh conversation.
      providerStopSession(teamId);
      clearLiveStateForTeam(teamId);
      const c = providerCreateConversation(teamId);
      switchToConversation(c);
    }
  }

  // ── Send / resume / stop ───────────────────────────────────
  // All three just delegate to the provider.  The provider owns the
  // AbortController and the SSE loop, so navigating away doesn't abort.
  function stopProcessing() {
    providerStopSession(teamId);
  }

  async function send() {
    if (!input.trim() || inputLocked || queryActive) return;
    const msg = input.trim();
    setInput("");

    let convoId = activeConvoId;
    if (!convoId) {
      const c = providerCreateConversation(teamId, msg.slice(0, 40));
      convoId = c.id;
      setActiveConvoId(convoId);
    }
    userNearBottom.current = true;
    await providerSendMessage(teamId, convoId, msg, {
      threadId: threadId || null,
      model: chatModel || undefined,
    });
  }

  async function resumeHITL(response: Record<string, any>) {
    if (!threadId) return;
    userNearBottom.current = true;
    await providerResumeHITL(teamId, threadId, response);
  }

  // ── Trace panel selection ──────────────────────────────────
  // Selecting a historical message's trace is purely local UI — the provider
  // keeps its own `activeTrace` for the current stream, so we only update the
  // selected-index here.  The render path prefers historical trace when no
  // live data is present.
  function selectMessage(i: number, m: Message) {
    if (m.role !== "assistant" || !m.trace) return;
    setSelectedMsgIndex(i);
    setHistoricalTrace(m.trace || null);
  }

  // When user switches to a new conversation, drop the historical override.
  useEffect(() => {
    setHistoricalTrace(null);
    setSelectedMsgIndex(null);
  }, [activeConvoId]);

  // When the user clicks a historical message to inspect its trace, we
  // suppress the live overlay so the clicked trace is clearly rendered.
  const hasLiveData = historicalTrace
    ? false
    : (liveSpans.length > 0 || trajectory.length > 0);
  const showTrace = queryActive || hasLiveData;
  // The trace the right panel should display: historical override wins over
  // provider's running `activeTrace`.
  const displayTrace = historicalTrace ?? activeTrace;
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
          {(() => {
            // Only show conversations that belong to the currently-selected
            // team so dev-team and sdlc_2_0 histories stay visually isolated.
            const visible = conversations.filter(c => !teamId || c.teamId === teamId);
            return (
              <>
                {visible.map(c => (
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
                {visible.length === 0 && hydrated && (
                  <div className="text-[11px] text-[var(--text-muted)] text-center py-6 px-3">
                    No conversations yet for this team
                  </div>
                )}
              </>
            );
          })()}
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
            {/* C4: when another team has a live stream (not the currently-viewed
                one), show a tiny live dot next to the switcher so the user knows
                "something is running in the background". */}
            {runningTeamIds.some(id => id !== teamId) && (
              <span
                className="relative inline-flex h-2 w-2"
                title={`Live chat on: ${runningTeamIds.filter(id => id !== teamId).join(", ")}`}
              >
                <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-60 animate-ping" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
              </span>
            )}
            <select value={teamId} onChange={e => setTeamId(e.target.value)} className="input !w-auto !text-[12px]">
              {teams.map(t => {
                const live = runningTeamIds.includes(t.id);
                return (
                  <option key={t.id} value={t.id}>
                    {live ? "● " : ""}{t.name}
                  </option>
                );
              })}
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

          <div ref={endRef} />
        </div>

        {/* ── Thinking overlay — sits BETWEEN messages and input, never displaces messages ── */}
        {inputLocked && thinkingSegments.length > 0 && (
          <div className="border-t border-[var(--border)] bg-[#f9f9fa]">
            <div className="px-4 pt-2 pb-1">
              <button onClick={() => setThinkingCollapsed(c => !c)}
                className="flex items-center gap-1.5 text-[11px] text-[var(--text-muted)] hover:text-[var(--text)] transition-colors w-full">
                <span className={`transition-transform text-[9px] shrink-0 ${thinkingCollapsed ? "" : "rotate-90"}`}>&#9654;</span>
                <span className="h-2 w-2 rounded-full bg-zinc-400 animate-pulse shrink-0" />
                <span className="font-medium">{activeAgent ? `${activeAgent} is thinking…` : "Thinking…"}</span>
                <span className="ml-1 text-[9px] text-zinc-400">
                  {thinkingSegments.reduce((n, s) => n + s.text.length, 0).toLocaleString()} chars
                  {thinkingSegments.length > 1 && ` · ${thinkingSegments.length} agents`}
                </span>
                <span className="ml-auto text-[9px] text-zinc-400">{thinkingCollapsed ? "click to expand" : "click to collapse"}</span>
              </button>
            </div>
            {!thinkingCollapsed && (
              <div className="px-4 pb-2">
                <LiveThinkingBox segments={thinkingSegments} />
              </div>
            )}
          </div>
        )}

        {/* Processing dots (when no thinking segments yet) */}
        {inputLocked && thinkingSegments.length === 0 && (
          <div className="border-t border-[var(--border)] px-4 py-2.5 bg-[#f9f9fa]">
            <div className="flex items-center gap-2.5">
              <div className="flex gap-1">
                <div className="h-2 w-2 rounded-full bg-zinc-400 animate-bounce [animation-delay:0ms]" />
                <div className="h-2 w-2 rounded-full bg-zinc-400 animate-bounce [animation-delay:150ms]" />
                <div className="h-2 w-2 rounded-full bg-zinc-400 animate-bounce [animation-delay:300ms]" />
              </div>
              <span className="text-[13px] text-[var(--text-muted)]">{statusText}</span>
            </div>
          </div>
        )}

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

        {/* Auto-resolved strategy badge */}
        {resolvedStrategy && (
          <div className="flex items-center gap-1.5 mb-3 px-2 py-1.5 rounded-lg bg-blue-50 border border-blue-200 text-[11px] text-blue-700">
            <span className="h-1.5 w-1.5 rounded-full bg-blue-500 flex-shrink-0" />
            <span><span className="font-medium">Auto →</span> {resolvedStrategy}</span>
          </div>
        )}

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
        {!hasLiveData && displayTrace && (
          <CompletedTrace trace={displayTrace} />
        )}

        {!showTrace && !displayTrace && (
          <div className="text-[12px] text-[var(--text-muted)] text-center py-10">
            Send a message — or click any response to inspect its trace
          </div>
        )}
      </div>
    </div>
  );
}


// ── Agent badge helper ────────────────────────────────────────────
const AGENT_COLORS: Record<string, string> = {
  planner:         "bg-violet-100 text-violet-700 border-violet-200",
  coder:           "bg-blue-100 text-blue-700 border-blue-200",
  tester:          "bg-emerald-100 text-emerald-700 border-emerald-200",
  devops:          "bg-orange-100 text-orange-700 border-orange-200",
  project_manager: "bg-amber-100 text-amber-700 border-amber-200",
  researcher:      "bg-teal-100 text-teal-700 border-teal-200",
  reviewer:        "bg-pink-100 text-pink-700 border-pink-200",
  data_analyst:    "bg-cyan-100 text-cyan-700 border-cyan-200",
  supervisor:      "bg-zinc-200 text-zinc-700 border-zinc-300",
};
function agentBadge(agent: string) {
  return AGENT_COLORS[agent] ?? "bg-zinc-100 text-zinc-600 border-zinc-200";
}


// ── Thinking segments renderer ────────────────────────────────────
// Shared by both live box and history — renders one block per agent.
function ThinkingSegmentsView({ segments, isLive }: { segments: ThinkingSegment[]; isLive?: boolean }) {
  return (
    <div className="space-y-2">
      {segments.map((seg, i) => (
        <div key={i}>
          {/* Agent label */}
          <div className="flex items-center gap-1.5 mb-1">
            <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${agentBadge(seg.agent)}`}>
              {seg.agent}
            </span>
            {isLive && i === segments.length - 1 && (
              <span className="text-[9px] text-zinc-400 animate-pulse">streaming…</span>
            )}
          </div>
          {/* Content */}
          <div className="thinking-md text-[12px] text-[#52525b] leading-relaxed pl-1 border-l-2 border-zinc-200">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{seg.text || " "}</ReactMarkdown>
            {isLive && i === segments.length - 1 && (
              <span className="inline-block h-3 w-0.5 bg-zinc-400 animate-pulse ml-0.5 align-middle" />
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

/** Parse thinkingContent which may be JSON (new) or a plain string (legacy). */
function parseThinkingContent(content: string): ThinkingSegment[] {
  try {
    const parsed = JSON.parse(content);
    if (Array.isArray(parsed)) return parsed as ThinkingSegment[];
  } catch { /* legacy plain text */ }
  return [{ agent: "unknown", text: content }];
}


// ── Live thinking box (during streaming) ─────────────────────────
function LiveThinkingBox({ segments }: { segments: ThinkingSegment[] }) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [userScrolled, setUserScrolled] = useState(false);

  useEffect(() => {
    const el = boxRef.current;
    if (!userScrolled && el) el.scrollTop = el.scrollHeight;
  }, [segments, userScrolled]);

  const handleScroll = () => {
    const el = boxRef.current;
    if (!el) return;
    setUserScrolled(el.scrollHeight - el.scrollTop - el.clientHeight > 40);
  };

  return (
    <div className="relative">
      <div ref={boxRef} onScroll={handleScroll}
        className="rounded-lg border border-[#e4e4e7] bg-white px-4 py-3 max-h-[40vh] overflow-y-auto">
        <ThinkingSegmentsView segments={segments} isLive />
      </div>
      {userScrolled && (
        <button
          onClick={() => { setUserScrolled(false); if (boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight; }}
          className="absolute bottom-2 right-3 text-[10px] bg-zinc-700 text-white px-2 py-0.5 rounded-full opacity-80 hover:opacity-100 transition-opacity shadow-sm"
        >
          ↓ latest
        </button>
      )}
    </div>
  );
}


// ── Thinking History ─────────────────────────────────────────────
function ThinkingHistory({ content }: { content: string }) {
  const [collapsed, setCollapsed] = useState(true);
  const segments = parseThinkingContent(content);
  const totalChars = segments.reduce((n, s) => n + s.text.length, 0);
  const agentLabels = [...new Set(segments.map(s => s.agent))];

  return (
    <div className="mb-2">
      <button
        onClick={e => { e.stopPropagation(); setCollapsed(c => !c); }}
        className="flex items-center gap-1.5 text-[11px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors mb-1"
      >
        <span className={`transition-transform text-[9px] ${collapsed ? "" : "rotate-90"}`}>&#9654;</span>
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--text-muted)]" />
        {collapsed ? "Show thinking" : "Hide thinking"}
        {/* Show mini agent badges when collapsed so user knows who thought */}
        <span className="flex items-center gap-0.5 ml-1">
          {agentLabels.map(a => (
            <span key={a} className={`text-[9px] px-1.5 py-0 rounded-full border ${agentBadge(a)}`}>{a}</span>
          ))}
        </span>
        <span className="ml-1 text-[9px] text-zinc-400">{totalChars.toLocaleString()} chars</span>
      </button>
      {!collapsed && (
        <div className="rounded-xl border border-[#e4e4e7] bg-[#fafafa] px-4 py-3 mb-2 max-h-[600px] overflow-y-auto">
          <ThinkingSegmentsView segments={segments} />
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
