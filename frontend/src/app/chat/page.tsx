"use client";
import { useEffect, useState, useRef } from "react";
import { api } from "@/lib/api";

interface ToolCall { tool: string; args: Record<string, any> }
interface SpanInfo { id: string; name: string; span_type: string; tokens_in: number; tokens_out: number; cost: number; status: string }
interface TraceInfo { trace_id: string; total_latency_ms: number; total_tokens: number; total_cost: number; spans: SpanInfo[] }
interface Message {
  role: "user" | "assistant"; content: string; agent?: string;
  toolCalls?: ToolCall[]; trace?: TraceInfo;
}

const TYPE_DOT: Record<string, string> = {
  routing: "bg-blue-500",
  llm_call: "bg-purple-500",
  tool_call: "bg-green-500",
};

export default function ChatPage() {
  const [teams, setTeams] = useState<any[]>([]);
  const [teamId, setTeamId] = useState("default");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTrace, setActiveTrace] = useState<TraceInfo | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => { api.teams.list().then(setTeams); }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function send() {
    if (!input.trim() || loading) return;
    const msg = input.trim();
    setInput("");
    setMessages(p => [...p, { role: "user", content: msg }]);
    setLoading(true);
    setActiveTrace(null);
    try {
      const r = await api.teams.chat(teamId, msg);
      const newMsg: Message = { role: "assistant", content: r.response, agent: r.agent_used, toolCalls: r.tool_calls, trace: r.trace };
      setMessages(p => [...p, newMsg]);
      setActiveTrace(r.trace);
    } catch (e: any) {
      setMessages(p => [...p, { role: "assistant", content: `Error: ${e.message}` }]);
    } finally { setLoading(false); }
  }

  const latestTrace = activeTrace || messages.filter(m => m.trace).slice(-1)[0]?.trace;

  return (
    <div className="flex gap-5 h-[calc(100vh-4rem)]">
      {/* Chat Panel - 60% */}
      <div className="flex flex-col flex-[3]">
        <div className="flex items-center justify-between mb-3">
          <h1 className="text-xl font-semibold">Chat</h1>
          <select value={teamId} onChange={e => setTeamId(e.target.value)} className="input !w-auto">
            {teams.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
          </select>
        </div>

        <div className="flex-1 overflow-y-auto space-y-2.5 pb-3">
          {messages.length === 0 && (
            <div className="text-center text-[var(--text-muted)] py-16">
              <p className="text-base">Ask the agent to do something</p>
              <p className="text-sm mt-1.5 opacity-70">Try: &quot;Read main.py&quot; or &quot;Run the tests&quot;</p>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[85%] rounded-xl px-4 py-2.5 ${m.role === "user" ? "bg-[var(--accent)] text-white" : "card !p-3"}`}>
                {m.agent && <div className="text-[11px] text-[var(--accent)] font-medium mb-0.5">{m.agent}</div>}
                <div className="text-[13px] whitespace-pre-wrap leading-relaxed">{m.content}</div>
                {m.toolCalls && m.toolCalls.length > 0 && (
                  <div className="mt-2 pt-1.5 border-t border-[var(--border)] space-y-0.5">
                    {m.toolCalls.map((tc, j) => (
                      <div key={j} className="flex items-center gap-1 text-[11px]">
                        <span className="text-[var(--success)]">&#10003;</span>
                        <span className="font-mono">{tc.tool}</span>
                      </div>
                    ))}
                  </div>
                )}
                {m.trace && (
                  <button onClick={() => setActiveTrace(m.trace!)}
                    className="flex gap-3 mt-1.5 text-[10px] text-[var(--text-muted)] hover:text-[var(--accent)] transition-colors">
                    <span>{m.trace.total_latency_ms.toFixed(0)}ms</span>
                    <span>{m.trace.total_tokens} tokens</span>
                    <span>${m.trace.total_cost.toFixed(4)}</span>
                    <span className="underline">{m.trace.spans.length} spans</span>
                  </button>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="card !p-3 text-[13px] text-[var(--text-muted)] flex items-center gap-2">
                <div className="h-3 w-3 border-2 border-[var(--border)] border-t-[var(--accent)] rounded-full animate-spin" />
                Working...
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        <div className="flex gap-2 pt-2.5 border-t border-[var(--border)]">
          <input value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && send()} disabled={loading}
            placeholder="Ask the agent..." className="input flex-1" />
          <button onClick={send} disabled={loading} className="btn-primary">Send</button>
        </div>
      </div>

      {/* Trace Panel - 40% */}
      <div className="flex-[2] border-l border-[var(--border)] pl-5 overflow-y-auto">
        <h2 className="text-sm font-medium text-[var(--text-muted)] mb-3">Trace Inspector</h2>

        {loading && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 p-2 rounded-lg bg-[var(--accent-light)] text-[var(--accent)] text-xs">
              <div className="h-2.5 w-2.5 bg-[var(--accent)] rounded-full animate-pulse" />
              Processing request...
            </div>
          </div>
        )}

        {latestTrace ? (
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="p-2 rounded-lg bg-[var(--bg)]">
                <div className="text-lg font-semibold">{latestTrace.total_latency_ms.toFixed(0)}<span className="text-xs text-[var(--text-muted)]">ms</span></div>
                <div className="text-[10px] text-[var(--text-muted)]">Latency</div>
              </div>
              <div className="p-2 rounded-lg bg-[var(--bg)]">
                <div className="text-lg font-semibold">{latestTrace.total_tokens}</div>
                <div className="text-[10px] text-[var(--text-muted)]">Tokens</div>
              </div>
              <div className="p-2 rounded-lg bg-[var(--bg)]">
                <div className="text-lg font-semibold">${latestTrace.total_cost.toFixed(4)}</div>
                <div className="text-[10px] text-[var(--text-muted)]">Cost</div>
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="text-[11px] text-[var(--text-muted)] uppercase tracking-wide">Spans</div>
              {latestTrace.spans.map((s, i) => (
                <div key={s.id} className="flex items-center gap-2 p-2 rounded-lg border border-[var(--border)] bg-[var(--bg-card)] text-xs">
                  <div className={`h-2 w-2 rounded-full flex-shrink-0 ${TYPE_DOT[s.span_type] || "bg-gray-400"}`} />
                  <div className="flex-1 min-w-0">
                    <div className="font-medium truncate">{s.name}</div>
                    <div className="text-[var(--text-muted)] text-[10px]">{s.span_type}</div>
                  </div>
                  <div className="text-right text-[10px] text-[var(--text-muted)] flex-shrink-0">
                    {s.tokens_in + s.tokens_out > 0 && <div>{s.tokens_in + s.tokens_out} tok</div>}
                    {s.cost > 0 && <div>${s.cost.toFixed(4)}</div>}
                  </div>
                  <div className={`h-1.5 w-1.5 rounded-full flex-shrink-0 ${s.status === "completed" ? "bg-green-500" : "bg-red-500"}`} />
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="text-sm text-[var(--text-muted)] text-center py-12">
            Send a message to see the trace
          </div>
        )}
      </div>
    </div>
  );
}
