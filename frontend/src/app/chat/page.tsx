"use client";
import { useEffect, useState, useRef } from "react";
import { api } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  agent?: string;
  toolCalls?: { tool: string; args: Record<string, any> }[];
  trace?: { trace_id: string; total_latency_ms: number; total_tokens: number; total_cost: number; spans: any[] };
}

export default function ChatPage() {
  const [teams, setTeams] = useState<any[]>([]);
  const [teamId, setTeamId] = useState("default");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => { api.teams.list().then(setTeams); }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function send() {
    if (!input.trim() || loading) return;
    const msg = input.trim();
    setInput("");
    setMessages(p => [...p, { role: "user", content: msg }]);
    setLoading(true);
    try {
      const r = await api.teams.chat(teamId, msg);
      setMessages(p => [...p, { role: "assistant", content: r.response, agent: r.agent_used, toolCalls: r.tool_calls, trace: r.trace }]);
    } catch (e: any) {
      setMessages(p => [...p, { role: "assistant", content: `Error: ${e.message}` }]);
    } finally { setLoading(false); }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">Chat</h1>
        <select value={teamId} onChange={e => setTeamId(e.target.value)} className="input !w-auto">
          {teams.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 pb-4">
        {messages.length === 0 && (
          <div className="text-center text-[var(--text-muted)] py-20">
            <p className="text-lg">Ask the agent to do something</p>
            <p className="text-sm mt-2">Try: &quot;Read main.py&quot; or &quot;Run the tests&quot;</p>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[80%] rounded-xl px-4 py-3 ${m.role === "user" ? "bg-[var(--accent)] text-white" : "card"}`}>
              {m.agent && <div className="text-[11px] text-[var(--accent)] font-medium mb-1">{m.agent}</div>}
              <div className="text-sm whitespace-pre-wrap leading-relaxed">{m.content}</div>

              {m.toolCalls && m.toolCalls.length > 0 && (
                <div className="mt-3 pt-2 border-t border-[var(--border)] space-y-1">
                  {m.toolCalls.map((tc, j) => (
                    <div key={j} className="flex items-center gap-1.5 text-[11px]">
                      <span className="text-[var(--success)]">✓</span>
                      <span className="font-mono text-[var(--text)]">{tc.tool}</span>
                      <span className="text-[var(--text-muted)] truncate">
                        ({Object.entries(tc.args || {}).map(([k, v]) => `${k}="${String(v).slice(0, 25)}"`).join(", ")})
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {m.trace && (
                <div className="flex gap-4 mt-2 text-[10px] text-[var(--text-muted)]">
                  <span>{m.trace.total_latency_ms.toFixed(0)}ms</span>
                  <span>{m.trace.total_tokens} tokens</span>
                  <span>${m.trace.total_cost.toFixed(4)}</span>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="card px-4 py-3 text-sm text-[var(--text-muted)] flex items-center gap-2">
              <div className="h-3 w-3 border-2 border-[var(--border)] border-t-[var(--accent)] rounded-full animate-spin" />
              Working...
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="flex gap-2 pt-3 border-t border-[var(--border)]">
        <input value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && send()} disabled={loading}
          placeholder="Ask the agent..." className="input flex-1" />
        <button onClick={send} disabled={loading} className="btn-primary">Send</button>
      </div>
    </div>
  );
}
