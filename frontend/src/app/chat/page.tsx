"use client";
import { useEffect, useState, useRef } from "react";
import { api } from "@/lib/api";

interface ToolCall { tool: string; args: Record<string, any> }
interface TraceSpan { id: string; name: string; span_type: string; tokens_in: number; tokens_out: number; cost: number; status: string }
interface Message {
  role: "user" | "assistant";
  content: string;
  agent?: string;
  toolCalls?: ToolCall[];
  trace?: { trace_id: string; total_latency_ms: number; total_tokens: number; total_cost: number; spans: TraceSpan[] };
}

export default function ChatPage() {
  const [teams, setTeams] = useState<any[]>([]);
  const [selectedTeam, setSelectedTeam] = useState("default");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => { api.teams.list().then(setTeams); }, []);
  useEffect(() => { scrollRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function send() {
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: userMsg }]);
    setLoading(true);

    try {
      const result = await api.teams.chat(selectedTeam, userMsg);
      setMessages(prev => [...prev, {
        role: "assistant",
        content: result.response,
        agent: result.agent_used,
        toolCalls: result.tool_calls,
        trace: result.trace,
      }]);
    } catch (e: any) {
      setMessages(prev => [...prev, { role: "assistant", content: `Error: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      {/* Team Selector */}
      <div className="flex items-center gap-3 mb-4">
        <span className="text-sm text-gray-400">Team:</span>
        <select value={selectedTeam} onChange={e => setSelectedTeam(e.target.value)}
          className="bg-gray-800 border border-gray-700 rounded-md px-3 py-1.5 text-sm">
          {teams.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 pb-4">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-2xl rounded-xl px-4 py-3 ${msg.role === "user" ? "bg-blue-600 text-white" : "bg-gray-800 border border-gray-700"}`}>
              {msg.agent && (
                <div className="text-xs text-blue-400 mb-1 font-medium">{msg.agent}</div>
              )}
              <div className="text-sm whitespace-pre-wrap">{msg.content}</div>

              {/* Tool Calls Trace */}
              {msg.toolCalls && msg.toolCalls.length > 0 && (
                <div className="mt-3 pt-2 border-t border-gray-700 space-y-1">
                  <div className="text-xs text-gray-500 font-medium">Agent Workflow:</div>
                  {msg.toolCalls.map((tc, j) => (
                    <div key={j} className="flex items-center gap-2 text-xs">
                      <span className="text-green-400">&#10003;</span>
                      <span className="text-gray-300 font-mono">{tc.tool}</span>
                      <span className="text-gray-600">({Object.entries(tc.args || {}).map(([k, v]) => `${k}="${String(v).slice(0, 30)}"`).join(", ")})</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Trace Summary */}
              {msg.trace && (
                <div className="mt-2 flex gap-3 text-xs text-gray-500">
                  <span>{msg.trace.total_latency_ms.toFixed(0)}ms</span>
                  <span>{msg.trace.total_tokens} tokens</span>
                  <span>${msg.trace.total_cost.toFixed(4)}</span>
                  <span>{msg.trace.spans.length} spans</span>
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-sm text-gray-400">
              <div className="flex items-center gap-2">
                <div className="animate-spin h-4 w-4 border-2 border-gray-600 border-t-blue-400 rounded-full" />
                Working...
              </div>
            </div>
          </div>
        )}
        <div ref={scrollRef} />
      </div>

      {/* Input */}
      <div className="flex gap-3 pt-3 border-t border-gray-800">
        <input value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && send()}
          placeholder="Ask the agent to do something..."
          disabled={loading}
          className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-blue-500" />
        <button onClick={send} disabled={loading}
          className="px-6 py-2.5 bg-blue-600 rounded-lg text-sm font-medium hover:bg-blue-500 disabled:opacity-50">
          Send
        </button>
      </div>
    </div>
  );
}
