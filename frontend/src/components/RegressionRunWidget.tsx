"use client";
import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  useRegressionRun, RunSession, LiveCaseSession,
} from "@/contexts/RegressionRunContext";

// ── Helpers ───────────────────────────────────────────────────────────────────

const AGENT_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  coder:           { bg: "bg-blue-50",   text: "text-blue-700",   border: "border-blue-200" },
  planner:         { bg: "bg-purple-50", text: "text-purple-700", border: "border-purple-200" },
  reviewer:        { bg: "bg-amber-50",  text: "text-amber-700",  border: "border-amber-200" },
  devops:          { bg: "bg-teal-50",   text: "text-teal-700",   border: "border-teal-200" },
  researcher:      { bg: "bg-cyan-50",   text: "text-cyan-700",   border: "border-cyan-200" },
  qa:              { bg: "bg-rose-50",   text: "text-rose-700",   border: "border-rose-200" },
  data_analyst:    { bg: "bg-lime-50",   text: "text-lime-700",   border: "border-lime-200" },
  project_manager: { bg: "bg-orange-50", text: "text-orange-700", border: "border-orange-200" },
  supervisor:      { bg: "bg-zinc-50",   text: "text-zinc-700",   border: "border-zinc-200" },
};

function agentStyle(agent: string) {
  return AGENT_COLORS[agent] ?? { bg: "bg-zinc-50", text: "text-zinc-600", border: "border-zinc-200" };
}

const mdComponents = {
  code({ className, children, ...props }: any) {
    const isInline = !className;
    return isInline
      ? <code className="bg-zinc-100 text-zinc-700 px-1 py-0.5 rounded text-[11px] font-mono" {...props}>{children}</code>
      : <pre className="bg-zinc-900 text-zinc-100 p-2 rounded text-[10px] overflow-x-auto my-1"><code>{children}</code></pre>;
  },
  p: ({ children }: any) => <p className="mb-1 last:mb-0 leading-relaxed text-[12px]">{children}</p>,
  ul: ({ children }: any) => <ul className="list-disc pl-4 space-y-0.5 text-[12px]">{children}</ul>,
  ol: ({ children }: any) => <ol className="list-decimal pl-4 space-y-0.5 text-[12px]">{children}</ol>,
  li: ({ children }: any) => <li>{children}</li>,
};

function StatusDot({ status }: { status: RunSession["status"] }) {
  if (status === "running") return <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse inline-block" />;
  if (status === "done") return <span className="h-2 w-2 rounded-full bg-emerald-500 inline-block" />;
  if (status === "error") return <span className="h-2 w-2 rounded-full bg-red-500 inline-block" />;
  if (status === "stopped") return <span className="h-2 w-2 rounded-full bg-zinc-400 inline-block" />;
  return null;
}

// ── ThinkingBox (per agent turn, archived) ────────────────────────────────────

function ThinkingBox({ segments }: { segments: { agent: string; text: string }[] }) {
  const [open, setOpen] = useState(false);
  if (segments.length === 0) return null;
  return (
    <div className="my-1 rounded border border-zinc-200 bg-zinc-50/80 overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-left hover:bg-zinc-100 transition-colors">
        <span className="text-zinc-400 text-[9px]">{open ? "▼" : "▶"}</span>
        <span className="text-[10px] text-zinc-500 font-medium">Thinking ({segments.length} agent{segments.length > 1 ? "s" : ""})</span>
      </button>
      {open && (
        <div className="px-2.5 pb-2 space-y-2 max-h-60 overflow-y-auto">
          {segments.map((seg, i) => {
            const s = agentStyle(seg.agent);
            return (
              <div key={i}>
                <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded ${s.bg} ${s.text} border ${s.border}`}>{seg.agent}</span>
                <div className="mt-1 text-[11px] text-zinc-600 leading-relaxed whitespace-pre-wrap">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>{seg.text}</ReactMarkdown>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── Live thinking box (current turn) ─────────────────────────────────────────

function LiveThinkingBox({ segments }: { segments: { agent: string; text: string }[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => { ref.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [segments]);
  if (segments.length === 0) return null;
  const current = segments[segments.length - 1];
  const s = agentStyle(current.agent);
  return (
    <div className={`rounded border ${s.border} ${s.bg} px-2.5 py-2 space-y-1 max-h-40 overflow-y-auto`}>
      <div className="flex items-center gap-1.5">
        <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded ${s.bg} ${s.text} border ${s.border}`}>{current.agent}</span>
        <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
      </div>
      <div className="text-[11px] text-zinc-600 leading-relaxed">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>{current.text || "…"}</ReactMarkdown>
      </div>
      <div ref={ref} />
    </div>
  );
}

// ── Tool call bubble ──────────────────────────────────────────────────────────

function ToolBubble({ tool, args, output, agent }: { tool: string; args: Record<string, any>; output?: string; agent?: string }) {
  const [open, setOpen] = useState(false);
  const s = agentStyle(agent || "");
  return (
    <div className="flex items-start gap-1.5 text-[11px]">
      <div className={`rounded px-1.5 py-0.5 text-[9px] font-semibold border ${s.bg} ${s.text} ${s.border} shrink-0`}>{agent || "agent"}</div>
      <button
        onClick={() => setOpen(o => !o)}
        className="flex-1 text-left rounded bg-zinc-100 border border-zinc-200 px-2 py-1 hover:bg-zinc-200 transition-colors">
        <div className="flex items-center gap-1">
          <span className="text-zinc-400">⚙</span>
          <span className="font-medium text-zinc-700">{tool}</span>
          {output && <span className="ml-auto text-[9px] text-emerald-600">✓</span>}
        </div>
        {open && (
          <div className="mt-1.5 space-y-1">
            {Object.keys(args).length > 0 && (
              <div className="text-[10px] text-zinc-500 bg-white rounded border border-zinc-200 p-1.5 font-mono break-all">
                {JSON.stringify(args, null, 2).slice(0, 400)}
              </div>
            )}
            {output && (
              <div className="text-[10px] text-zinc-500 bg-white rounded border border-zinc-200 p-1.5 font-mono break-all max-h-24 overflow-y-auto">
                {output.slice(0, 600)}
              </div>
            )}
          </div>
        )}
      </button>
    </div>
  );
}

// ── Case detail view ──────────────────────────────────────────────────────────

function CaseDetailView({ c }: { c: LiveCaseSession }) {
  const bottomRef = useRef<HTMLDivElement>(null);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [c.trajectory, c.thinkingSegments, c.toolCalls]);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Prompt */}
      <div className="px-3 py-2 border-b border-zinc-100 shrink-0">
        <div className="text-[9px] font-semibold text-zinc-400 uppercase tracking-wide mb-0.5">Prompt</div>
        <p className="text-[11px] text-zinc-600 line-clamp-2">{c.prompt}</p>
      </div>

      {/* Chat-style event stream */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-2">

        {/* Archived thinking segments (one block per agent turn) */}
        {c.thinkingHistory.map((segs, i) => (
          <ThinkingBox key={i} segments={segs} />
        ))}

        {/* Tool calls so far */}
        {c.toolCalls.map((tc, i) => (
          <ToolBubble key={i} tool={tc.tool} args={tc.args} output={tc.output} agent={tc.agent} />
        ))}

        {/* Live thinking (current agent's stream) */}
        {c.thinkingSegments.length > 0 && c.status === "running" && (
          <LiveThinkingBox segments={c.thinkingSegments} />
        )}

        {/* Result (when done) */}
        {c.result && (
          <div className="space-y-2 pt-1 border-t border-zinc-100">
            {/* Trajectory pattern */}
            {c.result.trajectory_pattern.length > 0 && (
              <div className="flex flex-wrap gap-1 items-center">
                {c.result.trajectory_pattern.map((ag, i) => {
                  const s = agentStyle(ag);
                  return (
                    <React.Fragment key={i}>
                      {i > 0 && <span className="text-zinc-300 text-[10px]">→</span>}
                      <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded border ${s.bg} ${s.text} ${s.border}`}>{ag}</span>
                    </React.Fragment>
                  );
                })}
              </div>
            )}

            {/* Criteria */}
            {Object.entries(c.result.trace_assertions).map(([k, v]: [string, any]) => (
              <div key={k} className="flex items-start gap-1.5 text-[11px]">
                <span className={v.passed ? "text-emerald-500 shrink-0" : "text-red-500 shrink-0"}>{v.passed ? "✓" : "✗"}</span>
                <span className="text-zinc-600 font-medium shrink-0">{k.replace(/_/g, " ")}</span>
                <span className="text-zinc-400 text-[10px] truncate">{v.reason}</span>
              </div>
            ))}

            {/* Scores */}
            {Object.keys(c.result.deepeval_scores).filter(k => !k.endsWith("_reason")).length > 0 && (
              <div className="flex flex-wrap gap-1">
                {Object.entries(c.result.deepeval_scores)
                  .filter(([k]) => !k.endsWith("_reason"))
                  .map(([k, v]: [string, any]) => (
                    <span key={k} className={`text-[10px] px-1.5 py-0.5 rounded border ${parseFloat(v) >= 0.6 ? "bg-emerald-50 border-emerald-200 text-emerald-700" : "bg-red-50 border-red-200 text-red-700"}`}>
                      {k.replace(/_/g, " ")} {parseFloat(v).toFixed(2)}
                    </span>
                  ))}
                <span className={`text-[10px] px-1.5 py-0.5 rounded border ${c.result.semantic_similarity >= 0.5 ? "bg-emerald-50 border-emerald-200 text-emerald-700" : "bg-amber-50 border-amber-200 text-amber-700"}`}>
                  sim {(c.result.semantic_similarity * 100).toFixed(0)}%
                </span>
              </div>
            )}

            {/* Final output */}
            {c.result.actual_output && (
              <div className="rounded border border-zinc-200 bg-white">
                <div className="px-2 py-1 border-b border-zinc-100 text-[10px] font-medium text-zinc-500">Final Output</div>
                <div className="px-2 py-1.5 text-[11px] text-zinc-700 max-h-48 overflow-y-auto">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>{c.result.actual_output}</ReactMarkdown>
                </div>
              </div>
            )}
            {c.result.error && <div className="text-[11px] text-red-600 font-medium">Error: {c.result.error}</div>}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Footer stats (latency, cost) when done */}
      {c.result && (
        <div className="px-3 py-1.5 border-t border-zinc-100 flex items-center gap-3 text-[10px] text-zinc-400 shrink-0">
          <span>{Math.round(c.result.latency_ms / 1000)}s</span>
          <span>${c.result.actual_cost.toFixed(4)}</span>
          <span className={`font-semibold ${c.result.passed ? "text-emerald-600" : "text-red-600"}`}>
            {c.result.passed ? "PASS" : "FAIL"}
          </span>
        </div>
      )}
    </div>
  );
}

// ── Main widget ───────────────────────────────────────────────────────────────

export default function RegressionRunWidget() {
  const {
    sessions, activeSessionId, activeCaseId, isAnyRunning,
    setActiveSessionId, setActiveCaseId, stopRun, clearSessions,
  } = useRegressionRun();

  const [expanded, setExpanded] = useState(false);

  // Auto-expand when a run starts
  useEffect(() => {
    if (isAnyRunning) setExpanded(true);
  }, [isAnyRunning]);

  const activeSession = sessions.find(s => s.sessionId === activeSessionId) ?? sessions[sessions.length - 1] ?? null;
  const casesArr = activeSession ? Object.values(activeSession.cases).sort((a, b) => a.index - b.index) : [];
  const activeCase = activeSession && activeCaseId ? activeSession.cases[activeCaseId] ?? casesArr[0] ?? null : casesArr[0] ?? null;

  // ── Collapsed badge ──────────────────────────────────────────────────────────

  if (!expanded) {
    if (sessions.length === 0) return null;
    const runningCount = sessions.filter(s => s.status === "running").length;
    const latestSession = sessions[sessions.length - 1];
    const passedCount = Object.values(latestSession?.cases ?? {}).filter(c => c.status === "pass").length;
    const total = Object.values(latestSession?.cases ?? {}).length;
    return (
      <button
        onClick={() => setExpanded(true)}
        className="fixed bottom-4 left-4 z-50 flex items-center gap-2 px-3 py-2 rounded-full bg-zinc-900 text-white text-[12px] font-medium shadow-lg hover:bg-zinc-800 transition-all">
        {isAnyRunning
          ? <><span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />{passedCount}/{total} tests running</>
          : <><span className="h-2 w-2 rounded-full bg-emerald-500" />{latestSession?.summary ? `${latestSession.summary.num_passed}/${latestSession.summary.total_cases} passed` : "Run complete"}</>}
        {runningCount > 1 && <span className="px-1.5 py-0.5 rounded-full bg-zinc-700 text-[10px]">×{runningCount}</span>}
      </button>
    );
  }

  // ── Expanded widget ──────────────────────────────────────────────────────────

  return (
    <div className="fixed bottom-4 left-4 z-50 flex flex-col rounded-2xl shadow-2xl bg-white border border-zinc-200 overflow-hidden"
      style={{ width: 700, height: 560 }}>

      {/* ── Header ── */}
      <div className="flex items-center gap-2 px-3 py-2.5 bg-zinc-900 text-white shrink-0">
        <span className="text-[12px] font-semibold flex-1">Regression Run</span>

        {/* Session tabs */}
        <div className="flex items-center gap-1 overflow-x-auto max-w-[320px]">
          {sessions.map(s => (
            <button key={s.sessionId}
              onClick={() => { setActiveSessionId(s.sessionId); setActiveCaseId(null); }}
              className={`flex items-center gap-1.5 px-2 py-1 rounded text-[10px] font-medium shrink-0 transition-colors ${
                s.sessionId === activeSession?.sessionId
                  ? "bg-white text-zinc-900"
                  : "text-zinc-400 hover:text-white hover:bg-zinc-700"
              }`}>
              <StatusDot status={s.status} />
              <span className="truncate max-w-[80px]">{s.label}</span>
            </button>
          ))}
        </div>

        {/* Stop button */}
        {activeSession?.status === "running" && (
          <button onClick={() => stopRun(activeSession.sessionId)}
            className="px-2 py-1 rounded text-[10px] font-medium bg-red-600 hover:bg-red-700 text-white transition-colors shrink-0">
            Stop
          </button>
        )}

        {/* Clear all (when nothing running) */}
        {!isAnyRunning && (
          <button onClick={clearSessions}
            className="px-2 py-1 rounded text-[10px] font-medium text-zinc-400 hover:text-white transition-colors shrink-0">
            Clear
          </button>
        )}

        {/* Collapse */}
        <button onClick={() => setExpanded(false)}
          className="text-zinc-400 hover:text-white transition-colors shrink-0 ml-1 text-lg leading-none">
          ×
        </button>
      </div>

      {/* ── Run summary row ── */}
      {activeSession && (
        <div className="flex items-center gap-3 px-3 py-1.5 bg-zinc-50 border-b border-zinc-100 text-[11px] text-zinc-500 shrink-0">
          {activeSession.runId && (
            <span className="font-mono text-zinc-400">{activeSession.runId.slice(0, 8)}</span>
          )}
          <span>
            {Object.values(activeSession.cases).filter(c => c.status === "pass").length}/
            {Object.values(activeSession.cases).length} passed
          </span>
          {activeSession.summary && (
            <>
              <span>${(activeSession.summary.total_cost || 0).toFixed(4)}</span>
              <span>{Math.round((activeSession.summary.avg_latency_ms || 0) / 1000)}s avg</span>
            </>
          )}
          <span className="ml-auto">
            {activeSession.status === "running" && <span className="text-emerald-600 font-medium">Running…</span>}
            {activeSession.status === "done" && <span className="text-emerald-600 font-medium">Done</span>}
            {activeSession.status === "stopped" && <span className="text-zinc-400 font-medium">Stopped</span>}
            {activeSession.status === "error" && <span className="text-red-600 font-medium">Error</span>}
          </span>
        </div>
      )}

      {/* ── Body: case list + detail ── */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left: case list */}
        <div className="w-48 border-r border-zinc-100 flex flex-col overflow-hidden shrink-0">
          <div className="overflow-y-auto flex-1">
            {casesArr.length === 0 && (
              <div className="text-[11px] text-zinc-400 p-3 italic">Waiting for tests…</div>
            )}
            {casesArr.map(c => {
              const isActive = c.caseId === activeCase?.caseId;
              const statusColor = c.status === "pass" ? "text-emerald-600" : c.status === "fail" ? "text-red-600" : c.status === "error" ? "text-orange-600" : c.status === "running" ? "text-indigo-600" : "text-zinc-400";
              const statusIcon = c.status === "pass" ? "✓" : c.status === "fail" ? "✗" : c.status === "error" ? "!" : c.status === "running" ? "…" : "○";
              return (
                <button key={c.caseId}
                  onClick={() => setActiveCaseId(c.caseId)}
                  className={`w-full text-left px-2.5 py-2 border-b border-zinc-50 transition-colors ${isActive ? "bg-zinc-900 text-white" : "hover:bg-zinc-50"}`}>
                  <div className="flex items-center gap-1.5">
                    <span className={`text-[10px] font-bold shrink-0 ${isActive ? "text-white" : statusColor}`}>{statusIcon}</span>
                    <span className={`text-[11px] font-medium truncate ${isActive ? "text-white" : "text-zinc-700"}`}>{c.caseLabel}</span>
                  </div>
                  {c.status === "running" && (
                    <div className="mt-1 flex flex-wrap gap-0.5">
                      {c.trajectory.filter(t => t.status === "active").map((t, i) => {
                        const ag = t.action.startsWith("tool:") ? t.action.slice(5) : t.agent;
                        const isToolActive = t.action.startsWith("tool:");
                        return (
                          <span key={i} className={`text-[9px] px-1 py-0.5 rounded font-mono ${isActive ? "bg-zinc-700 text-zinc-300" : "bg-indigo-50 text-indigo-600 border border-indigo-100"} ${isToolActive ? "italic" : ""}`}>
                            {isToolActive ? `⚙ ${ag}` : ag}
                          </span>
                        );
                      })}
                    </div>
                  )}
                  {c.result && (
                    <div className={`text-[9px] mt-0.5 ${isActive ? "text-zinc-400" : "text-zinc-400"}`}>
                      {Math.round(c.result.latency_ms / 1000)}s · ${c.result.actual_cost.toFixed(3)}
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Right: case detail */}
        <div className="flex-1 overflow-hidden">
          {activeCase
            ? <CaseDetailView c={activeCase} />
            : (
              <div className="flex items-center justify-center h-full text-[12px] text-zinc-400">
                Select a test case to see details
              </div>
            )
          }
        </div>
      </div>
    </div>
  );
}
