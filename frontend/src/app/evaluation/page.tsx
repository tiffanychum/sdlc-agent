"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

interface EvalRun {
  id: string; model: string; prompt_version: string; num_tasks: number;
  task_completion_rate: number; routing_accuracy: number;
  avg_tool_call_accuracy: number; avg_latency_ms: number | null;
  total_cost: number; created_at: string;
}

export default function EvaluationPage() {
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [selectedRuns, setSelectedRuns] = useState<string[]>([]);
  const [comparison, setComparison] = useState<any>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => { loadRuns(); }, []);

  async function loadRuns() { setRuns(await api.eval.runs()); }

  async function runEval() {
    setRunning(true);
    try {
      await api.eval.run("default");
      await loadRuns();
    } finally { setRunning(false); }
  }

  function toggleSelect(id: string) {
    setSelectedRuns(prev =>
      prev.includes(id) ? prev.filter(r => r !== id) : prev.length < 2 ? [...prev, id] : [prev[1], id]
    );
    setComparison(null);
  }

  async function compareSelected() {
    if (selectedRuns.length !== 2) return;
    const result = await api.eval.compareRuns(selectedRuns[0], selectedRuns[1]);
    setComparison(result);
  }

  const chartData = runs.slice(0, 10).reverse().map(r => ({
    model: `${r.model.slice(0, 12)}(${r.id.slice(0, 4)})`,
    completion: +(r.task_completion_rate * 100).toFixed(1),
    routing: +(r.routing_accuracy * 100).toFixed(1),
    tool_accuracy: +(r.avg_tool_call_accuracy * 100).toFixed(1),
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Evaluation</h1>
        <div className="flex gap-3">
          {selectedRuns.length === 2 && (
            <button onClick={compareSelected} className="px-4 py-2 text-sm bg-yellow-600/20 text-yellow-400 border border-yellow-500/30 rounded-lg hover:bg-yellow-600/30">
              Compare Selected ({selectedRuns.length})
            </button>
          )}
          <button onClick={runEval} disabled={running}
            className="px-4 py-2 text-sm bg-blue-600 rounded-lg hover:bg-blue-500 disabled:opacity-50">
            {running ? "Running..." : "Run Evaluation"}
          </button>
        </div>
      </div>

      {/* Comparison Chart */}
      {chartData.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <h3 className="text-sm font-semibold mb-3 text-gray-400">Performance Across Runs</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="model" stroke="#6B7280" fontSize={10} />
              <YAxis stroke="#6B7280" fontSize={11} domain={[0, 100]} />
              <Tooltip contentStyle={{ backgroundColor: "#1F2937", border: "1px solid #374151", borderRadius: 8 }} />
              <Legend />
              <Bar dataKey="completion" name="Completion %" fill="#10B981" radius={[2, 2, 0, 0]} />
              <Bar dataKey="routing" name="Routing %" fill="#3B82F6" radius={[2, 2, 0, 0]} />
              <Bar dataKey="tool_accuracy" name="Tool Accuracy %" fill="#8B5CF6" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Regression Comparison */}
      {comparison && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <h3 className="font-semibold mb-3">Regression Analysis: {comparison.run_a} vs {comparison.run_b}</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 text-xs uppercase">
                <th className="text-left py-2">Metric</th>
                <th className="text-right py-2">Before</th>
                <th className="text-right py-2">After</th>
                <th className="text-right py-2">Delta</th>
                <th className="text-right py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(comparison.comparison).map(([metric, data]: [string, any]) => (
                <tr key={metric} className="border-t border-gray-800">
                  <td className="py-2 text-gray-300">{metric.replace(/_/g, " ")}</td>
                  <td className="text-right">{(data.before * 100).toFixed(1)}%</td>
                  <td className="text-right">{(data.after * 100).toFixed(1)}%</td>
                  <td className={`text-right ${data.delta >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {data.delta >= 0 ? "+" : ""}{(data.delta * 100).toFixed(1)}%
                  </td>
                  <td className="text-right">
                    {data.regression
                      ? <span className="text-xs px-2 py-0.5 bg-red-600/20 text-red-400 rounded-full">REGRESSION</span>
                      : <span className="text-xs px-2 py-0.5 bg-green-600/20 text-green-400 rounded-full">OK</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Runs Table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-xs uppercase bg-gray-800/50">
              <th className="text-left px-4 py-3">Select</th>
              <th className="text-left px-4 py-3">Run ID</th>
              <th className="text-left px-4 py-3">Model</th>
              <th className="text-right px-4 py-3">Tasks</th>
              <th className="text-right px-4 py-3">Completion</th>
              <th className="text-right px-4 py-3">Routing</th>
              <th className="text-right px-4 py-3">Tool Acc.</th>
              <th className="text-right px-4 py-3">Latency</th>
              <th className="text-right px-4 py-3">Date</th>
            </tr>
          </thead>
          <tbody>
            {runs.map(r => (
              <tr key={r.id} className={`border-t border-gray-800 hover:bg-gray-800/30 ${selectedRuns.includes(r.id) ? "bg-blue-600/10" : ""}`}>
                <td className="px-4 py-3">
                  <input type="checkbox" checked={selectedRuns.includes(r.id)} onChange={() => toggleSelect(r.id)}
                    className="rounded border-gray-600" />
                </td>
                <td className="px-4 py-3 font-mono text-xs">{r.id}</td>
                <td className="px-4 py-3">{r.model}</td>
                <td className="text-right px-4 py-3">{r.num_tasks}</td>
                <td className="text-right px-4 py-3">{(r.task_completion_rate * 100).toFixed(0)}%</td>
                <td className="text-right px-4 py-3">{(r.routing_accuracy * 100).toFixed(0)}%</td>
                <td className="text-right px-4 py-3">{(r.avg_tool_call_accuracy * 100).toFixed(0)}%</td>
                <td className="text-right px-4 py-3">{r.avg_latency_ms ? `${r.avg_latency_ms.toFixed(0)}ms` : "-"}</td>
                <td className="text-right px-4 py-3 text-gray-500 text-xs">{r.created_at?.slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {runs.length === 0 && <div className="text-center py-10 text-gray-500">No evaluation runs yet. Click &quot;Run Evaluation&quot; to start.</div>}
      </div>
    </div>
  );
}
