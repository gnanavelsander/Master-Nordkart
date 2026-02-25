"use client";

import { useEffect, useState } from "react";
import type { FeatureCollection, Geometry } from "geojson";
import MapClient from "./components/MapClient";
import ResultTable from "./ResultTable";

export default function Page() {
  const [geojson, setGeojson] = useState<FeatureCollection | undefined>();
  const [aoi, setAoi] = useState<Geometry | null>(null);
  const [scrollActive, setScrollActive] = useState(false);





  /**making react hooks to send message from the user input */
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [lastResult, setLastResult] = useState<any>(null);
  const [rightTab, setRightTab] = useState<"sql" | "queries" | "metrics" | "errors">("sql");

  const [errors, setErrors] = useState<string[]>([]);






  // chat history — stored locally, not in Firestore
  const [messages, setMessages] = useState<{ role: "user" | "agent"; text: string; results?: any[] }[]>([]);

  // which model result is shown on map
  const [activeModel, setActiveModel] = useState<string | null>(null);

  // table view per model
  const [tableModel, setTableModel] = useState<string | null>(null);





  /**take user input**/
  /**using react hooks */
  async function sendToBackend() {
    if (!message.trim()) return;

    const userMessage = message;

    // add user message to chat
    setMessages((prev) => [...prev, { role: "user", text: userMessage }]);
    setMessage("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userMessage,
          models: selectedModels,
          mode: "parallel",
          aoi,
        }),
      });
      console.log("status", res.status);



      //convert response to jason
      const data = await res.json();


      setLastResult(data.results);

      setErrors(
  data.results.flatMap((r: any) => r.all_errors ?? [])
);      

      // get best model
      const best = data.results.find((r: any)=>r.model_id === data.best_model)

        // Set best model as default on the map 
       if (best) {
        setActiveModel(best.model_id);
        if (best.geojson) setGeojson(best.geojson);
      }

      // add agent response to chat
      setMessages((prev) => [
        ...prev,
        {
          role: "agent",
          text: best
            ? `Results found. Best model: ${best.model_id} (${best.latency_s}s, ${best.attempts} forsøk)`
            : "No model was able to generate valid SQL.",
          results: data.results,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }










  const ALL_MODELS = ["gpt-35-turbo", "gpt-4.1-nano", "gpt-4", "gpt-4o", "GPT-4o-mini"];
  const [selectedModels, setSelectedModels] = useState<string[]>(["gpt-35-turbo", "gpt-4.1-nano", "gpt-4", "gpt-4o", "GPT-4o-mini"]);

  const modelLength = selectedModels.length;


  function handleDraw(aoiGeo: Geometry) {
    setAoi(aoiGeo);
  } 

  //Updates the map with that model’s spatial result.
  function showOnMap(r: any) {
    setActiveModel(r.model_id);
    if (r.geojson) setGeojson(r.geojson);
  }

 // states
const [leftView, setLeftView] = useState<"map" | "table">("map");
const [tableData, setTableData] = useState<any[]>([]);
const [tableModelName, setTableModelName] = useState<string>("");

// show tael function
async function showTable(r: any) {
    // always fetch — don't rely on cache
    const res = await fetch("http://localhost:8000/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sql: r.sql }),
    });

    const data = await res.json();

    if (data.error) {
        console.error("execute error:", data.error);
        return;
    }

    r.rows = data.rows;
    setLastResult((prev: any) => [...prev]);
    setTableData(data.rows);
    setTableModelName(r.model_id);
    setLeftView("table");
}





  function toggleModel(model: string) {
    setSelectedModels((prev) =>
      prev.includes(model)
        ? prev.filter((m) => m !== model)
        : [...prev, model]
    );
  }

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-6">Nordkart Agent</h2>

      <div className="grid grid-cols-[1fr_700px] gap-3">

        {/* map + chat */}
        <div className="flex flex-col items-center gap-6">

       {/* MAP or TABLE */}
<div className="w-[800px]">

    {/* back to map button */}
    {leftView === "table" && (
        <button
            onClick={() => setLeftView("map")}
            className="mb-2 px-3 py-1 text-xs border border-gray-600 text-gray-300 rounded hover:bg-zinc-700"
        >
            Back to map
        </button>
    )}

    {leftView === "map" && (
        <MapClient geojson={geojson} onDraw={handleDraw} />
    )}

    {leftView === "table" && (
        <ResultTable
            modelId={tableModelName}
            rows={tableData}
            onClose={() => setLeftView("map")}
        />
    )}
</div>
        


        {/* TABLE — */}
        {tableModel && lastResult && (
            <ResultTable
                modelId={tableModel}
                rows={lastResult.find((r: any) => r.model_id === tableModel)?.rows ?? []}
                onClose={() => setTableModel(null)}
            />
        )}














          {/* CHAT */}
          <aside className="w-[800px] rounded-xl border p-4 h-[400px] flex flex-col relative">

            {/* messages */}
<div className="flex-1 overflow-y-auto mb-4 space-y-3 w-full custom-scroll">
                <p className="text-sm text-gray-600">What spatial query would you like to run?</p>

              {messages.map((msg, i) => (
                <div key={i} className={msg.role === "user" ? "text-right" : "text-left"}>
                  <div className="bg-zinc-800 px-3 py-2 rounded-lg inline-block text-white text-sm">
                    {msg.text}
                  </div>
{/* metrics table under agent reply */}
{msg.role === "agent" && msg.results && (
  <div className="mt-2">
    <table className="w-full text-xs">
      <thead>
        <tr className="border-b border-gray-700">
          <th className="text-left py-1 text-orange-400 font-mono">Modell</th>
          <th className="text-left py-1 text-cyan-400 font-mono">Valid</th>
          <th className="text-left py-1 text-cyan-400 font-mono">Attempts</th>
          <th className="text-left py-1 text-cyan-400 font-mono">Latency</th>
          <th className="text-left py-1 text-cyan-400 font-mono">First-pass</th>



        </tr>
      </thead>
      <tbody>
        {msg.results.map((r: any) => (
          <tr key={r.model_id} className="border-b border-gray-800">
            <td className="py-1 text-orange-300 font-mono">{r.model_id}</td>
            <td className={`py-1 font-mono ${r.valid ? "text-green-400" : "text-red-400"}`}>
              {r.valid ? "Yes" : "No"}
            </td>
            <td className={`py-1 font-mono ${
              r.attempts === 1 ? "text-green-400" :
              r.attempts === 2 ? "text-yellow-400" :
              "text-red-400"
            }`}>
              {r.attempts}
            </td>
            <td className="py-1 text-cyan-300 font-mono">{r.latency_s}s</td>
            <td className="py-1 text-cyan-300 font-mono">
              {r.attempts === 1 ? "Yes" : "No"}
            </td>
      


          </tr>
        ))}
      </tbody>
    </table>
  </div>
)}
</div>
))}

{loading && (
  <p className="text-xs text-gray-500 animate-pulse">{modelLength>1?"Running models...":"Running model ..."}</p>
)}
</div>
            {/* MODELS */}
            <div className="right-4 bottom-16 flex gap-3">
              {ALL_MODELS.map((model) => (
                <label key={model} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(model)}
                    onChange={() => toggleModel(model)}
                  />
                  {model}
                </label>
              ))}
            </div>

            {/* Input area */}
            <div className="border-t pt-4">
              <div className="flex gap-2 items-end relative w-full">
                <textarea
                  rows={1}
                  value={message}
                  onChange={(e) => {
                    setMessage(e.target.value);
                    const el = e.currentTarget;
                    const maxHeight = 200;
                    el.style.height = "auto";
                    const next = Math.min(el.scrollHeight, maxHeight);
                    el.style.height = next + "px";
                    setScrollActive(el.scrollHeight > maxHeight);
                  }}
                  placeholder="Skriv melding..."
                  onInput={(e) => {
                    const el = e.currentTarget;
                    const maxHeight = 200;
                    el.style.height = "auto";
                    const next = Math.min(el.scrollHeight, maxHeight);
                    el.style.height = next + "px";
                    // show scrollbar only when max reached
                    setScrollActive(el.scrollHeight > maxHeight);
                  }}
                  className={[
                    "flex-1 h-12 max-h-[200px] resize-none rounded-lg border border-gray-200 px-4 py-2",
                    "focus:outline-none focus:ring-2 focus:ring-blue-500",
                    "text-white bg-black",
                    scrollActive ? "overflow-y-auto" : "overflow-y-hidden",
                  ].join(" ")}
                />

                <button
                  onClick={sendToBackend}
                  disabled={loading}
                  className="bg-black border absolute right-1.5 bottom-1.5 border-gray-300 w-7 h-7 rounded-lg flex items-center justify-center"
                >
                  <img src="/logo.svg" alt="Send" className="w-full h-full object-contain" />
                </button>
              </div>
            </div>
          </aside>
        </div>

        {/* SQL / queries / metrics panel */}
        <div className="sticky top-6 w-[500px] h-[85vh] flex flex-col border border-gray-700 rounded-xl bg-[#0d0d0d]">

          {/* Tabs */}
          <div className="flex border-b border-gray-700">
            <button
              onClick={() => setRightTab("sql")}
              className={`flex-1 p-2 text-sm ${rightTab === "sql" ? "text-white" : "text-gray-400"}`}
            >
              SQL
            </button>
                     <button
    onClick={() => setRightTab("metrics")}
    className={`flex-1 p-2 text-sm ${rightTab === "metrics" ? "text-white" : "text-gray-400"}`}
  >
    Metrics
  </button>
            <button
              onClick={() => setRightTab("queries")}
              className={`flex-1 p-2 text-sm ${rightTab === "queries" ? "text-white" : "text-gray-400"}`}
            >
              Queries
            </button>
                 <button
              onClick={() => setRightTab("errors")}
              className={`flex-1 p-2 text-sm ${rightTab === "errors" ? "text-white" : "text-gray-400"}`}
            >
              Errors 
            </button>
            
     
     
      
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto custom-scroll p-3">

            {rightTab === "sql" && (
              <div className="space-y-6">
                {lastResult?.map((r: any) => (
                  <div
                    key={r.model_id}
                    className={`border rounded-xl p-4 flex flex-col gap-2 ${
                      activeModel === r.model_id ? "border-blue-500" : "border-gray-700"
                    }`}
                  >
                    {/* modell header */}
                    <div className="flex items-center justify-between shrink-0">
                      <h3 className="text-sm font-semibold text-green-400">
                        {r.model_id}
                      </h3>
                      <div className="flex items-center gap-2 text-xs text-gray-400">
                        <span>{r.attempts} forsøk</span>
                        <span>{r.latency_s}s</span>
                      </div>
                    </div>

                 {/* SQL */}
<div className="overflow-hidden">
  <pre className="text-xs text-white whitespace-pre-wrap break-words w-full">
    {r.sql || r.all_errors?.join("\n")}
  </pre>
</div>

                    {/* knapper */}
                    {r.valid && (
                      <div className="flex gap-2 mt-1">
                        <button
                          onClick={() => showOnMap(r)}
                          className={`px-3 py-1 rounded text-xs border ${
                            activeModel === r.model_id
                              ? "bg-blue-600 border-blue-500 text-white"
                              : "border-gray-600 text-gray-300 hover:bg-zinc-700"
                          }`}
                        >
                          Vis på kart
                        </button>
                        <button
                          onClick={() => showTable(r)}
                          className={`px-3 py-1 rounded text-xs border ${
                            tableModel === r.model_id
                              ? "bg-green-700 border-green-600 text-white"
                              : "border-gray-600 text-gray-300 hover:bg-zinc-700"
                          }`}
                        >
                          Vis tabell
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {rightTab === "queries" && (
              <div className="text-white text-sm space-y-2">
                {messages
                  .filter((msg) => msg.role === "user")
                  .map((msg, i) => (
                    <div key={i} className="text-left">
                      <div className="bg-zinc-800 px-3 py-2 rounded-lg inline-block">
                        {msg.text}
                      </div>
                    </div>
                  ))}
              </div>
            )}

       {rightTab === "metrics" && (
  <div className="space-y-4">

    {/* comparison table — current run */}


    {/* aggregated per model */}
    {lastResult?.map((r: any) => (
      r.aggregated && r.aggregated.total > 1 && (
        <div key={r.model_id} className="border border-gray-700 rounded-xl p-4">
          <h3 className="text-xs font-semibold text-green-400 mb-3">
            {r.model_id} — aggregated ({r.aggregated.total} runs)
          </h3>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="text-gray-400">First-pass</div>
            <div className="text-cyan-300 font-mono">
              {((r.aggregated.first_pass / r.aggregated.total) * 100).toFixed(1)}%
            </div>
            <div className="text-gray-400">Repaired</div>
            <div className="text-cyan-300 font-mono">
              {((r.aggregated.repaired / r.aggregated.total) * 100).toFixed(1)}%
            </div>
            <div className="text-gray-400">Avg latency</div>
            <div className="text-cyan-300 font-mono">
              {(r.aggregated.total_latency / r.aggregated.total).toFixed(2)}s
            </div>
          </div>
        </div>
      )
    ))}

  </div>
)}


{rightTab === "errors" && (
  <div className="space-y-4">

    {/* errors per model — current run */}
    {lastResult?.map((r: any) => (
      <div key={r.model_id} className="border border-gray-700 rounded-xl p-4">
        <h3 className="text-xs font-semibold text-green-400 mb-3">
          {r.model_id}
        </h3>

        {r.all_errors && r.all_errors.length > 0 ? (
          <div className="space-y-2">
            {r.all_errors.map((err: string, i: number) => (
              <div key={i} className="text-xs text-red-400 font-mono bg-zinc-900 px-3 py-2 rounded-lg">
                Attempt {i + 1}: {err}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-gray-500">No errors</p>
        )}

      </div>
    ))}

  </div>
)}







          </div>
        </div>
      </div>
    </div>
  );
}