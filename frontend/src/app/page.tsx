"use client";

import { useEffect, useState } from "react";
import type { Feature, FeatureCollection, Geometry } from "geojson";
import MapClient from "./components/MapClient";
import ResultTable from "./ResultTable";

type ChatMessage = { role: "user" | "agent"; text: string; results?: any[] };
type ConvTurn = { question: string; answer: string | null; sql: string | null; summary: string | null };

const PROMPT_OPTIONS = ["prompt_v1", "prompt_v2", "prompt_v3", "prompt_v4"] as const;
type PromptName = (typeof PROMPT_OPTIONS)[number];

function getOrCreateAssistantSessionId(): string {
  if (typeof window === "undefined") return "demo-session";
  const key = "assistant-demo-session-id";
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;
  const nextId =
    typeof window.crypto !== "undefined" && "randomUUID" in window.crypto
      ? window.crypto.randomUUID()
      : `demo-${Date.now()}`;
  window.localStorage.setItem(key, nextId);
  return nextId;
}

function getAssistantApiBases(): string[] {
  const configured =
    process.env.NEXT_PUBLIC_EVALUATION_API_BASE?.trim() ||
    process.env.NEXT_PUBLIC_ASSISTANT_API_BASE?.trim();
  const bases = new Set<string>();

  if (configured) {
    bases.add(configured.replace(/\/$/, ""));
  }

  if (typeof window !== "undefined") {
    const { protocol, hostname } = window.location;
    bases.add(`${protocol}//${hostname}:8000`);
    bases.add(`${protocol}//localhost:8000`);
  } else {
    bases.add("http://localhost:8000");
  }

  return Array.from(bases);
}

async function fetchAssistant(path: string, init?: RequestInit): Promise<Response> {
  let lastError: unknown = null;

  for (const base of getAssistantApiBases()) {
    try {
      return await fetch(`${base}${path}`, init);
    } catch (error) {
      lastError = error;
    }
  }

  throw new Error(
    lastError instanceof Error
      ? `${lastError.message}\n\nStart evaluation backend with: uvicorn backend.app:app --reload --port 8000`
      : "Could not reach evaluation backend. Start it with: uvicorn backend.app:app --reload --port 8000"
  );
}

export default function Page() {
  const [sessionId, setSessionId] = useState<string>("demo-session");
  const [geojson, setGeojson] = useState<FeatureCollection | undefined>();
  const [aoi, setAoi] = useState<Geometry | null>(null);
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);
  const [scrollActive, setScrollActive] = useState(false);





  /**making react hooks to send message from the user input */
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [lastResult, setLastResult] = useState<any>(null);
  const [rightTab, setRightTab] = useState<"sql" | "prompt" | "log" | "errors">("sql");
  const [promptContent, setPromptContent] = useState<string>("");
  const [promptSaved, setPromptSaved] = useState(false);
  const [selectedPrompt, setSelectedPrompt] = useState<PromptName>("prompt_v4");
  // per-step edited SQL: { [modelId]: { [stepKey]: string } }
  const [editedSql, setEditedSql] = useState<Record<string, Record<string, string>>>({});
  const [expandedStep, setExpandedStep] = useState<Record<string, string | null>>({});







  // chat history — stored locally, not in Firestore
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  // conversation history sent to backend for agent memory
  const [convHistory, setConvHistory] = useState<ConvTurn[]>([]);

  // which model result is shown on map
  const [activeModel, setActiveModel] = useState<string | null>(null);

  // table view per model
  const [tableModel, setTableModel] = useState<string | null>(null);
  const runMode = "assistant";

  function pickPrimaryResult(results: any[], bestModelId?: string | null) {
    if (!results.length) return null;
    return (
      results.find((r: any) => r.final_answer) ??
      results.find((r: any) => r.executed || r.task_completed) ??
      results.find((r: any) => r.valid_sql) ??
      results.find((r: any) => r.model_id === bestModelId) ??
      results[0]
    );
  }





  /**take user input**/
  /**using react hooks */
  async function sendToBackend() {
    if (!message.trim()) return;

    const userMessage = message;

    // add user message to chat
    setMessages((prev) => [...prev, { role: "user", text: userMessage }]);
    setMessage("");
    setGeojson(undefined); // clear old map drawing before new query
    setLoading(true);

    try {
      const res = await fetchAssistant("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userMessage,
          session_id: sessionId,
          models: selectedModels,
          mode: runMode,
          prompt: selectedPrompt,
          aoi,
          selectedFeature,
          history: convHistory,
        }),
      });
      console.log("status", res.status);



      //convert response to jason
      const data = await res.json();
      if (data.session_id) {
        const nextSessionId = String(data.session_id);
        setSessionId(nextSessionId);
        if (typeof window !== "undefined") {
          window.localStorage.setItem("assistant-demo-session-id", nextSessionId);
        }
      }
      if (!res.ok || data.error) {
        const errorText = data.error ? String(data.error) : `Request failed with status ${res.status}`;
        setMessages((prev) => [
          ...prev,
          { role: "agent", text: `Jeg fikk ikke kjørt forespørselen ennå.\n\n${errorText}` },
        ]);
        return;
      }

      const results = Array.isArray(data.results) ? data.results : [];
      setLastResult(results);

      const best = pickPrimaryResult(results, data.best_model);

        // Set chosen result as default on the map 
      if (best) {
        setActiveModel(best.model_id);
        setGeojson(best.geojson || undefined);
      }

      const bestError =
        best?.all_errors && best.all_errors.length > 0 ? String(best.all_errors[0]) : null;
      const bestRanAnything = Boolean(best?.executed || best?.valid_sql || best?.task_completed || best?.iterations);
      const agentText = best?.final_answer
        ? best.final_answer
        : bestError
        ? `Jeg fikk ikke kjørt oppgaven helt ennå.\n\n${bestError}`
        : bestRanAnything
        ? "Jeg jobber videre med dette, men har ikke et fullført svar ennå."
        : "Ingen modell klarte å fullføre oppgaven.";

      // add agent response to chat
      setMessages((prev) => [
        ...prev,
        {
          role: "agent",
          text: agentText,
          results: results,
        },
      ]);

      // update conversation history for agent memory
      setConvHistory((prev) => [
        ...prev,
        {
          question: userMessage,
          answer: best?.final_answer ?? null,
          sql: best?.final_sql ?? null,
          summary: best
            ? `Query executed by ${best.model_id}. Rows returned: ${best.result_data?.row_count ?? "unknown"}.`
            : "No result.",
        },
      ]);
    } catch (error) {
      const errorText = error instanceof Error ? error.message : "Failed to fetch";
      setMessages((prev) => [
        ...prev,
        { role: "agent", text: `Jeg fikk ikke kontakt med assistant-backenden.\n\n${errorText}` },
      ]);
    } finally {
      setLoading(false);
    }
  }










  const [ALL_MODELS, setAllModels] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);

  useEffect(() => {
    setSessionId(getOrCreateAssistantSessionId());
  }, []);

  useEffect(() => {
    if (rightTab === "prompt") {
      fetchAssistant(`/prompt/${selectedPrompt}`)
        .then((r) => r.json())
        .then((data) => setPromptContent(data.content ?? ""))
        .catch(() => {});
    }
  }, [rightTab, selectedPrompt]);

  useEffect(() => {
    fetchAssistant("/models")
      .then((r) => r.json())
      .then((data) => {
        const models: string[] = data.models ?? [];
        setAllModels(models);
        setSelectedModels(models.slice(0, 3));
      })
      .catch(() => {});
  }, []);

  const modelLength = selectedModels.length;


  function handleDraw(aoiGeo: Geometry | null) {
    if (aoiGeo === null) {
      console.log("[AOI] cleared");
      setAoi(null);
    } else {
      console.log("[AOI] drawn geometry:", JSON.stringify(aoiGeo, null, 2));
      setAoi(aoiGeo);
    }
  }

  //Updates the map with that model’s spatial result.
  function showOnMap(r: any) {
    setActiveModel(r.model_id);
    setGeojson(r.geojson || undefined);
    setSelectedFeature(null);
  }

 // states
const [leftView, setLeftView] = useState<"map" | "table">("map");
const [tableData, setTableData] = useState<any[]>([]);
const [tableModelName, setTableModelName] = useState<string>("");

// show tael function
function showTable(r: any) {
    const rows = r.result_data?.rows ?? [];
    setTableData(rows);
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
    <div className="h-screen overflow-hidden bg-black px-6 pt-8 pb-6">
      <h2 className="text-2xl font-bold mb-6 text-white">Nordkart Agent</h2>

      <div className="mx-auto grid h-[calc(100vh-96px)] w-full max-w-[1920px] grid-cols-[minmax(0,1fr)_660px] gap-10 items-start">

        {/* map + chat */}
        <div className="flex h-full flex-col items-center gap-6 pt-2">

       {/* MAP or TABLE */}
<div className="w-[980px] max-w-full flex-1 min-h-0">

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
        <div className="h-full overflow-hidden rounded-2xl border border-zinc-800 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
          <MapClient geojson={geojson} onDraw={handleDraw} onFeatureSelect={setSelectedFeature} />
        </div>
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
          <aside className="w-[980px] max-w-full rounded-2xl border border-zinc-700 bg-[#050505] p-4 h-[450px] flex flex-col relative shrink-0">

            {/* messages */}
<div className="flex-1 overflow-y-auto mb-4 space-y-3 w-full custom-scroll">
                <p className="text-sm text-gray-600">What spatial query would you like to run?</p>
                {selectedFeature && (
                  <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
                    Valgt objekt: {
                      String(
                        (selectedFeature.properties as any)?.name ??
                        (selectedFeature.properties as any)?.navn ??
                        (selectedFeature.properties as any)?.informasjon ??
                        (selectedFeature.properties as any)?.objid ??
                        "ukjent objekt"
                      )
                    }
                  </div>
                )}

              {messages.map((msg, i) => (
                <div key={i} className="text-left">
                  <div className={`px-3 py-2 rounded-lg inline-block text-sm whitespace-pre-wrap ${msg.role === "user" ? "bg-zinc-700 text-gray-200" : "bg-zinc-900 text-white"}`}>
                    {msg.text}
                  </div>
                </div>
))}

{loading && (
  <p className="text-xs text-gray-500 animate-pulse">{modelLength>1?"Running models...":"Running model ..."}</p>
)}
</div>
            {/* MODELS */}
            <div className="right-4 bottom-16 flex flex-wrap gap-6">
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
        <div className="sticky top-8 justify-self-end w-[660px] min-w-0 h-full overflow-hidden flex flex-col border border-gray-700 rounded-2xl bg-[#0d0d0d]">

          {/* Tabs */}
          <div className="flex border-b border-gray-700">
            {(["sql", "prompt", "log", "errors"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setRightTab(tab)}
                className={`flex-1 p-2 text-sm ${rightTab === tab ? "text-white border-b border-white" : "text-gray-400"}`}
              >
                {tab === "sql" ? "Steps" : tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden custom-scroll p-3">

            {/* STEPS TAB */}
            {rightTab === "sql" && (
              <div className="space-y-6">
                {lastResult?.map((r: any) => (
                  <div key={r.model_id} className={`border rounded-xl p-3 flex flex-col gap-2 ${activeModel === r.model_id ? "border-blue-500" : "border-gray-700"}`}>
                    {/* model header */}
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-semibold text-green-400">{r.model_id}</h3>
                      <div className="flex items-center gap-2 text-xs text-gray-400">
                        <span>{r.iterations} iter</span>
                        <span>{r.latency_s}s</span>
                        {r.task_completed && <span className="text-green-400">✓</span>}
                      </div>
                    </div>

                    {/* Steps */}
                    {r.steps && r.steps.length > 0 ? (
                      <div className="space-y-2">
                        {r.steps.map((step: any) => {
                          const stepKey = `${r.model_id}-${step.step}`;
                          const isExpanded = expandedStep[r.model_id] === String(step.step);
                          const sql = editedSql[r.model_id]?.[step.step] ?? step.sql ?? "";
                          const statusColor = step.status === "ok" ? "text-green-400" : step.status === "error" ? "text-red-400" : "text-yellow-400";
                          return (
                            <div key={stepKey} className="border border-gray-700 rounded-lg overflow-hidden">
                              {/* step header — click to expand */}
                              <button
                                className="w-full flex items-center gap-2 px-3 py-2 text-xs hover:bg-zinc-800 text-left"
                                onClick={() => setExpandedStep(prev => ({ ...prev, [r.model_id]: isExpanded ? null : String(step.step) }))}
                              >
                                <span className="bg-zinc-700 text-gray-300 rounded px-1.5 py-0.5 font-mono">#{step.step}</span>
                                <span className="text-blue-400 font-mono">{step.action ?? "?"}</span>
                                {step.database && <span className="text-gray-500">{step.database}</span>}
                                <span className={`ml-auto font-mono ${statusColor}`}>{step.status ?? "pending"}</span>
                                {step.rows_returned != null && <span className="text-gray-400">{step.rows_returned} rows</span>}
                                {step.sql_repairs > 0 && <span className="text-yellow-500">{step.sql_repairs} repair</span>}
                              </button>

                              {/* expanded: editable SQL */}
                              {isExpanded && (
                                <div className="border-t border-gray-700 p-2 flex flex-col gap-2">
                                  <textarea
                                    className="w-full text-xs font-mono text-white bg-zinc-900 border border-gray-600 rounded p-2 resize-y min-h-[100px]"
                                    value={sql}
                                    onChange={(e) => setEditedSql(prev => ({
                                      ...prev,
                                      [r.model_id]: { ...(prev[r.model_id] ?? {}), [step.step]: e.target.value }
                                    }))}
                                  />
                                  {step.error && (
                                    <p className="text-xs text-red-400 font-mono bg-zinc-900 px-2 py-1 rounded">{step.error}</p>
                                  )}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <p className="text-xs text-gray-500">Ingen steg registrert.</p>
                    )}

                    {/* action buttons */}
                    {(r.executed || r.valid_sql) && (
                      <div className="flex gap-2 mt-1">
                        <button onClick={() => showOnMap(r)} className={`px-3 py-1 rounded text-xs border ${activeModel === r.model_id ? "bg-blue-600 border-blue-500 text-white" : "border-gray-600 text-gray-300 hover:bg-zinc-700"}`}>
                          Vis på kart
                        </button>
                        <button onClick={() => showTable(r)} className={`px-3 py-1 rounded text-xs border ${tableModel === r.model_id ? "bg-green-700 border-green-600 text-white" : "border-gray-600 text-gray-300 hover:bg-zinc-700"}`}>
                          Vis tabell
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* PROMPT TAB */}
            {rightTab === "prompt" && (
              <div className="flex flex-col gap-3 h-full">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400">Prompt</span>
                    <select
                      value={selectedPrompt}
                      onChange={(e) => {
                        setSelectedPrompt(e.target.value as PromptName);
                        setPromptSaved(false);
                      }}
                      className="rounded border border-gray-700 bg-zinc-900 px-2 py-1 text-xs text-white"
                    >
                      {PROMPT_OPTIONS.map((promptName) => (
                        <option key={promptName} value={promptName}>
                          {promptName}.txt
                        </option>
                      ))}
                    </select>
                  </div>
                  <button
                    onClick={async () => {
                      await fetchAssistant(`/prompt/${selectedPrompt}`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ content: promptContent }),
                      });
                      setPromptSaved(true);
                      setTimeout(() => setPromptSaved(false), 2000);
                    }}
                    className="px-3 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded"
                  >
                    {promptSaved ? "Lagret ✓" : "Lagre"}
                  </button>
                </div>
                <textarea
                  className="flex-1 w-full text-xs font-mono text-white bg-zinc-900 border border-gray-700 rounded p-3 resize-none min-h-[500px]"
                  value={promptContent}
                  onChange={(e) => setPromptContent(e.target.value)}
                />
              </div>
            )}

            {/* LOG TAB */}
            {rightTab === "log" && (
              <div className="h-full min-w-0 space-y-4">
                {!lastResult && <p className="text-xs text-gray-500">Ingen kjøring ennå.</p>}
                {lastResult?.map((r: any) => (
                  <div key={r.model_id} className="w-full min-w-0 overflow-hidden rounded-xl border border-gray-800 bg-zinc-950/50 p-3">
                    {r.full_log ? (
                      <pre className="w-full min-w-0 overflow-x-auto whitespace-pre-wrap break-all text-xs font-mono leading-relaxed text-gray-300">
                        {r.full_log}
                      </pre>
                    ) : (
                      <p className="text-xs text-gray-500">Ingen logg tilgjengelig.</p>
                    )}
                  </div>
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
