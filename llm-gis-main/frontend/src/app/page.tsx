"use client";

import { useEffect, useState } from "react";
import type { Feature, FeatureCollection, Geometry } from "geojson";
import MapClient from "./components/MapClient";
import ResultTable from "./ResultTable";

type ChatMessage = { role: "user" | "agent"; text: string; results?: any[] };
type ConvTurn = {
  question: string;
  answer: string | null;
  sql: string | null;
  summary: string | null;
};

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

  const [message, setMessage] = useState("");
  const [scrollActive, setScrollActive] = useState(false);
  const [loading, setLoading] = useState(false);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [convHistory, setConvHistory] = useState<ConvTurn[]>([]);

  const [lastResult, setLastResult] = useState<any[] | null>(null);
  const [activeModel, setActiveModel] = useState<string | null>(null);

  const [rightPanelOpen, setRightPanelOpen] = useState(false);
  const [rightTab, setRightTab] = useState<"sql" | "prompt" | "log" | "errors">("sql");
  const [promptContent, setPromptContent] = useState<string>("");
  const [promptSaved, setPromptSaved] = useState(false);
  const [selectedPrompt, setSelectedPrompt] = useState<PromptName>("prompt_v4");

  const [editedSql, setEditedSql] = useState<Record<string, Record<string, string>>>({});
  const [expandedStep, setExpandedStep] = useState<Record<string, string | null>>({});

  const [allModels, setAllModels] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);

  const [leftView, setLeftView] = useState<"map" | "table">("map");
  const [tableData, setTableData] = useState<any[]>([]);
  const [tableModelName, setTableModelName] = useState<string>("");

  const runMode = "assistant";
  const modelLength = selectedModels.length;

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

  function handleDraw(aoiGeo: Geometry | null) {
    if (aoiGeo === null) {
      console.log("[AOI] cleared");
      setAoi(null);
    } else {
      console.log("[AOI] drawn geometry:", JSON.stringify(aoiGeo, null, 2));
      setAoi(aoiGeo);
    }
  }

  function showOnMap(r: any) {
    setActiveModel(r.model_id);
    setGeojson(r.geojson || undefined);
    setSelectedFeature(null);
    setLeftView("map");
  }

  function showTable(r: any) {
    const rows = r.result_data?.rows ?? r.rows ?? [];
    setTableData(rows);
    setTableModelName(r.model_id);
    setLeftView("table");
  }

  function toggleModel(model: string) {
    setSelectedModels((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model]
    );
  }

  async function sendToBackend() {
    if (!message.trim()) return;

    const userMessage = message;

    setMessages((prev) => [...prev, { role: "user", text: userMessage }]);
    setMessage("");
    setGeojson(undefined);
    setLoading(true);
    setLeftView("map");

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
          {
            role: "agent",
            text: `Jeg fikk ikke kjørt forespørselen ennå.\n\n${errorText}`,
          },
        ]);

        return;
      }

      const results = Array.isArray(data.results) ? data.results : [];
      setLastResult(results);

      const best = pickPrimaryResult(results, data.best_model);

      if (best) {
        setActiveModel(best.model_id);
        setGeojson(best.geojson || undefined);
      }

      const bestError =
        best?.all_errors && best.all_errors.length > 0 ? String(best.all_errors[0]) : null;

      const bestRanAnything = Boolean(
        best?.executed || best?.valid_sql || best?.task_completed || best?.iterations
      );

      const agentText = best?.final_answer
        ? best.final_answer
        : bestError
        ? `Jeg fikk ikke kjørt oppgaven helt ennå.\n\n${bestError}`
        : bestRanAnything
        ? "Jeg jobber videre med dette, men har ikke et fullført svar ennå."
        : "Ingen modell klarte å fullføre oppgaven.";

      setMessages((prev) => [
        ...prev,
        {
          role: "agent",
          text: agentText,
          results,
        },
      ]);

      setConvHistory((prev) => [
        ...prev,
        {
          question: userMessage,
          answer: best?.final_answer ?? null,
          sql: best?.final_sql ?? null,
          summary: best
            ? `Query executed by ${best.model_id}. Rows returned: ${
                best.result_data?.row_count ?? "unknown"
              }.`
            : "No result.",
        },
      ]);
    } catch (error) {
      const errorText = error instanceof Error ? error.message : "Failed to fetch";

      setMessages((prev) => [
        ...prev,
        {
          role: "agent",
          text: `Jeg fikk ikke kontakt med assistant-backenden.\n\n${errorText}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

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

  const activeResult = lastResult?.find((r: any) => r.model_id === activeModel) ?? lastResult?.[0];

  return (
    <main className="relative h-screen w-screen overflow-hidden bg-[#F4FBF7] text-[#034E31]">
      {/* Map layer */}
      <div className="absolute inset-0">
        <MapClient geojson={geojson} onDraw={handleDraw} onFeatureSelect={setSelectedFeature} />
      </div>

      {/* Soft overlay for readability */}
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-white/25 via-transparent to-white/10" />

      {/* Top navbar */}
      <header className="absolute left-6 right-6 top-5 z-[900] flex items-center justify-between rounded-full border border-[#034E31]/10 bg-white/90 px-5 py-3 shadow-[0_18px_55px_rgba(3,78,49,0.14)] backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[#034E31] text-sm font-bold text-white">
            N
          </div>

          <div>
            <h1 className="text-sm font-bold leading-tight text-[#034E31]">Norkart Agent</h1>
            <p className="text-xs text-[#034E31]/60">Natural-language GIS assistant</p>
          </div>
        </div>

        <nav className="hidden items-center gap-2 md:flex">
          <button className="rounded-full bg-[#C8EBDC]/60 px-4 py-2 text-xs font-semibold text-[#034E31]">
            Map
          </button>
          <button className="rounded-full px-4 py-2 text-xs font-medium text-[#034E31]/65 hover:bg-[#C8EBDC]/40">
            Evaluation
          </button>
          <button className="rounded-full px-4 py-2 text-xs font-medium text-[#034E31]/65 hover:bg-[#C8EBDC]/40">
            History
          </button>
        </nav>

        <div className="flex items-center gap-2">
          <span className="hidden rounded-full border border-[#034E31]/10 bg-white px-3 py-2 text-xs text-[#034E31]/70 sm:inline">
            {selectedPrompt}
          </span>

          <div className="h-2.5 w-2.5 rounded-full bg-[#24BD76] shadow-[0_0_0_6px_rgba(36,189,118,0.16)]" />
        </div>
      </header>

      {/* Table overlay */}
      {leftView === "table" && (
        <section className="absolute bottom-8 left-8 top-24 z-[850] w-[min(980px,calc(100vw-96px))]">
          <ResultTable
            modelId={tableModelName}
            rows={tableData}
            onClose={() => setLeftView("map")}
          />
        </section>
      )}

      {/* Left session card */}
      <section className="absolute left-6 top-24 z-[850] hidden w-[310px] rounded-[28px] border border-[#034E31]/10 bg-white/90 p-4 shadow-[0_20px_60px_rgba(3,78,49,0.16)] backdrop-blur xl:block">
        <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#24BD76]">
          Active session
        </p>

        <h2 className="mt-2 text-lg font-bold text-[#034E31]">GIS workflow</h2>

        <p className="mt-2 text-sm leading-relaxed text-[#034E31]/65">
          Ask a spatial question, select models, inspect reasoning steps, and view the result on the
          map.
        </p>

        <div className="mt-4 grid grid-cols-3 gap-2">
          <div className="rounded-2xl bg-[#F4FBF7] p-3">
            <p className="text-[10px] uppercase tracking-wide text-[#034E31]/50">Models</p>
            <p className="mt-1 text-lg font-bold">{selectedModels.length}</p>
          </div>

          <div className="rounded-2xl bg-[#F4FBF7] p-3">
            <p className="text-[10px] uppercase tracking-wide text-[#034E31]/50">Runs</p>
            <p className="mt-1 text-lg font-bold">{lastResult?.length ?? 0}</p>
          </div>

          <div className="rounded-2xl bg-[#F4FBF7] p-3">
            <p className="text-[10px] uppercase tracking-wide text-[#034E31]/50">AOI</p>
            <p className="mt-1 text-lg font-bold">{aoi ? "✓" : "–"}</p>
          </div>
        </div>

        {selectedFeature && (
          <div className="mt-4 rounded-2xl border border-[#24BD76]/25 bg-[#C8EBDC]/40 p-3">
            <p className="text-[10px] font-semibold uppercase tracking-wide text-[#034E31]/60">
              Selected feature
            </p>
            <p className="mt-1 truncate text-sm font-semibold text-[#034E31]">
              {String(
                (selectedFeature.properties as any)?.name ??
                  (selectedFeature.properties as any)?.navn ??
                  (selectedFeature.properties as any)?.informasjon ??
                  (selectedFeature.properties as any)?.objid ??
                  "Unknown object"
              )}
            </p>
          </div>
        )}
      </section>

      {/* Floating command area */}
      <section className="absolute bottom-6 left-1/2 z-[900] w-[min(920px,calc(100vw-48px))] -translate-x-1/2">
        <div className="rounded-[32px] border border-[#034E31]/10 bg-white/92 p-4 shadow-[0_24px_80px_rgba(3,78,49,0.22)] backdrop-blur">
          <div className="mb-3 flex max-h-[150px] flex-col gap-2 overflow-y-auto custom-scroll">
            {messages.length === 0 && (
              <div>
                <p className="text-sm font-semibold text-[#034E31]">
                  What spatial query would you like to run?
                </p>
                <p className="mt-1 text-xs text-[#034E31]/55">
                  Example: Count all buildings in Kristiansand municipality.
                </p>
              </div>
            )}

            {messages.slice(-4).map((msg, i) => (
              <div
                key={i}
                className={`max-w-[85%] rounded-2xl px-4 py-2 text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "ml-auto bg-[#034E31] text-white"
                    : "mr-auto bg-[#F4FBF7] text-[#034E31]"
                }`}
              >
                <p className="whitespace-pre-wrap">{msg.text}</p>
              </div>
            ))}

            {loading && (
              <p className="text-xs font-medium text-[#24BD76] animate-pulse">
                {modelLength > 1 ? "Running models..." : "Running model..."}
              </p>
            )}
          </div>

          <div className="mb-3 flex flex-wrap gap-2">
            {allModels.map((model) => {
              const selected = selectedModels.includes(model);

              return (
                <button
                  key={model}
                  onClick={() => toggleModel(model)}
                  className={`rounded-full border px-3 py-1.5 text-xs font-medium transition ${
                    selected
                      ? "border-[#24BD76] bg-[#C8EBDC]/70 text-[#034E31]"
                      : "border-[#034E31]/10 bg-white text-[#034E31]/55 hover:bg-[#F4FBF7]"
                  }`}
                >
                  {selected ? "✓ " : ""}
                  {model}
                </button>
              );
            })}
          </div>

          <div className="flex items-end gap-3 rounded-[24px] border border-[#034E31]/10 bg-[#F4FBF7] p-2">
            <textarea
              rows={1}
              value={message}
              onChange={(e) => {
                setMessage(e.target.value);

                const el = e.currentTarget;
                const maxHeight = 180;
                el.style.height = "auto";

                const next = Math.min(el.scrollHeight, maxHeight);
                el.style.height = next + "px";
                setScrollActive(el.scrollHeight > maxHeight);
              }}
              onInput={(e) => {
                const el = e.currentTarget;
                const maxHeight = 180;
                el.style.height = "auto";

                const next = Math.min(el.scrollHeight, maxHeight);
                el.style.height = next + "px";
                setScrollActive(el.scrollHeight > maxHeight);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendToBackend();
                }
              }}
              placeholder="Ask a GIS question..."
              className={[
                "min-h-12 flex-1 resize-none bg-transparent px-4 py-3 text-sm text-[#034E31]",
                "placeholder:text-[#034E31]/45 focus:outline-none",
                scrollActive ? "overflow-y-auto" : "overflow-y-hidden",
              ].join(" ")}
            />

            <button
              onClick={sendToBackend}
              disabled={loading || !message.trim()}
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[#24BD76] text-lg font-bold text-white shadow-[0_10px_25px_rgba(36,189,118,0.35)] transition hover:bg-[#20aa69] disabled:cursor-not-allowed disabled:opacity-50"
              title="Send"
            >
              ↑
            </button>
          </div>
        </div>
      </section>

      {/* Button to open right panel */}
      {!rightPanelOpen && (
        <button
          onClick={() => setRightPanelOpen(true)}
          className="absolute right-6 top-24 z-[920] rounded-full border border-[#034E31]/10 bg-white/95 px-5 py-3 text-sm font-semibold text-[#034E31] shadow-[0_18px_55px_rgba(3,78,49,0.18)] backdrop-blur transition hover:bg-[#F4FBF7]"
        >
          Reasoning panel
        </button>
      )}

      {/* Right intelligence panel */}
      {rightPanelOpen && (
        <aside className="absolute bottom-6 right-6 top-24 z-[900] flex w-[500px] flex-col overflow-hidden rounded-[32px] border border-[#034E31]/10 bg-white/94 shadow-[0_24px_80px_rgba(3,78,49,0.20)] backdrop-blur">
          <div className="border-b border-[#034E31]/10 p-5">
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#24BD76]">
              Intelligence panel
            </p>

            <div className="mt-2 flex items-start justify-between gap-4">
              <div>
                <h2 className="text-xl font-bold text-[#034E31]">Reasoning & results</h2>
                <p className="mt-1 text-sm text-[#034E31]/60">
                  Inspect model steps, SQL, logs and errors.
                </p>
              </div>

              <button
                onClick={() => setRightPanelOpen(false)}
                className="rounded-full border border-[#034E31]/10 bg-white px-3 py-2 text-xs font-semibold text-[#034E31] transition hover:bg-[#C8EBDC]/40"
              >
                Close
              </button>
            </div>

            {activeResult && (
              <div className="mt-4 rounded-2xl bg-[#F4FBF7] px-4 py-3">
                <p className="text-[10px] uppercase tracking-wide text-[#034E31]/50">
                  Active model
                </p>
                <p className="mt-1 truncate text-xs font-semibold text-[#034E31]">
                  {activeResult.model_id}
                </p>
              </div>
            )}
          </div>

          <div className="flex border-b border-[#034E31]/10 px-3">
            {(["sql", "prompt", "log", "errors"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setRightTab(tab)}
                className={`flex-1 border-b-2 px-2 py-3 text-sm font-medium transition ${
                  rightTab === tab
                    ? "border-[#24BD76] text-[#034E31]"
                    : "border-transparent text-[#034E31]/45 hover:text-[#034E31]"
                }`}
              >
                {tab === "sql" ? "Steps" : tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden p-4 custom-scroll">
            {rightTab === "sql" && (
              <div className="space-y-4">
                {!lastResult && (
                  <div className="rounded-3xl border border-dashed border-[#034E31]/15 bg-[#F4FBF7]/70 p-5">
                    <p className="text-sm font-semibold text-[#034E31]">No run yet</p>
                    <p className="mt-1 text-sm text-[#034E31]/60">
                      Ask a question to see model steps, SQL validation and execution results.
                    </p>
                  </div>
                )}

                {lastResult?.map((r: any) => (
                  <div
                    key={r.model_id}
                    className={`rounded-3xl border p-4 transition ${
                      activeModel === r.model_id
                        ? "border-[#24BD76] bg-[#F4FBF7]"
                        : "border-[#034E31]/10 bg-white"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <h3 className="truncate text-sm font-bold text-[#034E31]">{r.model_id}</h3>
                        <p className="mt-1 text-xs text-[#034E31]/55">
                          {r.iterations ?? "–"} iterations
                          {r.latency_s != null ? ` · ${r.latency_s}s` : ""}
                        </p>
                      </div>

                      <div className="flex shrink-0 items-center gap-2">
                        {r.task_completed ? (
                          <span className="rounded-full bg-[#C8EBDC] px-2 py-1 text-xs font-semibold text-[#034E31]">
                            Completed
                          </span>
                        ) : (
                          <span className="rounded-full bg-amber-100 px-2 py-1 text-xs font-semibold text-amber-700">
                            Incomplete
                          </span>
                        )}
                      </div>
                    </div>

                    {r.final_answer && (
                      <div className="mt-3 rounded-2xl bg-white p-3 text-sm leading-relaxed text-[#034E31]/80">
                        {r.final_answer}
                      </div>
                    )}

                    {r.steps && r.steps.length > 0 ? (
                      <div className="mt-4 space-y-2">
                        {r.steps.map((step: any) => {
                          const stepKey = `${r.model_id}-${step.step}`;
                          const isExpanded = expandedStep[r.model_id] === String(step.step);
                          const sql = editedSql[r.model_id]?.[step.step] ?? step.sql ?? "";

                          const statusClass =
                            step.status === "ok"
                              ? "bg-[#C8EBDC] text-[#034E31]"
                              : step.status === "error"
                              ? "bg-red-100 text-red-700"
                              : "bg-amber-100 text-amber-700";

                          return (
                            <div
                              key={stepKey}
                              className="overflow-hidden rounded-2xl border border-[#034E31]/10 bg-white"
                            >
                              <button
                                className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs hover:bg-[#F4FBF7]"
                                onClick={() =>
                                  setExpandedStep((prev) => ({
                                    ...prev,
                                    [r.model_id]: isExpanded ? null : String(step.step),
                                  }))
                                }
                              >
                                <span className="rounded-full bg-[#034E31] px-2 py-1 font-mono text-white">
                                  #{step.step}
                                </span>

                                <span className="font-mono font-semibold text-[#24BD76]">
                                  {step.action ?? "?"}
                                </span>

                                {step.database && (
                                  <span className="truncate text-[#034E31]/45">
                                    {step.database}
                                  </span>
                                )}

                                <span
                                  className={`ml-auto rounded-full px-2 py-1 font-semibold ${statusClass}`}
                                >
                                  {step.status ?? "pending"}
                                </span>
                              </button>

                              {isExpanded && (
                                <div className="border-t border-[#034E31]/10 p-3">
                                  <textarea
                                    className="min-h-[110px] w-full resize-y rounded-2xl border border-[#034E31]/10 bg-[#F4FBF7] p-3 font-mono text-xs text-[#034E31] focus:outline-none focus:ring-2 focus:ring-[#24BD76]/30"
                                    value={sql}
                                    onChange={(e) =>
                                      setEditedSql((prev) => ({
                                        ...prev,
                                        [r.model_id]: {
                                          ...(prev[r.model_id] ?? {}),
                                          [step.step]: e.target.value,
                                        },
                                      }))
                                    }
                                  />

                                  {step.error && (
                                    <p className="mt-2 rounded-2xl bg-red-50 px-3 py-2 font-mono text-xs text-red-700">
                                      {step.error}
                                    </p>
                                  )}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <p className="mt-3 text-xs text-[#034E31]/50">No steps registered.</p>
                    )}

                    {(r.executed || r.valid_sql) && (
                      <div className="mt-4 flex gap-2">
                        <button
                          onClick={() => showOnMap(r)}
                          className={`rounded-full border px-4 py-2 text-xs font-semibold transition ${
                            activeModel === r.model_id
                              ? "border-[#24BD76] bg-[#24BD76] text-white"
                              : "border-[#034E31]/10 bg-white text-[#034E31] hover:bg-[#F4FBF7]"
                          }`}
                        >
                          Show map
                        </button>

                        <button
                          onClick={() => showTable(r)}
                          className="rounded-full border border-[#034E31]/10 bg-white px-4 py-2 text-xs font-semibold text-[#034E31] transition hover:bg-[#F4FBF7]"
                        >
                          Show table
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {rightTab === "prompt" && (
              <div className="flex h-full flex-col gap-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-[#034E31]/60">Prompt</span>

                    <select
                      value={selectedPrompt}
                      onChange={(e) => {
                        setSelectedPrompt(e.target.value as PromptName);
                        setPromptSaved(false);
                      }}
                      className="rounded-full border border-[#034E31]/10 bg-[#F4FBF7] px-3 py-2 text-xs text-[#034E31] focus:outline-none"
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
                    className="rounded-full bg-[#24BD76] px-4 py-2 text-xs font-semibold text-white hover:bg-[#20aa69]"
                  >
                    {promptSaved ? "Saved ✓" : "Save"}
                  </button>
                </div>

                <textarea
                  className="min-h-[500px] flex-1 resize-none rounded-3xl border border-[#034E31]/10 bg-[#F4FBF7] p-4 font-mono text-xs leading-relaxed text-[#034E31] focus:outline-none focus:ring-2 focus:ring-[#24BD76]/30"
                  value={promptContent}
                  onChange={(e) => setPromptContent(e.target.value)}
                />
              </div>
            )}

            {rightTab === "log" && (
              <div className="space-y-4">
                {!lastResult && <p className="text-sm text-[#034E31]/50">No run yet.</p>}

                {lastResult?.map((r: any) => (
                  <div
                    key={r.model_id}
                    className="overflow-hidden rounded-3xl border border-[#034E31]/10 bg-[#F4FBF7] p-4"
                  >
                    <p className="mb-2 text-xs font-bold text-[#034E31]">{r.model_id}</p>

                    {r.full_log ? (
                      <pre className="overflow-x-auto whitespace-pre-wrap break-all font-mono text-xs leading-relaxed text-[#034E31]/75">
                        {r.full_log}
                      </pre>
                    ) : (
                      <p className="text-xs text-[#034E31]/50">No log available.</p>
                    )}
                  </div>
                ))}
              </div>
            )}

            {rightTab === "errors" && (
              <div className="space-y-4">
                {!lastResult && <p className="text-sm text-[#034E31]/50">No run yet.</p>}

                {lastResult?.map((r: any) => (
                  <div
                    key={r.model_id}
                    className="rounded-3xl border border-[#034E31]/10 bg-white p-4"
                  >
                    <h3 className="mb-3 text-sm font-bold text-[#034E31]">{r.model_id}</h3>

                    {r.all_errors && r.all_errors.length > 0 ? (
                      <div className="space-y-2">
                        {r.all_errors.map((err: string, i: number) => (
                          <div
                            key={i}
                            className="rounded-2xl bg-red-50 px-3 py-2 font-mono text-xs text-red-700"
                          >
                            Attempt {i + 1}: {err}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-xs text-[#034E31]/50">No errors</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>
      )}
    </main>
  );
}