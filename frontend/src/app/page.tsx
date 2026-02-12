"use client";

import { useEffect, useState } from "react";
import type { FeatureCollection, Geometry } from "geojson";
import MapClient from "./components/MapClient";

import { collection, addDoc, serverTimestamp, getDocs,  query, orderBy,onSnapshot } from "firebase/firestore";


 import { db } from "@/lib/firebase";


export default function Page() {
  const [geojson, setGeojson] = useState<FeatureCollection | undefined>();




  const [aoi, setAoi] = useState<Geometry | null>(null);
  const [scrollActive, setScrollActive] = useState(false);




  /**makeing react hooks to send massage from the user input  */
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);


  const [lastResult, setLastResult] = useState<any>(null);



  const [rightTab, setRightTab] = useState<"sql" | "chat">("sql");



  const [messages, setMessages] = useState<any[]>([]);



  /**take  user input**/
  /**using react  hooks  */
  async function sendToBackend() {
    if (!message.trim()) return;

    setLoading(true);
    try {
      const res  = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: message,
          models: selectedModels,
          mode: "parallel",
          aoi,
        }),
      });
      console.log("status", res.status);

      const data = await res.json ();
      await addDoc(collection(db, "queries"), {
        question: message,
        results: data.results,
        createdAt: serverTimestamp(),
      });

async function loadQueries() {
  const snap = await getDocs(collection(db, "queries"));
  snap.forEach(doc => {
    console.log(doc.id, doc.data());
  });
}

      const combined = data.results
  .map((r: any) =>
    `=== ${r.model} ===\n${r.sql || r.errors?.join("\n")}\n`
  )
  .join("\n");

setLastResult(data.results);


    setMessage("");
  } finally {
    setLoading(false);
  }
}




  const ALL_MODELS = ["gpt-35-turbo","gpt-4.1-nano","gpt-4","gpt-4o","GPT-4o-mini"];

  const [selectedModels, setSelectedModels] = useState<string[]>(["gpt-35-turbo"]);

  function handleDraw(aoiGeo: Geometry) {
    setAoi(aoiGeo);
  }



// const createChat = async () => {
//   const chatRef = await addDoc(collection(db, "chats"), {
//     createdAt: serverTimestamp(),
//   });
// };

useEffect(() => {
  const q = query(
    collection(db, "queries"),
    orderBy("createdAt", "asc")
  );

  const unsubscribe = onSnapshot(q, (snapshot) => {
    const msgs = snapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data(),
    }));
    setMessages(msgs);
  });

  return () => unsubscribe();
}, []);

  function toggleModel(model: string) {
  setSelectedModels(prev =>
    prev.includes(model)
      ? prev.filter(m => m !== model)
      : [...prev, model]
  );
}


return (
  <div className="p-4">
    <h2 className="text-xl font-bold mb-6">Nordkart Agent</h2>

<div className="grid grid-cols-[1fr_700px] gap-3">
      
      {/* map + chat */}
      <div className="flex flex-col items-center gap-6">
        {/* MAP */}
        <div className="w-[800px]">
          <MapClient geojson={geojson} onDraw={handleDraw} />
        </div>

        {/* CHAT */}
        <aside className="w-[800px] rounded-xl border p-4 h-[400px] flex flex-col relative">
          {/* messages */}
          <div className="flex-1 overflow-y-auto mb-4">
            <p className="text-sm text-gray-600">Hei! Hvordan kan jeg hjelpe deg?</p>
          </div>

          {/* MODELS*/}
<div className=" right-4 bottom-16 flex gap-3">
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

{/* SQL / queries */}
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
      onClick={() => setRightTab("chat")}
      className={`flex-1 p-2 text-sm ${rightTab === "chat" ? "text-white" : "text-gray-400"}`}
    >
      queries
    </button>
  </div>

  {/* Content */}
  <div className="flex-1 overflow-y-auto p-3">
    {rightTab === "sql" ? (
      <div className="space-y-6">
        {lastResult?.map((r: any) => (
          <div
            key={r.model}
            className="border border-gray-700 rounded-xl p-4 h-[180px] flex flex-col"
          >
            <h3 className="text-sm font-semibold text-green-400 mb-2 shrink-0">
              {r.model}
            </h3>

            <div className="flex-1 overflow-auto">
              <pre className="text-xs text-white whitespace-pre">
                {r.sql || r.errors?.join("\n")}
              </pre>
            </div>
          </div>
        ))}
      </div>
    ) : (
    <div className="text-white text-sm space-y-2">
  {messages.map((msg) => (
    <div
      key={msg.id}
      className={msg.role === "user" ? "text-right" : "text-left"}
    >
      <div className="bg-zinc-800 px-3 py-2 rounded-lg inline-block">
        {msg.text || msg.question}
      </div>
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
