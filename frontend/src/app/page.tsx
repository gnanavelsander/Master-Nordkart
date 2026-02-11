"use client";

import { useState } from "react";
import type { FeatureCollection, Geometry } from "geojson";
import MapClient from "./components/MapClient";

export default function Page() {
  const [geojson, setGeojson] = useState<FeatureCollection | undefined>();
  const [aoi, setAoi] = useState<Geometry | null>(null);
  const [scrollActive, setScrollActive] = useState(false);


  const ALL_MODELS = ["gpt-35-turbo","gpt-4.1-nano","gpt-4","gpt-4o","gpt-4o-mini"];

  const [selectedModels, setSelectedModels] = useState<string[]>(["gpt-35-turbo"]);

  function handleDraw(aoiGeo: Geometry) {
    setAoi(aoiGeo);
  }

  function toggleModel(model: string) {
  setSelectedModels(prev =>
    prev.includes(model)
      ? prev.filter(m => m !== model)
      : [...prev, model]
  );
}


  return (

    
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Nordkart Agent</h2>



      <div className="flex flex-col items-center gap-6">
        {/* MAP */}
        <div className="w-[800px]">
          <MapClient geojson={geojson} onDraw={handleDraw} />
        </div>

        {/* PANEL */}
        <aside className="w-[800px] rounded-xl border p-4 h-[400px] flex flex-col ">
          {/* Chat messages area */}
          <div className="flex-1 overflow-y-auto mb-4">
            <p className="text-sm text-gray-600">
              Hei! Hvordan kan jeg hjelpe deg?
            </p>
          </div>

  
          {/* Input area */}
          <div className="border-t pt-4">
            <div className="flex gap-2 items-end relative w-full">
              <textarea
                rows={1}
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

              <button className="bg-black  border absolute right-1.5 bottom-1.5 border-gray-300 w-7 h-7 rounded-lg  flex items-center justify-center shrink-0">
                <img
                  src="/logo.svg"
                  alt="Send"
                  className="w-full h-full object-contain"
                />
              </button>
            </div>
          </div>
        </aside>

                  {/* models */}


        <div className="absolute bottom-[200px] right-[400px] flex flex-col gap-3 items-start">
          {ALL_MODELS.map(model => (
            <label key={model} className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={selectedModels.includes(model)}
                onChange={() => toggleModel(model)}
              />
              {model}
            </label>
          ))}
        </div>

      </div>
    </div>
  );
}
