export type RunMode = "sequential" | "parallel";

export type QueryRequest = {
  question: string;
  models: string[];
  mode: RunMode;
  aoi?: GeoJSON.Polygon | GeoJSON.MultiPolygon;
};

// oppdatert — matcher nå hva backend faktisk returnerer
export type ModelResult = {
  model_id: string;       // var "model"
  sql: string;
  valid: boolean;         // var "accuracy: number"
  attempts: number;       // nytt
  latency_s: number;      // nytt
  all_errors: string[];   // var "errors"
  rows?: Record<string, any>[];     // nytt — rådata for tabell
  geojson?: GeoJSON.FeatureCollection; // nytt — for kart
};

export type QueryResponse = {
  best_model: string;
  results: ModelResult[];
  execution_time_sec?: number;
};

export async function runQuery(payload: QueryRequest): Promise<QueryResponse> {
  const url = process.env.NEXT_PUBLIC_BACKEND_URL!;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Backend error: ${res.status}`);
  }
  return res.json();
}