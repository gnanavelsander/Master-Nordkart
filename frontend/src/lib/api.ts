export type RunMode = "sequential" | "parallel";

export type QueryRequest = {
  question: string;
  models: string[];
  mode: RunMode;
  aoi?: GeoJSON.Polygon | GeoJSON.MultiPolygon;
};

export type ModelResult = {
  model: string;
  sql: string;
  accuracy: number; // 0..1
  errors: string[];
};

export type QueryResponse = {
  best_model: string;
  results: ModelResult[];
  geojson?: GeoJSON.FeatureCollection;
  table?: Record<string, any>[];
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
