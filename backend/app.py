from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any, Optional
from psycopg_pool import AsyncConnectionPool
from Client.AzureChatOpenAI import run_all
from firebase_admin import firestore
from backend.firebase_admin_init import fs
import os

from backend.mcp_c_s.mcp_client import query_postgis




DATABASE_URL = os.getenv("PGCONN_STRING")

# shared pool across all requests
pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # open pool on startup, close on shutdown
    global pool
    async with AsyncConnectionPool(DATABASE_URL, min_size=2, max_size=10) as p:
        pool = p
        yield



#create  a fastapi instance
app = FastAPI(lifespan=lifespan)



#initiale middelware, which front orgin are allowed  
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)



#test api
@app.get("/")
def health():
    return {"ok": True}




#post methode
@app.post("/query")
async def run(payload: dict[str, Any]):
    question = payload["question"]
    models = payload["models"]
    mode = payload.get("mode", "parallel")
    aoi: Optional[dict[str, Any]] = payload.get("aoi")

    # run models sequentially or in parallel
    if mode == "sequential":
        results = []
        for m in models:
            r = (await run_all(question, [m], pool))[0]
            results.append(r)
    else:
        results = await run_all(question, models, pool)

    # build clean result list, handle exceptions
    clean = []
    for r, m in zip(results, models):
        if isinstance(r, Exception):
            clean.append({
                "model_id":   m,
                "sql":        "",
                "valid":      False,
                "attempts":   0,
                "latency_s":  0,
                "all_errors": [str(r)],
            })
        else:
            clean.append({
                "model_id":   r["model_id"],
                "sql":        r["sql"],
                "valid":      r["valid"],
                "attempts":   r["attempts"],
                "latency_s":  r["latency_s"],
                "all_errors": r["all_errors"],
            })

    # store raw data and update metrics in Firestore
    for r in clean:
        try:
            # store raw evaluation
            fs.collection("evaluations").add({
                "question":   question,
                "model_id":   r["model_id"],
                "valid":      r["valid"],
                "attempts":   r["attempts"],
                "latency_s":  r["latency_s"],
                "sql":        r["sql"],
                "all_errors": r["all_errors"],
                "timestamp":  firestore.SERVER_TIMESTAMP,
            })

            # update aggregated metrics
            ref = fs.collection("metrics").document(r["model_id"])
            ref.set({
                "total":         firestore.Increment(1),
                "first_pass": firestore.Increment(1 if r["valid"] and r["attempts"] == 1 else 0),
                "repaired": firestore.Increment(1 if r["valid"] and r["attempts"] > 1 else 0),
                "total_latency": firestore.Increment(r["latency_s"]),
                "success": firestore.Increment(1 if r["valid"] else 0),
            }, merge=True)

        except Exception as e:
            print(f"Firestore error for {r['model_id']}: {e}")

    # attach aggregated metrics per model
    for r in clean:
        doc = fs.collection("metrics").document(r["model_id"]).get()
        data = doc.to_dict()
        r["aggregated"] = data if data else None

    # pick best model by fewest attempts then lowest latency
    best = sorted(
        [r for r in clean if r["valid"]],
        key=lambda r: (r["attempts"], r["latency_s"])
    )
    best_model = best[0]["model_id"] if best else ""

    return {"best_model": best_model, "results": clean}




import math
import pandas as pd 
import json

def round_coords(coords, decimals=10):
    if isinstance(coords[0], list):
        return [round_coords(c, decimals) for c in coords]
    return [round(c, decimals) for c in coords]


def rows_to_geojson(rows: list[dict]) -> dict:
    features = []
    for row in rows:
        geom_str = row.get("geojson_geom") or row.get("wkt_geom")
        if not geom_str:
            continue

        try:
            geom = json.loads(geom_str)
            # round coordinates to 10 decimals
            if "coordinates" in geom:
                geom["coordinates"] = round_coords(geom["coordinates"])
        except Exception:
            continue

        # all columns except geometry as properties
        props = {k: v for k, v in row.items() if k not in ("geojson_geom", "wkt_geom", "geom")}

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": props
        })

    return {"type": "FeatureCollection", "features": features}


@app.post("/execute")
async def execute(payload: dict[str, Any]):
    sql = payload.get("sql")
    if not sql:
        return {"error": "No SQL provided"}

    try:
        df = await query_postgis(sql)

        # fast NaN handling — pandas handles it internally
        rows = json.loads(df.to_json(orient="records"))
        geojson = rows_to_geojson(rows)

        return {"rows": rows, "geojson": geojson}
    except Exception as e:
        print(f"execute error: {e}")
        return {"error": str(e)}