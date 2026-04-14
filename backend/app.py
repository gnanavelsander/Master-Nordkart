import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent / "evaluation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVAL_DIR))

from backend.evaluation.pipeline_runner import run_all
from backend.evaluation.agent_helpers import load_prompt
from backend.evaluation.agent_models import ModelParams, AgentRunError, PipelineResult
from backend.mcp.client.client import mcp_client
from config import ALL_MODELS

logger = logging.getLogger(__name__)

_GEOM_KEYS = frozenset({"geom", "geometry", "geometri", "shape"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    try:
        await mcp_client.close()
    except Exception as e:
        logger.warning("mcp_client.close() failed on shutdown: %s", e)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def _rows_to_geojson(result_data: Any) -> dict | None:
    if not isinstance(result_data, dict):
        return None
    rows = result_data.get("rows") or []
    if not rows:
        return None

    features = []
    for row in rows:
        geom_raw = None
        geom_key = None
        for key in row:
            if key.lower() in _GEOM_KEYS:
                geom_raw = row[key]
                geom_key = key
                break
        if geom_raw is None:
            continue
        try:
            geom = json.loads(geom_raw) if isinstance(geom_raw, str) else geom_raw
            if not isinstance(geom, dict):
                continue
            props = {k: v for k, v in row.items() if k != geom_key}
            features.append({"type": "Feature", "geometry": geom, "properties": props})
        except Exception:
            continue

    if not features:
        return None
    return {"type": "FeatureCollection", "features": features}


def _serialise(r: PipelineResult | AgentRunError, model_id: str) -> dict:
    if isinstance(r, AgentRunError):
        snap = r.snapshot or {}
        return {
            "model_id": model_id,
            "final_answer": None,
            "final_sql": snap.get("final_sql"),
            "valid_sql": False,
            "executed": False,
            "task_completed": False,
            "iterations": snap.get("iterations", 0),
            "sql_repairs_used": snap.get("sql_repairs_used", 0),
            "latency_s": 0.0,
            "all_errors": snap.get("all_errors", [str(r)]),
            "result_data": None,
            "geojson": None,
        }

    geojson = _rows_to_geojson(r.result_data)
    return {
        "model_id": r.model_id,
        "final_answer": r.final_answer,
        "final_sql": r.final_sql,
        "valid_sql": r.valid_sql,
        "executed": r.executed,
        "task_completed": r.task_completed,
        "iterations": r.iterations,
        "sql_repairs_used": r.sql_repairs_used,
        "latency_s": r.latency_s,
        "all_errors": r.all_errors or [],
        "result_data": r.result_data,
        "geojson": geojson,
    }


def _pick_best(results: list[dict]) -> str:
    def _score(r: dict) -> tuple:
        return (
            -int(r.get("task_completed", False)),
            -int(r.get("executed", False)),
            -int(r.get("valid_sql", False)),
            r.get("latency_s", 9999),
        )
    candidates = [r for r in results if r.get("task_completed") or r.get("executed") or r.get("valid_sql")]
    pool = candidates or results
    return min(pool, key=_score)["model_id"] if pool else ""


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/")
def health():
    return {"ok": True}


@app.get("/models")
def list_models():
    return {"models": ALL_MODELS}


@app.post("/query")
async def run_query(payload: dict[str, Any]):
    question = (payload.get("question") or "").strip()
    if not question:
        return {"error": "No question provided"}

    model_ids: list[str] = payload.get("models") or ALL_MODELS
    prompt_name: str = payload.get("prompt") or "prompt_v1"

    params = ModelParams(
        temperature=0.0,
        max_tokens=4000,
        timeout=300,
        max_retries=0,
        max_iterations=50,
        max_sql_repairs=3,
        max_restarts=2,
    )

    try:
        prompt = load_prompt(prompt_name)
    except ValueError as e:
        return {"error": str(e)}

    raw_results = await run_all(
        user_input=question,
        model_ids=model_ids,
        params=params,
        prompt=prompt,
    )

    results = [_serialise(r, mid) for r, mid in zip(raw_results, model_ids)]
    best_model = _pick_best(results)

    return {"best_model": best_model, "results": results}
