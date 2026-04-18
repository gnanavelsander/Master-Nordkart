import hashlib
import json
import logging
import re
import sys
import time as _time
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
from backend.evaluation.agent_helpers import load_prompt, _short, _dedup
from backend.evaluation.agent_models import ModelParams, AgentRunError, PipelineResult
from backend.evaluation.reporting import (
    extract_agent_trace,
    _active_errors_for_result,
    _collect_step_stats,
    _failed_step_summary,
    _step_issue_summaries,
    _sorted_step_items,
)
from backend.mcp.client.client import mcp_client
from config import ALL_MODELS

logger = logging.getLogger(__name__)

_GEOM_KEYS = frozenset({
    "geom", "geometry", "geometri", "shape", "the_geom", "wkb_geometry", "geom_4326",
    # common PostGIS geometry column names used in Norwegian GIS data
    "omrade", "grense", "senterlinje", "vannvei",
    # agent-generated alias from ST_AsGeoJSON(...)
    "geom_geojson", "geometry_geojson",
})
_PROMPTS_DIR = Path(__file__).resolve().parent / "evaluation" / "prompts"

# ─────────────────────────────────────────
# Query result cache
# ─────────────────────────────────────────
_QUERY_CACHE: dict[str, dict] = {}
_CACHE_TTL_S = 1800   # 30 minutes
_CACHE_MAX   = 200    # max entries before LRU eviction


def _cache_key(question: str, aoi: Any, prompt_name: str, model_ids: list[str]) -> str:
    parts = [
        question.lower().strip(),
        json.dumps(aoi, sort_keys=True, ensure_ascii=False) if aoi else "",
        prompt_name,
        ",".join(sorted(model_ids)),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _cache_get(key: str) -> dict | None:
    entry = _QUERY_CACHE.get(key)
    if not entry:
        return None
    if _time.time() - entry["ts"] > _CACHE_TTL_S:
        _QUERY_CACHE.pop(key, None)
        return None
    return entry["result"]


def _cache_set(key: str, result: dict) -> None:
    if len(_QUERY_CACHE) >= _CACHE_MAX:
        oldest = min(_QUERY_CACHE, key=lambda k: _QUERY_CACHE[k]["ts"])
        _QUERY_CACHE.pop(oldest, None)
    _QUERY_CACHE[key] = {"result": result, "ts": _time.time()}


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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

import struct as _struct


def _srid_from_ewkb(data: bytes) -> int | None:
    """Extract SRID from an EWKB byte string (PostGIS extended WKB)."""
    if len(data) < 9:
        return None
    try:
        byte_order = data[0]
        endian = "<" if byte_order == 1 else ">"
        wkb_type = _struct.unpack(f"{endian}I", data[1:5])[0]
        if wkb_type & 0x20000000:  # SRID flag
            return _struct.unpack(f"{endian}I", data[5:9])[0]
    except Exception:
        pass
    return None


def _shp_to_4326_geojson(shp: Any, srid: int | None) -> dict | None:
    """Convert a shapely geometry to a GeoJSON dict, transforming to EPSG:4326 when needed."""
    try:
        from shapely.geometry import mapping
        if srid and srid != 4326:
            from pyproj import Transformer
            from shapely.ops import transform
            t = Transformer.from_crs(f"EPSG:{srid}", "EPSG:4326", always_xy=True)
            shp = transform(t.transform, shp)
        return json.loads(json.dumps(mapping(shp)))
    except Exception:
        return None


def _geom_raw_to_dict(geom_raw: Any) -> dict | None:
    """Convert any geometry representation to a GeoJSON dict in EPSG:4326."""
    if geom_raw is None:
        return None
    # Already a dict (GeoJSON) — assumed 4326 when returned by ST_AsGeoJSON
    if isinstance(geom_raw, dict):
        return geom_raw
    # JSON string produced by ST_AsGeoJSON — already 4326
    if isinstance(geom_raw, str):
        s = geom_raw.strip()
        if s.startswith("{"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        # WKB hex (EWKB from asyncpg) or WKT — must transform to 4326
        try:
            from shapely import wkb as shapely_wkb, wkt as shapely_wkt
            hex_s = s[2:] if s.startswith(("\\x", "0x")) else s
            if len(hex_s) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in hex_s):
                try:
                    raw = bytes.fromhex(hex_s)
                    shp = shapely_wkb.loads(raw)
                    srid = _srid_from_ewkb(raw)
                    return _shp_to_4326_geojson(shp, srid)
                except Exception:
                    pass
            try:
                shp = shapely_wkt.loads(s)
                return _shp_to_4326_geojson(shp, None)
            except Exception:
                pass
        except ImportError:
            pass
        return None
    # Bytes / memoryview → EWKB from asyncpg
    if isinstance(geom_raw, (bytes, memoryview)):
        try:
            from shapely import wkb as shapely_wkb
            raw = bytes(geom_raw) if isinstance(geom_raw, memoryview) else geom_raw
            shp = shapely_wkb.loads(raw)
            srid = _srid_from_ewkb(raw)
            return _shp_to_4326_geojson(shp, srid)
        except Exception:
            pass
    # Shapely geometry object — CRS unknown, return as-is
    if hasattr(geom_raw, "geom_type"):
        try:
            return _shp_to_4326_geojson(geom_raw, None)
        except Exception:
            pass
    return None


_MAP_FEATURE_LIMIT = 2000  # max features sent to browser for map rendering


def _rows_to_geojson(result_data: Any) -> dict | None:
    if not isinstance(result_data, dict):
        return None
    rows = result_data.get("rows") or []
    if not rows:
        return None

    total_rows = len(rows)
    # prefer the geom_col detected during step execution
    geom_col_hint = result_data.get("geom_col")
    # CRS from geopandas-based steps (buffer, spatial_join, etc.) — used to reproject shapely objects
    crs_str = result_data.get("crs")
    crs_srid: int | None = None
    if crs_str and isinstance(crs_str, str):
        import re as _re
        m = _re.search(r"(\d{4,5})$", crs_str)
        if m:
            crs_srid = int(m.group(1))

    features = []
    for row in rows:
        geom_raw = None
        geom_key = None

        # 1. use detected geom col if present
        if geom_col_hint and geom_col_hint in row:
            geom_raw = row[geom_col_hint]
            geom_key = geom_col_hint
        else:
            # 2. fall back to known names and _geojson suffix aliases
            for key in row:
                k = key.lower()
                if k in _GEOM_KEYS or k.endswith("_geojson"):
                    geom_raw = row[key]
                    geom_key = key
                    break

        if geom_raw is None:
            continue
        try:
            # Pass the step CRS for shapely objects so they get reprojected to 4326
            if hasattr(geom_raw, "geom_type") and crs_srid and crs_srid != 4326:
                geom = _shp_to_4326_geojson(geom_raw, crs_srid)
            else:
                geom = _geom_raw_to_dict(geom_raw)
            if not isinstance(geom, dict):
                continue
            props = {k: v for k, v in row.items() if k != geom_key}
            features.append({"type": "Feature", "geometry": geom, "properties": props})
        except Exception:
            continue

        if len(features) >= _MAP_FEATURE_LIMIT:
            break

    if not features:
        return None

    fc: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if total_rows > len(features):
        # Let the frontend know how many rows exist vs how many are rendered
        fc["total_features"] = total_rows
        fc["rendered_features"] = len(features)
    return fc


def _result_to_geojson(result_data: Any) -> dict | None:
    if not isinstance(result_data, dict):
        return None

    direct_geojson = result_data.get("geojson")
    if isinstance(direct_geojson, dict):
        geojson_type = str(direct_geojson.get("type") or "")
        if geojson_type == "FeatureCollection":
            return direct_geojson
        if geojson_type == "Feature":
            return {"type": "FeatureCollection", "features": [direct_geojson]}

    return _rows_to_geojson(result_data)


def _extract_log(run_logs: list) -> list:
    """Compact log for frontend — one entry per iteration."""
    entries = []
    for run in (run_logs or []):
        for it in (run.get("iterations") or []):
            tools = []
            for t in (it.get("tools") or []):
                tools.append({
                    "tool": t.get("tool"),
                    "args": t.get("args"),
                    "result": t.get("result"),
                    "step": t.get("step"),
                    "status": t.get("status"),
                    "executed": t.get("executed"),
                    "validated": t.get("validated"),
                })
            entries.append({
                "iteration": it.get("iteration"),
                "ai_content": (it.get("ai_content") or "")[:500],
                "had_error": it.get("had_error", False),
                "tokens": it.get("tokens"),
                "tools": tools,
            })
    return entries


def _extract_steps(run_logs: list) -> list:
    if not run_logs:
        return []
    # use last run's step_registry
    last_run = run_logs[-1]
    step_registry = last_run.get("step_registry") or {}
    steps = []
    for step_key, step_data in sorted(
        step_registry.items(),
        key=lambda x: int(str(x[0])) if str(x[0]).isdigit() else 0,
    ):
        steps.append({
            "step": step_key,
            "action": step_data.get("action"),
            "database": step_data.get("database"),
            "sql": step_data.get("sql"),
            "status": step_data.get("status"),
            "validated": step_data.get("validated", False),
            "executed": step_data.get("executed", False),
            "rows_returned": step_data.get("rows_returned"),
            "sql_repairs": step_data.get("sql_repairs", 0),
            "error": step_data.get("error"),
        })
    return steps


def _build_log_text(r: PipelineResult) -> str:
    lines: list[str] = []
    W = 110

    def L(s: str = ""):
        lines.append(s)

    L("=" * W)
    L(f"FINAL MODEL REPORT: {r.model_id}")
    L("=" * W)

    L(f"  valid_sql          : {r.valid_sql}")
    L(f"  executed           : {r.executed}")
    L(f"  task_completed     : {r.task_completed}")
    L(f"  iterations         : {r.iterations}")
    L(f"  sql_repairs_used   : {r.sql_repairs_used}")
    L(f"  restarts_used      : {r.restarts_used}")
    L(f"  latency_s          : {r.latency_s:.3f}")
    if r.tokens:
        L(f"  tokens             : in={r.tokens.prompt_tokens} out={r.tokens.completion_tokens} total={r.tokens.total_tokens}")

    retrieval: dict = {}
    if r.run_logs:
        retrieval = dict((r.run_logs[-1] or {}).get("retrieval", {}))
    if retrieval:
        L(f"  retrieval_score    : {retrieval.get('top_score')}")
        L(f"  retrieval_scoped   : {retrieval.get('scope_active')}")
        L(f"  retrieval_candidates: {[c.get('full_name') for c in retrieval.get('candidates', [])]}")

    step_registry, sql_repairs_by_step, step_retries_by_step = _collect_step_stats(r.run_logs)
    failed_summary = _failed_step_summary(r.run_logs)
    active_step_issues, recovered_step_issues = _step_issue_summaries(r.run_logs)

    if step_registry:
        L()
        L("  STEP REGISTRY:")
        for step_no, meta in _sorted_step_items(step_registry):
            L(f"    Step {step_no}:")
            L(f"      action          : {meta.get('action')}")
            L(f"      database        : {meta.get('database')}")
            L(f"      validated       : {meta.get('validated')}")
            L(f"      executed        : {meta.get('executed')}")
            L(f"      sql_repairs     : {sql_repairs_by_step.get(step_no, 0)}")
            L(f"      step_retries    : {step_retries_by_step.get(step_no, 0)}")
            L(f"      rows_returned   : {meta.get('rows_returned')}")
            L(f"      status          : {meta.get('status')}")
            L(f"      sql             : {_strip_aoi_from_sql(str(meta.get('sql') or ''))}")

    active_errors = [issue.get("error") for issue in active_step_issues if issue.get("error")]
    if not active_errors and not r.task_completed:
        active_errors = _dedup(r.all_errors or [])
    if active_errors:
        L()
        L("  ERRORS:")
        for err in _dedup(active_errors):
            L(f"    - {_sanitize_aoi_text(str(err))}")

    trace = extract_agent_trace(r)
    L()
    L("  TRACE:")
    L(f"    databases explored : {trace.databases_explored}")
    L(f"    tables explored    : {trace.tables_explored}")
    L(f"    validated          : {trace.validated_before_exec}")
    L(f"    executed ok        : {trace.executed_successfully}")
    L(f"    total tool calls   : {trace.total_tool_calls}")
    L(f"    tool call counts   : {trace.tool_call_counts}")

    if failed_summary and failed_summary.get("status") != "ok":
        L()
        L("  FAILED STEP SUMMARY:")
        L(f"    step              : {failed_summary.get('step')}")
        L(f"    action            : {failed_summary.get('action')}")
        L(f"    database          : {failed_summary.get('database')}")
        L(f"    status            : {failed_summary.get('status')}")
        L(f"    sql_repairs       : {failed_summary.get('sql_repairs')}")
        L(f"    step_retries      : {failed_summary.get('step_retries')}")
        L(f"    error             : {_sanitize_aoi_text(str(failed_summary.get('error') or ''))}")
        L(f"    sql               : {_strip_aoi_from_sql(str(failed_summary.get('sql') or ''))}")

    if recovered_step_issues:
        L()
        L("  RECOVERED STEPS:")
        for item in recovered_step_issues:
            L(f"    step              : {item.get('step')}")
            L(f"    action            : {item.get('action')}")
            L(f"    database          : {item.get('database')}")
            L(f"    sql_repairs       : {item.get('sql_repairs')}")
            L(f"    step_retries      : {item.get('step_retries')}")
            L(f"    recovered_from    : {_sanitize_aoi_text(str(item.get('error') or ''))}")
            L(f"    sql               : {_strip_aoi_from_sql(str(item.get('sql') or ''))}")

    if trace.executions:
        L()
        L("  EXECUTIONS:")
        for i, ex in enumerate(trace.executions):
            L(f"    Execution {i}:")
            L(f"      step            : {ex['step']}")
            L(f"      action          : {ex['action']}")
            L(f"      success         : {ex['success']}")
            L(f"      database        : {ex['database']}")
            L(f"      rows_returned   : {ex['rows_returned']}")
            L(f"      sql             : {_short(_strip_aoi_from_sql(str(ex['sql'] or '')), 300)}")
            if not ex["success"]:
                L(f"      error           : {_sanitize_aoi_text(str(ex['error'] or ''))}")

    if r.run_logs:
        L()
        L("  RUN LOGS:")
        for run in r.run_logs:
            L(f"  Run {run.get('run_index', 0)}:")
            L(f"      attempt         : {run.get('attempt_number', 0)}")
            L(f"      sql_repairs     : {run.get('sql_repairs_used', 0)}")
            L(f"      final_sql       : {_short(_strip_aoi_from_sql(str(run.get('final_sql') or '(none)')), 300)}")
            atk = run.get("attempt_tokens")
            if atk:
                L(f"      tokens          : in={atk.get('prompt_tokens', 0)} out={atk.get('completion_tokens', 0)} total={atk.get('total_tokens', 0)}")
            if run.get("validation_error"):
                L(f"      validation_error: {_sanitize_aoi_text(str(run['validation_error']))}")
            if run.get("guard_error"):
                L(f"      guard_error     : {_sanitize_aoi_text(str(run['guard_error']))}")
            if run.get("execution_error"):
                L(f"      execution_error : {_sanitize_aoi_text(str(run['execution_error']))}")

            for entry in run.get("iterations", []):
                tok = entry.get("tokens", {})
                L(f"      Iteration {entry['iteration']}:")
                L(f"        tokens              : in={tok.get('prompt_tokens', 0)} out={tok.get('completion_tokens', 0)} total={tok.get('total_tokens', 0)}")
                L(f"        ai_content          : {_short(_sanitize_aoi_text(str(entry.get('ai_content', ''))), 400)}")
                L(f"        tool_calls_declared : {len(entry.get('tool_calls', []))}")
                L(f"        had_error           : {entry.get('had_error', False)}")
                L(f"        failed_step         : {entry.get('failed_step')}")
                if entry.get("iteration_error"):
                    L(f"        iteration_error     : {entry.get('iteration_error')}")
                for t in entry.get("tools", []):
                    L(f"          TOOL   : {t.get('tool')}")
                    L(f"          ARGS   : {_short(_sanitize_aoi_text(str(t.get('args', ''))), 200)}")
                    L(f"          RESULT : {_short(_sanitize_aoi_text(str(t.get('result', ''))), 200)}")
                    L(f"          STEP   : {t.get('step')}")
                    L(f"          STATUS : {t.get('status')}")
                    L(f"          VALID  : {t.get('validated')}")
                    L(f"          EXEC   : {t.get('executed')}")
                    L()

    L()
    L("  FINAL:")
    L(f"    answer            : {_sanitize_aoi_text(str(r.final_answer or '(none)'))}")
    L(f"    sql               : {_short(_strip_aoi_from_sql(str(r.final_sql or '(none)')), 300)}")

    L()
    L("=" * W)
    return "\n".join(lines)


def _serialise(r: PipelineResult | AgentRunError, model_id: str) -> dict:
    if isinstance(r, AgentRunError):
        snap = r.snapshot or {}
        return {
            "model_id": model_id,
            "final_answer": None,
            "final_sql": _sanitize_aoi_payload(snap.get("final_sql"), sql_field=True),
            "valid_sql": False,
            "executed": False,
            "task_completed": False,
            "iterations": snap.get("iterations", 0),
            "sql_repairs_used": snap.get("sql_repairs_used", 0),
            "latency_s": 0.0,
            "all_errors": _sanitize_aoi_payload(snap.get("all_errors", [str(r)])),
            "result_data": None,
            "geojson": None,
            "steps": [],
            "log": [],
            "full_log": "\n".join(
                str(_sanitize_aoi_text(str(err))) for err in (snap.get("all_errors", [str(r)]))
            ),
        }

    geojson = _result_to_geojson(r.result_data)
    steps = _extract_steps(r.run_logs or [])
    log = _extract_log(r.run_logs or [])
    full_log = _build_log_text(r)
    return {
        "model_id": r.model_id,
        "final_answer": _sanitize_aoi_payload(r.final_answer),
        "final_sql": _sanitize_aoi_payload(r.final_sql, sql_field=True),
        "valid_sql": r.valid_sql,
        "executed": r.executed,
        "task_completed": r.task_completed,
        "iterations": r.iterations,
        "sql_repairs_used": r.sql_repairs_used,
        "latency_s": r.latency_s,
        "all_errors": _sanitize_aoi_payload(r.all_errors or []),
        "result_data": r.result_data,
        "geojson": geojson,
        "steps": _sanitize_aoi_payload(steps),
        "log": _sanitize_aoi_payload(log),
        "full_log": full_log,
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


@app.get("/prompt/{name}")
def get_prompt(name: str):
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        return {"error": f"Prompt '{name}' not found"}
    return {"name": name, "content": path.read_text(encoding="utf-8")}


@app.post("/prompt/{name}")
def save_prompt(name: str, payload: dict[str, Any]):
    content = payload.get("content", "")
    if not content.strip():
        return {"error": "Content cannot be empty"}
    path = _PROMPTS_DIR / f"{name}.txt"
    path.write_text(content, encoding="utf-8")
    logger.info("[PROMPT] Saved prompt '%s' (%d chars)", name, len(content))
    return {"ok": True, "name": name}


@app.get("/models")
def list_models():
    return {"models": ALL_MODELS}


_AOI_SQL_PATTERN = re.compile(
    r"ST_GeomFromText\s*\(\s*'POLYGON"
    r"|ST_GeomFromGeoJSON\s*\(\s*'"
    r"|ST_SetSRID\s*\(\s*ST_GeomFromGeoJSON"
    r"|ST_MakeEnvelope\s*\([0-9. ,\-]+,\s*4326"
    r"|&&\s*ST_Transform\s*\(\s*ST_MakeEnvelope"
    r"|ST_Intersects\s*\([^)]*ST_SetSRID\s*\(\s*ST_GeomFromGeoJSON",
    re.IGNORECASE | re.DOTALL,
)

_AOI_SQL_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"ST_GeomFromText\s*\(\s*'POLYGON[^']*'\s*,\s*4326\s*\)", re.IGNORECASE | re.DOTALL),
        "ST_GeomFromText('[AOI polygon omitted]', 4326)",
    ),
    (
        re.compile(
            r"ST_SetSRID\s*\(\s*ST_GeomFromGeoJSON\s*\(\s*'.*?'\s*\)\s*,\s*4326\s*\)",
            re.IGNORECASE | re.DOTALL,
        ),
        "ST_SetSRID(ST_GeomFromGeoJSON('[AOI GeoJSON omitted]'), 4326)",
    ),
    (
        re.compile(r"ST_GeomFromGeoJSON\s*\(\s*'.*?'\s*\)", re.IGNORECASE | re.DOTALL),
        "ST_GeomFromGeoJSON('[AOI GeoJSON omitted]')",
    ),
    (
        re.compile(r"ST_MakeEnvelope\s*\([^)]*,\s*4326\s*\)", re.IGNORECASE | re.DOTALL),
        "ST_MakeEnvelope([AOI bbox omitted], 4326)",
    ),
]


def _strip_aoi_from_sql(sql: str) -> str:
    """Remove AOI polygon coordinates from SQL so the LLM cannot reuse them."""
    if _AOI_SQL_PATTERN.search(sql):
        return "[SQL omitted — contained AOI polygon coordinates]"
    return sql


def _sanitize_aoi_text(value: str) -> str:
    """Keep messages readable while removing raw AOI geometry and bbox payloads."""
    text = value
    for pattern, replacement in _AOI_SQL_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def _sanitize_aoi_payload(value: Any, *, sql_field: bool = False) -> Any:
    if isinstance(value, str):
        if sql_field:
            return _strip_aoi_from_sql(value)
        return _sanitize_aoi_text(value)
    if isinstance(value, list):
        return [_sanitize_aoi_payload(item, sql_field=sql_field) for item in value]
    if isinstance(value, dict):
        sanitized: dict[Any, Any] = {}
        for key, item in value.items():
            child_sql_field = sql_field or str(key).lower() in {"sql", "final_sql"}
            sanitized[key] = _sanitize_aoi_payload(item, sql_field=child_sql_field)
        return sanitized
    return value


def _history_to_context(history: list | None, has_aoi: bool = False) -> str | None:
    if not history:
        return None
    lines = ["Previous conversation (use this as context for the current question):"]
    for i, turn in enumerate(history, 1):
        sql_raw = turn.get("sql") or ""
        turn_had_aoi = bool(sql_raw and _AOI_SQL_PATTERN.search(sql_raw))

        lines.append(f"\nTurn {i}:")
        lines.append(f"  User: {turn.get('question', '')}")

        if turn.get("answer"):
            if turn_had_aoi and not has_aoi:
                # The previous answer referenced a spatial area that no longer exists —
                # drop it so the LLM doesn't think the area is still active.
                lines.append("  Answer: [previous answer omitted — it referenced a spatial filter that is no longer active]")
            else:
                lines.append(f"  Answer: {turn['answer']}")

        if sql_raw:
            lines.append(f"  SQL: {_strip_aoi_from_sql(sql_raw.strip().replace(chr(10), ' ')[:300])}")
        if turn.get("summary"):
            lines.append(f"  Summary: {turn['summary']}")
    return "\n".join(lines)


def _aoi_to_context(aoi: Any) -> str | None:
    if not aoi or not isinstance(aoi, dict):
        return None
    try:
        from shapely.geometry import shape
        geom = shape(aoi)
        minx, miny, maxx, maxy = geom.bounds
        wkt = geom.wkt
        geojson_str = json.dumps(aoi, ensure_ascii=False)
        geom_label = "MultiPolygon" if aoi.get("type") == "MultiPolygon" else "Polygon"
        return (
            f"The user has drawn {geom_label.lower()} area(s) of interest on the map (EPSG:4326 / WGS84).\n"
            f"Use this to spatially constrain all queries.\n\n"
            f"Bounding box: ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326)\n"
            f"{geom_label} WKT:  ST_GeomFromText('{wkt}', 4326)\n"
            f"{geom_label} JSON: ST_GeomFromGeoJSON('{geojson_str}')\n\n"
            f"When the data is in a different SRID, apply ST_Transform:\n"
            f"  ST_Intersects(geom, ST_Transform(ST_GeomFromGeoJSON('{geojson_str}'), <target_srid>))\n\n"
            f"For site suitability queries (find safe building area, avoid hazard zones, suggest alternative location):\n"
            f"  Use ST_Buffer(ST_GeomFromGeoJSON('{geojson_str}')::geography, <radius_m>)::geometry to create a search ring,\n"
            f"  then ST_Difference(search_ring, ST_Union(hazard_geom)) to isolate the safe area."
        )
    except Exception as e:
        logger.warning("Failed to parse AOI geometry: %s", e)
        return None


def _selected_feature_to_context(feature: Any) -> str | None:
    if not feature or not isinstance(feature, dict):
        return None
    props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
    geom = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else None
    if not props and not geom:
        return None

    lines = ["Selected map feature (use this as context if the user asks about 'this', 'denne', 'that object', or similar):"]
    if props:
        keep_keys = [
            "id", "objid", "gid", "name", "navn", "informasjon", "objtype",
            "distance_m", "duration_s", "building_count",
        ]
        summary = {k: props[k] for k in keep_keys if k in props}
        if not summary:
            summary = dict(list(props.items())[:8])
        lines.append(f"  Properties: {json.dumps(summary, ensure_ascii=False, default=str)}")
    if geom:
        lines.append(f"  Geometry type: {geom.get('type')}")
    lines.append(
        "Interpret follow-up questions that refer to 'this/denne/den' as referring to this selected feature. "
        "If the user asks a clearly new standalone query with new target words, treat that as a new task."
    )
    return "\n".join(lines)


@app.post("/query")
async def run_query(payload: dict[str, Any]):
    question = (payload.get("question") or "").strip()
    if not question:
        return {"error": "No question provided"}

    model_ids: list[str] = payload.get("models") or ALL_MODELS
    mode: str = "eval"
    prompt_name: str = payload.get("prompt") or "prompt_v1"
    aoi_context = _aoi_to_context(payload.get("aoi"))
    if aoi_context:
        logger.info("[AOI] Spatial filter injected:\n%s", aoi_context)
    else:
        logger.info("[AOI] No area of interest drawn — running without spatial filter")

    history_context = _history_to_context(payload.get("history"), has_aoi=bool(aoi_context))
    if history_context:
        logger.info("[HISTORY] Conversation history injected (%d turns)", len(payload.get("history", [])))
    selected_feature_context = _selected_feature_to_context(payload.get("selectedFeature"))
    if selected_feature_context:
        history_context = f"{history_context}\n\n{selected_feature_context}" if history_context else selected_feature_context

    ck = _cache_key(question, payload.get("aoi"), prompt_name, model_ids)
    cached = _cache_get(ck)
    if cached:
        logger.info("[CACHE] Hit for question='%s'", question[:80])
        return {**cached, "from_cache": True}

    params = ModelParams(
        temperature=0.0,
        max_tokens=4000,
        timeout=300,
        max_retries=0,
        max_iterations=50,
        max_sql_repairs=3,
        max_step_retries=3,
        max_replans=2,
    )

    prompt = load_prompt(prompt_name)

    raw_results = await run_all(
        user_input=question,
        model_ids=model_ids,
        params=params,
        prompt=prompt,
        aoi_context=aoi_context,
        history_context=history_context,
    )

    results = [_serialise(r, mid) for r, mid in zip(raw_results, model_ids)]
    best_model = _pick_best(results)

    response = {"best_model": best_model, "results": results, "mode": mode}

    # Only cache successful responses that returned data
    best = next((r for r in results if r["model_id"] == best_model), None)
    if best and (best.get("task_completed") or best.get("executed")):
        _cache_set(ck, response)
        logger.info("[CACHE] Stored result for question='%s'", question[:80])

    return response


@app.get("/debug/catalog")
async def debug_catalog():
    result = await mcp_client.call_tool(
        "get_database_catalog",
        {"databaseName": "matrikkelenbygning"}
    )
    return {"result": result}
