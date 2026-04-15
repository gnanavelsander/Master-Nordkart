import asyncio
import json
import logging
import re
import time
from dataclasses import asdict
from typing import Any

import requests as _requests

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from backend.evaluation.agent_models import AgentRunError, ModelParams, PipelineResult, TokenUsage
from backend.mcp.client.tools_discovery import get_database_catalog, get_table_schema, prefetch_databases, run_postgis_query
from backend.tools.langchain_tools import (
    kommune_where_clause,
    tool_get_available_databases,
    tool_get_available_schemas,
    tool_get_available_tables,
    tool_get_database_catalog,
    tool_run_postgis_query,
    tool_get_table_schema,
)
from backend.embeddings.retrieve_layers import build_retrieval_context, retrieve_database_candidates
from config import (
    ALL_MODELS,
    AZURE_FOUNDRY_ENDPOINT,
    AZURE_FOUNDRY_MODEL_IDS,
    AZURE_MODEL_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_MODEL_IDS,
    EMBEDDINGS_TABLE_INDEX_PATH,
    EMBEDDINGS_TOP_K,
    OLLAMA_MODEL_IDS,
)

logger = logging.getLogger(__name__)

ASSISTANT_MAX_STALL_ITERATIONS = 6
ASSISTANT_MAX_REPEAT_ERROR = 3
ASSISTANT_ALLOWED_DATABASES: tuple[str, ...] = ("postgis", "brannstasjoner")
_CATALOG_STOPWORDS = {
    "vis", "vise", "tegn", "tegne", "kart", "map", "på", "paa", "med", "for",
    "som", "til", "fra", "over", "under", "inside", "within", "show", "draw",
    "the", "and", "med", "alle", "over", "oslo",
}
_ASSISTANT_DISCOVERY_CACHE: dict[str, str] = {}


STEP_DICT_FIELDS: frozenset[str] = frozenset({
    "step", "step_number", "action", "database", "sql", "validSql",
    "source_step", "left_source", "right_source",
    "join_type", "left_geom", "right_geom",
    "distance_meters", "buffer_distance",
    "filters", "filter", "select_columns", "output_columns",
    "group_by", "aggregation_type", "metric_column", "metric_as",
    "order_by", "order_direction", "limit",
    "target_crs", "join_keys", "join_key",
    "schema_name", "table_name",
})


ASSISTANT_RUNTIME_PROMPT = """
You are a practical GIS assistant with access to a single PostGIS database.

Your only data tool is tool_run_postgis_query. Use it for every data task.
NEVER use tool_execute_query_step or any other step tools — they do not exist in this mode.

Workflow for every task:
1. Call tool_get_database_catalog (databaseName="postgis") to find the relevant schema and table.
2. Call tool_get_table_schema to verify the exact column names including the geometry column.
3. If the task targets a municipality, call kommune_where_clause to get the spatial filter clause.
4. Write a single PostGIS SQL query and execute it with tool_run_postgis_query.
5. If you get 0 rows, inspect the filters and retry with relaxed conditions.

SQL rules:
- Geometry column MUST be named exactly "geometry":
    ST_AsGeoJSON(ST_Transform(<geom_col>, 4326)) AS geometry
  If SRID is already 4326: ST_AsGeoJSON(<geom_col>) AS geometry
- For mixed geometry types, use UNION ALL so every row has a "geometry" column.
- Do not reference kommune_exact_where_clause, kommune_context, or tool_get_kommune_context as SQL objects.
- Allowed databases: postgis and brannstasjoner only.

General rules:
- Stop as soon as you have executed a successful query and answer based on the row count.
- If a query fails, fix the column name or filter and retry immediately.
- Do not explain what you plan to do — just do it.
- If conversation context includes a selected map feature and the user asks about "this/denne/den", treat it as a follow-up about that selected object.
- If the user clearly names a new target or new task, treat it as a new query instead of forcing the previous object context.
""".strip()

_ORS_PROFILES = frozenset({
    "foot-walking", "foot-hiking",
    "driving-car", "driving-hgv",
    "cycling-regular", "cycling-road", "cycling-mountain", "cycling-electric",
    "wheelchair",
})


def _tool_get_route(coordinates: list, profile: str = "foot-walking") -> str:
    """
    Calculate a route between two or more waypoints using OpenRouteService.

    Returns a GeoJSON FeatureCollection with the route line and summary
    (distance_m, duration_s).

    Args:
        coordinates: List of [longitude, latitude] pairs, e.g.
                     [[10.74, 59.91], [10.75, 59.92]]. At least 2 required.
        profile: Travel mode — foot-walking (default), foot-hiking,
                 driving-car, driving-hgv,
                 cycling-regular, cycling-road, cycling-mountain, cycling-electric,
                 wheelchair.
    """
    from config import ORS_API_KEY
    if not ORS_API_KEY:
        return json.dumps({"error": "ORS_API_KEY is not configured."})
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        return json.dumps({"error": "coordinates must be a list of at least two [lon, lat] pairs."})
    if profile not in _ORS_PROFILES:
        profile = "foot-walking"

    url = f"https://api.openrouteservice.org/v2/directions/{profile}/geojson"
    headers = {
        "Accept": "application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8",
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json; charset=utf-8",
    }
    try:
        resp = _requests.post(url, json={"coordinates": coordinates}, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except _requests.HTTPError:
        return json.dumps({"error": f"ORS HTTP {resp.status_code}: {resp.text[:300]}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

    features = data.get("features") or []
    rows = []
    for feat in features:
        summary = (feat.get("properties") or {}).get("summary") or {}
        rows.append({
            "geometry": json.dumps(feat["geometry"]),
            "distance_m": summary.get("distance"),
            "duration_s": summary.get("duration"),
        })

    return json.dumps({
        "executed": True,
        "profile": profile,
        "geojson": data,
        "rows": rows,
        "row_count": len(rows),
        "geom_col": "geometry",
    })


from langchain.tools import tool as _lc_tool

@_lc_tool
def tool_get_route(coordinates: list, profile: str = "foot-walking") -> str:
    """
    Calculate a route between two or more waypoints using OpenRouteService.

    Returns a GeoJSON FeatureCollection with the route line and summary
    (distance_m, duration_s).

    Args:
        coordinates: List of [longitude, latitude] pairs, e.g.
                     [[10.74, 59.91], [10.75, 59.92]]. At least 2 required.
        profile: Travel mode — foot-walking (default), foot-hiking,
                 driving-car, driving-hgv,
                 cycling-regular, cycling-road, cycling-mountain, cycling-electric,
                 wheelchair.
    """
    return _tool_get_route(coordinates, profile)


def _assistant_tools():
    return [
        kommune_where_clause,
        tool_get_available_databases,
        tool_get_available_schemas,
        tool_get_available_tables,
        tool_get_table_schema,
        tool_get_database_catalog,
        tool_run_postgis_query,
        tool_get_route,
    ]


def _get_model(model_id: str, params: ModelParams):
    tools = _assistant_tools()
    dep = AZURE_MODEL_DEPLOYMENT.get(model_id)

    model_kwargs = {}
    if params.top_p is not None:
        model_kwargs["top_p"] = params.top_p
    if params.reasoning_effort is not None:
        model_kwargs["reasoning_effort"] = params.reasoning_effort

    if model_id in AZURE_OPENAI_MODEL_IDS:
        return AzureChatOpenAI(
            api_key=AZURE_OPENAI_KEY,
            azure_deployment=dep,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version="2024-12-01-preview",
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            timeout=params.timeout,
            max_retries=params.max_retries,
            model_kwargs=model_kwargs,
        ).bind_tools(tools, tool_choice=params.tool_choice)

    if model_id in AZURE_FOUNDRY_MODEL_IDS:
        return ChatOpenAI(
            model=dep,
            api_key=AZURE_OPENAI_KEY,
            base_url=AZURE_FOUNDRY_ENDPOINT,
            temperature=params.temperature,
            timeout=params.timeout,
            max_retries=params.max_retries,
            model_kwargs=model_kwargs,
        ).bind_tools(tools, tool_choice=params.tool_choice)

    if model_id in OLLAMA_MODEL_IDS:
        return ChatOllama(model=model_id, temperature=params.temperature).bind_tools(tools)

    raise ValueError(f"Unknown model_id: {model_id}")


def _extract_tokens(ai: Any) -> TokenUsage | None:
    usage = getattr(ai, "usage_metadata", None) or getattr(ai, "response_metadata", {}).get("token_usage", {})
    if not usage:
        return None
    if isinstance(usage, dict):
        pt = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        ct = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        tt = usage.get("total_tokens", 0) or (pt + ct)
        return TokenUsage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)
    pt = getattr(usage, "input_tokens", 0)
    ct = getattr(usage, "output_tokens", 0)
    tt = getattr(usage, "total_tokens", 0) or (pt + ct)
    return TokenUsage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)


def _parse_json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _build_human_in_loop_message(
    user_input: str,
    last_error: str | None,
    discovered_databases: list[str],
    discovered_tables: list[str],
) -> str:
    if last_error:
        return f"Klarte ikke å fullføre forespørselen. Feil: {last_error[:200]}"
    return "Klarte ikke å fullføre forespørselen. Prøv å formulere den på nytt."


def _build_execution_required_message(
    user_input: str,
    last_error: str | None,
    discovered_databases: list[str],
    discovered_tables: list[str],
) -> str:
    if last_error:
        return f"Fant ikke data for forespørselen. Feil: {last_error[:200]}"
    return "Fant ikke data for forespørselen i databasen."


def _truncate_value(value: Any, max_len: int = 240) -> Any:
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + "...[truncated]"
    return value


def _compact_tool_result_for_model(tool_name: str, parsed: Any, raw_result: str) -> str:
    if not isinstance(parsed, dict):
        if len(raw_result) > 12000:
            return json.dumps({"tool": tool_name, "note": "result too large for model context"})
        return raw_result

    if tool_name == "tool_get_database_catalog":
        compact_databases: list[dict[str, Any]] = []
        for db in (parsed.get("databases") or [])[:3]:
            schemas_out: list[dict[str, Any]] = []
            for schema in (db.get("schemas") or [])[:6]:
                tables = []
                for table in (schema.get("tables") or [])[:8]:
                    tables.append({
                        "table": table.get("table"),
                        "recommended_geometry_column": table.get("recommended_geometry_column"),
                    })
                schemas_out.append({
                    "schema": schema.get("schema"),
                    "tables": tables,
                })
            compact_databases.append({
                "database": db.get("database"),
                "description": db.get("description"),
                "schemas": schemas_out,
            })
        return json.dumps({"databases": compact_databases}, ensure_ascii=False, default=str)

    compact = dict(parsed)
    rows = compact.pop("rows", None)
    sample_rows: list[dict[str, Any]] = []
    if isinstance(rows, list):
        for row in rows[:5]:
            if isinstance(row, dict):
                sample_rows.append({k: _truncate_value(v) for k, v in row.items()})
            else:
                sample_rows.append({"value": _truncate_value(row)})
        compact["sample_rows"] = sample_rows
        compact["result_truncated_for_model"] = True

    if "geometry" in compact:
        compact["geometry"] = _truncate_value(compact["geometry"])

    encoded = json.dumps(compact, ensure_ascii=False, default=str)
    if len(encoded) > 12000:
        reduced = {
            "executed": compact.get("executed"),
            "database": compact.get("database"),
            "row_count": compact.get("row_count"),
            "rows_returned": compact.get("rows_returned"),
            "columns": compact.get("columns"),
            "result_truncated_for_model": True,
        }
        if sample_rows:
            reduced["sample_rows"] = sample_rows
        return json.dumps(reduced, ensure_ascii=False, default=str)
    return encoded


def _is_map_intent(text: str | None) -> bool:
    value = str(text or "").lower()
    return any(token in value for token in [
        "kart",
        "map",
        "vis ",
        "vise ",
        "tegn",
        "draw",
    ])


def _looks_like_reprompt(text: str | None) -> bool:
    value = str(text or "").lower()
    return any(token in value for token in [
        "vil du",
        "reply with",
        "svar kort",
        "interaktivt kart",
        "featurecollection",
        "nedlast",
        "first 50",
        "first 200",
    ])


_KNOWN_GEOM_COLUMNS = frozenset({"geom", "geometri", "shape", "the_geom", "wkb_geometry", "wkt_geometry"})


def _normalise_query_result(result: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure the result has a column named exactly 'geometry' containing GeoJSON strings.
    Renames any known geometry column alias to 'geometry' and sets geom_col hint.
    Does not re-project — the agent is responsible for returning 4326 GeoJSON.
    No-ops if 'geometry' already present or no geometry column detected.
    """
    rows = result.get("rows")
    if not rows or not isinstance(rows, list):
        return result
    first = rows[0]
    if not isinstance(first, dict):
        return result
    if "geometry" in first:
        # Already correct; just ensure the hint is set
        return {**result, "geom_col": "geometry"} if result.get("geom_col") != "geometry" else result
    # Find a geometry-like column
    geom_key = next((k for k in first if k.lower() in _KNOWN_GEOM_COLUMNS), None)
    if geom_key is None:
        return result
    # Rename that column to 'geometry' in every row
    renamed_rows = [{("geometry" if k == geom_key else k): v for k, v in row.items()} for row in rows]
    cols = result.get("columns") or []
    renamed_cols = [("geometry" if c == geom_key else c) for c in cols]
    return {**result, "rows": renamed_rows, "columns": renamed_cols, "geom_col": "geometry"}


def _extract_count_value(result_data: dict[str, Any] | None) -> int | None:
    if not isinstance(result_data, dict):
        return None
    raw_rows = result_data.get("rows") or []
    if not (isinstance(raw_rows, list) and raw_rows and isinstance(raw_rows[0], dict)):
        return None
    row = raw_rows[0]
    for key, value in row.items():
        key_lower = str(key).lower()
        if key_lower.endswith("_count") or key_lower in {"count", "antall", "building_count"}:
            try:
                return int(value)
            except Exception:
                return None
    return None


def _is_route_result(result_data: dict[str, Any] | None) -> bool:
    if not isinstance(result_data, dict):
        return False
    if result_data.get("database") == "route":
        return True
    raw_rows = result_data.get("rows") or []
    if not (isinstance(raw_rows, list) and raw_rows and isinstance(raw_rows[0], dict)):
        return False
    row = raw_rows[0]
    return "distance_m" in row and "duration_s" in row and "geometry" in row


def _has_drawable_geometry(result_data: dict[str, Any] | None) -> bool:
    if not isinstance(result_data, dict):
        return False
    rows = result_data.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return False
    first = rows[0]
    return isinstance(first, dict) and "geometry" in first


def _build_success_answer(user_input: str, result_data: dict[str, Any] | None) -> str:
    rows = 0
    if isinstance(result_data, dict):
        rows = int(result_data.get("row_count") or result_data.get("rows_returned") or 0)
        count_value = _extract_count_value(result_data)
        if count_value is not None:
            return f"Det er {count_value} boliger i det valgte området."
        if _is_route_result(result_data):
            raw_rows = result_data.get("rows") or []
            first = raw_rows[0] if isinstance(raw_rows, list) and raw_rows else {}
            distance_m = first.get("distance_m") if isinstance(first, dict) else None
            duration_s = first.get("duration_s") if isinstance(first, dict) else None
            if distance_m is not None and duration_s is not None:
                try:
                    distance_km = round(float(distance_m) / 1000, 1)
                    minutes = round(float(duration_s) / 60)
                    return f"Jeg har vist ruten på kartet. Lengde: {distance_km} km. Ca. {minutes} min."
                except Exception:
                    return "Jeg har vist ruten på kartet."
            return "Jeg har vist ruten på kartet."

    if _is_map_intent(user_input):
        if rows > 0:
            return f"Jeg har vist {rows} objekter på kartet."
        return "Jeg har vist resultatet på kartet."

    if rows > 0:
        return f"Jeg fant {rows} objekter i resultatet."
    return "Jeg har fullført forespørselen."


def _requested_exact_municipality(text: str | None) -> str | None:
    """Extract a Norwegian municipality name from the user input, e.g. 'i Oslo' → 'Oslo'."""
    value = str(text or "")
    # Match "i <Place>" — Norwegian preposition for location
    match = re.search(r'\bi\s+([A-ZÆØÅ][a-zæøå]{2,}(?:\s+[A-ZÆØÅ][a-zæøå]+)*)', value)
    if match:
        return match.group(1)
    # Fallback: lowercase check for well-known cities
    lower = value.lower()
    for city in ("oslo", "bergen", "trondheim", "stavanger", "grimstad", "tromsø", "kristiansand"):
        if city in lower:
            return city.capitalize()
    return None


def _is_direct_flood_map_request(text: str | None) -> bool:
    value = str(text or "").lower()
    return _is_map_intent(value) and any(token in value for token in ["flom", "flomsone", "flomsoner"])


def _is_aoi_building_count_request(text: str | None, aoi_wkt: str | None) -> bool:
    value = str(text or "").lower()
    if not aoi_wkt:
        return False
    return any(token in value for token in ["hvor mange", "count", "antall"]) and any(
        token in value for token in ["bolig", "boliger", "bygning", "bygninger", "buildings"]
    )


def _is_buildings_in_flood_request(text: str | None) -> bool:
    value = str(text or "").lower()
    return any(token in value for token in ["bolig", "boliger", "bygning", "bygninger", "buildings"]) and any(
        token in value for token in ["flom", "flomsone", "flomsoner"]
    )


def _is_arealbruk_skogbonitet_request(text: str | None) -> bool:
    value = str(text or "").lower()
    return "arealbruk" in value and "skogbonitet" in value


def _parse_route_request(text: str | None) -> tuple[str, str] | None:
    value = str(text or "").strip()
    value = re.sub(r"^\s*vis\s+", "", value, flags=re.IGNORECASE)
    match = re.search(r"\brute\s+fra\s+(.+?)\s+til\s+(.+)$", value, re.IGNORECASE)
    if not match:
        return None
    origin = match.group(1).strip(" .")
    destination = match.group(2).strip(" .")
    destination = re.sub(r"\s+på\s+kart\s*$", "", destination, flags=re.IGNORECASE)
    destination = re.sub(r"\s+og\s+denne\s*$", "", destination, flags=re.IGNORECASE)
    destination = destination.strip(" .")
    if not origin or not destination:
        return None
    return origin, destination


def _is_route_request(text: str | None) -> bool:
    return _parse_route_request(text) is not None


def _build_municipality_start_message(user_input: str) -> str | None:
    if _is_route_request(user_input):
        return (
            "This is a route request. Prefer tool_get_route directly. "
            "Do not use tool_run_postgis_query for route tasks unless the route tool fails."
        )
    municipality = _requested_exact_municipality(user_input)
    if not municipality:
        return None
    return (
        f"The request targets {municipality.title()}. Before building the final SQL query, "
        "prefer kommune_where_clause with the discovered geometry column and SQL alias if you use one. "
        "Do not write SQL that refers to kommune_exact_where_clause, kommune_context, or tool_get_kommune_context as database objects."
    )




def _infer_geom_reference_from_sql(sql: str | None) -> tuple[str, str] | None:
    value = str(sql or "")
    patterns = [
        r"ST_AsGeoJSON\s*\(\s*ST_Transform\(\s*([A-Za-z0-9_\.]+)\s*,\s*4326\s*\)\s*\)",
        r"ST_Transform\(\s*([A-Za-z0-9_\.]+)\s*,\s*4326\s*\)",
        r"ST_Intersects\(\s*([A-Za-z0-9_\.]+)\s*,",
        # Catch common geometry column names when used directly in SELECT
        r"SELECT\b[^F]+\b([A-Za-z0-9_]+)\.(geom|geometry|the_geom|wkb_geometry|shape)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, re.IGNORECASE)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 2:
            # alias.column captured separately
            return groups[0].strip(), groups[1].strip()
        ref = groups[0]
        if "." in ref:
            alias, column = ref.rsplit(".", 1)
            return alias.strip(), column.strip()
        return "", ref.strip()
    return None


def _inject_exact_municipality_clause(sql: str, clause: str) -> str:
    stripped = sql.strip().rstrip(";")
    if re.search(r"\bWHERE\b", stripped, re.IGNORECASE):
        return re.sub(r"\bWHERE\b", f"WHERE ({clause}) AND ", stripped, count=1, flags=re.IGNORECASE) + ";"
    return stripped + f" WHERE {clause};"


def _normalize_tool_args(name: str, raw_args: dict, run_id: str) -> dict:
    return dict(raw_args or {})


def _candidate_database_order(candidates: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in candidates:
        database = str(item.get("database") or "").strip()
        if not database or database in seen:
            continue
        seen.add(database)
        ordered.append(database)
    return ordered


def _catalog_keywords(user_input: str) -> list[str]:
    parts = re.findall(r"[a-zA-ZæøåÆØÅ0-9_]+", str(user_input or "").lower())
    keywords: list[str] = []
    for part in parts:
        if len(part) < 4 or part in _CATALOG_STOPWORDS:
            continue
        if part not in keywords:
            keywords.append(part)
    return keywords[:6]


def _filter_catalog_payload(payload: dict[str, Any], user_input: str) -> dict[str, Any]:
    keywords = _catalog_keywords(user_input)
    if not keywords:
        return payload

    filtered_databases: list[dict[str, Any]] = []
    for db in payload.get("databases", []) or []:
        db_match = any(k in str(db.get("database") or "").lower() or k in str(db.get("description") or "").lower() for k in keywords)
        schemas_out: list[dict[str, Any]] = []
        for schema in db.get("schemas", []) or []:
            matched_tables = []
            for table in schema.get("tables", []) or []:
                haystack = " ".join([
                    str(db.get("database") or ""),
                    str(db.get("description") or ""),
                    str(schema.get("schema") or ""),
                    str(table.get("table") or ""),
                    str(table.get("description") or ""),
                ]).lower()
                if db_match or any(k in haystack for k in keywords):
                    matched_tables.append(table)
            if matched_tables:
                schemas_out.append({"schema": schema.get("schema"), "tables": matched_tables})
        if schemas_out:
            filtered_databases.append({
                "database": db.get("database"),
                "description": db.get("description"),
                "schemas": schemas_out,
            })

    return {"databases": filtered_databases} if filtered_databases else payload


def _cache_key(tool_name: str, args: dict[str, Any], user_input: str | None = None) -> str:
    payload = {"tool": tool_name, "args": args}
    if user_input is not None and tool_name == "tool_get_database_catalog":
        payload["user_input"] = user_input
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def _is_allowed_database(database_name: str | None) -> bool:
    return str(database_name or "").strip() in ASSISTANT_ALLOWED_DATABASES


def _blocked_database_result(database_name: str | None) -> str:
    return json.dumps(
        {
            "error": (
                f"Database '{database_name}' is not available in assistant mode. "
                f"Use only: {', '.join(ASSISTANT_ALLOWED_DATABASES)}."
            )
        }
    )


def _find_catalog_tables(
    catalog: dict[str, Any],
    *,
    preferred_table_tokens: list[str],
    sample_suffix: str = "_sample",
) -> list[tuple[str, str, str]]:
    matches: list[tuple[int, str, str, str]] = []
    for db in catalog.get("databases", []) or []:
        for schema in db.get("schemas", []) or []:
            schema_name = str(schema.get("schema") or "").strip()
            for table in schema.get("tables", []) or []:
                table_name = str(table.get("table") or "").strip()
                if not table_name or table_name.endswith(sample_suffix):
                    continue
                geom_col = str(table.get("recommended_geometry_column") or "").strip()
                if not geom_col:
                    continue
                haystack = " ".join([
                    table_name.lower(),
                    str(table.get("description") or "").lower(),
                ])
                score = sum(3 for token in preferred_table_tokens if token in table_name.lower())
                score += sum(1 for token in preferred_table_tokens if token in haystack)
                if score > 0:
                    matches.append((score, schema_name, table_name, geom_col))
    matches.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(schema_name, table_name, geom_col) for _, schema_name, table_name, geom_col in matches]


async def _run_sql_result(sql: str, database: str = "postgis") -> dict[str, Any]:
    rows = await run_postgis_query(sql=sql, databaseName=database)
    result = {
        "executed": True,
        "database": database,
        "row_count": len(rows),
        "rows_returned": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "rows": rows,
    }
    return _normalise_query_result(result)


async def _discover_preferred_table(
    *,
    preferred_tokens: list[str],
    discovered_databases: list[str],
    discovered_tables: list[str],
) -> tuple[str, str, str] | None:
    catalog = await get_database_catalog(databaseName="postgis")
    matches = _find_catalog_tables(catalog, preferred_table_tokens=preferred_tokens)
    if not matches:
        return None
    schema_name, table_name, geom_col = matches[0]
    if "postgis" not in discovered_databases:
        discovered_databases.append("postgis")
    if table_name not in discovered_tables:
        discovered_tables.append(table_name)
    return schema_name, table_name, geom_col


async def _try_direct_aoi_building_count_query(
    user_input: str,
    discovered_databases: list[str],
    discovered_tables: list[str],
    aoi_wkt: str | None,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if not _is_aoi_building_count_request(user_input, aoi_wkt):
        return None, None, None
    try:
        table_info = await _discover_preferred_table(
            preferred_tokens=["buildings", "building", "bygning", "bygninger", "bolig", "boliger"],
            discovered_databases=discovered_databases,
            discovered_tables=discovered_tables,
        )
        if not table_info or not aoi_wkt:
            return None, None, "Could not find a buildings table for AOI count."
        schema_name, table_name, geom_col = table_info
        aoi_clause = _aoi_filter_clause(f"b.{geom_col}", aoi_wkt, geom_type="polygon")
        sql = (
            "SELECT COUNT(*)::bigint AS building_count "
            f"FROM {schema_name}.{table_name} AS b "
            f"WHERE {aoi_clause};"
        )
        return await _run_sql_result(sql), sql, None
    except Exception as e:
        return None, None, str(e)


async def _try_direct_buildings_in_flood_query(
    user_input: str,
    discovered_databases: list[str],
    discovered_tables: list[str],
    aoi_wkt: str | None = None,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    municipality = _requested_exact_municipality(user_input)
    if not municipality or not _is_buildings_in_flood_request(user_input):
        return None, None, None
    try:
        building_info = await _discover_preferred_table(
            preferred_tokens=["buildings", "building", "bygning", "bygninger", "bolig", "boliger"],
            discovered_databases=discovered_databases,
            discovered_tables=discovered_tables,
        )
        flood_info = await _discover_preferred_table(
            preferred_tokens=["flomsoner", "flomsone", "flood"],
            discovered_databases=discovered_databases,
            discovered_tables=discovered_tables,
        )
        if not building_info or not flood_info:
            return None, None, "Could not find building or flood-zone table."
        b_schema, b_table, b_geom = building_info
        f_schema, f_table, f_geom = flood_info
        kommune_clause = kommune_where_clause.func(
            name=municipality,
            geom_column=b_geom,
            table_alias="b",
        )
        if not kommune_clause or str(kommune_clause).startswith("-- ERROR:"):
            return None, None, f"Could not build municipality clause for {municipality}."
        where = f"({kommune_clause}) AND ST_Intersects(b.{b_geom}, f.{f_geom})"
        if aoi_wkt:
            where += f" AND {_aoi_filter_clause(f'b.{b_geom}', aoi_wkt, geom_type='polygon')}"
        sql = (
            "SELECT row_number() OVER () AS id, "
            f"ST_AsGeoJSON(ST_Transform(b.{b_geom}, 4326)) AS geometry "
            f"FROM {b_schema}.{b_table} AS b "
            f"JOIN {f_schema}.{f_table} AS f "
            f"ON ST_Intersects(b.{b_geom}, f.{f_geom}) "
            f"WHERE {where};"
        )
        return await _run_sql_result(sql), sql, None
    except Exception as e:
        return None, None, str(e)


async def _try_direct_arealbruk_skogbonitet_query(
    user_input: str,
    discovered_databases: list[str],
    discovered_tables: list[str],
    aoi_wkt: str | None = None,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    municipality = _requested_exact_municipality(user_input)
    if not municipality or not _is_arealbruk_skogbonitet_request(user_input):
        return None, None, None
    try:
        table_info = await _discover_preferred_table(
            preferred_tokens=["arealbruk_skogbonitet", "skogbonitet", "arealbruk"],
            discovered_databases=discovered_databases,
            discovered_tables=discovered_tables,
        )
        if not table_info:
            return None, None, "Could not find arealbruk/skogbonitet table."
        schema_name, table_name, geom_col = table_info
        kommune_clause = kommune_where_clause.func(
            name=municipality,
            geom_column=geom_col,
            table_alias="a",
        )
        if not kommune_clause or str(kommune_clause).startswith("-- ERROR:"):
            return None, None, f"Could not build municipality clause for {municipality}."
        where = kommune_clause
        if aoi_wkt:
            where = f"({where}) AND {_aoi_filter_clause(f'a.{geom_col}', aoi_wkt, geom_type='polygon')}"
        sql = (
            "SELECT row_number() OVER () AS id, "
            f"ST_AsGeoJSON(ST_Transform(a.{geom_col}, 4326)) AS geometry "
            f"FROM {schema_name}.{table_name} AS a "
            f"WHERE {where};"
        )
        return await _run_sql_result(sql), sql, None
    except Exception as e:
        return None, None, str(e)


def _geocode_place(query: str) -> tuple[float, float] | None:
    try:
        resp = _requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "jsonv2", "limit": 1},
            headers={"User-Agent": "nordkart-assistant/1.0"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or not data:
            return None
        item = data[0]
        return float(item["lon"]), float(item["lat"])
    except Exception:
        return None


async def _try_direct_route_query(
    user_input: str,
    discovered_databases: list[str],
    discovered_tables: list[str],
    aoi_wkt: str | None = None,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    parsed = _parse_route_request(user_input)
    if not parsed:
        return None, None, None
    origin, destination = parsed
    origin_coords = _geocode_place(origin)
    dest_coords = _geocode_place(destination)
    if not origin_coords or not dest_coords:
        return None, None, f"Could not geocode route endpoints: {origin} -> {destination}"
    raw = _tool_get_route([list(origin_coords), list(dest_coords)], "foot-walking")
    parsed_result = _parse_json_maybe(raw)
    if not isinstance(parsed_result, dict):
        return None, None, "Route tool returned unexpected response."
    if parsed_result.get("error"):
        return None, None, str(parsed_result["error"])
    result = {
        "executed": True,
        "database": "route",
        "row_count": parsed_result.get("row_count", 0),
        "rows_returned": parsed_result.get("row_count", 0),
        "columns": ["geometry", "distance_m", "duration_s"],
        "rows": parsed_result.get("rows", []),
        "geom_col": "geometry",
        "geojson": parsed_result.get("geojson"),
        "profile": parsed_result.get("profile"),
    }
    return result, f"route: {origin} -> {destination}", None


_GEOM_COLUMN_CANDIDATES = ("geom", "geometry", "the_geom", "wkb_geometry", "shape", "wkt_geometry")

_AOI_WKT_RE = re.compile(r"Polygon:\s*ST_GeomFromText\('([^']+)',\s*4326\)", re.IGNORECASE)


def _extract_aoi_wkt(aoi_context: str | None) -> str | None:
    """Extract the WKT polygon from the aoi_context string produced by app._aoi_to_context."""
    if not aoi_context:
        return None
    m = _AOI_WKT_RE.search(aoi_context)
    return m.group(1) if m else None


def _aoi_filter_clause(col_ref: str, wkt: str, geom_type: str = "polygon") -> str:
    """
    Return a PostGIS predicate that uses the actual AOI polygon (not bbox).
    - points  → ST_Within  (must be fully inside)
    - others  → ST_Intersects with real polygon (not the envelope)
    The transform adapts to whatever SRID the data column uses.
    """
    transform = f"ST_Transform(ST_GeomFromText('{wkt}', 4326), ST_SRID({col_ref}))"
    if geom_type and geom_type.lower() in ("point", "multipoint"):
        return f"ST_Within({col_ref}, {transform})"
    return f"ST_Intersects({col_ref}, {transform})"


def _aoi_system_message(aoi_context: str) -> str:
    """
    Replace the bbox-only hint in aoi_context with polygon-first predicate guidance.
    The original string contains the WKT under 'Polygon: ST_GeomFromText(...)'.
    """
    wkt = _extract_aoi_wkt(aoi_context)
    if not wkt:
        return aoi_context
    return (
        "The user has drawn an area of interest (AOI) on the map (EPSG:4326 / WGS84). "
        "All queries MUST be spatially constrained to this AOI.\n\n"
        f"AOI polygon (WKT): {wkt!r}\n\n"
        "Use the ACTUAL polygon, NOT the bounding box. Choose predicate by geometry type:\n"
        "  • points   → ST_Within(geom, ST_Transform(ST_GeomFromText('<wkt>', 4326), ST_SRID(geom)))\n"
        "  • polygons / lines (must be fully inside) → ST_Within(...)\n"
        "  • polygons / lines (overlap allowed)      → ST_Intersects(geom, ST_Transform(ST_GeomFromText('<wkt>', 4326), ST_SRID(geom)))\n"
        "Never filter by ST_MakeEnvelope or && alone when an AOI polygon is available."
    )


async def _discover_geom_column(schema_name: str, table_name: str, database: str) -> str | None:
    """Return the first geometry-like column found in the table, or None."""
    try:
        result = await get_table_schema(databaseName=database, schemaName=schema_name, table_name=table_name)
        if isinstance(result, dict):
            for col in result.get("columns", []):
                col_name = str(col.get("column_name") or "").lower()
                col_type = str(col.get("data_type") or "").lower()
                if col_name in _GEOM_COLUMN_CANDIDATES or "geometry" in col_type or "geography" in col_type:
                    return col.get("column_name")
    except Exception:
        pass
    return None


async def _try_direct_municipality_map_query(
    user_input: str,
    discovered_databases: list[str],
    discovered_tables: list[str],
    aoi_wkt: str | None = None,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    municipality = _requested_exact_municipality(user_input)
    if not municipality or not _is_direct_flood_map_request(user_input):
        return None, None, None

    try:
        catalog = await get_database_catalog(databaseName="postgis")
        catalog_matches = _find_catalog_tables(
            catalog,
            preferred_table_tokens=["flomsoner", "flomsone", "flood"],
        )

        # Build hardcoded fallbacks using common flood table names + geometry column names
        hardcoded_tables = ["flomsoner", "flomsone", "nve_flomsoner", "flood_zones"]
        fallback_matches: list[tuple[str, str, str]] = []
        for tname in hardcoded_tables:
            for gcol in ("geom", "geometry", "the_geom", "wkb_geometry"):
                fallback_matches.append(("public", tname, gcol))

        candidates: list[tuple[str, str, str | None]] = []
        for item in catalog_matches + fallback_matches:
            if item not in candidates:
                candidates.append(item)

        # Also add catalog tables with flood-related names but no geometry column recorded
        # (geometry_columns may not be populated) — geom column will be discovered at query time
        for db in catalog.get("databases", []) or []:
            for schema in db.get("schemas", []) or []:
                for table in schema.get("tables", []) or []:
                    tname = str(table.get("table") or "").lower()
                    if any(tok in tname for tok in ["flomsone", "flood"]) and not table.get("recommended_geometry_column"):
                        entry: tuple[str, str, str | None] = (
                            str(schema.get("schema") or "public"),
                            str(table.get("table") or tname),
                            None,
                        )
                        if entry not in candidates:
                            candidates.append(entry)

        if not candidates:
            return None, None, "Could not find a flood-zone table in postgis."

        errors: list[str] = []
        for schema_name, table_name, geom_col in candidates[:8]:

            if "postgis" not in discovered_databases:
                discovered_databases.append("postgis")
            if table_name not in discovered_tables:
                discovered_tables.append(table_name)

            # If no geometry column is known, try to discover it
            if not geom_col:
                geom_col = await _discover_geom_column(schema_name, table_name, "postgis")
                if not geom_col:
                    errors.append(f"{schema_name}.{table_name}: no geometry column found")
                    continue

            kommune_clause = kommune_where_clause.func(
                name=municipality.title(),
                geom_column=geom_col,
                table_alias="f",
            )
            if not kommune_clause or str(kommune_clause).startswith("-- ERROR:"):
                errors.append(f"{schema_name}.{table_name}: could not build municipality clause")
                continue

            where = kommune_clause
            if aoi_wkt:
                # Flood zones are polygons — use ST_Intersects with actual polygon, not bbox
                aoi_clause = _aoi_filter_clause(f"f.{geom_col}", aoi_wkt, geom_type="polygon")
                where = f"({kommune_clause}) AND {aoi_clause}"
            sql = (
                "SELECT row_number() OVER () AS id, "
                f"ST_AsGeoJSON(ST_Transform(f.{geom_col}, 4326)) AS geometry "
                f"FROM {schema_name}.{table_name} AS f "
                f"WHERE {where};"
            )
            try:
                rows = await run_postgis_query(sql=sql, databaseName="postgis")
                result_data = {
                    "executed": True,
                    "database": "postgis",
                    "row_count": len(rows),
                    "rows_returned": len(rows),
                    "columns": list(rows[0].keys()) if rows else ["id", "geometry"],
                    "rows": rows,
                }
                return result_data, sql, None
            except Exception as e:
                errors.append(f"{schema_name}.{table_name}: {e}")

        return None, None, "; ".join(errors[:3]) or "Direct municipality map query failed."
    except Exception as e:
        return None, None, str(e)


async def _invoke_assistant_tool(
    tool_map: dict[str, Any],
    name: str,
    args: dict[str, Any],
    user_input: str,
) -> str:
    route_request = _is_route_request(user_input)
    if name == "tool_get_available_databases":
        return json.dumps(list(ASSISTANT_ALLOWED_DATABASES))

    cacheable_tools = {
        "tool_get_database_catalog",
        "tool_get_available_schemas",
        "tool_get_available_tables",
        "tool_get_table_schema",
    }
    cache_key = _cache_key(name, args, user_input)
    if name in cacheable_tools and cache_key in _ASSISTANT_DISCOVERY_CACHE:
        return _ASSISTANT_DISCOVERY_CACHE[cache_key]

    if name == "tool_get_database_catalog":
        database_name = args.get("databaseName")
        if database_name:
            if not _is_allowed_database(database_name):
                return _blocked_database_result(database_name)
            tool = tool_map.get(name)
            result_str = await tool.ainvoke(args)
            parsed = _parse_json_maybe(result_str)
            if isinstance(parsed, dict):
                result_str = json.dumps(_filter_catalog_payload(parsed, user_input), ensure_ascii=False, default=str)
            _ASSISTANT_DISCOVERY_CACHE[cache_key] = result_str
            return result_str

        tool = tool_map.get(name)
        merged: list[dict[str, Any]] = []
        for allowed_db in ASSISTANT_ALLOWED_DATABASES:
            result_str = await tool.ainvoke({"databaseName": allowed_db})
            parsed = _parse_json_maybe(result_str)
            if isinstance(parsed, dict):
                merged.extend(parsed.get("databases", []))
        result_str = json.dumps(_filter_catalog_payload({"databases": merged}, user_input), ensure_ascii=False, default=str)
        _ASSISTANT_DISCOVERY_CACHE[cache_key] = result_str
        return result_str

    database_name = args.get("databaseName")
    if name in {
        "tool_get_available_schemas",
        "tool_get_available_tables",
        "tool_get_table_schema",
        "tool_run_postgis_query",
    }:
        if database_name and not _is_allowed_database(database_name):
            return _blocked_database_result(database_name)

    exact_municipality = _requested_exact_municipality(user_input)
    if exact_municipality and name == "tool_run_postgis_query" and not route_request:
        sql = str(args.get("sql") or "")
        if "kommune_exact_where_clause" in sql or "kommune_context" in sql or "tool_get_kommune_context" in sql:
            return json.dumps({
                "executed": False,
                "database": args.get("databaseName"),
                "error": (
                    "Query blocked in assistant mode: do not reference kommune_exact_where_clause "
                    "or kommune_context inside SQL. Use kommune_where_clause and paste the returned SQL clause into WHERE."
                ),
            })
        # Auto-inject municipality filter if the SQL has no spatial filter yet
        if "ST_MakeEnvelope" not in sql and "&&" not in sql and "ST_Intersects" not in sql:
            geom_ref = _infer_geom_reference_from_sql(sql)
            if geom_ref is not None:
                table_alias, geom_column = geom_ref
                kommune_clause = kommune_where_clause.func(
                    name=exact_municipality,
                    geom_column=geom_column,
                    table_alias=table_alias,
                )
                if not str(kommune_clause).startswith("-- ERROR:"):
                    logger.info("[ASSISTANT] auto-injecting municipality clause for '%s'", exact_municipality)
                    rewritten_args = dict(args)
                    rewritten_args["sql"] = _inject_exact_municipality_clause(sql, kommune_clause)
                    tool = tool_map.get(name)
                    return await tool.ainvoke(rewritten_args)

    if route_request and name == "tool_run_postgis_query":
        return json.dumps({
            "executed": False,
            "database": args.get("databaseName"),
            "error": "Route requests should use tool_get_route directly, not tool_run_postgis_query.",
        })

    tool = tool_map.get(name)
    result = await tool.ainvoke(args)
    if name in cacheable_tools:
        _ASSISTANT_DISCOVERY_CACHE[cache_key] = result
    return result



def _empty_run_log() -> dict[str, Any]:
    return {
        "attempt_number": 0,
        "run_index": 0,
        "iterations": [],
        "step_registry": {},
        "discovered_databases": [],
        "discovered_tables": [],
        "sql_repairs_used": 0,
        "step_retry_counts": {},
        "sql_repair_counts": {},
        "attempt_tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "retrieval": {"scope_active": False, "top_score": None, "candidates": []},
    }


def _update_attempt_tokens(run_log: dict[str, Any], usage: TokenUsage | None) -> None:
    if usage is None:
        return
    tokens = dict(run_log.get("attempt_tokens") or {})
    tokens["prompt_tokens"] = tokens.get("prompt_tokens", 0) + usage.prompt_tokens
    tokens["completion_tokens"] = tokens.get("completion_tokens", 0) + usage.completion_tokens
    tokens["total_tokens"] = tokens.get("total_tokens", 0) + usage.total_tokens
    run_log["attempt_tokens"] = tokens


async def run_one_assistant(
    user_input: str,
    model_id: str,
    params: ModelParams,
    aoi_context: str | None = None,
    history_context: str | None = None,
    extra_instructions: str | None = None,
) -> PipelineResult | AgentRunError:
    t0 = time.time()
    model = _get_model(model_id, params)
    run_log = _empty_run_log()
    run_id = f"assistant_{int(time.time() * 1000)}_{model_id}".replace("-", "_")
    discovered_databases: list[str] = []
    discovered_tables: list[str] = []
    final_answer: str | None = None
    final_sql: str | None = None
    last_result: dict[str, Any] | None = None
    last_error: str | None = None
    stall_iterations = 0
    error_streak = 0
    last_error_signature: str | None = None

    retrieval_candidates: list[dict[str, Any]] = []
    retrieval_context = ""
    try:
        retrieval_candidates = [
            item for item in retrieve_database_candidates(
            user_input,
            k=EMBEDDINGS_TOP_K,
            index_path=EMBEDDINGS_TABLE_INDEX_PATH,
        ) if _is_allowed_database(item.get("database"))]
        retrieval_context = build_retrieval_context(retrieval_candidates)
        run_log["retrieval"] = {
            "scope_active": bool(retrieval_candidates),
            "top_score": retrieval_candidates[0]["score"] if retrieval_candidates else None,
            "candidates": retrieval_candidates,
        }
        preferred_databases = _candidate_database_order(retrieval_candidates) or list(ASSISTANT_ALLOWED_DATABASES)
        if preferred_databases:
            await prefetch_databases(preferred_databases)
    except Exception as e:
        logger.warning("assistant runtime retrieval failed for %s: %s", model_id, e)

    messages = [SystemMessage(content=ASSISTANT_RUNTIME_PROMPT)]
    if extra_instructions:
        messages.append(SystemMessage(content=extra_instructions))
    if retrieval_context:
        messages.append(SystemMessage(content=retrieval_context))
    municipality_start = _build_municipality_start_message(user_input)
    if municipality_start:
        messages.append(SystemMessage(content=municipality_start))
    if history_context:
        messages.append(SystemMessage(content=history_context))
    aoi_wkt = _extract_aoi_wkt(aoi_context)
    if aoi_context and not _is_route_request(user_input):
        messages.append(SystemMessage(content=_aoi_system_message(aoi_context)))
    else:
        if _is_route_request(user_input):
            messages.append(SystemMessage(content=(
                "This route request has no active AOI. Ignore any previous AOI, polygon, bbox, or municipality filter from earlier conversation."
            )))
        else:
            messages.append(SystemMessage(content=(
                "No area of interest is active for this request. "
                "Do not apply any spatial polygon filter from previous queries. "
                "Do not reuse any ST_GeomFromText, ST_MakeEnvelope, or polygon WKT coordinates seen in conversation history."
            )))
    messages.append(HumanMessage(content=user_input))

    try:
        for direct_fn in (
            _try_direct_route_query,
            _try_direct_aoi_building_count_query,
            _try_direct_buildings_in_flood_query,
            _try_direct_arealbruk_skogbonitet_query,
            _try_direct_municipality_map_query,
        ):
            direct_result, direct_sql, direct_error = await direct_fn(
                user_input=user_input,
                discovered_databases=discovered_databases,
                discovered_tables=discovered_tables,
                aoi_wkt=aoi_wkt,
            )
            if direct_result is not None:
                logger.info("[ASSISTANT] direct path OK — %d rows, sql: %s", direct_result.get("row_count", 0), (direct_sql or "")[:120])
                run_log["discovered_databases"] = discovered_databases
                run_log["discovered_tables"] = discovered_tables
                run_log["iterations"] = []
                final_answer = _build_success_answer(user_input, direct_result)
                return PipelineResult(
                    model_id=model_id,
                    final_answer=final_answer,
                    final_sql=direct_sql,
                    result_data=direct_result,
                    valid_sql=bool(direct_sql),
                    executed=True,
                    iterations=0,
                    sql_repairs_used=0,
                    restarts_used=0,
                    latency_s=round(time.time() - t0, 3),
                    all_errors=[],
                    params=params,
                    tokens=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                    run_logs=[run_log],
                    task_completed=True,
                    answer_based_on_result=True,
                )
            if direct_error:
                logger.info("[ASSISTANT] direct path failed: %s", direct_error)
                last_error = direct_error

        if last_error:
            logger.info("[ASSISTANT] falling back to LLM agent after direct path errors: %s", last_error)

        for iteration in range(1, params.max_iterations + 1):
            before_db_count = len(discovered_databases)
            before_table_count = len(discovered_tables)
            ai = await model.ainvoke(messages)
            usage = _extract_tokens(ai)
            _update_attempt_tokens(run_log, usage)
            iteration_entry: dict[str, Any] = {
                "iteration": iteration,
                "ai_content": (ai.content or "").strip(),
                "tool_calls": list(ai.tool_calls or []),
                "tools": [],
                "had_error": False,
            }
            run_log["iterations"].append(iteration_entry)
            messages.append(ai)

            if not ai.tool_calls:
                final_answer = (ai.content or "").strip() or final_answer
                logger.info("[ASSISTANT] iter %d — no tool calls, agent finished. answer: %s", iteration, (final_answer or "")[:120])
                break

            logger.info("[ASSISTANT] iter %d — %d tool call(s): %s", iteration, len(ai.tool_calls), [tc["name"] for tc in ai.tool_calls])

            tool_map = {t.name: t for t in _assistant_tools()}
            for idx, tc in enumerate(ai.tool_calls, start=1):
                tool = tool_map.get(tc["name"])
                raw_args = tc.get("args") or {}
                args = _normalize_tool_args(tc["name"], raw_args, run_id)
                if tool is None:
                    result_str = json.dumps({"error": f"Unknown tool: {tc['name']}"})
                    iteration_entry["had_error"] = True
                    iteration_entry["iteration_error"] = result_str
                    logger.warning("[ASSISTANT]   [%d/%d] UNKNOWN tool: %s", idx, len(ai.tool_calls), tc["name"])
                else:
                    logger.info("[ASSISTANT]   [%d/%d] → %s  args=%s", idx, len(ai.tool_calls), tc["name"], str(raw_args)[:200])
                    try:
                        result_str = await _invoke_assistant_tool(tool_map, tc["name"], args, user_input)
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})
                        iteration_entry["had_error"] = True
                        iteration_entry["iteration_error"] = result_str

                parsed = _parse_json_maybe(result_str)

                # log result summary (not full rows)
                if isinstance(parsed, dict):
                    if parsed.get("error"):
                        logger.warning("[ASSISTANT]   [%d/%d] ✗ %s error: %s", idx, len(ai.tool_calls), tc["name"], str(parsed["error"])[:200])
                    elif parsed.get("executed"):
                        logger.info("[ASSISTANT]   [%d/%d] ✓ %s executed, rows=%s", idx, len(ai.tool_calls), tc["name"], parsed.get("rows_returned") or parsed.get("row_count"))
                    else:
                        logger.info("[ASSISTANT]   [%d/%d] ✓ %s → keys=%s", idx, len(ai.tool_calls), tc["name"], list(parsed.keys())[:8])
                else:
                    logger.info("[ASSISTANT]   [%d/%d] ✓ %s → %s", idx, len(ai.tool_calls), tc["name"], str(result_str)[:120])

                iteration_entry["tools"].append({
                    "tool": tc["name"],
                    "args": raw_args,
                    "normalized_args": args,
                    "result": result_str,
                })
                messages.append(
                    ToolMessage(
                        content=_compact_tool_result_for_model(tc["name"], parsed, result_str),
                        tool_call_id=tc.get("id") or f"fallback_{idx}",
                    )
                )

                if tc["name"] == "tool_get_database_catalog" and isinstance(parsed, dict):
                    for db in parsed.get("databases", []):
                        db_name = db.get("database")
                        if db_name and db_name not in discovered_databases:
                            discovered_databases.append(db_name)
                        for schema in db.get("schemas", []):
                            for table in schema.get("tables", []):
                                table_name = table.get("table")
                                if table_name and table_name not in discovered_tables:
                                    discovered_tables.append(table_name)

                if tc["name"] == "tool_get_available_databases" and isinstance(parsed, list):
                    for db_name in parsed:
                        if isinstance(db_name, str) and db_name not in discovered_databases:
                            discovered_databases.append(db_name)

                if tc["name"] == "tool_get_available_tables" and isinstance(parsed, dict):
                    for table_name in parsed.get("tables", []):
                        if isinstance(table_name, str) and table_name not in discovered_tables:
                            discovered_tables.append(table_name)

                if tc["name"] == "tool_run_postgis_query" and isinstance(parsed, dict):
                    if parsed.get("executed", False):
                        last_result = _normalise_query_result(parsed)
                        final_sql = args.get("sql") or final_sql
                    else:
                        last_error = str(parsed.get("error") or "")
                        iteration_entry["had_error"] = True
                        iteration_entry["iteration_error"] = last_error

                if tc["name"] == "tool_get_route" and isinstance(parsed, dict):
                    if parsed.get("executed", False):
                        last_result = {**parsed, "geom_col": "geometry"}
                    elif parsed.get("error"):
                        last_error = str(parsed["error"])
                        iteration_entry["had_error"] = True
                        iteration_entry["iteration_error"] = last_error

            progress_made = (
                len(discovered_databases) > before_db_count
                or len(discovered_tables) > before_table_count
                or last_result is not None
            )

            if progress_made:
                stall_iterations = 0
                error_streak = 0
                last_error_signature = None
            else:
                stall_iterations += 1
                current_error = str(iteration_entry.get("iteration_error") or last_error or "").strip()
                if current_error and current_error == last_error_signature:
                    error_streak += 1
                elif current_error:
                    last_error_signature = current_error
                    error_streak = 1

            if stall_iterations >= ASSISTANT_MAX_STALL_ITERATIONS or error_streak >= ASSISTANT_MAX_REPEAT_ERROR:
                final_answer = _build_human_in_loop_message(
                    user_input=user_input,
                    last_error=last_error_signature or last_error,
                    discovered_databases=discovered_databases,
                    discovered_tables=discovered_tables,
                )
                logger.info(
                    "assistant runtime early stop model=%s stall_iterations=%s error_streak=%s",
                    model_id,
                    stall_iterations,
                    error_streak,
                )
                break

        run_log["step_registry"] = {}
        run_log["discovered_databases"] = discovered_databases
        run_log["discovered_tables"] = discovered_tables

        expanded_result = last_result
        executed = bool(expanded_result and isinstance(expanded_result, dict) and expanded_result.get("executed"))
        if executed and _is_map_intent(user_input) and not _has_drawable_geometry(expanded_result):
            last_error = "Query executed, but no drawable geometry was returned."
            executed = False
        if executed:
            if not final_answer or _looks_like_reprompt(final_answer):
                final_answer = _build_success_answer(user_input, expanded_result)
        else:
            final_answer = _build_execution_required_message(
                user_input=user_input,
                last_error=last_error,
                discovered_databases=discovered_databases,
                discovered_tables=discovered_tables,
            )
        task_completed = bool(executed and final_answer)
        all_errors = [last_error] if last_error else []
        tokens = run_log.get("attempt_tokens") or {}

        return PipelineResult(
            model_id=model_id,
            final_answer=final_answer,
            final_sql=final_sql,
            result_data=expanded_result,
            valid_sql=bool(final_sql),
            executed=executed,
            iterations=len(run_log["iterations"]),
            sql_repairs_used=0,
            restarts_used=0,
            latency_s=round(time.time() - t0, 3),
            all_errors=all_errors,
            params=params,
            tokens=TokenUsage(
                prompt_tokens=tokens.get("prompt_tokens", 0),
                completion_tokens=tokens.get("completion_tokens", 0),
                total_tokens=tokens.get("total_tokens", 0),
            ),
            run_logs=[run_log],
            task_completed=task_completed,
            answer_based_on_result=task_completed and expanded_result is not None,
        )
    except Exception as e:
        return AgentRunError(str(e), snapshot={"all_errors": [str(e)], "run_logs": [run_log]})


async def run_all_assistant(
    user_input: str,
    model_ids: list[str] | None,
    params: ModelParams,
    aoi_context: str | None = None,
    history_context: str | None = None,
    extra_instructions: str | None = None,
) -> list[PipelineResult | AgentRunError]:
    selected_models = model_ids or ALL_MODELS
    tasks = [
        run_one_assistant(
            user_input=user_input,
            model_id=model_id,
            params=params,
            aoi_context=aoi_context,
            history_context=history_context,
            extra_instructions=extra_instructions,
        )
        for model_id in selected_models
    ]
    return await asyncio.gather(*tasks)
