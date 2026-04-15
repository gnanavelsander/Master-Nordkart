from langchain.tools import tool
from functools import lru_cache
from pathlib import Path
import requests
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ORS_API_KEY

from backend.mcp.client.tools_discovery import (
    get_database_catalog,
    validate_sql_via_mcp,
    get_available_schemas,
    get_available_tables,
    get_table_schema,
    get_available_databases,
    run_postgis_query,
)
from backend.mcp.client.tools_steps import (
    execute_query_step,
    execute_filter_step,
    execute_select_columns_step,
    execute_spatial_join_step,
    execute_attribute_join_step,
    execute_buffer_step,
    execute_nearest_step,
    execute_distance_step,
    execute_aggregate_step,
    execute_sort_step,
    execute_limit_step,
    execute_clip_step,
    execute_union_step,
    execute_intersection_step,
    execute_difference_step,
    execute_dissolve_step,
    execute_centroid_step,
    execute_transform_crs_step,
)


def _safe_json(data) -> str:
    """Serialize to JSON string, always safe for the agent to parse."""
    return json.dumps(data, ensure_ascii=False, default=str)


def _err_json(step: int | None, action: str, msg: str) -> str:
    """Consistent error JSON returned to agent when a tool fails."""
    return _safe_json({"executed": False, "step": step, "action": action, "error": str(msg)})


@lru_cache(maxsize=64)
def _fetch_kommune_geometry(name: str) -> tuple[str, dict, tuple[float, float, float, float]]:
    r = requests.get(
        "https://api.kartverket.no/kommuneinfo/v1/sok",
        params={"knavn": name},
        timeout=10,
    )
    r.raise_for_status()
    kommuner = r.json().get("kommuner", [])
    if not kommuner:
        raise ValueError(f"No municipality found for '{name}'")

    kommune_nr = kommuner[0]["kommunenummer"]

    r = requests.get(
        f"https://api.kartverket.no/kommuneinfo/v1/kommuner/{kommune_nr}/omrade",
        timeout=10,
    )
    r.raise_for_status()
    omrade = r.json()["omrade"]
    coords = omrade["coordinates"]

    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for polygon in coords:
        for ring in polygon:
            for lon, lat in ring:
                minx = min(minx, lon)
                miny = min(miny, lat)
                maxx = max(maxx, lon)
                maxy = max(maxy, lat)

    return kommune_nr, omrade, (minx, miny, maxx, maxy)


# # --------------------------------------------------
# # Input schemas
# # --------------------------------------------------

# class ValidateSqlInput(BaseModel):
#     sql: str
#     databaseName: str
#     step_number: int











# --------------------------------------------------
# Discovery tools
# --------------------------------------------------

@tool
async def tool_get_available_databases() -> str:
    """Get all available database names."""
    return _safe_json(await get_available_databases())


@tool
async def tool_get_available_schemas(databaseName: str) -> str:
    """Get available schemas for a given database."""
    return _safe_json(await get_available_schemas(databaseName=databaseName))


@tool
async def tool_get_available_tables(databaseName: str, schemaName: str) -> str:
    """Get available tables for a given schema in a given database."""
    return _safe_json(await get_available_tables(databaseName=databaseName, schemaName=schemaName))


@tool
async def tool_get_table_schema(databaseName: str, schemaName: str, table_name: str) -> str:
    """Get column names and data types for a specific table."""
    return _safe_json(await get_table_schema(databaseName=databaseName, schemaName=schemaName, table_name=table_name))


@tool
async def tool_get_database_catalog(
    databaseName: str | None = None,
    schemaName: str | None = None,
) -> str:
    """Get database catalog with schemas, tables, geometry columns and SRID info."""
    return _safe_json(await get_database_catalog(databaseName=databaseName, schemaName=schemaName))


@tool
async def tool_run_postgis_query(sql: str, databaseName: str) -> str:
    """Execute a direct SQL query against one PostGIS database for simple single-database assistant tasks."""
    try:
        rows = await run_postgis_query(sql=sql, databaseName=databaseName)
        columns = list(rows[0].keys()) if rows else []
        return _safe_json({
            "executed": True,
            "database": databaseName,
            "row_count": len(rows),
            "rows_returned": len(rows),
            "columns": columns,
            "rows": rows,
        })
    except Exception as e:
        return _safe_json({
            "executed": False,
            "database": databaseName,
            "error": str(e),
        })


# --------------------------------------------------
# Validation tool
# --------------------------------------------------

@tool
async def tool_validate_sql(sql: str, databaseName: str, step_number: int) -> str:
    """Validate SQL syntax using EXPLAIN through MCP and return JSON."""
    return _safe_json(
        await validate_sql_via_mcp(
            sql=sql,
            databaseName=databaseName,
            step_number=step_number,
        )
    )
# --------------------------------------------------
# Municipality filter tool
# --------------------------------------------------

@tool
def kommune_where_clause(name: str, geom_column: str, table_alias: str = "") -> str:
    """
    Return a PostGIS WHERE clause that restricts results exactly to a Norwegian municipality
    using the official municipality polygon from Kartverket.

    Uses a bbox index scan (&&) combined with ST_Intersects against the actual municipality
    polygon for precise clipping. Works for any source SRID via ST_SRID().

    Args:
        name: Norwegian municipality name (e.g. 'Oslo', 'Bergen', 'Grimstad').
        geom_column: Geometry column name in the target table.
        table_alias: Optional SQL table alias (e.g. 'b' → 'b.geom'). Leave empty for no alias.
    """
    try:
        _kommune_nr, omrade, (minx, miny, maxx, maxy) = _fetch_kommune_geometry(name)
        col_ref = f"{table_alias}.{geom_column}" if table_alias else geom_column
        geom_json = json.dumps(omrade, ensure_ascii=False).replace("'", "''")

        return (
            f"{col_ref} && ST_Transform(ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326), ST_SRID({col_ref})) "
            f"AND ST_Intersects("
            f"{col_ref}, "
            f"ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON('{geom_json}'), 4326), ST_SRID({col_ref}))"
            f")"
        )
    except Exception as e:
        return f"-- ERROR: kommune_where_clause failed for '{name}': {e}"


@tool
def kommune_exact_where_clause(name: str, geom_column: str, table_alias: str = "") -> str:
    """
    Return an exact PostGIS municipality filter using the municipality polygon from Kartverket.

    This is more precise than kommune_where_clause because it uses the actual kommune geometry,
    not only the municipality bounding box.
    """
    try:
        _kommune_nr, omrade, (minx, miny, maxx, maxy) = _fetch_kommune_geometry(name)
        col_ref = f"{table_alias}.{geom_column}" if table_alias else geom_column
        geom_json = json.dumps(omrade, ensure_ascii=False).replace("'", "''")

        return (
            f"{col_ref} && ST_Transform("
            f"ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326), ST_SRID({col_ref})) "
            f"AND ST_Intersects("
            f"{col_ref}, "
            f"ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON('{geom_json}'), 4326), ST_SRID({col_ref}))"
            f")"
        )
    except Exception as e:
        return f"-- ERROR: kommune_exact_where_clause failed for '{name}': {e}"


@tool
def tool_get_kommune_context(name: str, geom_column: str, table_alias: str = "") -> str:
    """
    Get structured municipality context from Kartverket kommuneinfo:
    municipality number, bbox, and both coarse and exact SQL clauses for a geometry column.
    """
    try:
        kommune_nr, omrade, (minx, miny, maxx, maxy) = _fetch_kommune_geometry(name)
        col_ref = f"{table_alias}.{geom_column}" if table_alias else geom_column
        geom_json = json.dumps(omrade, ensure_ascii=False).replace("'", "''")
        bbox_clause = (
            f"{col_ref} && ST_Transform("
            f"ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326), ST_SRID({col_ref}))"
        )
        exact_clause = (
            f"{bbox_clause} "
            f"AND ST_Intersects("
            f"{col_ref}, "
            f"ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON('{geom_json}'), 4326), ST_SRID({col_ref}))"
            f")"
        )
        return _safe_json({
            "name": name,
            "kommunenummer": kommune_nr,
            "bbox": {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy},
            "bbox_clause": bbox_clause,
            "exact_clause": exact_clause,
        })
    except Exception as e:
        return _safe_json({"error": str(e)})




# --------------------------------------------------
# Step tools — all catch exceptions and return JSON
# --------------------------------------------------

@tool
async def tool_execute_query_step(run_id: str, step: dict) -> str:
    """Execute one validated SQL query step."""
    try:
        return _safe_json(await execute_query_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "query", str(e))


@tool
async def tool_execute_filter_step(run_id: str, step: dict) -> str:
    """Filter rows from a previous step."""
    try:
        return _safe_json(await execute_filter_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "filter", str(e))


@tool
async def tool_execute_select_columns_step(run_id: str, step: dict) -> str:
    """Select specific columns from a previous step."""
    try:
        return _safe_json(await execute_select_columns_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "select_columns", str(e))


@tool
async def tool_execute_spatial_join_step(run_id: str, step: dict) -> str:
    """Spatial join between two previous step results."""
    try:
        return _safe_json(await execute_spatial_join_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "spatial_join", str(e))


@tool
async def tool_execute_attribute_join_step(run_id: str, step: dict) -> str:
    """Attribute join between two previous step results."""
    try:
        return _safe_json(await execute_attribute_join_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "attribute_join", str(e))


@tool
async def tool_execute_buffer_step(run_id: str, step: dict) -> str:
    """Buffer geometries from a previous step."""
    try:
        return _safe_json(await execute_buffer_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "buffer", str(e))


@tool
async def tool_execute_nearest_step(run_id: str, step: dict) -> str:
    """Find nearest geometry from another step."""
    try:
        return _safe_json(await execute_nearest_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "nearest", str(e))


@tool
async def tool_execute_distance_step(run_id: str, step: dict) -> str:
    """Compute nearest distance to another step."""
    try:
        return _safe_json(await execute_distance_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "distance", str(e))


@tool
async def tool_execute_aggregate_step(run_id: str, step: dict) -> str:
    """Aggregate rows from a previous step."""
    try:
        return _safe_json(await execute_aggregate_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "aggregate", str(e))


@tool
async def tool_execute_sort_step(run_id: str, step: dict) -> str:
    """Sort rows from a previous step."""
    try:
        return _safe_json(await execute_sort_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "sort", str(e))


@tool
async def tool_execute_limit_step(run_id: str, step: dict) -> str:
    """Limit rows from a previous step."""
    try:
        return _safe_json(await execute_limit_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "limit", str(e))


@tool
async def tool_execute_clip_step(run_id: str, step: dict) -> str:
    """Clip one geometry layer by another."""
    try:
        return _safe_json(await execute_clip_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "clip", str(e))


@tool
async def tool_execute_union_step(run_id: str, step: dict) -> str:
    """Union all geometries from a previous step."""
    try:
        return _safe_json(await execute_union_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "union", str(e))


@tool
async def tool_execute_intersection_step(run_id: str, step: dict) -> str:
    """Overlay intersection between two geometry layers."""
    try:
        return _safe_json(await execute_intersection_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "intersection", str(e))


@tool
async def tool_execute_difference_step(run_id: str, step: dict) -> str:
    """Overlay difference between two geometry layers."""
    try:
        return _safe_json(await execute_difference_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "difference", str(e))


@tool
async def tool_execute_dissolve_step(run_id: str, step: dict) -> str:
    """Dissolve geometries by group."""
    try:
        return _safe_json(await execute_dissolve_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "dissolve", str(e))


@tool
async def tool_execute_centroid_step(run_id: str, step: dict) -> str:
    """Compute centroids for a geometry layer."""
    try:
        return _safe_json(await execute_centroid_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "centroid", str(e))


@tool
async def tool_execute_transform_crs_step(run_id: str, step: dict) -> str:
    """Transform geometries to another CRS."""
    try:
        return _safe_json(await execute_transform_crs_step(run_id, step))
    except Exception as e:
        return _err_json(step.get("step") if isinstance(step, dict) else None, "transform_crs", str(e))


# --------------------------------------------------
# Routing tool — OpenRouteService
# --------------------------------------------------

_ORS_PROFILES = {
    "foot-walking", "foot-hiking",
    "driving-car", "driving-hgv",
    "cycling-regular", "cycling-road", "cycling-mountain", "cycling-electric",
    "wheelchair",
}

_ORS_BASE = "https://api.openrouteservice.org/v2/directions"


@lru_cache(maxsize=256)
def _geocode_place(place: str) -> tuple[float, float] | None:
    if not place or not place.strip():
        return None
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": place, "format": "jsonv2", "limit": 1},
        headers={"User-Agent": "llm-gis-poc/1.0"},
        timeout=10,
    )
    resp.raise_for_status()
    hits = resp.json()
    if not hits:
        return None
    return float(hits[0]["lon"]), float(hits[0]["lat"])


@tool
def tool_get_route(
    coordinates: list | None = None,
    profile: str = "foot-walking",
    origin: str | None = None,
    destination: str | None = None,
) -> str:
    """
    Calculate a route between two or more waypoints using OpenRouteService.

    Returns a GeoJSON FeatureCollection with the route line geometry and
    summary properties (distance in metres, duration in seconds).

    Args:
        coordinates: Optional list of [longitude, latitude] pairs, e.g.
                     [[10.74, 59.91], [10.75, 59.92]].
                     At least two points are required if origin/destination are not provided.
        profile: Travel mode. One of:
                 foot-walking (default), foot-hiking,
                 driving-car, driving-hgv,
                 cycling-regular, cycling-road, cycling-mountain, cycling-electric,
                 wheelchair.
        origin: Optional start place name for geocoding, e.g. "Oslo S".
        destination: Optional end place name for geocoding, e.g. "Rådhuset i Oslo".
    """
    if not ORS_API_KEY:
        return _safe_json({"error": "ORS_API_KEY is not configured."})
    if (not isinstance(coordinates, list) or len(coordinates) < 2) and origin and destination:
        try:
            origin_coords = _geocode_place(origin)
            destination_coords = _geocode_place(destination)
        except Exception as e:
            return _safe_json({"error": f"Route geocoding failed: {e}"})
        if not origin_coords or not destination_coords:
            return _safe_json({"error": f"Could not geocode route endpoints: {origin} -> {destination}"})
        coordinates = [list(origin_coords), list(destination_coords)]
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        return _safe_json({"error": "Provide either coordinates or both origin and destination."})
    if profile not in _ORS_PROFILES:
        profile = "foot-walking"

    url = f"{_ORS_BASE}/{profile}/geojson"
    headers = {
        "Accept": "application/json",
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json; charset=utf-8",
    }
    body = {"coordinates": coordinates}

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as e:
        return _safe_json({"error": f"ORS HTTP {resp.status_code}: {resp.text[:300]}"})
    except Exception as e:
        return _safe_json({"error": str(e)})

    # Surface the summary in each feature's properties so the agent can read it
    features = data.get("features") or []
    for feat in features:
        props = feat.get("properties") or {}
        summary = props.get("summary") or {}
        feat["properties"] = {
            **props,
            "distance_m": summary.get("distance"),
            "duration_s": summary.get("duration"),
        }

    return _safe_json({
        "executed": True,
        "profile": profile,
        "geojson": data,
        "geom_col": "geometry",
        "rows": [
            {
                "geometry": json.dumps(feat["geometry"]),
                **{k: v for k, v in (feat.get("properties") or {}).items()},
            }
            for feat in features
        ],
        "row_count": len(features),
    })
