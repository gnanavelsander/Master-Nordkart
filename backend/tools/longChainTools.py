from langchain.tools import tool
from pydantic import BaseModel
from pathlib import Path
import requests
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.mcp_c_s.mcp_client import (
    get_database_catalog,
    validate_sql_via_mcp,
    get_available_schemas,
    get_available_tables,
    get_table_schema,
    get_available_databases,
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


# --------------------------------------------------
# Input schemas
# --------------------------------------------------

class ValidateSqlInput(BaseModel):
    sql: str
    databaseName: str
    step_number: int


# --------------------------------------------------
# Discovery tools
# --------------------------------------------------

@tool
async def tool_get_available_databases() -> list[str]:
    """Get all available database names."""
    return await get_available_databases()


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


# --------------------------------------------------
# Validation tool
# --------------------------------------------------

@tool(args_schema=ValidateSqlInput)
async def tool_validate_sql(sql: str, databaseName: str, step_number: int) -> str:
    """Validate SQL syntax using EXPLAIN. Returns JSON with validated=true/false."""
    return await validate_sql_via_mcp(sql=sql, databaseName=databaseName, step_number=step_number)


# --------------------------------------------------
# Municipality filter tool
# --------------------------------------------------

@tool
def kommune_where_clause(name: str, geom_column: str, table_alias: str = "") -> str:
    """
    Return a PostGIS WHERE clause that restricts results to a Norwegian municipality
    using a bounding box derived from Kartverket APIs.

    Args:
        name: Norwegian municipality name (e.g. 'Oslo', 'Bergen').
        geom_column: Geometry column name in the target table (e.g. 'representasjonspunkt', 'omrade', 'geometri').
        table_alias: Optional SQL table alias (e.g. 'b' gives 'b.omrade'). Leave empty for no alias.
    """
    r = requests.get(
        "https://api.kartverket.no/kommuneinfo/v1/sok",
        params={"knavn": name},
        timeout=10,
    )
    r.raise_for_status()
    kommune_nr = r.json()["kommuner"][0]["kommunenummer"]

    r = requests.get(
        f"https://api.test.kartverket.no/kommuneinfo/v1/kommuner/{kommune_nr}/omrade",
        timeout=10,
    )
    r.raise_for_status()
    coords = r.json()["omrade"]["coordinates"]

    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    for polygon in coords:
        for ring in polygon:
            for lon, lat in ring:
                minx = min(minx, lon)
                miny = min(miny, lat)
                maxx = max(maxx, lon)
                maxy = max(maxy, lat)

    # Build qualified column reference
    col_ref = f"{table_alias}.{geom_column}" if table_alias else geom_column

    return (
        f"{col_ref} && ST_Transform("
        f"ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326), 25833)"
    )


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