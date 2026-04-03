from typing import Literal

from langchain.tools import tool
import requests
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.mcp_c_s.mcp_client import (
    get_database_catalog,
    validate_sql_via_mcp,
    query_postgis,
    get_available_schemas,
    get_available_tables,
    get_table_schema,
    get_available_databases,
    run_multidb_plan,
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




@tool
async def tool_get_available_databases() -> list[str]:
    """Get all available database names."""
    return await get_available_databases()









@tool
async def execute_multidb_plan(plan: dict) -> str:
    """Execute a structured multi-database plan step by step."""
    return await run_multidb_plan(plan)


@tool
async def execute_postgis_query(sql: str, databaseName: str) -> str:
    """Execute a PostGIS SQL query via MCP and return JSON (string)."""
    rows = await query_postgis(sql, databaseName=databaseName)

    # Return JSON string, not a pretty table (safer + smaller)
    return json.dumps(rows[:5], ensure_ascii=False, default=str)



@tool
async def tool_get_available_schemas(databaseName: str) -> str:
    """Get available schemas for a given database."""
    return await get_available_schemas(databaseName=databaseName)



@tool
async def tool_get_available_tables(databaseName: str, schemaName: str) -> str:
    """Get available tables for a given schema in a given database."""
    return await get_available_tables(
        databaseName=databaseName,
        schemaName=schemaName,
    )






@tool
async def tool_get_table_schema(
    databaseName: str,
    schemaName: str,
    table_name: str,
) -> str:
    """Get column names and data types for a specific table."""
    return await get_table_schema(
        databaseName=databaseName,
        schemaName=schemaName,
        table_name=table_name,
    )





from pydantic import BaseModel, Field
from langchain.tools import tool

class QueryStep(BaseModel):
    step: int
    action: Literal["query"]
    database: str
    sql: str
    validSql: Literal["ok"]

class ExecuteQueryStepInput(BaseModel):
    run_id: str
    step: QueryStep

class ValidateSqlInput(BaseModel):
    sql: str
    databaseName: str
    step_number: int










# @tool
# async def tool_validate_sql(
#     sql: str,
#     databaseName: str,
# ) -> str:
#     """Validate SQL syntax using EXPLAIN. Returns ok or error message."""
#     return await validate_sql_via_mcp(sql, databaseName=databaseName)


@tool(args_schema=ValidateSqlInput)
async def tool_validate_sql(sql: str, databaseName: str, step_number: int) -> str:
    """Validate SQL syntax using EXPLAIN. Returns ok or error message."""
    return await validate_sql_via_mcp(
        sql=sql,
        databaseName=databaseName,
        step_number=step_number,
    )









@tool
def kommune_where_clause(name: str) -> str:
    """
    Return a PostGIS WHERE clause that restricts results to a Norwegian municipality
    using a bounding box derived from Kartverket APIs.
    """

    # Look up municipality number by name
    r = requests.get(
        "https://api.kartverket.no/kommuneinfo/v1/sok",
        params={"knavn": name},
        timeout=10,
    )
    r.raise_for_status()
    kommune_nr = r.json()["kommuner"][0]["kommunenummer"]

    # Fetch municipality geometry (MultiPolygon)
    r = requests.get(
        f"https://api.test.kartverket.no/kommuneinfo/v1/kommuner/{kommune_nr}/omrade",
        timeout=10,
    )
    r.raise_for_status()
    coords = r.json()["omrade"]["coordinates"]

    # Compute bounding box from MultiPolygon
    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    for polygon in coords:
        for ring in polygon:
            for lon, lat in ring:
                minx = min(minx, lon)
                miny = min(miny, lat)
                maxx = max(maxx, lon)
                maxy = max(maxy, lat)

    # Return PostGIS WHERE clause using transformed bounding box
    return f"""
a.geom && ST_Transform(
  ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326),
  25833
)
""".strip()





@tool
def get_available_gis_operations():
    """Return supported GIS operations for query building."""
    return [
        "intersects",
        "within",
        "contains",
        "touches",
        "overlaps",
        "buffer",
        "distance",
        "area",
        "centroid",
        "union",
        "intersection",
        "count",
        "group_by",
    ]












# @tool
# async def tool_execute_query_step(run_id: str, step: dict) -> str:
#     """Execute one validated SQL query step."""
#     return json.dumps(await execute_query_step(run_id, step), ensure_ascii=False, default=str)




@tool(args_schema=ExecuteQueryStepInput)
async def tool_execute_query_step(run_id: str, step: QueryStep) -> str:
    """Execute one validated SQL query step."""
    return json.dumps(
        await execute_query_step(run_id, step.model_dump()),
        ensure_ascii=False,
        default=str,
    )









@tool
async def tool_execute_filter_step(run_id: str, step: dict) -> str:
    """Filter rows from a previous step."""
    return json.dumps(await execute_filter_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_select_columns_step(run_id: str, step: dict) -> str:
    """Select specific columns from a previous step."""
    return json.dumps(await execute_select_columns_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_spatial_join_step(run_id: str, step: dict) -> str:
    """Spatial join between two previous step results."""
    return json.dumps(await execute_spatial_join_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_attribute_join_step(run_id: str, step: dict) -> str:
    """Attribute join between two previous step results."""
    return json.dumps(await execute_attribute_join_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_buffer_step(run_id: str, step: dict) -> str:
    """Buffer geometries from a previous step."""
    return json.dumps(await execute_buffer_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_nearest_step(run_id: str, step: dict) -> str:
    """Find nearest geometry from another step."""
    return json.dumps(await execute_nearest_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_distance_step(run_id: str, step: dict) -> str:
    """Compute nearest distance to another step."""
    return json.dumps(await execute_distance_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_aggregate_step(run_id: str, step: dict) -> str:
    """Aggregate rows from a previous step."""
    return json.dumps(await execute_aggregate_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_sort_step(run_id: str, step: dict) -> str:
    """Sort rows from a previous step."""
    return json.dumps(await execute_sort_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_limit_step(run_id: str, step: dict) -> str:
    """Limit rows from a previous step."""
    return json.dumps(await execute_limit_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_clip_step(run_id: str, step: dict) -> str:
    """Clip one geometry layer by another."""
    return json.dumps(await execute_clip_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_union_step(run_id: str, step: dict) -> str:
    """Union all geometries from a previous step."""
    return json.dumps(await execute_union_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_intersection_step(run_id: str, step: dict) -> str:
    """Overlay intersection between two geometry layers."""
    return json.dumps(await execute_intersection_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_difference_step(run_id: str, step: dict) -> str:
    """Overlay difference between two geometry layers."""
    return json.dumps(await execute_difference_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_dissolve_step(run_id: str, step: dict) -> str:
    """Dissolve geometries by group."""
    return json.dumps(await execute_dissolve_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_centroid_step(run_id: str, step: dict) -> str:
    """Compute centroids for a geometry layer."""
    return json.dumps(await execute_centroid_step(run_id, step), ensure_ascii=False, default=str)


@tool
async def tool_execute_transform_crs_step(run_id: str, step: dict) -> str:
    """Transform geometries to another CRS."""
    return json.dumps(await execute_transform_crs_step(run_id, step), ensure_ascii=False, default=str)





@tool
async def tool_get_database_catalog(
    databaseName: str | None = None,
    schemaName: str | None = None,
) -> str:
    """Get database catalog with schemas, tables, columns, geometry columns and SRID info."""
    return json.dumps(
        await get_database_catalog(databaseName=databaseName, schemaName=schemaName),
        ensure_ascii=False,
        default=str,
    )