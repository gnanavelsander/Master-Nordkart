from langchain.tools import tool
import requests
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.mcp_c_s.mcp_client import (
   
    validate_sql_via_mcp,
    query_postgis,
    get_available_schemas,
    get_available_tables,
    get_table_schema,
    get_available_databases
)





@tool
async def tool_get_available_databases() -> list[str]:
    """Get all available database names."""
    return await get_available_databases()

























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











@tool
async def tool_validate_sql(
    sql: str,
    databaseName: str,
) -> str:
    """Validate SQL syntax using EXPLAIN. Returns ok or error message."""
    return await validate_sql_via_mcp(sql, databaseName=databaseName)


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