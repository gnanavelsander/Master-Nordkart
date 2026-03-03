from langchain.tools import tool
import requests


import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.mcp_c_s.mcp_client import (
    get_available_tables,    
    get_table_schema,
    validate_sql_via_mcp,
    query_postgis,
    
)

@tool
async def execute_postgis_query(sql: str) -> str:
    """Execute a PostGIS SQL query via MCP and return JSON (string)."""

    df = await query_postgis(sql)

    # Return JSON string, not a pretty table (safer + smaller)
    return df.head(5).to_json(orient="records", force_ascii=False)




@tool
async def tool_get_available_tables() -> str:
    """Get all available tables in the PostGIS database."""
    return await get_available_tables()

@tool
async def tool_get_table_schema(table_name: str) -> str:
    """Get column names and data types for a specific table."""
    return await get_table_schema(table_name)

@tool
async def tool_validate_sql(sql: str) -> str:
    """Validate SQL syntax using EXPLAIN. Returns ok or error message."""
    return await validate_sql_via_mcp(sql)





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




