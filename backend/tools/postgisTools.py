from langchain.tools import tool


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