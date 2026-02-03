from langchain.tools import tool


import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp_c_s.mcp_client import query_postgis

@tool
async def execute_postgis_query(sql: str) -> str:
    """Execute a PostGIS SQL query via MCP and return JSON (string)."""

    df = await query_postgis(sql)

    # Return JSON string, not a pretty table (safer + smaller)
    return df.to_json(orient="records", force_ascii=False)