import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import os
import traceback

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

from core.planner import process_user_input
from core.sql_builder import plan_to_sql, enrich_plan




# from mcp_c_s.mcp_client import query_postgis 
# from ui.map_utils import showMap



# project_root = os.path.dirname(os.path.abspath(__file__))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# parent_dir = os.path.dirname(project_root)
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)


from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from database import Database
from config import PGCONN_STRING
from log_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)




@dataclass
class AppContext:
    db: Database

@asynccontextmanager
async def app_lifespan(server: FastMCP):
    logger.info("Starting server lifespan")
    conn_str = os.environ.get("PGCONN_STRING")
    if not conn_str:
        logger.error("PGCONN_STRING not set")
        raise ValueError("PGCONN_STRING environment variable not set")
    
    logger.info("Initializing database connection")
    db = Database(conn_str).connect()
    try:
        logger.info("Server ready, yielding app context")
        yield AppContext(db=db)
    finally:
        logger.info("Cleaning up database connection")
        db.disconnect()



# ---------- MCP server ----------
mcp = FastMCP("postgis-server", lifespan=app_lifespan)


# ---------- Tool ----------
@mcp.tool(
    name="run_postgis_query",
    description="Execute SQL query on PostGIS database and return results as JSON"
)
# def run_postgis_query(
#     ctx: Context[ServerSession, AppContext],
#     sql: str
# ) -> str:
#     """Execute a SQL query and return results as JSON string."""
#     logger.info(f"Tool called with SQL: {sql[:100]}...")
#     try:
#         df = ctx.request_context.lifespan_context.db.query(sql)
#         result = df.to_dict(orient="records")
#         logger.info(f"Returning {len(result)} records")
#         return json.dumps(result, default=str)
#     except Exception as e:
#         logger.error(f"Tool execution failed: {e}")
#         return f"Error executing tool run_postgis_query: {str(e)}"




@mcp.tool()
def ping() -> str:
     return "pong"


# @mcp.tool()
# def make_plan(user_text: str) -> str:
#     plan = process_user_input(user_text)   # din eksisterende planner
#     return json.dumps(plan)

# @mcp.tool()
# def plan_to_sql_tool(plan_json: str) -> str:
#     plan = json.loads(plan_json)
#     sql = plan_to_sql(plan)                # din eksisterende builder
#     return sql

# @mcp.tool()
# def run_postgis_query(sql: str) -> str:
#     rows = db.query(sql)                   # din eksisterende db
#     return json.dumps(rows)



@mcp.tool(
    name="gis_query",
    description="End-to-end GIS query: user_text -> plan -> SQL -> PostGIS rows"
)
def gis_query(
    ctx: Context[ServerSession, AppContext],
    user_query: str,
    k: int = 8
) -> str:
    try:
        plan = process_user_input(user_query)
        plan  = enrich_plan(plan)
        
        sql = plan_to_sql(plan)

        df = ctx.request_context.lifespan_context.db.query(sql)
        rows = df.to_dict(orient="records")

        return json.dumps({
            "ok": True,
            "data": {
                "plan": plan,
                "sql": sql,
                "rows": rows
            }
        }, default=str)

    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})





















if __name__ == "__main__":
    try:
        mcp.run()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
