from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import sys
import os

from pathlib import Path


# project_root = os.path.dirname(os.path.abspath(__file__))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# parent_dir = os.path.dirname(project_root)
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)


from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parents[2] / ".env"
loaded = load_dotenv(env_path, override=True)

if not loaded:
    raise RuntimeError(f".env not found at {env_path}")

if not os.getenv("PGCONN_STRING"):
    raise RuntimeError("PGCONN_STRING missing after loading .env")



ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from backend.database import PoolManager
from config import PGCONN_STRING
from backend.log_utils import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


#workshop = os.environ.get("PGCONN_STRING") or PGCONN_STRING



@dataclass
class AppContext:
    db: PoolManager
    conn_ids: dict[str, str]




#create app contex manger
@asynccontextmanager
async def app_lifespan(server: FastMCP):

    db = PoolManager()

    conn_ids = {
        "007workshop": await db.connect(os.environ.get("PGCONN_STRING") or PGCONN_STRING)
    }

    try:
        logger.info("Server ready, yielding app context")
        yield AppContext(db=db, conn_ids=conn_ids)
    finally:
        logger.info("Cleaning up database connection")
        await db.close_all()



# ---------- MCP server ----------
mcp = FastMCP("postgis-server", lifespan=app_lifespan)


# ---------- Tool ----------
@mcp.tool(
    name="run_postgis_query",
    description="Execute SQL query on PostGIS database and return results as JSON"
)
async def run_postgis_query(
    ctx: Context[ServerSession, AppContext],
    sql: str,
    databaseName: str
) -> str:
    """Execute a SQL query and return results as JSON string."""
    logger.info(f"Tool called with SQL: {sql[:100]}...")
    try:
        conID = ctx.request_context.lifespan_context.conn_ids[databaseName]
        result = await ctx.request_context.lifespan_context.db.query(conID, sql)

        logger.info(f"Returning {len(result)} records")
        return json.dumps(result, default=str)
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return f"Error executing tool run_postgis_query: {str(e)}"







@mcp.tool(
    name="get_available_tables",
    description="Get available tables from the public schema"
)
async def get_available_tables(
    ctx: Context[ServerSession, AppContext],
    databaseName: str,
) -> str:
    try:
        conn_id = ctx.request_context.lifespan_context.conn_ids[databaseName]

        rows = await ctx.request_context.lifespan_context.db.query(
            conn_id,
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_schema, table_name
            """
        )

        tables = [f"{row['table_schema']}.{row['table_name']}" for row in rows]

        return ", ".join(tables)

    except Exception as e:
        return f"Error: {e}"




@mcp.tool(
    name="get_table_schema",
    description="Get column names and types for a given table in the public schema"
)
async def get_table_schema(
    ctx: Context[ServerSession, AppContext],
    table_name: str,
    databaseName: str,
) -> str:
    try:
        conn_id = ctx.request_context.lifespan_context.conn_ids[databaseName]

        rows = await ctx.request_context.lifespan_context.db.query(
            conn_id,
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = $1
            ORDER BY ordinal_position
            """,
            (table_name,)
        )

        if not rows:
            return f"Table '{table_name}' not found."

        return "\n".join(
            f"{row['column_name']}: {row['data_type']}"
            for row in rows
        )

    except Exception as e:
        return f"Error: {e}"










@mcp.tool(
    name="validate_sql_explain",
    description="Validate SQL syntax using EXPLAIN. Returns ok or error message."
)
async def validate_sql_explain(
    ctx: Context[ServerSession, AppContext],
    sql: str,
    databaseName: str
) -> str:
    try:
        conID = ctx.request_context.lifespan_context.conn_ids[databaseName]
        await ctx.request_context.lifespan_context.db.query(
            conID,
            f"EXPLAIN (FORMAT JSON) {sql}"
        )
        return "ok"
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    logger.info("Starting MCP server")
    print(f"Running MCP server...", flush=False)
    mcp.run()