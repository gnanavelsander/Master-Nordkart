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
from config import PGCONN_STRING, PGCONN_MATRIKKEL, PGCONN_BRANNSTASJONER, PGCONN_FLOMSONER,PGCONN_FLOMAKTSOMRADER, PGCONN_KVIKKLEIRE, PGCONN_MATRIKKELENEIENDOMSKARTTEIG, PGCONN_SNOSKREDAKTSOMHETSKART, PGCONN_AKTSOMHETSKART_JORD_FLOMSKRED, PGCONN_KOMMUNER
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

    poolManager = PoolManager()

    conn_ids = {
       # "007workshop": await poolManager.connect(PGCONN_STRING),
        "matrikkelenbygning": await poolManager.connect(PGCONN_MATRIKKEL),



        "flomaktsomrader": await poolManager.connect(PGCONN_FLOMAKTSOMRADER),
        "flomsoner": await poolManager.connect(PGCONN_FLOMSONER),
        "brannstasjoner": await poolManager.connect(PGCONN_BRANNSTASJONER),

        "kvikkleire": await poolManager.connect(PGCONN_KVIKKLEIRE),
        "snoskredaktsomhetskart": await poolManager.connect(PGCONN_SNOSKREDAKTSOMHETSKART),
        "matrikkeleneiendomskartteig": await poolManager.connect(PGCONN_MATRIKKELENEIENDOMSKARTTEIG),
        "aktsomhetskart_jord_flomskred": await poolManager.connect(PGCONN_AKTSOMHETSKART_JORD_FLOMSKRED),
        "kommuner": await poolManager.connect(PGCONN_KOMMUNER),



    }

    try:
        logger.info("Server ready, yielding app context")
        yield AppContext(db=poolManager, conn_ids=conn_ids)
    finally:
        logger.info("Cleaning up database connection")
        await poolManager.close_all()



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
    name="get_available_schemas",
    description="Get available schemas for a given database"
)
async def get_available_schemas(
    ctx: Context[ServerSession, AppContext],
    databaseName: str,
) -> str:
    try:
        conn_id = ctx.request_context.lifespan_context.conn_ids[databaseName]

        rows = await ctx.request_context.lifespan_context.db.query(
            conn_id,
            """
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
            ORDER BY schema_name;
            """
        )

        return json.dumps({
            "databaseName": databaseName,
            "schemas": [row["schema_name"] for row in rows]
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)





@mcp.tool(
    name="get_available_databases",
    description="Get available database names"
)
async def get_available_databases(
    ctx: Context[ServerSession, AppContext],
) -> str:
    try:
        db_names = list(ctx.request_context.lifespan_context.conn_ids.keys())

        return json.dumps({
            "databases": db_names
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
















@mcp.tool(
    name="get_available_tables",
    description="Get available tables for a given schema in a given database"
)
async def get_available_tables(
    ctx: Context[ServerSession, AppContext],
    databaseName: str,
    schemaName: str,
) -> str:
    try:
        conn_id = ctx.request_context.lifespan_context.conn_ids[databaseName]

        rows = await ctx.request_context.lifespan_context.db.query(
            conn_id,
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema = $1
            ORDER BY table_name;
            """,
            (schemaName,)
        )

        return json.dumps({
            "databaseName": databaseName,
            "schemaName": schemaName,
            "tables": [row["table_name"] for row in rows]
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)






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





@mcp.tool(
    name="get_table_schema",
    description="Get column names and types for a given table in a given schema"
)
async def get_table_schema(
    ctx: Context[ServerSession, AppContext],
    databaseName: str,
    schemaName: str,
    table_name: str,
) -> str:
    try:
        conn_id = ctx.request_context.lifespan_context.conn_ids[databaseName]

        rows = await ctx.request_context.lifespan_context.db.query(
            conn_id,
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = $1
              AND table_name = $2
            ORDER BY ordinal_position;
            """,
            (schemaName, table_name)
        )

        if not rows:
            return json.dumps({
                "databaseName": databaseName,
                "schemaName": schemaName,
                "table_name": table_name,
                "columns": [],
                "error": f"Table '{schemaName}.{table_name}' not found."
            }, ensure_ascii=False)

        return json.dumps({
            "databaseName": databaseName,
            "schemaName": schemaName,
            "table_name": table_name,
            "columns": [
                {
                    "column_name": row["column_name"],
                    "data_type": row["data_type"]
                }
                for row in rows
            ]
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)




if __name__ == "__main__":
    logger.info("Starting MCP server")
    mcp.run()