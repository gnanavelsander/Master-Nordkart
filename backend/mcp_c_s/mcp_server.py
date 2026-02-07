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
from backend.database import Database
from config import PGCONN_STRING
from backend.log_utils import setup_logging


setup_logging()
logger = logging.getLogger(__name__)




@dataclass
class AppContext:
    db: Database



@asynccontextmanager
async def app_lifespan(server: FastMCP):
    logger.info("Starting server lifespan")
    conn_str = os.environ.get("PGCONN_STRING") or PGCONN_STRING
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
def run_postgis_query(
    ctx: Context[ServerSession, AppContext],
    sql: str
) -> str:
    """Execute a SQL query and return results as JSON string."""
    logger.info(f"Tool called with SQL: {sql[:100]}...")
    try:
        df = ctx.request_context.lifespan_context.db.query(sql)
        result = df.to_dict(orient="records")
        logger.info(f"Returning {len(result)} records")
        return json.dumps(result, default=str)
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return f"Error executing tool run_postgis_query: {str(e)}"




if __name__ == "__main__":
    logger.info("Starting MCP server")
    print(f"Running MCP server...", flush=False)
    mcp.run()