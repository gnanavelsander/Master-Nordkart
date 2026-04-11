from contextlib import asynccontextmanager
from pathlib import Path
import sys
from mcp.server.fastmcp import FastMCP
from backend.mcp.server.models import AppContext






from backend.database import PoolManager
from config import (
    PGCONN_MATRIKKEL, PGCONN_BRANNSTASJONER, PGCONN_FLOMSONER,
    PGCONN_FLOMAKTSOMRADER, PGCONN_KVIKKLEIRE,
    PGCONN_MATRIKKELENEIENDOMSKARTTEIG, PGCONN_SNOSKREDAKTSOMHETSKART,
    PGCONN_AKTSOMHETSKART_JORD_FLOMSKRED, PGCONN_KOMMUNER,
)
import os, tempfile, uuid

@asynccontextmanager
async def app_lifespan(server):
    pool_manager = PoolManager()
    duck_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.duckdb")
    conn_ids = {
        "matrikkelenbygning":            await pool_manager.connect(PGCONN_MATRIKKEL),
        "flomaktsomrader":               await pool_manager.connect(PGCONN_FLOMAKTSOMRADER),
        "flomsoner":                     await pool_manager.connect(PGCONN_FLOMSONER),
        "brannstasjoner":                await pool_manager.connect(PGCONN_BRANNSTASJONER),
        "kvikkleire":                    await pool_manager.connect(PGCONN_KVIKKLEIRE),
        "snoskredaktsomhetskart":        await pool_manager.connect(PGCONN_SNOSKREDAKTSOMHETSKART),
        "matrikkeleneiendomskartteig":   await pool_manager.connect(PGCONN_MATRIKKELENEIENDOMSKARTTEIG),
        "aktsomhetskart_jord_flomskred": await pool_manager.connect(PGCONN_AKTSOMHETSKART_JORD_FLOMSKRED),
        "kommuner":                      await pool_manager.connect(PGCONN_KOMMUNER),
    }
    try:
        yield AppContext(db=pool_manager, conn_ids=conn_ids, step_results={}, duckdb_path=duck_path)
    finally:
        await pool_manager.close_all()
        if os.path.exists(duck_path):
            os.remove(duck_path)

mcp = FastMCP("postgis-server", lifespan=app_lifespan)