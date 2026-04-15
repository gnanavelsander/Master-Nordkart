from contextlib import asynccontextmanager
from pathlib import Path
import sys
from mcp.server.fastmcp import FastMCP
from backend.mcp.server.models import AppContext






from backend.database import PoolManager
from config import (
    PGCONN_FYLKER_GIS, PGCONN_MATRIKKEL, PGCONN_BRANNSTASJONER, PGCONN_FLOMSONER,
    PGCONN_FLOMAKTSOMRADER, PGCONN_KVIKKLEIRE,
    PGCONN_MATRIKKELENEIENDOMSKARTTEIG, PGCONN_SNOSKREDAKTSOMHETSKART,
    PGCONN_AREALRESSURSKART_AR50,
    PGCONN_AKTSOMHETSKART_JORD_FLOMSKRED, PGCONN_KOMMUNER, PGCONN_STATISTISKRUTENETT1KM_GIS, PGCONN_STEDSNAVN_GIS, PGCONN_TRAFIKKMENGDE_GIS,
    PGCONN_STRING,
)
import os, tempfile, uuid

@asynccontextmanager
async def app_lifespan(server):
    pool_manager = PoolManager()
    duck_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.duckdb")
    conn_ids = {
        "postgis": "postgis",
        "matrikkelenbygning": "matrikkelenbygning",
        "flomaktsomrader": "flomaktsomrader",
        "flomsoner": "flomsoner",
        "brannstasjoner": "brannstasjoner",
        "kvikkleire": "kvikkleire",
        "snoskredaktsomhetskart": "snoskredaktsomhetskart",
        "matrikkeleneiendomskartteig": "matrikkeleneiendomskartteig",
        "aktsomhetskart_jord_flomskred": "aktsomhetskart_jord_flomskred",
        "kommuner": "kommuner",
        "fylker": "fylker",
        "statistiskrutenett1km": "statistiskrutenett1km",
        "trafikkmengde": "trafikkmengde",
        "stedsnavn": "stedsnavn",
        "arealressurskart_ar50": "arealressurskart_ar50",
    }
    pool_manager.register("postgis", PGCONN_STRING)
    pool_manager.register("matrikkelenbygning", PGCONN_MATRIKKEL)
    pool_manager.register("flomaktsomrader", PGCONN_FLOMAKTSOMRADER)
    pool_manager.register("flomsoner", PGCONN_FLOMSONER)
    pool_manager.register("brannstasjoner", PGCONN_BRANNSTASJONER)
    pool_manager.register("kvikkleire", PGCONN_KVIKKLEIRE)
    pool_manager.register("snoskredaktsomhetskart", PGCONN_SNOSKREDAKTSOMHETSKART)
    pool_manager.register("matrikkeleneiendomskartteig", PGCONN_MATRIKKELENEIENDOMSKARTTEIG)
    pool_manager.register("aktsomhetskart_jord_flomskred", PGCONN_AKTSOMHETSKART_JORD_FLOMSKRED)
    pool_manager.register("kommuner", PGCONN_KOMMUNER)
    pool_manager.register("fylker", PGCONN_FYLKER_GIS)
    pool_manager.register("statistiskrutenett1km", PGCONN_STATISTISKRUTENETT1KM_GIS)
    pool_manager.register("trafikkmengde", PGCONN_TRAFIKKMENGDE_GIS)
    pool_manager.register("stedsnavn", PGCONN_STEDSNAVN_GIS)
    pool_manager.register("arealressurskart_ar50", PGCONN_AREALRESSURSKART_AR50)





    try:
        yield AppContext(db=pool_manager, conn_ids=conn_ids, step_results={}, duckdb_path=duck_path)
    finally:
        await pool_manager.close_all()
        if os.path.exists(duck_path):
            os.remove(duck_path)

mcp = FastMCP("postgis-server", lifespan=app_lifespan)
