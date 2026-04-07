from contextlib import asynccontextmanager
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

import json
import logging
import os
import sys

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field
from shapely import wkb as shapely_wkb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.database import PoolManager
from backend.log_utils import setup_logging
from config import (
    PGCONN_STRING,
    PGCONN_MATRIKKEL,
    PGCONN_BRANNSTASJONER,
    PGCONN_FLOMSONER,
    PGCONN_FLOMAKTSOMRADER,
    PGCONN_KVIKKLEIRE,
    PGCONN_MATRIKKELENEIENDOMSKARTTEIG,
    PGCONN_SNOSKREDAKTSOMHETSKART,
    PGCONN_AKTSOMHETSKART_JORD_FLOMSKRED,
    PGCONN_KOMMUNER,
)

env_path = Path(__file__).resolve().parents[2] / ".env"
loaded = load_dotenv(env_path, override=True)

if not loaded:
    raise RuntimeError(f".env not found at {env_path}")

if not os.getenv("PGCONN_STRING"):
    raise RuntimeError("PGCONN_STRING missing after loading .env")

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    db: PoolManager
    conn_ids: dict[str, str]
    step_results: dict[str, dict[int, list[dict]]]


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    poolManager = PoolManager()

    conn_ids = {
        "matrikkelenbygning":            await poolManager.connect(PGCONN_MATRIKKEL),
        "flomaktsomrader":               await poolManager.connect(PGCONN_FLOMAKTSOMRADER),
        "flomsoner":                     await poolManager.connect(PGCONN_FLOMSONER),
        "brannstasjoner":                await poolManager.connect(PGCONN_BRANNSTASJONER),
        "kvikkleire":                    await poolManager.connect(PGCONN_KVIKKLEIRE),
        "snoskredaktsomhetskart":        await poolManager.connect(PGCONN_SNOSKREDAKTSOMHETSKART),
        "matrikkeleneiendomskartteig":   await poolManager.connect(PGCONN_MATRIKKELENEIENDOMSKARTTEIG),
        "aktsomhetskart_jord_flomskred": await poolManager.connect(PGCONN_AKTSOMHETSKART_JORD_FLOMSKRED),
        "kommuner":                      await poolManager.connect(PGCONN_KOMMUNER),
    }

    try:
        yield AppContext(
            db=poolManager,
            conn_ids=conn_ids,
            step_results={},
        )
    finally:
        logger.info("Cleaning up database connection")
        await poolManager.close_all()


mcp = FastMCP("postgis-server", lifespan=app_lifespan)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _err(step: int, action: str, msg: str) -> str:
    """Consistent error response — always JSON with executed=False."""
    return json.dumps({
        "executed": False,
        "step": step,
        "action": action,
        "error": str(msg),
    })


def _ok(run_id: str, step: int, action: str, rows: list[dict]) -> str:
    """Summary returned to LLM. Raw rows stored server-side only."""
    columns = list(rows[0].keys()) if rows else []
    return json.dumps({
        "executed": True,
        "run_id": run_id,
        "step": step,
        "action": action,
        "rows_returned": len(rows),
        "columns": columns,
    }, default=str)


def _get_run_store(ctx: Context[ServerSession, AppContext], run_id: str) -> dict[int, list[dict]]:
    store = ctx.request_context.lifespan_context.step_results
    if run_id not in store:
        store[run_id] = {}
    return store[run_id]


def _get_rows(ctx: Context[ServerSession, AppContext], run_id: str, step_no: int) -> list[dict]:
    run_store = _get_run_store(ctx, run_id)
    rows = run_store.get(step_no)
    if rows is None:
        available = sorted(run_store.keys())
        raise ValueError(
            f"No data for step {step_no} in run {run_id}. "
            f"Available steps: {available}. "
            "Ensure the source step executed successfully before referencing it."
        )
    return rows


def _save_rows(ctx: Context[ServerSession, AppContext], run_id: str, step_no: int, rows: list[dict]) -> None:
    _get_run_store(ctx, run_id)[step_no] = rows


def _decode_geom(value):
    if value is None:
        return None
    if hasattr(value, "geom_type"):
        return value
    if isinstance(value, bytes):
        return shapely_wkb.loads(value)
    if isinstance(value, memoryview):
        return shapely_wkb.loads(bytes(value))
    if isinstance(value, str):
        return shapely_wkb.loads(bytes.fromhex(value))
    return value


def _to_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def _detect_geom_col(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None

    preferred = ["geom", "geometry", "wkb_geom"]
    for col in preferred:
        if col in df.columns:
            sample = df[col].dropna()
            if not sample.empty:
                try:
                    _decode_geom(sample.iloc[0])
                    return col
                except Exception:
                    pass

    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        try:
            geom = _decode_geom(sample.iloc[0])
            if hasattr(geom, "geom_type"):
                return col
        except Exception:
            continue

    return None


def _to_gdf(rows: list[dict], geom_col: str | None) -> gpd.GeoDataFrame:
    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[])

    detected_geom = geom_col or _detect_geom_col(df)
    if detected_geom is None:
        raise ValueError(
            f"No geometry column found. Available columns: {list(df.columns)}"
        )

    if detected_geom not in df.columns:
        raise ValueError(
            f"Geometry column '{detected_geom}' not found. Available: {list(df.columns)}"
        )

    df[detected_geom] = df[detected_geom].apply(_decode_geom)

    non_null = df[detected_geom].dropna()
    if not non_null.empty:
        bad = [v for v in non_null.head(5) if not hasattr(v, "geom_type")]
        if bad:
            raise ValueError(
                f"Column '{detected_geom}' could not be decoded as geometry."
            )

    gdf = gpd.GeoDataFrame(df, geometry=detected_geom, crs="EPSG:25833")
    return gdf.set_geometry(detected_geom)
# --------------------------------------------------
# Plan model
# --------------------------------------------------

class PlanStep(BaseModel):
    step: int
    action: Literal[
        "query", "filter", "select_columns", "spatial_join", "attribute_join",
        "buffer", "nearest", "distance", "aggregate", "sort", "limit",
        "clip", "union", "intersection", "difference", "dissolve", "centroid", "transform_crs",
    ]
    database: Optional[str] = None
    sql: Optional[str] = None
    validSql: Optional[str] = None
    source_step: Optional[int] = None
    left_source: Optional[int] = None
    right_source: Optional[int] = None
    join_type: Optional[Literal["intersects", "within", "contains", "dwithin", "touches", "overlaps"]] = None
    left_geom: Optional[str] = None
    right_geom: Optional[str] = None
    distance_meters: Optional[float] = None
    buffer_distance: Optional[float] = None
    filters: list[str] = Field(default_factory=list)
    select_columns: list[str] = Field(default_factory=list)
    output_columns: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    aggregation_type: Optional[Literal["count", "sum", "avg", "min", "max"]] = None
    metric_column: Optional[str] = None
    metric_as: Optional[str] = None
    order_by: Optional[str] = None
    order_direction: Optional[Literal["asc", "desc"]] = None
    limit: Optional[int] = None  # None means no limit — agent must set explicitly if needed
    target_crs: Optional[str] = None
    join_keys: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    query_mode: Literal["single_db", "multi_db"]
    steps: list[PlanStep] = Field(default_factory=list)


# --------------------------------------------------
# DB descriptions
# --------------------------------------------------

DB_DESCRIPTIONS = {
    "matrikkelenbygning":            "Building and cadastral building data.",
    "flomsoner":                     "Flood zone related spatial layers.",
    "flomaktsomrader":               "Flood hazard / susceptibility areas.",
    "brannstasjoner":                "Fire station locations and related data.",
    "kvikkleire":                    "Quick clay hazard data.",
    "snoskredaktsomhetskart":        "Snow avalanche susceptibility data.",
    "matrikkeleneiendomskartteig":   "Property parcel / cadastral parcel data.",
    "aktsomhetskart_jord_flomskred": "Soil and flood landslide susceptibility data.",
    "kommuner":                      "Municipality boundaries and municipality reference data.",
}


# --------------------------------------------------
# Base query tool
# --------------------------------------------------

@mcp.tool(
    name="run_postgis_query",
    description="Execute SQL query on PostGIS database and return results as JSON",
)
async def run_postgis_query(
    ctx: Context[ServerSession, AppContext],
    sql: str,
    databaseName: str,
) -> str:
    logger.info(f"Tool called with SQL: {sql[:100]}...")
    try:
        conID = ctx.request_context.lifespan_context.conn_ids[databaseName]
        result = await ctx.request_context.lifespan_context.db.query(conID, sql)
        logger.info(f"Returning {len(result)} records")
        return json.dumps(result, default=str)
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return json.dumps({"error": str(e)})


# --------------------------------------------------
# Discovery tools
# --------------------------------------------------

@mcp.tool(
    name="get_available_databases",
    description="Get available database names",
)
async def get_available_databases(
    ctx: Context[ServerSession, AppContext],
) -> str:
    try:
        # Returns plain list — consistent with agent _extract_databases_from_result
        return json.dumps(
            list(ctx.request_context.lifespan_context.conn_ids.keys()),
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)








@mcp.tool(
    name="get_available_schemas",
    description="Get available schemas for a given database",
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
            """,
        )
        return json.dumps(
            {"databaseName": databaseName, "schemas": [row["schema_name"] for row in rows]},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


















@mcp.tool(
    name="get_available_tables",
    description="Get available tables for a given schema in a given database",
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
            (schemaName,),
        )
        return json.dumps(
            {"databaseName": databaseName, "schemaName": schemaName, "tables": [row["table_name"] for row in rows]},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool(
    name="get_table_schema",
    description="Get column names and types for a given table in a given schema",
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
            (schemaName, table_name),
        )
        if not rows:
            return json.dumps({
                "databaseName": databaseName, "schemaName": schemaName,
                "table_name": table_name, "columns": [],
                "error": f"Table '{schemaName}.{table_name}' not found.",
            }, ensure_ascii=False)
        return json.dumps({
            "databaseName": databaseName, "schemaName": schemaName,
            "table_name": table_name,
            "columns": [{"column_name": r["column_name"], "data_type": r["data_type"]} for r in rows],
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool(
    name="get_database_catalog",
    description="Get lightweight database catalog with schemas, tables, geometry columns and SRID info.",
)
async def get_database_catalog(
    ctx: Context[ServerSession, AppContext],
    databaseName: Optional[str] = None,
    schemaName: Optional[str] = None,
) -> str:
    try:
        db_names = [databaseName] if databaseName else list(
            ctx.request_context.lifespan_context.conn_ids.keys()
        )
        catalog = []

        for db_name in db_names:
            conn_id = ctx.request_context.lifespan_context.conn_ids[db_name]
            params = []
            schema_filter_sql = ""
            if schemaName:
                schema_filter_sql = "AND t.table_schema = $1"
                params.append(schemaName)

            rows = await ctx.request_context.lifespan_context.db.query(
                conn_id,
                f"""
                WITH geom_info AS (
                    SELECT f_table_schema AS table_schema, f_table_name AS table_name,
                           f_geometry_column AS geometry_column, srid, type AS geometry_type
                    FROM geometry_columns
                )
                SELECT t.table_schema, t.table_name, g.geometry_column, g.srid, g.geometry_type
                FROM information_schema.tables t
                LEFT JOIN geom_info g
                  ON t.table_schema = g.table_schema AND t.table_name = g.table_name
                WHERE t.table_type = 'BASE TABLE'
                  AND t.table_schema NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
                  {schema_filter_sql}
                ORDER BY t.table_schema, t.table_name
                """,
                tuple(params) if params else None,
            )

            grouped = defaultdict(list)
            for row in rows:
                grouped[row["table_schema"]].append({
                    "table": row["table_name"],
                    "geometry_columns": (
                        [{"column_name": row["geometry_column"], "srid": row["srid"], "geometry_type": row["geometry_type"]}]
                        if row.get("geometry_column") else []
                    ),
                    "recommended_geometry_column": row.get("geometry_column"),
                })

            catalog.append({
                "database": db_name,
                "description": DB_DESCRIPTIONS.get(db_name, ""),
                "schemas": [{"schema": sch, "tables": tbls} for sch, tbls in grouped.items()],
            })

        return json.dumps({"databases": catalog}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool(
    name="validate_sql_explain",
    description="Validate SQL syntax using EXPLAIN. Returns JSON with validated=true/false.",
)
async def validate_sql_explain(
    ctx: Context[ServerSession, AppContext],
    sql: str,
    databaseName: str,
    step_number: int,
) -> str:
    try:
        sql_clean = (sql or "").strip()
        sql_upper = sql_clean.upper()

        if not sql_clean:
            return json.dumps({"validated": False, "step_number": step_number, "database": databaseName, "error": "Empty SQL."})

        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            return json.dumps({"validated": False, "step_number": step_number, "database": databaseName, "error": "Only SELECT or WITH statements are allowed."})

        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"]
        if any(word in sql_upper for word in forbidden):
            return json.dumps({"validated": False, "step_number": step_number, "database": databaseName, "error": "Forbidden SQL statement detected."})

        conID = ctx.request_context.lifespan_context.conn_ids[databaseName]
        await ctx.request_context.lifespan_context.db.query(conID, f"EXPLAIN (FORMAT JSON) {sql_clean}")

        return json.dumps({"validated": True, "step_number": step_number, "database": databaseName})

    except Exception as e:
        return json.dumps({"validated": False, "step_number": step_number, "database": databaseName, "error": str(e)})


# --------------------------------------------------
# Step tools
# --------------------------------------------------

@mcp.tool(name="execute_query_step", description="Execute one validated SQL query step.")
async def execute_query_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        if not s.database or not s.sql:
            return _err(s.step, "query", "step requires database and sql.")
        if (s.validSql or "").strip().lower() != "ok":
            return _err(s.step, "query", "sql must be validated before execution (validSql must be 'ok').")
        conn_id = ctx.request_context.lifespan_context.conn_ids[s.database]
        rows = await ctx.request_context.lifespan_context.db.query(conn_id, s.sql)
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "query", rows)
    except Exception as e:
        logger.error("execute_query_step failed step=%s: %r", s.step, e)
        return _err(s.step, "query", repr(e))


@mcp.tool(name="execute_filter_step", description="Filter rows from a previous step.")
async def execute_filter_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        df = _to_df(_get_rows(ctx, run_id, s.source_step))
        if not df.empty and s.filters:
            mask = pd.Series(True, index=df.index)
            for expr in s.filters:
                mask &= df.eval(expr)
            df = df[mask]
        rows = df.to_dict("records")
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "filter", rows)
    except Exception as e:
        return _err(s.step, "filter", str(e))


@mcp.tool(name="execute_select_columns_step", description="Select specific columns from a previous step.")
async def execute_select_columns_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        df = _to_df(_get_rows(ctx, run_id, s.source_step))
        cols = s.select_columns or s.output_columns
        if not df.empty and cols:
            df = df[cols]
        rows = df.to_dict("records")
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "select_columns", rows)
    except Exception as e:
        return _err(s.step, "select_columns", str(e))


@mcp.tool(name="execute_spatial_join_step", description="Spatial join between two previous step results.")
async def execute_spatial_join_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        left_rows = _get_rows(ctx, run_id, s.left_source)
        right_rows = _get_rows(ctx, run_id, s.right_source)
        left_gdf = _to_gdf(left_rows, s.left_geom)
        right_gdf = _to_gdf(right_rows, s.right_geom)


        logger.info(
        "Spatial join step=%s | left_geom_col=%s right_geom_col=%s | left_cols=%s right_cols=%s",
        s.step,
        left_gdf.geometry.name if left_gdf.geometry is not None else None,
        right_gdf.geometry.name if right_gdf.geometry is not None else None,
        list(left_gdf.columns),
        list(right_gdf.columns),
    )

        # Log geometry info to help debug mismatches
        if not left_gdf.empty and not right_gdf.empty:
            logger.info("Spatial join step=%s | left_rows=%s right_rows=%s | left_geom_types=%s right_geom_types=%s | join_type=%s",
                s.step, len(left_gdf), len(right_gdf),
                dict(left_gdf.geometry.geom_type.value_counts()),
                dict(right_gdf.geometry.geom_type.value_counts()),
                s.join_type,
            )

        if left_gdf.empty or right_gdf.empty:
            rows = []
        elif s.join_type == "dwithin":
            joined = gpd.sjoin(left_gdf, right_gdf, predicate="dwithin", how="inner", distance=s.distance_meters)
            rows = joined.drop(columns=["index_right"], errors="ignore").to_dict("records")
        else:
            joined = gpd.sjoin(left_gdf, right_gdf, predicate=s.join_type, how="inner")
            rows = joined.drop(columns=["index_right"], errors="ignore").to_dict("records")

        logger.info("Spatial join result step=%s | rows_returned=%s", s.step, len(rows))
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "spatial_join", rows)
    except Exception as e:
        return _err(s.step, "spatial_join", str(e))


@mcp.tool(name="execute_attribute_join_step", description="Attribute join between two previous step results.")
async def execute_attribute_join_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        left_df = _to_df(_get_rows(ctx, run_id, s.left_source))
        right_df = _to_df(_get_rows(ctx, run_id, s.right_source))
        if left_df.empty or right_df.empty:
            rows = []
        else:
            rows = left_df.merge(right_df, on=s.join_keys, how="inner").to_dict("records")
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "attribute_join", rows)
    except Exception as e:
        return _err(s.step, "attribute_join", str(e))


@mcp.tool(name="execute_buffer_step", description="Buffer geometries from a previous step.")
async def execute_buffer_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        gdf = _to_gdf(_get_rows(ctx, run_id, s.source_step), s.left_geom).copy()
        if not gdf.empty:
            gdf[s.left_geom] = gdf.geometry.buffer(s.buffer_distance)
            gdf = gdf.set_geometry(s.left_geom)
        rows = gdf.to_dict("records")
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "buffer", rows)
    except Exception as e:
        return _err(s.step, "buffer", str(e))


@mcp.tool(name="execute_nearest_step", description="Find nearest geometry from another step.")
async def execute_nearest_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        left_gdf = _to_gdf(_get_rows(ctx, run_id, s.left_source), s.left_geom)
        right_gdf = _to_gdf(_get_rows(ctx, run_id, s.right_source), s.right_geom)
        if left_gdf.empty or right_gdf.empty:
            rows = []
        else:
            out = gpd.sjoin_nearest(left_gdf, right_gdf, how="left", distance_col=s.metric_as or "distance")
            rows = out.to_dict("records")
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "nearest", rows)
    except Exception as e:
        return _err(s.step, "nearest", str(e))


@mcp.tool(name="execute_distance_step", description="Compute nearest distance to another step.")
async def execute_distance_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        left_gdf = _to_gdf(_get_rows(ctx, run_id, s.left_source), s.left_geom)
        right_gdf = _to_gdf(_get_rows(ctx, run_id, s.right_source), s.right_geom)
        if left_gdf.empty or right_gdf.empty:
            rows = []
        else:
            out = gpd.sjoin_nearest(left_gdf, right_gdf, how="left", distance_col="distance_m")
            rows = out.to_dict("records")
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "distance", rows)
    except Exception as e:
        return _err(s.step, "distance", str(e))


@mcp.tool(name="execute_aggregate_step", description="Aggregate rows from a previous step.")
async def execute_aggregate_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        rows = _get_rows(ctx, run_id, s.source_step)
        df = _to_df(rows)
        metric_as = s.metric_as or "metric"
        agg_type = s.aggregation_type or "count"

        if df.empty:
            out = pd.DataFrame()
        elif agg_type == "count":
            out = (
                df.groupby(s.group_by).size().reset_index(name=metric_as)
                if s.group_by
                else pd.DataFrame([{metric_as: len(df)}])
            )
        else:
            out = (
                df.groupby(s.group_by)[s.metric_column].agg(agg_type).reset_index(name=metric_as)
                if s.group_by
                else pd.DataFrame([{metric_as: getattr(df[s.metric_column], agg_type)()}])
            )

        if not out.empty and s.order_by and s.order_by in out.columns:
            out = out.sort_values(s.order_by, ascending=(s.order_direction == "asc"))
        # Only apply limit if explicitly set — None means return all rows
        if not out.empty and s.limit is not None:
            out = out.head(s.limit)

        rows = out.to_dict("records")
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "aggregate", rows)
    except Exception as e:
        return _err(s.step, "aggregate", str(e))


@mcp.tool(name="execute_sort_step", description="Sort rows from a previous step.")
async def execute_sort_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        rows = _get_rows(ctx, run_id, s.source_step)
        df = _to_df(rows)
        if df.empty:
            _save_rows(ctx, run_id, s.step, [])
            return _ok(run_id, s.step, "sort", [])
        if not s.order_by or s.order_by not in df.columns:
            return _err(s.step, "sort",
                        f"Column '{s.order_by}' not found. Available: {list(df.columns)}")
        df = df.sort_values(s.order_by, ascending=(s.order_direction == "asc"))
        rows = df.to_dict("records")
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "sort", rows)
    except Exception as e:
        return _err(s.step, "sort", str(e))


@mcp.tool(name="execute_limit_step", description="Limit rows from a previous step.")
async def execute_limit_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        df = _to_df(_get_rows(ctx, run_id, s.source_step))
        if s.limit is None:
            return _err(s.step, "limit", "limit step requires limit.")
        rows = df.head(s.limit).to_dict("records")
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "limit", rows)
    except Exception as e:
        return _err(s.step, "limit", str(e))


@mcp.tool(name="execute_clip_step", description="Clip one geometry layer by another.")
async def execute_clip_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        left_gdf = _to_gdf(_get_rows(ctx, run_id, s.left_source), s.left_geom)
        right_gdf = _to_gdf(_get_rows(ctx, run_id, s.right_source), s.right_geom)
        if left_gdf.empty or right_gdf.empty:
            rows = []
        else:
            rows = gpd.clip(left_gdf, right_gdf).to_dict("records")
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "clip", rows)
    except Exception as e:
        return _err(s.step, "clip", str(e))


@mcp.tool(name="execute_union_step", description="Union all geometries from a previous step.")
async def execute_union_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        gdf = _to_gdf(_get_rows(ctx, run_id, s.source_step), s.left_geom)
        if gdf.empty:
            rows = []
        else:
            # Store as WKB hex — same format as raw geometry columns from DB
            union_geom = gdf.unary_union
            rows = [{s.left_geom: union_geom.wkb.hex()}]
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "union", rows)
    except Exception as e:
        return _err(s.step, "union", str(e))


@mcp.tool(name="execute_intersection_step", description="Overlay intersection between two geometry layers.")
async def execute_intersection_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        left_gdf = _to_gdf(_get_rows(ctx, run_id, s.left_source), s.left_geom)
        right_gdf = _to_gdf(_get_rows(ctx, run_id, s.right_source), s.right_geom)
        if left_gdf.empty or right_gdf.empty:
            rows = []
        else:
            rows = gpd.overlay(left_gdf, right_gdf, how="intersection").to_dict("records")
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "intersection", rows)
    except Exception as e:
        return _err(s.step, "intersection", str(e))


@mcp.tool(name="execute_difference_step", description="Overlay difference between two geometry layers.")
async def execute_difference_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        left_gdf = _to_gdf(_get_rows(ctx, run_id, s.left_source), s.left_geom)
        right_gdf = _to_gdf(_get_rows(ctx, run_id, s.right_source), s.right_geom)
        if left_gdf.empty or right_gdf.empty:
            rows = []
        else:
            rows = gpd.overlay(left_gdf, right_gdf, how="difference").to_dict("records")
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "difference", rows)
    except Exception as e:
        return _err(s.step, "difference", str(e))


@mcp.tool(name="execute_dissolve_step", description="Dissolve geometries by group.")
async def execute_dissolve_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        gdf = _to_gdf(_get_rows(ctx, run_id, s.source_step), s.left_geom)
        if gdf.empty:
            rows = []
        else:
            out = gdf.dissolve(by=s.group_by[0]).reset_index() if s.group_by else gdf.dissolve().reset_index()
            rows = out.to_dict("records")
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "dissolve", rows)
    except Exception as e:
        return _err(s.step, "dissolve", str(e))


@mcp.tool(name="execute_centroid_step", description="Compute centroids for a geometry layer.")
async def execute_centroid_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        gdf = _to_gdf(_get_rows(ctx, run_id, s.source_step), s.left_geom).copy()
        if not gdf.empty:
            gdf[s.left_geom] = gdf.geometry.centroid
            gdf = gdf.set_geometry(s.left_geom)
        rows = gdf.to_dict("records")
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "centroid", rows)
    except Exception as e:
        return _err(s.step, "centroid", str(e))


@mcp.tool(name="execute_transform_crs_step", description="Transform geometries to another CRS.")
async def execute_transform_crs_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    s = PlanStep.model_validate(step)
    try:
        gdf = _to_gdf(_get_rows(ctx, run_id, s.source_step), s.left_geom)
        if not gdf.empty:
            gdf = gdf.to_crs(s.target_crs)
        rows = gdf.to_dict("records")
        _save_rows(ctx, run_id, s.step, rows)
        return _ok(run_id, s.step, "transform_crs", rows)
    except Exception as e:
        return _err(s.step, "transform_crs", str(e))


if __name__ == "__main__":
    mcp.run()