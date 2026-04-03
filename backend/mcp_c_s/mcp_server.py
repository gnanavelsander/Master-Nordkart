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
        return f"Error executing tool run_postgis_query: {str(e)}"


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
        db_names = list(ctx.request_context.lifespan_context.conn_ids.keys())
        return json.dumps({"databases": db_names}, ensure_ascii=False)
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
            {
                "databaseName": databaseName,
                "schemas": [row["schema_name"] for row in rows],
            },
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
            {
                "databaseName": databaseName,
                "schemaName": schemaName,
                "tables": [row["table_name"] for row in rows],
            },
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
            return json.dumps(
                {
                    "databaseName": databaseName,
                    "schemaName": schemaName,
                    "table_name": table_name,
                    "columns": [],
                    "error": f"Table '{schemaName}.{table_name}' not found.",
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "databaseName": databaseName,
                "schemaName": schemaName,
                "table_name": table_name,
                "columns": [
                    {
                        "column_name": row["column_name"],
                        "data_type": row["data_type"],
                    }
                    for row in rows
                ],
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


DB_DESCRIPTIONS = {
    "matrikkelenbygning": "Building and cadastral building data.",
    "flomsoner": "Flood zone related spatial layers.",
    "flomaktsomrader": "Flood hazard / susceptibility areas.",
    "brannstasjoner": "Fire station locations and related data.",
    "kvikkleire": "Quick clay hazard data.",
    "snoskredaktsomhetskart": "Snow avalanche susceptibility data.",
    "matrikkeleneiendomskartteig": "Property parcel / cadastral parcel data.",
    "aktsomhetskart_jord_flomskred": "Soil and flood landslide susceptibility data.",
    "kommuner": "Municipality boundaries and municipality reference data.",
}



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
                    SELECT
                        f_table_schema AS table_schema,
                        f_table_name AS table_name,
                        f_geometry_column AS geometry_column,
                        srid,
                        type AS geometry_type
                    FROM geometry_columns
                )
                SELECT
                    t.table_schema,
                    t.table_name,
                    g.geometry_column,
                    g.srid,
                    g.geometry_type
                FROM information_schema.tables t
                LEFT JOIN geom_info g
                  ON t.table_schema = g.table_schema
                 AND t.table_name = g.table_name
                WHERE t.table_type = 'BASE TABLE'
                  AND t.table_schema NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
                  {schema_filter_sql}
                ORDER BY t.table_schema, t.table_name
                """,
                tuple(params) if params else None,
            )

            grouped = defaultdict(list)

            for row in rows:
                sch = row["table_schema"]
                grouped[sch].append(
                    {
                        "table": row["table_name"],
                        "geometry_columns": (
                            [{
                                "column_name": row["geometry_column"],
                                "srid": row["srid"],
                                "geometry_type": row["geometry_type"],
                            }]
                            if row.get("geometry_column")
                            else []
                        ),
                        "recommended_geometry_column": row.get("geometry_column"),
                    }
                )

            schemas = [
                {"schema": sch, "tables": tables}
                for sch, tables in grouped.items()
            ]

            catalog.append(
                {
                    "database": db_name,
                    "description": DB_DESCRIPTIONS.get(db_name, ""),
                    "schemas": schemas,
                }
            )

        return json.dumps({"databases": catalog}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool(
    name="validate_sql_explain",
    description="Validate SQL syntax using EXPLAIN. Returns ok or error message.",
)
async def validate_sql_explain(
    ctx: Context[ServerSession, AppContext],
    sql: str,
    databaseName: str,
    step_number: int,
) -> str:
    try:
        conID = ctx.request_context.lifespan_context.conn_ids[databaseName]
        await ctx.request_context.lifespan_context.db.query(
            conID,
            f"EXPLAIN (FORMAT JSON) {sql}",
        )
        return json.dumps({
            "validated": True,
            "step_number": step_number,
            "database": databaseName,
        })
    except Exception as e:
        return f"Error: {e}"














# --------------------------------------------------
# Plan models
# --------------------------------------------------

class PlanStep(BaseModel):
    step: int
    action: Literal[
        "query",
        "filter",
        "select_columns",
        "spatial_join",
        "attribute_join",
        "buffer",
        "nearest",
        "distance",
        "aggregate",
        "sort",
        "limit",
        "clip",
        "union",
        "intersection",
        "difference",
        "dissolve",
        "centroid",
        "transform_crs",
    ]

    database: Optional[str] = None
    sql: Optional[str] = None
    validSql: Optional[str] = None

    source_step: Optional[int] = None
    left_source: Optional[int] = None
    right_source: Optional[int] = None

    join_type: Optional[
        Literal["intersects", "within", "contains", "dwithin", "touches", "overlaps"]
    ] = None
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
    limit: Optional[int] = 100

    target_crs: Optional[str] = None
    join_keys: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    query_mode: Literal["single_db", "multi_db"]
    steps: list[PlanStep] = Field(default_factory=list)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _get_run_store(ctx: Context[ServerSession, AppContext], run_id: str) -> dict[int, list[dict]]:
    store = ctx.request_context.lifespan_context.step_results
    if run_id not in store:
        store[run_id] = {}
    return store[run_id]


def _get_rows(ctx: Context[ServerSession, AppContext], run_id: str, step_no: int) -> list[dict]:
    run_store = _get_run_store(ctx, run_id)
    rows = run_store.get(step_no)
    if rows is None:
        raise ValueError(f"Missing source step {step_no}.")
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
    return pd.DataFrame(rows)


def _to_gdf(rows: list[dict], geom_col: str | None) -> gpd.GeoDataFrame:
    df = pd.DataFrame(rows)
    if geom_col is None:
        return gpd.GeoDataFrame(df)
    if geom_col not in df.columns:
        raise ValueError(f"Geometry column '{geom_col}' not found.")
    df[geom_col] = df[geom_col].apply(_decode_geom)
    return gpd.GeoDataFrame(df, geometry=geom_col, crs="EPSG:25833")


def _ok(run_id: str, step: int, action: str, rows: list[dict]) -> str:
    return json.dumps(
        {
            "executed": True,
            "run_id": run_id,
            "step": step,
            "action": action,
            "rows_returned": len(rows),
            "result": rows[:5],
        },
        default=str,
    )


# --------------------------------------------------
# Step tools
# --------------------------------------------------

@mcp.tool(name="execute_query_step", description="Execute one validated SQL query step.")
async def execute_query_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        if not step.database or not step.sql:
            return f"Error: step {step.step} requires database and sql."
        if (step.validSql or "").strip().lower() != "ok":
            return f"Error: step {step.step} sql must be validated."

        conn_id = ctx.request_context.lifespan_context.conn_ids[step.database]
        rows = await ctx.request_context.lifespan_context.db.query(conn_id, step.sql)
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "query", rows)
    except Exception as e:
        return f"Error executing execute_query_step: {e}"


@mcp.tool(name="execute_filter_step", description="Filter rows from a previous step.")
async def execute_filter_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        df = _to_df(_get_rows(ctx, run_id, step.source_step))
        if step.filters:
            mask = pd.Series(True, index=df.index)
            for expr in step.filters:
                mask &= df.eval(expr)
            df = df[mask]
        rows = df.to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "filter", rows)
    except Exception as e:
        return f"Error executing execute_filter_step: {e}"


@mcp.tool(name="execute_select_columns_step", description="Select specific columns from a previous step.")
async def execute_select_columns_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        df = _to_df(_get_rows(ctx, run_id, step.source_step))
        cols = step.select_columns or step.output_columns
        if cols:
            df = df[cols]
        rows = df.to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "select_columns", rows)
    except Exception as e:
        return f"Error executing execute_select_columns_step: {e}"


@mcp.tool(name="execute_spatial_join_step", description="Spatial join between two previous step results.")
async def execute_spatial_join_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        left_gdf = _to_gdf(_get_rows(ctx, run_id, step.left_source), step.left_geom)
        right_gdf = _to_gdf(_get_rows(ctx, run_id, step.right_source), step.right_geom)

        if step.join_type == "dwithin":
            joined = gpd.sjoin(
                left_gdf,
                right_gdf,
                predicate="dwithin",
                how="inner",
                distance=step.distance_meters,
            )
        else:
            joined = gpd.sjoin(left_gdf, right_gdf, predicate=step.join_type, how="inner")

        rows = joined.drop(columns=["index_right"], errors="ignore").to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "spatial_join", rows)
    except Exception as e:
        return f"Error executing execute_spatial_join_step: {e}"


@mcp.tool(name="execute_attribute_join_step", description="Attribute join between two previous step results.")
async def execute_attribute_join_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        left_df = _to_df(_get_rows(ctx, run_id, step.left_source))
        right_df = _to_df(_get_rows(ctx, run_id, step.right_source))
        merged = left_df.merge(right_df, on=step.join_keys, how="inner")
        rows = merged.to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "attribute_join", rows)
    except Exception as e:
        return f"Error executing execute_attribute_join_step: {e}"


@mcp.tool(name="execute_buffer_step", description="Buffer geometries from a previous step.")
async def execute_buffer_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        gdf = _to_gdf(_get_rows(ctx, run_id, step.source_step), step.left_geom).copy()
        gdf[step.left_geom] = gdf.geometry.buffer(step.buffer_distance)
        gdf = gdf.set_geometry(step.left_geom)
        rows = gdf.to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "buffer", rows)
    except Exception as e:
        return f"Error executing execute_buffer_step: {e}"


@mcp.tool(name="execute_nearest_step", description="Find nearest geometry from another step.")
async def execute_nearest_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        left_gdf = _to_gdf(_get_rows(ctx, run_id, step.left_source), step.left_geom)
        right_gdf = _to_gdf(_get_rows(ctx, run_id, step.right_source), step.right_geom)
        out = gpd.sjoin_nearest(left_gdf, right_gdf, how="left", distance_col=step.metric_as or "distance")
        rows = out.to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "nearest", rows)
    except Exception as e:
        return f"Error executing execute_nearest_step: {e}"


@mcp.tool(name="execute_distance_step", description="Compute nearest distance to another step.")
async def execute_distance_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        left_gdf = _to_gdf(_get_rows(ctx, run_id, step.left_source), step.left_geom)
        right_gdf = _to_gdf(_get_rows(ctx, run_id, step.right_source), step.right_geom)
        out = gpd.sjoin_nearest(left_gdf, right_gdf, how="left", distance_col="distance_m")
        rows = out.to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "distance", rows)
    except Exception as e:
        return f"Error executing execute_distance_step: {e}"


@mcp.tool(name="execute_aggregate_step", description="Aggregate rows from a previous step.")
async def execute_aggregate_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        df = _to_df(_get_rows(ctx, run_id, step.source_step))
        metric_as = step.metric_as or "metric"
        agg_type = step.aggregation_type or "count"

        if agg_type == "count":
            out = (
                df.groupby(step.group_by).size().reset_index(name=metric_as)
                if step.group_by
                else pd.DataFrame([{metric_as: len(df)}])
            )
        else:
            out = (
                df.groupby(step.group_by)[step.metric_column].agg(agg_type).reset_index(name=metric_as)
                if step.group_by
                else pd.DataFrame([{metric_as: getattr(df[step.metric_column], agg_type)()}])
            )

        if step.order_by:
            out = out.sort_values(step.order_by, ascending=(step.order_direction == "asc"))
        out = out.head(step.limit or 100)

        rows = out.to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "aggregate", rows)
    except Exception as e:
        return f"Error executing execute_aggregate_step: {e}"


@mcp.tool(name="execute_sort_step", description="Sort rows from a previous step.")
async def execute_sort_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        df = _to_df(_get_rows(ctx, run_id, step.source_step))
        df = df.sort_values(step.order_by, ascending=(step.order_direction == "asc"))
        rows = df.to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "sort", rows)
    except Exception as e:
        return f"Error executing execute_sort_step: {e}"


@mcp.tool(name="execute_limit_step", description="Limit rows from a previous step.")
async def execute_limit_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        df = _to_df(_get_rows(ctx, run_id, step.source_step)).head(step.limit or 100)
        rows = df.to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "limit", rows)
    except Exception as e:
        return f"Error executing execute_limit_step: {e}"


@mcp.tool(name="execute_clip_step", description="Clip one geometry layer by another.")
async def execute_clip_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        left_gdf = _to_gdf(_get_rows(ctx, run_id, step.left_source), step.left_geom)
        right_gdf = _to_gdf(_get_rows(ctx, run_id, step.right_source), step.right_geom)
        out = gpd.clip(left_gdf, right_gdf)
        rows = out.to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "clip", rows)
    except Exception as e:
        return f"Error executing execute_clip_step: {e}"


@mcp.tool(name="execute_union_step", description="Union all geometries from a previous step.")
async def execute_union_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        gdf = _to_gdf(_get_rows(ctx, run_id, step.source_step), step.left_geom)
        rows = [{step.left_geom: gdf.unary_union.wkt}]
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "union", rows)
    except Exception as e:
        return f"Error executing execute_union_step: {e}"


@mcp.tool(name="execute_intersection_step", description="Overlay intersection between two geometry layers.")
async def execute_intersection_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        left_gdf = _to_gdf(_get_rows(ctx, run_id, step.left_source), step.left_geom)
        right_gdf = _to_gdf(_get_rows(ctx, run_id, step.right_source), step.right_geom)
        out = gpd.overlay(left_gdf, right_gdf, how="intersection")
        rows = out.to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "intersection", rows)
    except Exception as e:
        return f"Error executing execute_intersection_step: {e}"


@mcp.tool(name="execute_difference_step", description="Overlay difference between two geometry layers.")
async def execute_difference_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        left_gdf = _to_gdf(_get_rows(ctx, run_id, step.left_source), step.left_geom)
        right_gdf = _to_gdf(_get_rows(ctx, run_id, step.right_source), step.right_geom)
        out = gpd.overlay(left_gdf, right_gdf, how="difference")
        rows = out.to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "difference", rows)
    except Exception as e:
        return f"Error executing execute_difference_step: {e}"


@mcp.tool(name="execute_dissolve_step", description="Dissolve geometries by group.")
async def execute_dissolve_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        gdf = _to_gdf(_get_rows(ctx, run_id, step.source_step), step.left_geom)
        out = gdf.dissolve(by=step.group_by[0]).reset_index() if step.group_by else gdf.dissolve().reset_index()
        rows = out.to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "dissolve", rows)
    except Exception as e:
        return f"Error executing execute_dissolve_step: {e}"


@mcp.tool(name="execute_centroid_step", description="Compute centroids for a geometry layer.")
async def execute_centroid_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        gdf = _to_gdf(_get_rows(ctx, run_id, step.source_step), step.left_geom).copy()
        gdf[step.left_geom] = gdf.geometry.centroid
        gdf = gdf.set_geometry(step.left_geom)
        rows = gdf.to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "centroid", rows)
    except Exception as e:
        return f"Error executing execute_centroid_step: {e}"


@mcp.tool(name="execute_transform_crs_step", description="Transform geometries to another CRS.")
async def execute_transform_crs_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    try:
        step = PlanStep.model_validate(step)
        gdf = _to_gdf(_get_rows(ctx, run_id, step.source_step), step.left_geom).to_crs(step.target_crs)
        rows = gdf.to_dict("records")
        _save_rows(ctx, run_id, step.step, rows)
        return _ok(run_id, step.step, "transform_crs", rows)
    except Exception as e:
        return f"Error executing execute_transform_crs_step: {e}"


if __name__ == "__main__":
    mcp.run()