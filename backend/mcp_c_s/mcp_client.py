import os
import sys
import json
import asyncio
import logging
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from backend.log_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _server_params() -> StdioServerParameters:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    server_script = os.path.join(current_dir, "mcp_server.py")

    return StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        env=dict(os.environ),
        cwd=project_root,
    )


class MCPClientManager:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._session: ClientSession | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()

    async def _run(self) -> None:
        """Holder stdio_client og ClientSession i én dedikert task."""
        try:
            async with stdio_client(_server_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    logger.info("MCP session ready")
                    await self._stop.wait()

        except Exception:
            logger.exception("MCP session crashed")
            self._ready.set() 

        finally:
            self._session = None

    async def connect(self) -> None:
        async with self._lock:
            if self._task is None:
                self._task = asyncio.create_task(self._run())

        await self._ready.wait()

        if self._session is None:
            raise RuntimeError("MCP session not available")

    async def close(self) -> None:
        async with self._lock:
            self._stop.set()

            if self._task is not None:
                await self._task
                self._task = None

            self._session = None
            self._ready = asyncio.Event()
            self._stop = asyncio.Event()

    async def call(self, tool_name: str, args: dict) -> str:
        if not self._ready.is_set():
            await self.connect()

        if self._session is None:
            raise RuntimeError("MCP session not available")

        res = await self._session.call_tool(tool_name, arguments=args)
        return res.content[0].text if res.content else ""


mcp_client = MCPClientManager()


async def mcp_call(tool_name: str, args: dict) -> str:
    return await mcp_client.call(tool_name, args)



async def query_postgis(sql: str, databaseName: str = "007workshop") -> list[dict]:
    try:
        json_str = await mcp_call(
            "run_postgis_query",
            {
                "sql": sql,
                "databaseName": databaseName,
            },
        )

        if not json_str:
            raise RuntimeError("Empty response from MCP")

        if json_str.startswith("Error"):
            raise RuntimeError(json_str)

        data = json.loads(json_str)

        if not isinstance(data, list):
            raise RuntimeError("MCP did not return a list of rows")

        return data

    except Exception as e:
        logger.exception("query_postgis failed: %s", sql)
        raise RuntimeError(f"query_postgis failed: {e}") from e
    









async def run_multidb_plan(plan: dict[str, Any]) -> dict:
    try:
        raw = await mcp_call("run_multidb_plan", {"plan": plan})

        print("RAW TYPE:", type(raw))
        print("RAW VALUE:", raw)

        if not raw:
            raise RuntimeError("Empty response from run_multidb_plan")

        if isinstance(raw, dict):
            return raw

        if isinstance(raw, str) and raw.startswith("Error:"):
            raise RuntimeError(raw)

        if isinstance(raw, str):
            return json.loads(raw)

        raise RuntimeError(f"Unexpected response type: {type(raw)}")

    except Exception as e:
        raise RuntimeError(f"run_multidb_plan failed: {e}") from e

async def get_available_schemas(databaseName: str) -> dict:
    try:
        raw = await mcp_call(
            "get_available_schemas",
            {"databaseName": databaseName},
        )
        if not raw:
            raise RuntimeError("Empty response from get_available_schemas")
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"get_available_schemas failed: {e}") from e


async def get_available_tables(databaseName: str, schemaName: str) -> dict:
    try:
        raw = await mcp_call(
            "get_available_tables",
            {
                "databaseName": databaseName,
                "schemaName": schemaName,
            },
        )
        if not raw:
            raise RuntimeError("Empty response from get_available_tables")
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"get_available_tables failed: {e}") from e


async def get_table_schema(databaseName: str, schemaName: str, table_name: str) -> dict:
    try:
        raw = await mcp_call(
            "get_table_schema",
            {
                "databaseName": databaseName,
                "schemaName": schemaName,
                "table_name": table_name,
            },
        )
        if not raw:
            raise RuntimeError("Empty response from get_table_schema")
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"get_table_schema failed: {e}") from e




async def get_available_databases() -> list[str]:
    raw = await mcp_call("get_available_databases", {})
    data = json.loads(raw)
    if "error" in data:
        raise RuntimeError(data["error"])
    return data.get("databases", [])




async def validate_sql_via_mcp(
    sql: str,
    databaseName: str,
    step_number: int,
    
) -> str:
    try:
        result = await mcp_call(
            "validate_sql_explain",
            {
                "sql": sql,
                "databaseName": databaseName,
                "step_number": step_number,
            },
        )

        if not result:
            raise RuntimeError("Empty response from validate_sql_explain")

        return result

    except Exception as e:
        raise RuntimeError(f"validate_sql_via_mcp failed: {e}") from e



async def execute_query_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call(
        "execute_query_step",
        {
            "run_id": run_id,
            "step": step,
        },
    )
    if not raw:
        raise RuntimeError("Empty response from execute_query_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)



async def execute_filter_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call(
        "execute_filter_step",
        {
            "run_id": run_id,
            "step": step,
        },
    )
    if not raw:
        raise RuntimeError("Empty response from execute_filter_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)




async def execute_select_columns_step(run_id: str, step:dict)->dict:
    raw = await mcp_call(
        "execute_select_columns_step",
        {
            "run_id": run_id,
            "step": step,
        },
    )
    if not raw:
        raise RuntimeError("Empty response from execute_select_columns_step")

    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)



async def execute_spatial_join_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_spatial_join_step", {"run_id": run_id, "step": step})
    if not raw:
        raise RuntimeError("Empty response from execute_spatial_join_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)


async def execute_attribute_join_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_attribute_join_step", {"run_id": run_id, "step": step})
    if not raw:
        raise RuntimeError("Empty response from execute_attribute_join_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)


async def execute_buffer_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_buffer_step", {"run_id": run_id, "step": step})
    if not raw:
        raise RuntimeError("Empty response from execute_buffer_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)


async def execute_nearest_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_nearest_step", {"run_id": run_id, "step": step})
    if not raw:
        raise RuntimeError("Empty response from execute_nearest_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)


async def execute_distance_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_distance_step", {"run_id": run_id, "step": step})
    if not raw:
        raise RuntimeError("Empty response from execute_distance_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)


async def execute_aggregate_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_aggregate_step", {"run_id": run_id, "step": step})
    if not raw:
        raise RuntimeError("Empty response from execute_aggregate_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)


async def execute_sort_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_sort_step", {"run_id": run_id, "step": step})
    if not raw:
        raise RuntimeError("Empty response from execute_sort_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)


async def execute_limit_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_limit_step", {"run_id": run_id, "step": step})
    if not raw:
        raise RuntimeError("Empty response from execute_limit_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)


async def execute_clip_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_clip_step", {"run_id": run_id, "step": step})
    if not raw:
        raise RuntimeError("Empty response from execute_clip_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)


async def execute_union_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_union_step", {"run_id": run_id, "step": step})
    if not raw:
        raise RuntimeError("Empty response from execute_union_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)


async def execute_intersection_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_intersection_step", {"run_id": run_id, "step": step})
    if not raw:
        raise RuntimeError("Empty response from execute_intersection_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)


async def execute_difference_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_difference_step", {"run_id": run_id, "step": step})
    if not raw:
        raise RuntimeError("Empty response from execute_difference_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)


async def execute_dissolve_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_dissolve_step", {"run_id": run_id, "step": step})
    if not raw:
        raise RuntimeError("Empty response from execute_dissolve_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)


async def execute_centroid_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_centroid_step", {"run_id": run_id, "step": step})
    if not raw:
        raise RuntimeError("Empty response from execute_centroid_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)


async def execute_transform_crs_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_transform_crs_step", {"run_id": run_id, "step": step})
    if not raw:
        raise RuntimeError("Empty response from execute_transform_crs_step")
    if raw.startswith("Error"):
        raise RuntimeError(raw)
    return json.loads(raw)


async def get_database_catalog(
    databaseName: str | None = None,
    schemaName: str | None = None,
) -> dict:
    try:
        args = {}
        if databaseName is not None:
            args["databaseName"] = databaseName
        if schemaName is not None:
            args["schemaName"] = schemaName

        raw = await mcp_call("get_database_catalog", args)

        if not raw:
            raise RuntimeError("Empty response from get_database_catalog")

        return json.loads(raw)

    except Exception as e:
        raise RuntimeError(f"get_database_catalog failed: {e}") from e