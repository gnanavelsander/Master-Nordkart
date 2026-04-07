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


def _parse(raw: str, tool_name: str) -> Any:
    """Parse JSON response from server. Server always returns JSON — never plain error strings."""
    if not raw:
        raise RuntimeError(f"Empty response from {tool_name}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{tool_name} returned non-JSON: {raw[:200]}") from e


def _parse_step(raw: str, tool_name: str) -> dict:
    """Parse a step tool response. Raises if executed=False."""
    result = _parse(raw, tool_name)
    if not isinstance(result, dict):
        raise RuntimeError(f"{tool_name} returned unexpected type: {type(result)}")
    if not result.get("executed", False):
        raise RuntimeError(f"{tool_name} failed: {result.get('error', 'unknown error')}")
    return result


# --------------------------------------------------
# Discovery tools
# --------------------------------------------------

async def get_available_databases() -> list[str]:
    raw = await mcp_call("get_available_databases", {})
    data = _parse(raw, "get_available_databases")
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(data["error"])
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response from get_available_databases: {data}")
    return data


async def get_available_schemas(databaseName: str) -> dict:
    raw = await mcp_call("get_available_schemas", {"databaseName": databaseName})
    return _parse(raw, "get_available_schemas")


async def get_available_tables(databaseName: str, schemaName: str) -> dict:
    raw = await mcp_call("get_available_tables", {"databaseName": databaseName, "schemaName": schemaName})
    return _parse(raw, "get_available_tables")


async def get_table_schema(databaseName: str, schemaName: str, table_name: str) -> dict:
    raw = await mcp_call("get_table_schema", {"databaseName": databaseName, "schemaName": schemaName, "table_name": table_name})
    return _parse(raw, "get_table_schema")


async def get_database_catalog(databaseName: str | None = None, schemaName: str | None = None) -> dict:
    args = {}
    if databaseName is not None:
        args["databaseName"] = databaseName
    if schemaName is not None:
        args["schemaName"] = schemaName
    raw = await mcp_call("get_database_catalog", args)
    return _parse(raw, "get_database_catalog")

async def validate_sql_via_mcp(sql: str, databaseName: str, step_number: int) -> dict:
    raw = await mcp_call(
        "validate_sql_explain",
        {"sql": sql, "databaseName": databaseName, "step_number": step_number},
    )
    result = _parse(raw, "validate_sql_explain")
    if not isinstance(result, dict):
        raise RuntimeError(f"validate_sql_explain returned unexpected type: {type(result)}")
    return result

# --------------------------------------------------
# Step tools — all use _parse_step (raises on executed=False)
# --------------------------------------------------

async def execute_query_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_query_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_query_step")


async def execute_filter_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_filter_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_filter_step")


async def execute_select_columns_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_select_columns_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_select_columns_step")


async def execute_spatial_join_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_spatial_join_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_spatial_join_step")


async def execute_attribute_join_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_attribute_join_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_attribute_join_step")


async def execute_buffer_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_buffer_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_buffer_step")


async def execute_nearest_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_nearest_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_nearest_step")


async def execute_distance_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_distance_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_distance_step")


async def execute_aggregate_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_aggregate_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_aggregate_step")


async def execute_sort_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_sort_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_sort_step")


async def execute_limit_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_limit_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_limit_step")


async def execute_clip_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_clip_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_clip_step")


async def execute_union_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_union_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_union_step")


async def execute_intersection_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_intersection_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_intersection_step")


async def execute_difference_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_difference_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_difference_step")


async def execute_dissolve_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_dissolve_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_dissolve_step")


async def execute_centroid_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_centroid_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_centroid_step")


async def execute_transform_crs_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_transform_crs_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_transform_crs_step")