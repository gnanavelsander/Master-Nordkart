import os
import sys
import json
import asyncio
import logging

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

        if not json_str or json_str.startswith("Error"):
            return []

        data = json.loads(json_str)
        return data if isinstance(data, list) else []

    except Exception:
        logger.exception("query_postgis failed: %s", sql)
        return []


async def get_available_tables(databaseName: str = "007workshop") -> str:
    return await mcp_call(
        "get_available_tables",
        {
            "databaseName": databaseName,
        },
    )


async def get_table_schema(
    table_name: str,
    databaseName: str = "007workshop",
    schemaName: str = "public",
) -> str:
    return await mcp_call(
        "get_table_schema",
        {
            "table_name": table_name,
            "databaseName": databaseName,
            "schemaName": schemaName,
        },
    )


async def validate_sql_via_mcp(
    sql: str,
    databaseName: str = "007workshop",
) -> str:
    return await mcp_call(
        "validate_sql_explain",
        {
            "sql": sql,
            "databaseName": databaseName,
        },
    )