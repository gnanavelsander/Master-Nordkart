import os
import sys
import json
import pandas as pd
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


def _server_params() -> StdioServerParameters:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    server_script = os.path.join(current_dir, "mcp_server.py")

    env = dict(os.environ)

    return StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        env=env,
        cwd=project_root,
    )


async def mcp_call(tool_name: str, args: dict) -> str:
    server_params = _server_params()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(tool_name, arguments=args)
            return res.content[0].text if res.content else ""


async def query_postgis(sql: str) -> pd.DataFrame:
    try:
        json_str = await mcp_call("run_postgis_query", {"sql": sql})

        if not json_str:
            return pd.DataFrame()

        if isinstance(json_str, str) and json_str.startswith("Error"):
            return pd.DataFrame()

        data = json.loads(json_str)
        return pd.DataFrame(data)

    except Exception:
        return pd.DataFrame()
