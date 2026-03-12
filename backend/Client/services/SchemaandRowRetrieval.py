

import sys, os, time, asyncio
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_MODEL_DEPLOYMENT



import json

def schema_and_row_retrieval(agentResult):
    for entry in reversed(agentResult.tool_calls):
        for tool in reversed(entry["tools"]):
            if tool["tool"] == "execute_postgis_query":
                raw = tool["result"]

                if not raw:
                    return False, "Empty tool result"

                if isinstance(raw, str) and raw.startswith("Error:"):
                    return False, raw

                try:
                    data = json.loads(raw)
                except Exception:
                    return False, "Tool result is not valid JSON"

                if isinstance(data, list) and len(data) > 0:
                    return True, raw

                return False, "No rows returned"

    return False, "No row retrieval tool result found"




