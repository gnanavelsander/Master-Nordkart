import sys
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
import asyncio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
)

from tools.kartverketAPI import kommune_where_clause
from tools.execute_postgis_query import execute_postgis_query
from mcp_c_s.mcp_client import query_postgis





PROMPT = (Path(__file__).parent / "prompt.txt").read_text(encoding="utf-8")


async def agentGis(userInput: str):
    """GIS Agent that executes queries via tool calling."""
    
    # Define available tools
    tools = [
        kommune_where_clause,
        #execute_postgis_query
    ]
    
    # Initialize model with tool binding
    model = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        temperature=0,
        max_tokens=2000,
        timeout=60,
        max_retries=2,
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-02-15-preview",
    ).bind_tools(tools)
    
    # Initialize conversation
    msgs = [
        SystemMessage(content=PROMPT),
        HumanMessage(content=userInput)
    ]
    
    # Tool execution loop
    tool_map = {t.name: t for t in tools}
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Get model response
        ai = await model.ainvoke(msgs)
        msgs.append(ai)
        
        # Check if there are tool calls
        if not ai.tool_calls:
            # No more tool calls - return final answer
            return ai.content
        




        # Execute all tool calls
        for tc in ai.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            
            print(f"🔧 Calling tool: {tool_name} with args: {tool_args}")
            
            try:
                # ALWAYS use ainvoke for async compatibility
                tool = tool_map[tool_name]
                result = await tool.ainvoke(tool_args)  # always use ainvoke
                
                print(f"Tool result: {str(result)[:200]}...")
                
                msgs.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"]
                ))
            except Exception as e:
                print(f"Tool error: {e}")
                msgs.append(ToolMessage(
                    content=f"Error executing tool: {str(e)}",
                    tool_call_id=tc["id"]
                ))
    
    return "Max iterations reached without completion."


async def main():
    userInput = "Find 10 buildings in Grimstad."

    print("Starting GIS Agent...")
    print("=" * 60)

    # Get SQL from LLM
    sql = await agentGis(userInput)


    print("=" * 60)

    print("\nGENERATED SQL:")
    print("=" * 60)
    print(sql)


    print("=" * 60)
    # Execute SQL outside LLM (MCP)
    result = await query_postgis(sql)

    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    print(result)





if __name__ == "__main__":
    asyncio.run(main())
