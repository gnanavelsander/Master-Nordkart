# import sys
# from pathlib import Path
# from langchain_openai import AzureChatOpenAI
# from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
# import asyncio

# ROOT = Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(ROOT))

# from config import (
#     AZURE_OPENAI_KEY,
#     AZURE_OPENAI_ENDPOINT,
#     AZURE_OPENAI_DEPLOYMENT,
# )

# from backend.tools.kartverketAPI import kommune_where_clause
# from backend.tools.execute_postgis_query import execute_postgis_query
# from backend.mcp_c_s.mcp_client import query_postgis





# PROMPT = (Path(__file__).parent / "prompt.txt").read_text(encoding="utf-8")


# async def agentGis(userInput: str):
#     """GIS Agent that executes queries via tool calling."""
    
#     # Define available tools
#     tools = [
#         kommune_where_clause,
#         #execute_postgis_query
#     ]
    
#     # Initialize model with tool binding
#     model = AzureChatOpenAI(
#         azure_deployment=AZURE_OPENAI_DEPLOYMENT,
#         azure_endpoint=AZURE_OPENAI_ENDPOINT,
#         temperature=0,
#         max_tokens=2000,
#         timeout=60,
#         max_retries=2,
#         api_key=AZURE_OPENAI_KEY,
#         api_version="2024-02-15-preview",
#     ).bind_tools(tools)
    
#     # Initialize conversation
#     msgs = [
#         SystemMessage(content=PROMPT),
#         HumanMessage(content=userInput)
#     ]
    
#     # Tool execution loop
#     tool_map = {t.name: t for t in tools}
#     max_iterations = 10
#     iteration = 0
    
#     while iteration < max_iterations:
#         iteration += 1
        
#         # Get model response
#         ai = await model.ainvoke(msgs)
#         msgs.append(ai)
        
#         # Check if there are tool calls
#         if not ai.tool_calls:
#             # No more tool calls - return final answer
#             return ai.content
        




#         # Execute all tool calls
#         for tc in ai.tool_calls:
#             tool_name = tc["name"]
#             tool_args = tc["args"]
            
#             print(f"🔧 Calling tool: {tool_name} with args: {tool_args}")
            
#             try:
#                 # ALWAYS use ainvoke for async compatibility
#                 tool = tool_map[tool_name]
#                 result = await tool.ainvoke(tool_args)  # always use ainvoke
                
#                 print(f"Tool result: {str(result)[:200]}...")
                
#                 msgs.append(ToolMessage(
#                     content=str(result),
#                     tool_call_id=tc["id"]
#                 ))
#             except Exception as e:
#                 print(f"Tool error: {e}")
#                 msgs.append(ToolMessage(
#                     content=f"Error executing tool: {str(e)}",
#                     tool_call_id=tc["id"]
#                 ))
    
#     return "Max iterations reached without completion."


# async def main():
#     userInput = "Find 10 buildings in Grimstad."

#     print("Starting GIS Agent...")
#     print("=" * 60)

#     # Get SQL from LLM
#     sql = await agentGis(userInput)


#     print("=" * 60)

#     print("\nGENERATED SQL:")
#     print("=" * 60)
#     print(sql)


#     print("=" * 60)
#     # Execute SQL outside LLM (MCP)
#     result = await query_postgis(sql)

#     print("\n" + "=" * 60)
#     print("FINAL RESULT:")
#     print("=" * 60)
#     print(result)





# if __name__ == "__main__":
#     asyncio.run(main())



import sys, os, time, asyncio
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT
from backend.tools.kartverketAPI import kommune_where_clause
from backend.mcp_c_s.mcp_client import query_postgis

PROMPT = (Path(__file__).parent / "prompt.txt").read_text(encoding="utf-8")

from config import AZURE_OPENAI_ENDPOINT, AZURE_MODEL_DEPLOYMENT




# get the model
def get_model(model_id: str, tools):
    dep =AZURE_MODEL_DEPLOYMENT[model_id]
    print("Using deployment:", dep)
    print("Endpoint:", AZURE_OPENAI_ENDPOINT)
    return AzureChatOpenAI(
        azure_deployment=dep,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        temperature=0,
        max_tokens=2000,
        timeout=60,
        max_retries=2,
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-02-15-preview",
    ).bind_tools(tools)



#runing agenet 
async def agentGis(userInput: str, model_id: str = "gpt4o-mini"):
    tools = [kommune_where_clause]
    model = get_model(model_id, tools)

    msgs = [SystemMessage(content=PROMPT), HumanMessage(content=userInput)]
    tool_map = {t.name: t for t in tools}

    for _ in range(10):
        ai = await model.ainvoke(msgs)
        msgs.append(ai)

        if not ai.tool_calls:
            return ai.content

        for tc in ai.tool_calls:
            tool = tool_map[tc["name"]]
            try:
                result = await tool.ainvoke(tc["args"])
                msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            except Exception as e:
                msgs.append(ToolMessage(content=f"Error executing tool: {e}", tool_call_id=tc["id"]))

    return "Max iterations reached without completion."


# 2) retrive data from database 
# async def run_pipeline(userInput: str, model_id: str):
#     t0 = time.time()
#     sql = await agentGis(userInput, model_id=model_id)
#     result = await query_postgis(sql)
#     return {"model_id": model_id, "sql": sql, "result": result, "latency_s": round(time.time() - t0, 3)}





async def sqlPipline(userInput: str, model_id: str):
    sql = await agentGis(userInput, model_id=model_id)
    return {
        "model_id": model_id,
        "sql": sql,
    }





# run mutilple parallet 
async def run_all(user_input: str, model_ids: list[str]):
    tasks = []

    for model_id in model_ids:
        task = sqlPipline(user_input, model_id)
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results





async def main():
    userInput = "Find 10 buildings in Grimstad."

    results = await run_all(
        userInput,
        ["gpt-35-turbo", "gpt-4.1-nano", "gpt-4", "gpt-4o", "GPT-4o-mini"]
    )

    for r in results:
        print("\n" + "=" * 60)
        if isinstance(r, Exception):
            print("MODEL ERROR:", type(r).__name__, str(r))
            continue

        print("MODEL:", r["model_id"])
        print("SQL:\n", r["sql"])

        if "latency_s" in r:
            print("Latency:", r["latency_s"], "s")
        if "result" in r:
            print("Result:\n", r["result"])



if __name__ == "__main__":
    asyncio.run(main())
