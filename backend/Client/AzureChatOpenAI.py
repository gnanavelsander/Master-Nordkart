import sys, os, time, asyncio
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from psycopg_pool import AsyncConnectionPool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_MODEL_DEPLOYMENT
from backend.tools.kartverketAPI import kommune_where_clause
from backend.tools.validate_sql import validate_sql_explain

PROMPT = (Path(__file__).parent / "prompt.txt").read_text(encoding="utf-8")
DATABASE_URL = os.getenv("PGCONN_STRING")


def get_model(model_id: str, tools):
    dep = AZURE_MODEL_DEPLOYMENT[model_id]
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
                msgs.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"]))

    return "Max iterations reached without completion."


async def sqlPipeline(question: str, model_id: str, pool: AsyncConnectionPool, max_attempts: int = 3):
    attempts = 0
    all_errors = []
    sql = None
    current_prompt = question
    t0 = time.time()

    while attempts < max_attempts:
        attempts += 1

        # Generate SQL using the LLM agent
        sql = await agentGis(current_prompt, model_id=model_id)

        # Borrow a database connection only for validation
        async with pool.connection() as conn:
            
            # Validate the SQL using EXPLAIN 

            v = await validate_sql_explain(conn, sql)


        # If validation succeeds, return result
        if v.ok:
            return {
                "model_id": model_id,
                "sql": sql,
                "attempts": attempts,
                "valid": True,
                "error": None,
                "all_errors": all_errors,
                "latency_s": round(time.time() - t0, 3),
            }
        

        # Collect validation errors
        all_errors.append(v.error)


        # Construct a repair prompt for the LLM
        current_prompt = f"""
You are fixing a broken PostGIS SQL query.

Original question: {question}
SQL that failed: {sql}
Postgres error: {v.error}

rewrite the SQL to fix the error. Return only the corrected SQL.
"""
    # Return failure result after  all attempts
    return {
        "model_id": model_id,
        "sql": sql,
        "attempts": attempts,
        "valid": False,
        "error": all_errors[-1] if all_errors else None,
        "all_errors": all_errors,
        "latency_s": round(time.time() - t0, 3),
    }


async def run_all(user_input: str, model_ids: list[str], pool: AsyncConnectionPool):
    tasks = [sqlPipeline(user_input, model_id, pool) for model_id in model_ids]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def main():
    userInput = "Find 10 buildings."

    async with AsyncConnectionPool(DATABASE_URL, min_size=2, max_size=10) as pool:
        results = await run_all(
            userInput,
            ["gpt-35-turbo", "gpt-4.1-nano", "gpt-4", "gpt-4o", "GPT-4o-mini"],
            pool
        )

    for r in results:
        print("\n" + "=" * 60)
        if isinstance(r, Exception):
            print("MODEL ERROR:", type(r).__name__, str(r))
            continue
        print("MODEL:   ", r["model_id"],"\n")

        print("VALID:   ", r["valid"])

        print("ATTEMPTS:", r["attempts"])


        print("LATENCY: ", r["latency_s"], "s")
        print("\n" + "#" * 20,"\n")

        print("SQL:\n",    r["sql"])
        if not r["valid"]:
            print("ALL ERRORS:", r["all_errors"])


if __name__ == "__main__":
    asyncio.run(main())