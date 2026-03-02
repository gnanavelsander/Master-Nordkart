import sys, os, time, asyncio
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from psycopg_pool import AsyncConnectionPool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_MODEL_DEPLOYMENT
from backend.tools.kartverketAPI import kommune_where_clause

from backend.tools.postgisTools import (execute_postgis_query, tool_validate_sql, tool_get_available_tables, tool_get_table_schema)

PROMPT = (Path(__file__).parent / "prompt.txt").read_text(encoding="utf-8")
DATABASE_URL = os.getenv("PGCONN_STRING")
from backend.firebase_admin_init import fs
from firebase_admin import firestore

from dataclasses import dataclass
from typing import Optional








#declre obejct to store paramkn,,,n,j
@dataclass
class ModelParams:
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 60
    max_retries: int = 2

    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None

    tool_choice:str = "auto"






# initialize model


def get_model(model_id: str, tools, params: Optional[ModelParams] = None):
    params = params or ModelParams()

    dep = AZURE_MODEL_DEPLOYMENT.get(model_id)
    if not dep:
        raise ValueError(f"Unknown model_id: {model_id}")

    kwargs = {
        "azure_deployment": dep,
        "azure_endpoint": AZURE_OPENAI_ENDPOINT,
        "api_key": AZURE_OPENAI_KEY,
        "api_version": "2024-02-15-preview",
        "temperature": params.temperature,
        "max_tokens": params.max_tokens,
        "timeout": params.timeout,
        "max_retries": params.max_retries,
    }

      # add only if it has a value 
    if params.top_p is not None:
        kwargs["top_p"] = params.top_p
    if params.presence_penalty is not None:
        kwargs["presence_penalty"] = params.presence_penalty
    if params.frequency_penalty is not None:
        kwargs["frequency_penalty"] = params.frequency_penalty
    if params.seed is not None:
        kwargs["seed"] = params.seed


    return AzureChatOpenAI(**kwargs).bind_tools(tools, tool_choice=params.tool_choice)







#cach the parameters, to reuse the same pamras and inilialize once, to avoid creating hhtp clint each time


_model_cache = {}

def get_model_cached (model_id, tools, params):
    

    params =  params or ModelParams()

    key =(model_id,tuple(sorted(params.__dict__.items())), tuple(t.name for t in tools))


    if key not in _model_cache: 
        model = get_model(model_id, tools=tools, params=params)
        _model_cache[key] = model 
        return model 
    else:
        return _model_cache[key]














async def agentGis(userInput: str, model_id: str = "gpt-4o-mini", params: Optional[ModelParams] = None):
    #deifine tools 
    tools = [kommune_where_clause, execute_postgis_query, tool_get_available_tables, tool_validate_sql, tool_get_table_schema]

    #call the modell, and  send model id and tools 
    model = get_model_cached(model_id, tools, params=params)
    

    msgs = [SystemMessage(content=PROMPT), HumanMessage(content=userInput)]


    #creatnig dictionart that map tool name
    tool_map = {t.name: t for t in tools}

    #store tool-call logs 
    tool_calls_log = []



    iterations = 0
    last_sql = None

    last_validated_sql = None
    last_validation_ok = False

    for _ in range(10):

        iterations += 1 

    
        ai = await model.ainvoke(msgs)
        msgs.append(ai)
        if ai.content and ai.content.strip():
            last_sql = ai.content.strip()

        print("her er ai result iteration:", iterations, "for model: ",model_id, ai, "\n")



        #if there is not tool call in this iteration, the model is returning the final answer
        if not ai.tool_calls:
            current_sql = (ai.content or "").strip()

            #evaualting whether the curremt sql is valid 

            validated = (
                last_validation_ok
                and last_validated_sql is not None
                and current_sql == last_validated_sql
            )

            agent_called_validate = (last_validated_sql is not None)

            return {
                "sql": current_sql,
                "iterations": iterations,
                "tool_calls": tool_calls_log,
                "valid": validated,
                "agent_validated": agent_called_validate,
                "all_errors": [] if validated else ["Final SQL was not validated"],
            }

        iteration_tools = []
        for tc in ai.tool_calls:
            tool = tool_map[tc["name"]]
            try:
                result = await tool.ainvoke(tc["args"])

             #track which sql was validated and wether validation was ok 

                if(tc["name"]=="tool_validate_sql"):
                    last_validated_sql = (tc["args"].get("sql") or "").strip()
                    last_validation_ok = str(result).strip().lower() == "ok"
             
                iteration_tools.append({
                    "tool":   tc["name"],
                    "args":   tc["args"],
                    "result": str(result),
                })
                msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            except Exception as e:
                iteration_tools.append({
                    "tool":   tc["name"],
                    "args":   tc["args"],
                    "result": f"Error: {e}",
                })
                msgs.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"]))



        # store the tool call logs for this iteration 
        tool_calls_log.append({
            "iteration": iterations,
            "tools":     iteration_tools,
        })


        #collect errors from toll result cross iterations 
        all_errors = [
                t["result"]
                for entry in tool_calls_log
                for t in entry["tools"]
                if isinstance(t.get("result"), str) and t["result"].startswith("Error")
            ]



    return {
        "sql":        last_sql or "Max iterations reached.",
        "iterations": iterations,
        "tool_calls": tool_calls_log,
        "valid":      False,
        "all_errors": all_errors or ["max iteration is reachd"],
    }








async def sqlPipeline(question: str, model_id: str, max_attempts: int = 3,  params: Optional[ModelParams] = None  ):
    attempts = 0
    all_errors = []
    result = None
    current_prompt = question
    t0 = time.time()





    while attempts < max_attempts:
        attempts += 1
        result = await agentGis(current_prompt, model_id=model_id, params= params)
        sql = result["sql"]

        #return the results if sql is valid
        if result["valid"]:
            return {
                "model_id":   model_id,
                "sql":        sql,
                "attempts":   attempts,
                "valid":      True,
                "all_errors": all_errors,
                "latency_s":  round(time.time() - t0, 3),
                "tool_calls": result["tool_calls"],
                "iterations": result["iterations"],
            }
        
        #if the result is not validate it restert the agent with a repair prompt that include the sql and errors

        all_errors.extend(result["all_errors"])
        current_prompt = f"""
You are fixing a broken PostGIS SQL query.
Original question: {question}
SQL that failed: {sql}
Errors: {result["all_errors"]}
Rewrite the SQL to fix the errors. Return only the corrected SQL.
"""

    return {
        "model_id":   model_id,
        "sql":        result["sql"] if result else None,
        "attempts":   attempts,
        "valid":      False,
        "all_errors": all_errors,
        "latency_s":  round(time.time() - t0, 3),
        "tool_calls": result["tool_calls"] if result else [],
        "iterations": result["iterations"] if result else 0,

    }







async def run_all(user_input: str, model_ids: list[str],  params: Optional[ModelParams] = None):
    tasks = [sqlPipeline(user_input, model_id, params=params ) for model_id in model_ids]
    return await asyncio.gather(*tasks, return_exceptions=True)





async def main():
    userInput = "find 10 building in grimstad."

    params =  ModelParams(temperature=0, timeout=60, max_tokens=2000)






    results = await run_all(userInput, ["gpt-35-turbo", "gpt-4.1-nano", "gpt-4", "gpt-4o", "gpt-4o-mini"], params= params)

    not_working = []

    for r in results:
        if isinstance(r, Exception):
            not_working.append(str(r))
            continue

        try:
            # # store to Firestore
            # fs.collection("evaluations").add({
            #     "question":   userInput,
            #     "model_id":   r["model_id"],
            #     "valid":      r["valid"],
            #     "attempts":   r["attempts"],
            #     "latency_s":  r["latency_s"],
            #     "sql":        r["sql"],
            #     "all_errors": r["all_errors"],
            #     "tools_log":  r.get("tool_calls", []),
            #     "timestamp":  firestore.SERVER_TIMESTAMP,
            # })

            # # update aggregated metrics
            # ref = fs.collection("metrics").document(r["model_id"])
            # ref.set({
            #     "total":         firestore.Increment(1),
            #     "first_pass":    firestore.Increment(1 if r["attempts"] == 1 else 0),
            #     "repaired":      firestore.Increment(1 if r["valid"] else 0),
            #     "total_latency": firestore.Increment(r["latency_s"]),
            # }, merge=True)

            # doc = ref.get()
            # data = doc.to_dict()

            # # print results
            # print("\n" + "=" * 60)
            # print("MODEL:   ", r["model_id"])
            # print("VALID:   ", r["valid"])
            # print("ATTEMPTS:", r["attempts"])
            # print("LATENCY: ", r["latency_s"], "s")

            # # tool call log
            # if r.get("tool_calls"):
            #     print("\n--- Tool calls ---")
            #     for entry in r["tool_calls"]:
            #         for t in entry["tools"]:
            #             if t["tool"] == "tool_validate_sql":
            #                 result = t.get("result", "")
            #                 if result == "ok":
            #                     print(f"  Iteration {entry['iteration']}: tool_validate_sql → ✓ ok")
            #                 else:
            #                     print(f"  Iteration {entry['iteration']}: tool_validate_sql → ✗ {result}")
            #             else:
            #                 print(f"  Iteration {entry['iteration']}: {t['tool']}({t['args']})")
            # else:
            #     print("\n--- No tool calls ---")

            # print("ITERATIONS:", r.get("iterations", "N/A"))

            # # aggregated metrics
            # if data:
            #     total = data["total"]
            #     print(f"\n--- Metrics ---")
            #     print(f"Total runs:  {total}")
            #     print(f"First-pass:  {round(data['first_pass'] / total * 100, 1)}%")
            #     print(f"Repaired:    {round(data['repaired'] / total * 100, 1)}%")
            #     print(f"Avg latency: {round(data['total_latency'] / total, 2)}s")

            # if not r["valid"]:
            #     print("ALL ERRORS:", r["all_errors"])

            print("SQL:\n", r["sql"])
            print("\n" + "#" * 20)

        except Exception as e:
            print(f"Failed to store evaluation: {e}")

    if not_working:
        print("\n--- Not working ---")
        for m in not_working:
            print(f"  {m}")


if __name__ == "__main__":
    asyncio.run(main())