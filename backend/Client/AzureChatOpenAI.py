import sys, os, time, asyncio
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_MODEL_DEPLOYMENT

from backend.tools.longChainTools import (execute_postgis_query,
                                           tool_validate_sql,
                                           tool_get_available_tables, 
                                           tool_get_table_schema, 
                                           kommune_where_clause)




PROMPT = (Path(__file__).parent / "prompt.txt").read_text(encoding="utf-8")
#DATABASE_URL = os.getenv("PGCONN_STRING")
from backend.firebase_admin_init import fs
from firebase_admin import firestore

from dataclasses import dataclass
from typing import Optional
from dataclasses import asdict








#declre obejct to store paramkn
@dataclass
class ModelParams:
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 60
    max_retries: int = 2

    seed: Optional[int] = None

    tool_choice:str = "auto"


@dataclass
class AgentResult:
    sql: str
    iterations: int
    tool_calls: list
    valid: bool
    agent_validated: bool
    all_errors: list[str]



@dataclass
class PipelineResult:
    model_id: str
    final_sql: str | None
    valid: bool
    attempts: int
    latency_s: float
    all_errors: list[str]
    params: ModelParams  
    agent: Optional[AgentResult] = None  




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

    all_errors = []

    iterations = 0
    last_sql = None

    last_validated_sql = None
    last_validation_ok = False

    for _ in range(10):

        iterations += 1 

        #send message to llm and wait to get the respons
        ai = await model.ainvoke(msgs)
        msgs.append(ai)

        # extract sql from llm
        #check the content exist and not white speace 
        if ai.content and ai.content.strip():
            last_sql = ai.content.strip()



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

            return AgentResult(
                    sql=current_sql,
                    iterations=iterations,
                    tool_calls=tool_calls_log,
                    valid=validated,
                    agent_validated=agent_called_validate,
                    all_errors=[] if validated else ["Final SQL was not validated"],
                )
        iteration_tools = []

        for tc in ai.tool_calls:
            tool = tool_map[tc["name"]]
            try:
                result = await tool.ainvoke(tc["args"])  # unnecessary to run all tools again; may only need validation

             #track which sql was validated and wether validation was ok 

                if(tc["name"]=="tool_validate_sql"):
                    last_validated_sql = (tc["args"].get("sql") or "").strip()
                    last_validation_ok = str(result).strip().lower() == "ok"
             
                iteration_tools.append({
                    "tool":   tc["name"],
                    "args":   tc["args"],
                    "result": str(result),
                })

                #reconsider this 
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


    return AgentResult(
        sql=last_sql or "Max iterations reached.",
        iterations=iterations,
        tool_calls=tool_calls_log,
        valid=False,
        agent_validated=False,
        all_errors=all_errors or ["Max iterations reached"],
    )







async def sqlPipeline(question: str, model_id: str, max_attempts: int = 3,  params: Optional[ModelParams] = None  ):
    attempts = 0
    all_errors = []
    result = None
    current_prompt = question
    t0 = time.time()





    while attempts < max_attempts:
        attempts += 1
        result = await agentGis(current_prompt, model_id=model_id, params= params)
        sql = result.sql

        #return the results if sql is valid
        if result.valid:
            return PipelineResult(
                model_id=model_id,
                final_sql=sql,
                valid=True,
                attempts=attempts,
                latency_s=round(time.time() - t0, 3),
                all_errors=all_errors, #which iteration gave which error
                params=params or ModelParams(),
                agent= result
            )
        
        #if the result is not validate it restert the agent with a repair prompt that include the sql and errors
        all_errors.extend(result.all_errors)

      #  all_errors.extend(result["all_errors"])
        current_prompt = f"""
You are fixing a broken PostGIS SQL query.
Original question: {question}
SQL that failed: {sql}
Errors: {result.all_errors}
Rewrite the SQL to fix the errors. Return only the corrected SQL.
"""
    return PipelineResult(
        model_id=model_id,
        final_sql=sql,
        valid=False,
        attempts=attempts,
        latency_s=round(time.time() - t0, 3),
        all_errors=all_errors,
        params=params or ModelParams(),
        agent= result
    )




#optimize runn_all 
async def run_all(user_input: str, model_ids: list[str],  params: Optional[ModelParams] = None):
    tasks = [sqlPipeline(user_input, model_id, params=params ) for model_id in model_ids]
    return await asyncio.gather(*tasks, return_exceptions=True)



def _short(s: str, n: int = 120) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."




def _dedup_keep_order(xs: list[str]) -> list[str]:
    seen, out = set(), []
    for x in xs or []:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _print_summary_table(query: str, results: list):
    print("\n" + "#" * 100)
    print("QUERY:")
    print(query)
    print("#" * 100 + "\n")

    print(f"{'MODEL':<15} {'VALID':<6} {'ATT':<4} {'LAT(s)':<8} {'ERROR'}")
    print("-" * 100)

    for r in results:
        if isinstance(r, Exception):
            print(f"{'ERROR':<15} {'False':<6} {'-':<4} {'-':<8} {_short(str(r), 200)}")
            continue

        err = "; ".join(_dedup_keep_order(r.all_errors))
        print(f"{r.model_id:<15} {str(r.valid):<6} {str(r.attempts):<4} {r.latency_s:<8.3f} {err}")








def _print_tool_logs(pr: "PipelineResult"):
    a = pr.agent
    if not a:
        print("No agent logs.\n")
        return

    print(f"AGENT ITERATIONS: {a.iterations}")
    print(f"AGENT VALID:      {a.valid}")
    print(f"AGENT VALIDATED:  {a.agent_validated}")

    if a.all_errors:
        print("AGENT ERRORS:")
        for e in a.all_errors:
            print(f"  - {e}")

    if not a.tool_calls:
        print("\n--- No tool calls ---\n")
        return

    print("\n--- Tool calls ---")
    for entry in a.tool_calls:
        it = entry.get("iteration")
        tools = entry.get("tools", [])
        for t in tools:
            tool = t.get("tool")
            args = t.get("args", {})
            res  = t.get("result", "")
            if tool == "tool_validate_sql":
                ok = (str(res).strip().lower() == "ok")
                print(f"  Iteration {it}: tool_validate_sql - {'ok' if ok else '✗ ' + _short(str(res), 200)}")
            else:
                print(f"  Iteration {it}: {tool}({args}) → {_short(str(res), 220)}")
    print()








def _print_details(results: list):
    print("\n" + "-" * 100)
    print("DETAILS PER MODEL")
    print("-" * 100 + "\n")

    for r in results:
        if isinstance(r, Exception):
            print("[ERROR]")
            print(str(r))
            print()
            continue

        print(f"[{r.model_id}]")
        print(f"VALID:    {r.valid}")
        print(f"ATTEMPTS: {r.attempts}")
        print(f"LATENCY:  {r.latency_s:.3f}s")

        if r.all_errors:
            print("ALL ERRORS:")
            for e in _dedup_keep_order(r.all_errors):
                print(f"  - {e}")
        else:
            print("ALL ERRORS: (none)")

        _print_tool_logs(r)

        print("FINAL SQL:")
        print(r.final_sql or "(no final_sql)")
        print("\n" + "#" * 30 + "\n")

async def store_run_firestore(query: str, params: ModelParams, results: list):


    #make  collection 
    run_ref = fs.collection("evaluation_runs").document()
    run_ref.set({
        "query": query,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "params": asdict(params),
    })

    batch = fs.batch()
    for r in results:
        if isinstance(r, Exception):
            doc = {
                "model_id": "ERROR",
                "valid": False,
                "attempts": -1,
                "latency_s": None,
                "final_sql": None,
                "all_errors": [str(r)],
                "agent": None,
            }
            mref = run_ref.collection("models").document("ERROR")
            batch.set(mref, doc)
            continue

        doc = asdict(r)
        # Make sure errors are deduped (optional)
        doc["all_errors"] = _dedup_keep_order(doc.get("all_errors") or [])
        if doc.get("agent") and doc["agent"].get("all_errors"):
            doc["agent"]["all_errors"] = _dedup_keep_order(doc["agent"]["all_errors"])

        mref = run_ref.collection("models").document(r.model_id)
        batch.set(mref, doc)

    batch.commit()
    return run_ref.id

async def main():
    params = ModelParams(temperature=0, timeout=60, max_tokens=2000)

    queries = "find 100 buildings in grimstad."

    





    model_ids = ["gpt-35-turbo", "gpt-4.1-nano", "gpt-4", "gpt-4o", "gpt-4o-mini"]

    results = await run_all(queries, model_ids, params=params)

    _print_summary_table(queries, results)
    _print_details(results)

    run_id = await store_run_firestore(queries, params, results)
    print(f"Saved to Firestore: evaluation_runs/{run_id}\n")

if __name__ == "__main__":
    asyncio.run(main())