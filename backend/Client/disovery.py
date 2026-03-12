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
                                           kommune_where_clause, tool_get_available_databases, tool_get_available_databases, tool_get_available_schemas)




PROMPT = (Path(__file__).parent / "prompt.txt").read_text(encoding="utf-8")
#DATABASE_URL = os.getenv("PGCONN_STRING")
from backend.firebase_admin_init import fs
from firebase_admin import firestore

from dataclasses import dataclass
from typing import Optional
from dataclasses import asdict


from backend.mcp_c_s.mcp_client import MCPClientManager





#declre obejct to store paramkn
@dataclass
class ModelParams:
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 60
    max_retries: int = 2


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




# # initialize model
# def get_model(model_id: str, tools, params: Optional[ModelParams] = None):
#     params = params or ModelParams()

#     dep = AZURE_MODEL_DEPLOYMENT.get(model_id)
#     if not dep:
#         raise ValueError(f"Unknown model_id: {model_id}")

#     kwargs = {
#         "azure_deployment": dep,
#         "azure_endpoint": AZURE_OPENAI_ENDPOINT,
#         "api_key": AZURE_OPENAI_KEY,
#         "api_version": "2024-02-15-preview",
#         "temperature": params.temperature,
#         "max_tokens": params.max_tokens,
#         "timeout": params.timeout,
#         "max_retries": params.max_retries,
#     }



#     return AzureChatOpenAI(**kwargs).bind_tools(tools, tool_choice=params.tool_choice)







# #cach the parameters, to reuse the same pamras and inilialize once, to avoid creating hhtp clint each time

# _model_cache = {}

# def get_model_cached (model_id, tools, params):
    

#     params =  params or ModelParams()

#     key =(model_id,tuple(sorted(params.__dict__.items())), tuple(t.name for t in tools))
    
#     if key not in _model_cache: 
#         model = get_model(model_id, tools=tools, params=params)
#         _model_cache[key] = model 

#         return model 
#     else:
#         return _model_cache[key]













# async def agentGis(userInput: str, model_id: str = "gpt-4o-mini", params: Optional[ModelParams] = None):



#     #deifine tools 
#     tools = [kommune_where_clause, execute_postgis_query, tool_get_available_tables, tool_validate_sql, tool_get_table_schema, tool_get_available_databases]

#     #call the modell, and  send model id and tools 
#     model = get_model_cached(model_id, tools, params=params)
    

#     msgs = [SystemMessage(content=PROMPT), HumanMessage(content=userInput)]


#     #creatnig dictionart that map tool name
#     tool_map = {t.name: t for t in tools}

#     #store tool-call logs 
#     tool_calls_log = []

#     all_errors = []

#     iterations = 0
#     last_sql = None

#     last_validated_sql = None
#     last_validation_ok = False

#     for _ in range(10):

#         iterations += 1 

#         #send message to llm and wait to get the respons
#         ai = await model.ainvoke(msgs)
#         msgs.append(ai)

#         # extract sql from llm
#         #check the content exist and not white speace 
#         if ai.content and ai.content.strip():
#             last_sql = ai.content.strip()



#         #if there is not tool call in this iteration, the model is returning the final answer
#         if not ai.tool_calls:
#             current_sql = (ai.content or "").strip()

#             #evaualting whether the curremt sql is valid 

#             validated = (
#                 last_validation_ok
#                 and last_validated_sql is not None
#                 and current_sql == last_validated_sql
#             )

#             agent_called_validate = (last_validated_sql is not None)

#             return AgentResult(
#                     sql=current_sql,
#                     iterations=iterations,
#                     tool_calls=tool_calls_log,
#                     valid=validated,
#                     agent_validated=agent_called_validate,
#                     all_errors=[] if validated else ["Final SQL was not validated"],
#                 )
#         iteration_tools = []

#         for tc in ai.tool_calls:
#             tool = tool_map[tc["name"]]
#             try:
#                 result = await tool.ainvoke(tc["args"])  # unnecessary to run all tools again; may only need validation

#              #track which sql was validated and wether validation was ok 

#                 if(tc["name"]=="tool_validate_sql"):
#                     last_validated_sql = (tc["args"].get("sql") or "").strip()
#                     last_validation_ok = str(result).strip().lower() == "ok"
             
#                 iteration_tools.append({
#                     "tool":   tc["name"],
#                     "args":   tc["args"],
#                     "result": str(result),
#                 })

#                 #reconsider this 
#                 msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
#             except Exception as e:
#                 iteration_tools.append({
#                     "tool":   tc["name"],
#                     "args":   tc["args"],
#                     "result": f"Error: {e}",
#                 })
#                 msgs.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"]))



#         # store the tool call logs for this iteration 
#         tool_calls_log.append({
#             "iteration": iterations,
#             "tools":     iteration_tools,
#         })


#         #collect errors from toll result cross iterations 
#         all_errors = [
#                 t["result"]
#                 for entry in tool_calls_log
#                 for t in entry["tools"]
#                 if isinstance(t.get("result"), str) and t["result"].startswith("Error")
#             ]


#     return AgentResult(
#         sql=last_sql or "Max iterations reached.",
#         iterations=iterations,
#         tool_calls=tool_calls_log,
#         valid=False,
#         agent_validated=False,
#         all_errors=all_errors or ["Max iterations reached"],
#     )







async def sqlPipeline(question: str, model_id: str, max_attempts: int = 3,  params: Optional[ModelParams] = None ):
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
        final_sql=False,
        valid=False,
        attempts=attempts,
        latency_s=round(time.time() - t0, 3),
        all_errors=all_errors,
        params=params or ModelParams(),
        agent= result
        
    )




#optimize runn_all (later)
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
    run_ref = fs.collection("evaluation_runsLongChain").document()
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
        # Make sure errors are deduped  

        doc["all_errors"] = _dedup_keep_order(doc.get("all_errors") or [])
        if doc.get("agent") and doc["agent"].get("all_errors"):
            doc["agent"]["all_errors"] = _dedup_keep_order(doc["agent"]["all_errors"])

        mref = run_ref.collection("models").document(r.model_id)
        batch.set(mref, doc)

    batch.commit()
    return run_ref.id

# async def main():
#     try:
#         params = ModelParams(temperature=0, timeout=60, max_tokens=2000)

#         queries = "Hvor mange fritidsboliger/næring/eneboliger totalt, i en kommune, tetthet, dbscan clusters."

        





# #        model_ids = ["gpt-35-turbo", "gpt-4.1-nano", "gpt-4", "gpt-4o", "gpt-4o-mini"]
#         model_ids = ["gpt-4o-mini"]


#         results = await run_all(queries, model_ids, params=params)

#         _print_summary_table(queries, results)
#         _print_details(results)

#         run_id = await store_run_firestore(queries, params, results)
#         print(f"Saved to Firestore: evaluation_runs/{run_id}\n")


#     finally:
      
#         MCPClientManager.close









prompt2 = (Path(__file__).parent / "prompt2.txt").read_text(encoding="utf-8")



import asyncio
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage







@dataclass
class ModelParams:
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 60
    max_retries: int = 2
    tool_choice: str = "auto"


@dataclass
class AgentResult:
    answer: str
    iterations: int
    tool_calls: list
    all_errors: list[str]




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



import asyncio
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from backend.Client.catalogBuilder import build_metadata_catalog
from backend.Client.services.SchemaandRowRetrieval import schema_and_row_retrieval
from backend.embeddings.embeddings import embed_text
from backend.embeddings.build_layer_index import build_table_description
from backend.embeddings.retrieve_layers import (
    retrieve_relevant_tables_with_embeddings,
    format_table_context,
)
from backend.Client.multiDb import run_multidb_pipeline
from backend.tools.longChainTools import (
    tool_get_available_databases,
    tool_get_available_schemas,
    tool_get_available_tables,
    tool_get_table_schema,
    execute_postgis_query,
    kommune_where_clause,
)

# disse må finnes hos deg
# from ... import get_model_cached, AgentResult, ModelParams, prompt2


class TableChoice(BaseModel):
    database: str
    schema: str
    table: str
    role: Literal["target", "context", "support"]


class SpatialAnalysisPlan(BaseModel):
    analysis_type: Literal["count", "density", "dbscan", "buffer", "intersect", "metadata"]
    databases: list[str]
    tables: list[TableChoice]
    kommune: str | None = None
    reasoning: str


def needs_multidb(plan: dict) -> bool:
    dbs = set(plan.get("databases", []))
    return len(dbs) > 1


async def build_spatial_plan(
    user_input: str,
    relevant_tables_context: str,
    model_id: str = "gpt-4o-mini",
    params: Optional["ModelParams"] = None,
) -> dict:
    planner_model = get_model_cached(model_id, tools=[], params=params)

    msgs = [
        SystemMessage(content=f"""
You are a spatial analysis planner.

Return ONLY valid JSON.
Do not use markdown.
Do not add explanations outside JSON.

JSON format:
{{
  "analysis_type": "count|density|dbscan|buffer|intersect|metadata",
  "databases": ["database1"],
  "tables": [
    {{
      "database": "database_name",
      "schema": "schema_name",
      "table": "table_name",
      "role": "target|context|support"
    }}
  ],
  "kommune": null,
  "reasoning": "short reason"
}}

Rules:
- Use only the tables listed below
- Do not invent databases, schemas, tables, or columns
- Choose the most likely target and context tables
- If the task is about listing tables/schemas/databases, use analysis_type="metadata"
- If a municipality is mentioned, put it in the kommune field

Relevant tables:
{relevant_tables_context}
"""),
        HumanMessage(content=user_input),
    ]

    ai = await planner_model.ainvoke(msgs)
    text = (ai.content or "").strip()

    try:
        return json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Planner did not return valid JSON: {text}") from e


async def simple_agent(
    user_input: str,
    msgs: list,
    relevant_tables_context: str,
    plan: dict,
    model_id: str = "gpt-4o-mini",
    params: Optional["ModelParams"] = None,
):
    tools = [
        tool_get_available_databases,
        tool_get_available_schemas,
        tool_get_available_tables,
        tool_get_table_schema,
        execute_postgis_query,
        kommune_where_clause,
    ]

    model = get_model_cached(model_id, tools, params=params)
    tool_map = {t.name: t for t in tools}

    local_msgs = msgs.copy()
    local_msgs.append(SystemMessage(content=f"""
You are a spatial analysis agent for PostGIS.

Relevant tables:
{relevant_tables_context}

Spatial plan:
{json.dumps(plan, ensure_ascii=False, indent=2)}

Rules:
- Do not invent databases, schemas, tables, columns, or category values
- Follow the spatial plan
- Prefer the retrieved tables
- Use tools only when needed
- If categorical values are uncertain, first run an inspection query
- Inspection queries should use DISTINCT or GROUP BY with COUNT and LIMIT
- After inspection, continue to the final query if possible
- Do not assume the geometry column is named geom
- If a table has another geometry column, use that column
- If a table has no usable geometry column, do not use kommune_where_clause
- If no geometry column exists, filter by kommunenavn or kommunenummer instead
- If enough information exists, do not stop at explanation only
- Do not end with only metadata listing unless the task is clearly metadata-only
- Return the final answer only after executing the necessary query when possible
"""))
    local_msgs.append(HumanMessage(content=user_input))

    tool_calls_log = []
    all_errors = []

    for i in range(10):
        ai = await model.ainvoke(local_msgs)
        local_msgs.append(ai)

        if not ai.tool_calls:
            return AgentResult(
                answer=(ai.content or "").strip(),
                iterations=i + 1,
                tool_calls=tool_calls_log,
                all_errors=all_errors,
            )

        iteration_tools = []

        for tc in ai.tool_calls:
            tool = tool_map[tc["name"]]
            print(f"Tool: {tc['name']} args={tc['args']}")

            try:
                result = await tool.ainvoke(tc["args"])
                print(f"Result: {result}")

                iteration_tools.append({
                    "tool": tc["name"],
                    "args": tc["args"],
                    "result": str(result),
                })
                local_msgs.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

            except Exception as e:
                err = f"Error: {e}"
                iteration_tools.append({
                    "tool": tc["name"],
                    "args": tc["args"],
                    "result": err,
                })
                local_msgs.append(
                    ToolMessage(content=err, tool_call_id=tc["id"])
                )
                all_errors.append(err)

        tool_calls_log.append({
            "iteration": i + 1,
            "tools": iteration_tools,
        })

    return AgentResult(
        answer="Max iterations reached.",
        iterations=10,
        tool_calls=tool_calls_log,
        all_errors=all_errors or ["Max iterations reached"],
    )


async def build_and_save_table_index() -> None:
    catalog = await build_metadata_catalog(
        get_available_databases=tool_get_available_databases,
        get_available_schemas=tool_get_available_schemas,
        get_available_tables=tool_get_available_tables,
        get_table_schema=tool_get_table_schema,
    )

    index = []
    for item in catalog:
        desc = build_table_description(item)
        vec = embed_text(desc)

        index.append({
            "database": item["database"],
            "schema": item["schema"],
            "table": item["table"],
            "full_name": item["full_name"],
            "columns": item["columns"],
            "description": desc,
            "embedding": vec,
        })

    output_path = Path(__file__).parent / "table_index.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    print(f"Index size: {len(index)}")
    print(f"Saved to: {output_path}")


async def main():
    params = ModelParams()
    model_id = "gpt-4o"
    msgs = [SystemMessage(content=prompt2)]

    if not (Path(__file__).parent / "table_index.json").exists():
        await build_and_save_table_index()

    while True:
        query = input("\nSpørsmål (q for å avslutte): ").strip()
        if query.lower() == "q":
            break

        tables = retrieve_relevant_tables_with_embeddings(query, k=8)

        print("\nRetrieved tables:")
        for t in tables:
            print(t["full_name"], t["score"])

        relevant_tables_context = format_table_context(tables)
        print("\nRelevant tables context:")
        print(relevant_tables_context)

        plan = await build_spatial_plan(
            query,
            relevant_tables_context,
            model_id=model_id,
            params=params,
        )

        print("\nSpatial plan:")
        print(json.dumps(plan, ensure_ascii=False, indent=2))

        if needs_multidb(plan):
            print("\nRunning multi-db pipeline...")
            plan_with_sql, outputs = await run_multidb_pipeline(
                user_input=query,
                relevant_tables_context=relevant_tables_context,
                model_id=model_id,
                params=params,
            )
            print("\nMulti-db plan with SQL:")
            print(json.dumps(plan_with_sql, ensure_ascii=False, indent=2))
            print("\nStep outputs:")
            print(outputs)
            continue

        result = await simple_agent(
            query,
            msgs,
            relevant_tables_context,
            plan,
            model_id=model_id,
            params=params,
        )

        test, text = schema_and_row_retrieval(result)
        print("\nevaluation:", test, text)
        print("\nSvar:")
        print(result.answer)


if __name__ == "__main__":
    asyncio.run(main())