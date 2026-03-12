import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.tools.longChainTools import (
    tool_get_table_schema,
    execute_postgis_query,
)


class ExecutionStep(BaseModel):
    step: int
    database: str
    schema: str | None = None
    table: str | None = None
    operation: Literal[
        "inspect_schema",
        "inspect_values",
        "query_rows",
        "aggregate",
        "spatial_filter",
        "join_with_previous",
    ]
    goal: str
    sql: str | None = None
    input_from_step: int | None = None


class MultiDBPlan(BaseModel):
    needs_multi_db: bool
    final_goal: str
    steps: list[ExecutionStep]


async def build_multidb_plan(
    user_input: str,
    relevant_tables_context: str,
    model_id: str = "gpt-4o-mini",
    params=None,
) -> dict:
    from backend.Client.AzureChatOpenAI import get_model_cached

    model = get_model_cached(model_id, tools=[], params=params)

    msgs = [
        SystemMessage(content=f"""
You are a spatial analysis planner.

Return ONLY valid JSON.
Do not use markdown.
Do not add explanations outside JSON.

JSON format:
{{
  "needs_multi_db": true,
  "final_goal": "short text",
  "steps": [
    {{
      "step": 1,
      "database": "db_name",
      "schema": "schema_name",
      "table": "table_name",
      "operation": "inspect_schema|inspect_values|query_rows|aggregate|spatial_filter|join_with_previous",
      "goal": "short goal",
      "sql": null,
      "input_from_step": null
    }}
  ]
}}

Rules:
- Break the task into executable steps
- Use one database per step
- If a later step depends on a previous step, set input_from_step
- Do not invent databases, schemas, tables, or columns
- Prefer the provided relevant tables
- If category values are uncertain, add an inspect_values step first
- If geometry filtering is needed, make that explicit in a spatial_filter step
- If the task can be solved in one database, set needs_multi_db to false

Relevant tables:
{relevant_tables_context}
"""),
        HumanMessage(content=user_input),
    ]

    ai = await model.ainvoke(msgs)
    text = (ai.content or "").strip()

    try:
        return json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Planner did not return valid JSON: {text}") from e


async def fill_sql_for_steps(
    user_input: str,
    plan: dict,
    relevant_tables_context: str,
    model_id: str = "gpt-4o-mini",
    params=None,
) -> dict:
    from backend.Client.AzureChatOpenAI import get_model_cached

    model = get_model_cached(model_id, tools=[], params=params)

    msgs = [
        SystemMessage(content=f"""
You are a PostGIS SQL writer.

Return ONLY valid JSON.
Do not use markdown.
Do not add explanations outside JSON.

Fill the sql field for each step that needs SQL.
Leave sql as null for inspect_schema.

Rules:
- Do not invent databases, schemas, tables, or columns
- Use only names found in the plan or relevant tables context
- Use one database per step
- If a step depends on input_from_step, write SQL that can later be combined with the previous step output
- If category values are uncertain, write an inspection query first
- Do not assume geometry column is named geom
- Use municipality attribute filters if no usable geometry column exists

Relevant tables:
{relevant_tables_context}

Plan:
{json.dumps(plan, ensure_ascii=False, indent=2)}
"""),
        HumanMessage(content=user_input),
    ]

    ai = await model.ainvoke(msgs)
    text = (ai.content or "").strip()

    try:
        return json.loads(text)
    except Exception as e:
        raise RuntimeError(f"SQL filler did not return valid JSON: {text}") from e


async def execute_multidb_plan(plan: dict) -> dict:
    step_outputs = {}

    for step in plan["steps"]:
        step_no = step["step"]
        op = step["operation"]

        if op == "inspect_schema":
            result = await tool_get_table_schema.ainvoke({
                "databaseName": step["database"],
                "schemaName": step["schema"],
                "table_name": step["table"],
            })
            step_outputs[step_no] = result
            continue

        sql = step.get("sql")
        if not sql:
            raise RuntimeError(f"Step {step_no} has no SQL: {step}")

        result = await execute_postgis_query.ainvoke({
            "sql": sql,
            "databaseName": step["database"],
        })
        step_outputs[step_no] = result

    return step_outputs


async def run_multidb_pipeline(
    user_input: str,
    relevant_tables_context: str,
    model_id: str = "gpt-4o-mini",
    params=None,
) -> tuple[dict, dict]:
    plan = await build_multidb_plan(
        user_input=user_input,
        relevant_tables_context=relevant_tables_context,
        model_id=model_id,
        params=params,
    )

    plan_with_sql = await fill_sql_for_steps(
        user_input=user_input,
        plan=plan,
        relevant_tables_context=relevant_tables_context,
        model_id=model_id,
        params=params,
    )

    outputs = await execute_multidb_plan(plan_with_sql)
    return plan_with_sql, outputs