import asyncio
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from backend.tools.longChainTools import (execute_postgis_query,
                                           tool_validate_sql,
                                           tool_get_available_tables, 
                                           tool_get_table_schema, 
                                           kommune_where_clause, tool_get_available_databases, get_available_databases)

PROMPT = (Path(__file__).parent / "prompt.txt").read_text(encoding="utf-8")






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


async def simple_agent(
    user_input: str,
    msgs: list,
    model_id: str = "gpt-4o-mini",
    params: Optional[ModelParams] = None,
):
    tools = [
        get_available_databases,
        tool_get_available_tables,
        tool_get_table_schema,
    ]

    model = get_model_cached(model_id, tools, params=params)
    tool_map = {t.name: t for t in tools}

    msgs.append(HumanMessage(content=user_input))

    tool_calls_log = []
    all_errors = []

    for i in range(10):
        ai = await model.ainvoke(msgs)
        msgs.append(ai)

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
            try:
                result = await tool.ainvoke(tc["args"])
                iteration_tools.append({
                    "tool": tc["name"],
                    "args": tc["args"],
                    "result": str(result),
                })
                msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            except Exception as e:
                err = f"Error: {e}"
                iteration_tools.append({
                    "tool": tc["name"],
                    "args": tc["args"],
                    "result": err,
                })
                msgs.append(ToolMessage(content=err, tool_call_id=tc["id"]))
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


async def main():
    params = ModelParams()
    model_id = "gpt-4o-mini"

    msgs = [SystemMessage(content=prompt2)]

    while True:
        query = input("\nSpørsmål (q for å avslutte): ").strip()
        if query.lower() == "q":
            break

        result = await simple_agent(query, msgs, model_id=model_id, params=params)

        print("\nSvar:")
        print(result.answer)


if __name__ == "__main__":
    asyncio.run(main())