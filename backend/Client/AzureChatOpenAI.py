import sys
import time
import json
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Any

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import (
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_FOUNDRY_KEY,
    AZURE_FOUNDRY_ENDPOINT,
    AZURE_MODEL_DEPLOYMENT,
    AZURE_OPENAI_MODEL_IDS,
    AZURE_FOUNDRY_MODEL_IDS,
    OLLAMA_MODEL_IDS,
)
from backend.tools.longChainTools import (
    execute_postgis_query,
    tool_validate_sql,
    tool_get_available_tables,
    tool_get_table_schema,
    kommune_where_clause,
    tool_get_available_databases,
)
from AzureBlobStorage import (
    upload,
    query_hash,
    query_meta_exists,
    _update_queries_index,
    _update_runs_index,
)

PROMPT = (Path(__file__).parent / "prompt.txt").read_text(encoding="utf-8")


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        if other is None:
            return
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


@dataclass
class AgentTrace:
    tables_explored: list[str] = field(default_factory=list)
    tables_in_result: list[str] = field(default_factory=list)
    databases_explored: list[str] = field(default_factory=list)
    databases_in_result: list[str] = field(default_factory=list)
    validated_before_exec: bool = False
    first_attempt_success: bool = False
    total_tool_calls: int = 0
    total_executions: int = 0
    executions: list[dict] = field(default_factory=list)


@dataclass
class ModelParams:
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 60
    max_retries: int = 2
    tool_choice: str = "auto"
    max_iterations: int = 10
    max_attempts: int = 5
    top_p: float | None = None
    reasoning_effort: str | None = None


@dataclass
class AgentResult:
    answer: str
    sql: str | None
    result_data: Any | None
    executed: bool
    iterations: int
    tool_calls: list[dict]
    valid: bool
    all_errors: list[str]
    tokens: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class PipelineResult:
    model_id: str
    final_answer: str | None
    final_sql: str | None
    result_data: Any | None
    valid: bool
    executed: bool
    attempts: int
    latency_s: float
    all_errors: list[str]
    params: ModelParams
    agent: Optional[AgentResult] = None
    tokens: TokenUsage = field(default_factory=TokenUsage)
    attempt_tokens: list[dict] = field(default_factory=list)


@dataclass
class EvalRun:
    query: str
    model_ids: list[str]
    params: ModelParams = field(default_factory=ModelParams)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _short(s: str, n: int = 120) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _dedup(xs: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in xs or []:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_tokens(ai) -> TokenUsage:
    usage = getattr(ai, "usage_metadata", None) or getattr(ai, "response_metadata", {}).get("token_usage", {})

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
        total_tokens = usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    prompt_tokens = getattr(usage, "input_tokens", 0) or 0
    completion_tokens = getattr(usage, "output_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


# ── Model factory ─────────────────────────────────────────────────────────────

def get_model(model_id: str, tools, params: ModelParams):
    dep = AZURE_MODEL_DEPLOYMENT.get(model_id)
    if not dep:
        raise ValueError(f"Unknown model_id: {model_id}")

    model_kwargs = {}
    if params.top_p is not None:
        model_kwargs["top_p"] = params.top_p
    if params.reasoning_effort is not None:
        model_kwargs["reasoning_effort"] = params.reasoning_effort

    if model_id in AZURE_OPENAI_MODEL_IDS:
        model = AzureChatOpenAI(
            api_key=AZURE_OPENAI_KEY,
            azure_deployment=dep,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version="2024-12-01-preview",
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            timeout=params.timeout,
            max_retries=params.max_retries,
            model_kwargs=model_kwargs,
        )
        return model.bind_tools(tools, tool_choice=params.tool_choice)

    if model_id in AZURE_FOUNDRY_MODEL_IDS:
        model = ChatOpenAI(
            model=dep,
            api_key=AZURE_FOUNDRY_KEY,
            base_url=AZURE_FOUNDRY_ENDPOINT,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            timeout=params.timeout,
            max_retries=params.max_retries,
            model_kwargs=model_kwargs,
        )
        return model.bind_tools(tools, tool_choice=params.tool_choice)

    if model_id in OLLAMA_MODEL_IDS:
        model = ChatOllama(
            model=model_id,
            temperature=params.temperature,
        )
        return model.bind_tools(tools, tool_choice=params.tool_choice)

    raise ValueError(f"Unknown model_id: {model_id}")


# ── Agent trace extraction ────────────────────────────────────────────────────

def extract_agent_trace(result: PipelineResult) -> AgentTrace:
    if not result.agent or not result.agent.tool_calls:
        return AgentTrace(first_attempt_success=result.attempts == 1)

    tables: list[str] = []
    databases: list[str] = []
    executions: list[dict] = []
    validated_before_exec = False
    total_tool_calls = 0

    for entry in result.agent.tool_calls:
        for t in entry.get("tools", []):
            total_tool_calls += 1
            tool_name = t.get("tool")
            args = t.get("args", {})
            tool_result = t.get("result")

            if tool_name == "tool_get_table_schema":
                table = args.get("table_name")
                if table and table not in tables:
                    tables.append(table)

            elif tool_name == "tool_get_available_tables":
                db = args.get("databaseName") or args.get("database")
                if db and db not in databases:
                    databases.append(db)

            elif tool_name == "tool_get_available_databases":
                pass

            elif tool_name == "tool_validate_sql":
                validated_before_exec = True

            elif tool_name == "execute_postgis_query":
                db = args.get("databaseName") or args.get("database")
                sql = args.get("sql", "")
                ok = not str(tool_result).startswith("Error")

                if db and db not in databases:
                    databases.append(db)

                executions.append(
                    {
                        "iteration": entry.get("iteration"),
                        "database": db,
                        "sql": sql,
                        "success": ok,
                        "error": None if ok else str(tool_result),
                    }
                )

    successful = [e for e in executions if e["success"]]
    all_sql_text = " ".join((e["sql"] or "").lower() for e in successful)
    tables_in_result = [t for t in tables if t.lower() in all_sql_text]
    databases_in_result = list(dict.fromkeys(e["database"] for e in successful if e["database"]))

    return AgentTrace(
        tables_explored=tables,
        tables_in_result=tables_in_result,
        databases_explored=databases,
        databases_in_result=databases_in_result,
        validated_before_exec=validated_before_exec,
        first_attempt_success=result.attempts == 1,
        total_tool_calls=total_tool_calls,
        total_executions=len(executions),
        executions=executions,
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

async def agent(user_input: str, model_id: str, params: ModelParams) -> AgentResult:
    tools = [
        kommune_where_clause,
        execute_postgis_query,
        tool_get_available_tables,
        tool_validate_sql,
        tool_get_table_schema,
        tool_get_available_databases,
    ]
    tool_map = {t.name: t for t in tools}
    model = get_model(model_id, tools, params)

    msgs = [
        SystemMessage(content=PROMPT),
        HumanMessage(content=user_input),
    ]

    tool_calls_log: list[dict] = []
    total_tokens = TokenUsage()

    iterations = 0
    last_sql: str | None = None
    last_validated_sql: str | None = None
    last_validation_ok = False
    last_executed_result: Any | None = None
    last_execute_ok = False

    for _ in range(params.max_iterations):
        iterations += 1

        ai = await model.ainvoke(msgs)
        iter_tokens = _extract_tokens(ai)
        total_tokens.add(iter_tokens)
        msgs.append(ai)

        if not ai.tool_calls:
            current_answer = (ai.content or "").strip()
            valid = (
                last_validation_ok
                and last_validated_sql is not None
                and (current_answer == last_validated_sql or last_sql == last_validated_sql or last_execute_ok)
            )

            return AgentResult(
                answer=current_answer,
                sql=last_validated_sql or last_sql,
                result_data=last_executed_result,
                executed=last_execute_ok,
                iterations=iterations,
                tool_calls=tool_calls_log,
                valid=valid,
                all_errors=[] if last_execute_ok else ["Final answer was not based on executed result"],
                tokens=total_tokens,
            )

        iteration_tools = []

        for tc in ai.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool = tool_map[tool_name]

            try:
                result = await tool.ainvoke(tool_args)

                if tool_name == "tool_validate_sql":
                    last_validated_sql = (tool_args.get("sql") or "").strip()
                    last_validation_ok = str(result).strip().lower() == "ok"

                if tool_name == "execute_postgis_query":
                    if tool_args.get("sql"):
                        last_sql = tool_args["sql"]

                    try:
                        last_executed_result = json.loads(result) if isinstance(result, str) else result
                    except Exception:
                        last_executed_result = result

                    last_execute_ok = not str(result).startswith("Error")

                iteration_tools.append(
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "result": str(result),
                    }
                )
                msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

            except Exception as e:
                err = f"Error: {e}"
                iteration_tools.append(
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "result": err,
                    }
                )
                msgs.append(ToolMessage(content=err, tool_call_id=tc["id"]))

        tool_calls_log.append(
            {
                "iteration": iterations,
                "tools": iteration_tools,
                "tokens": asdict(iter_tokens),
            }
        )

    all_errors = [
        t["result"]
        for entry in tool_calls_log
        for t in entry.get("tools", [])
        if isinstance(t.get("result"), str) and t["result"].startswith("Error")
    ]

    return AgentResult(
        answer="Max iterations reached.",
        sql=last_validated_sql or last_sql,
        result_data=last_executed_result,
        executed=last_execute_ok,
        iterations=iterations,
        tool_calls=tool_calls_log,
        valid=last_validation_ok,
        all_errors=all_errors or ["Max iterations reached"],
        tokens=total_tokens,
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def pipeline(question: str, model_id: str, params: Optional[ModelParams] = None) -> PipelineResult:
    params = params or ModelParams()

    attempts = 0
    all_errors: list[str] = []
    result: AgentResult | None = None
    prompt = question
    t0 = time.time()

    total_tokens = TokenUsage()
    attempt_tokens: list[dict] = []

    while attempts < params.max_attempts:
        attempts += 1

        result = await agent(prompt, model_id=model_id, params=params)
        total_tokens.add(result.tokens)
        attempt_tokens.append(asdict(result.tokens))

        if result.valid and result.executed:
            return PipelineResult(
                model_id=model_id,
                final_answer=result.answer,
                final_sql=result.sql,
                result_data=result.result_data,
                valid=True,
                executed=True,
                attempts=attempts,
                latency_s=round(time.time() - t0, 3),
                all_errors=_dedup(all_errors),
                params=params,
                agent=result,
                tokens=total_tokens,
                attempt_tokens=attempt_tokens,
            )

        all_errors.extend(result.all_errors)

        prompt = f"""
You are fixing a failed spatial analysis run.
Original question: {question}
Last SQL: {result.sql}
Errors: {result.all_errors}

Fix the SQL and execute it. Do not guess results.
""".strip()

    return PipelineResult(
        model_id=model_id,
        final_answer=result.answer if result else None,
        final_sql=result.sql if result else None,
        result_data=result.result_data if result else None,
        valid=False,
        executed=False,
        attempts=attempts,
        latency_s=round(time.time() - t0, 3),
        all_errors=_dedup(all_errors),
        params=params,
        agent=result,
        tokens=total_tokens,
        attempt_tokens=attempt_tokens,
    )


async def run_all(user_input: str, model_ids: list[str], params: Optional[ModelParams] = None) -> list:
    results = []

    for i in range(0, len(model_ids), 5):
        batch = model_ids[i : i + 5]
        print(f"\nRunning batch: {batch}")

        batch_results = await asyncio.gather(
            *[pipeline(user_input, model_id, params=params) for model_id in batch],
            return_exceptions=True,
        )
        results.extend(batch_results)

        if i + 5 < len(model_ids):
            print("Waiting 15s before next batch...")
            await asyncio.sleep(15)

    return results


# ── Printing ──────────────────────────────────────────────────────────────────

def _print_summary(query: str, results: list) -> None:
    print("\n" + "#" * 100)
    print(f"QUERY: {query}")
    print("#" * 100 + "\n")
    print(f"{'MODEL':<20} {'VALID':<6} {'EXEC':<6} {'ATT':<4} {'LAT(s)':<8} {'TOOLS':<6} ERROR")
    print("-" * 120)

    for r in results:
        if isinstance(r, Exception):
            print(f"{'ERROR':<20} {'False':<6} {'False':<6} {'-':<4} {'-':<8} {'-':<6} {_short(str(r), 200)}")
            continue

        trace = extract_agent_trace(r)
        err = "; ".join(_dedup(r.all_errors))
        print(f"{r.model_id:<20} {str(r.valid):<6} {str(r.executed):<6} {str(r.attempts):<4} {r.latency_s:<8.3f} {trace.total_tool_calls:<6} {err}")


def _print_details(results: list) -> None:
    print("\n" + "-" * 100)
    print("DETAILS PER MODEL")
    print("-" * 100 + "\n")

    for r in results:
        if isinstance(r, Exception):
            print(f"[ERROR] {r}\n")
            continue

        trace = extract_agent_trace(r)

        print(f"[{r.model_id}]")
        print(f"VALID:    {r.valid}")
        print(f"EXECUTED: {r.executed}")
        print(f"ATTEMPTS: {r.attempts}")
        print(f"LATENCY:  {r.latency_s:.3f}s")

        if r.tokens:
            print(f"TOKENS:   in={r.tokens.prompt_tokens}  out={r.tokens.completion_tokens}  total={r.tokens.total_tokens}")

        if r.all_errors:
            print("ERRORS:")
            for e in _dedup(r.all_errors):
                print(f"  - {e}")

        print("\nTRACE:")
        print(f"  databases explored:  {trace.databases_explored}")
        print(f"  databases in result: {trace.databases_in_result}")
        print(f"  tables explored:     {trace.tables_explored}")
        print(f"  tables in result:    {trace.tables_in_result}")
        print(f"  validated:           {trace.validated_before_exec}")
        print(f"  first attempt:       {trace.first_attempt_success}")
        print(f"  total tool calls:    {trace.total_tool_calls}")
        print(f"  total executions:    {trace.total_executions}")
        print("  executions:")
        for e in trace.executions:
            print(f"    iter {e['iteration']}: {e['success']} db={e['database']} sql={_short(e['sql'], 80)}")
            if not e["success"]:
                print(f"      error: {_short(e['error'], 100)}")

        if r.agent and r.agent.tool_calls:
            print(f"\nAGENT ITERATIONS: {r.agent.iterations}")
            print("--- Tool calls ---")
            for entry in r.agent.tool_calls:
                tok = entry.get("tokens", {})
                print(
                    f"\nIteration {entry['iteration']}: "
                    f"in={tok.get('prompt_tokens', 0)} "
                    f"out={tok.get('completion_tokens', 0)} "
                    f"total={tok.get('total_tokens', 0)}"
                )
                for t in entry["tools"]:
                    print(f"  TOOL:   {t['tool']}")
                    print(f"  ARGS:   {t['args']}")
                    if t["tool"] == "tool_validate_sql":
                        ok = str(t["result"]).strip().lower() == "ok"
                        print(f"  RESULT: {'ok' if ok else 'fail: ' + _short(t['result'])}")
                    else:
                        print(f"  RESULT: {_short(t['result'])}")
                    print("  " + "-" * 60)

        print(f"\nFINAL ANSWER: {r.final_answer or '(none)'}")
        print(f"FINAL SQL:\n{r.final_sql or '(none)'}")
        print(f"RESULT DATA: {r.result_data if r.result_data is not None else '(none)'}")
        print("\n" + "#" * 30 + "\n")


# ── Storage ───────────────────────────────────────────────────────────────────

def store_run(run: EvalRun, results: list) -> str:
    q_hash = query_hash(run.query)
    run_id = str(int(time.time()))
    base = f"queries/{q_hash}/runs/{run_id}"

    _update_queries_index(q_hash, run.query)

    if not query_meta_exists(q_hash):
        upload(
            f"queries/{q_hash}/meta.json",
            {
                "query": run.query,
                "query_hash": q_hash,
            },
        )

    summary = []
    for r in results:
        if isinstance(r, Exception):
            continue

        trace = extract_agent_trace(r)
        summary.append(
            {
                "model_id": r.model_id,
                "valid": r.valid,
                "executed": r.executed,
                "attempts": r.attempts,
                "latency_s": r.latency_s,
                "tokens": asdict(r.tokens) if r.tokens else None,
                "errors": _dedup(r.all_errors),
                "total_tool_calls": trace.total_tool_calls,
                "total_executions": trace.total_executions,
                "trace": asdict(trace),
            }
        )

    upload(
        f"{base}/meta.json",
        {
            "run_id": run_id,
            "query": run.query,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "params": asdict(run.params),
            "model_ids": run.model_ids,
            "summary": summary,
        },
    )

    # Adjust this call if your helper has a different signature.
    _update_runs_index(q_hash, run_id, summary, run.params)

    for r in results:
        if isinstance(r, Exception):
            upload(f"{base}/ERROR.json", {"error": str(r)})
            continue

        doc = asdict(r)
        doc["all_errors"] = _dedup(doc.get("all_errors") or [])

        if doc.get("agent") and doc["agent"].get("all_errors"):
            doc["agent"]["all_errors"] = _dedup(doc["agent"]["all_errors"])

        upload(f"{base}/{r.model_id}.json", doc)

        trace = extract_agent_trace(r)
        upload(f"{base}/{r.model_id}_trace.json", asdict(trace))

    print(f"\n{'=' * 60}")
    print(f"SAVED TO BLOB: {q_hash}/{run_id}")
    print(f"  {base}/meta.json")
    for r in results:
        if isinstance(r, Exception):
            print(f"  {base}/ERROR.json")
        else:
            print(f"  {base}/{r.model_id}.json")
            print(f"  {base}/{r.model_id}_trace.json")
    print(f"{'=' * 60}\n")

    return f"{q_hash}/{run_id}"


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    run = EvalRun(
        query="Find 10 buildings in Oslo that are within 100 meters of a flood zone."

        
        model_ids=[
            "gpt-35-turbo",
            "gpt-4",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-nano",
            "gpt-5-mini",
        ],
        params=ModelParams(
            temperature=0.0,
            max_tokens=2000,
            timeout=120,
            max_retries=0,
            max_iterations=10,
            max_attempts=5,
        ),
    )

    results = await run_all(run.query, run.model_ids, run.params)
    _print_summary(run.query, results)
    _print_details(results)
    store_run(run, results)


if __name__ == "__main__":
    asyncio.run(main())