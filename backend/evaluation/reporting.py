import asyncio
import logging
from pathlib import Path
import sys
import time
import uuid
from dataclasses import asdict
import json


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))




from agent_models import EvalRun, PipelineResult, AgentTrace, AgentRunError
from agent_helpers import (
    _dedup,
    _empty_token_dict,
    _short,
    load_prompt,
)
from backend.storage.AzureBlobStorage import (
    upload,
    query_hash,
    query_meta_exists,
    _update_queries_index,
    _update_runs_index,
)


logger = logging.getLogger(__name__)

def extract_agent_trace(result: PipelineResult) -> AgentTrace:
    tables, databases = [], []
    total_tool_calls = 0
    tool_call_counts: dict[str, int] = {}
    executions: list[dict] = []
    merged_step_registry: dict = {}

    if not result.run_logs:
        return AgentTrace(
            [], [], [], [],
            result.valid_sql,
            result.executed,
            0,
            [],
            {},
        )

    for run in result.run_logs:
        for db in run.get("discovered_databases", []):
            if db not in databases:
                databases.append(db)

        for tbl in run.get("discovered_tables", []):
            if tbl not in tables:
                tables.append(tbl)

        for step_no, meta in (run.get("step_registry") or {}).items():
            merged_step_registry[step_no] = meta

        for entry in (run.get("iterations") or []):
            for t in entry.get("tools", []):
                total_tool_calls += 1
                tool_name = t.get("tool")
                if tool_name:
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

    for step_no, meta in sorted(
        merged_step_registry.items(),
        key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0])
    ):
        executions.append({
            "iteration": None,
            "step": step_no,
            "database": meta.get("database"),
            "sql": meta.get("sql"),
            "success": meta.get("executed", False),
            "error": None if meta.get("executed", False) else meta.get("status"),
            "rows_returned": meta.get("rows_returned"),
            "status": meta.get("status"),
        })

    successful = [e for e in executions if e["success"]]
    all_sql_text = " ".join((e.get("sql") or "").lower() for e in successful)

    tables_in_result = [t for t in tables if t.lower() in all_sql_text]
    databases_in_result = list(dict.fromkeys(
        e["database"] for e in successful if e.get("database")
    ))

    query_steps_trace = [
        s for s in merged_step_registry.values()
        if s.get("action") == "query"
    ]

    if query_steps_trace:
        validated = all(s.get("validated", False) for s in query_steps_trace)
        executed_successfully = all(
            s.get("executed", False) for s in merged_step_registry.values()
        )
    else:
        validated = result.valid_sql
        executed_successfully = result.executed

    return AgentTrace(
        tables_explored=tables,
        tables_in_result=tables_in_result,
        databases_explored=databases,
        databases_in_result=databases_in_result,
        validated_before_exec=validated,
        executed_successfully=executed_successfully,
        total_tool_calls=total_tool_calls,
        executions=executions,
        tool_call_counts=tool_call_counts,
    )





def _print_summary(query: str, results: list):
    logger.info("\n" + "#" * 130)
    logger.info("QUERY")
    logger.info("  %s", query)
    logger.info("#" * 130)

    logger.info(
        "%-20s %-8s %-8s %-8s %-6s %-6s %-6s %-6s %-10s %-8s %-10s %s",
        "MODEL",
        "SQL_OK",
        "EXEC",
        "TASK",
        "ITER",
        "REP",
        "RST",
        "API",
        "LATENCY",
        "TOOLS",
        "TOKENS",
        "ERROR",
    )
    logger.info("-" * 130)

    for r in results:
        if isinstance(r, Exception):
            logger.info(
                "%-20s %-8s %-8s %-8s %-6s %-6s %-6s %-6s %-10s %-8s %-10s %s",
                "ERROR",
                "False",
                "False",
                "False",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                _short(str(r), 300),
            )
            continue

        trace = extract_agent_trace(r)
        err = "; ".join(_dedup(r.all_errors)) or "-"
        total_tokens = r.tokens.total_tokens if r.tokens else "-"

        logger.info(
            "%-20s %-8s %-8s %-8s %-6s %-6s %-6s %-6s %-10s %-8s %-10s %s",
            r.model_id,
            str(r.valid_sql),
            str(r.executed),
            str(r.task_completed),
            r.iterations,
            r.sql_repairs_used,
            r.restarts_used,
            r.total_api_calls,
            f"{r.latency_s:.3f}s",
            trace.total_tool_calls,
            str(total_tokens),
            err,
        )
def _print_details(results: list):
    logger.info("\n" + "=" * 110)
    logger.info("DETAILS PER MODEL")
    logger.info("=" * 110)

    for r in results:
        logger.info("\n" + "-" * 110)

        if isinstance(r, Exception):
            logger.info("ERROR")
            logger.info("  message            : %s", r)
            continue

        trace = extract_agent_trace(r)
        logger.info("[%s]", r.model_id)
        logger.info("  valid_sql          : %s", r.valid_sql)
        logger.info("  executed           : %s", r.executed)
        logger.info("  iterations         : %s", r.iterations)
        logger.info("  sql_repairs        : %s", r.sql_repairs_used)
        logger.info("  restarts           : %s", r.restarts_used)
        logger.info("  latency_s          : %.3f", r.latency_s)
        logger.info("  api_calls          : %s", r.total_api_calls)

        step_registry = {}
        sql_repairs_by_step = {}
        step_retries_by_step = {}

        for run in (r.run_logs or []):
            step_registry.update(run.get("step_registry", {}))
            for step_no, count in (run.get("sql_repair_counts") or {}).items():
                sql_repairs_by_step[step_no] = count
            for step_no, count in (run.get("step_retry_counts") or {}).items():
                step_retries_by_step[step_no] = count


        if step_registry:
            logger.info("\n  STEP REGISTRY:")
            for step_no, meta in sorted(
                step_registry.items(),
                key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0])
            ):
                logger.info("    Step %s:", step_no)
                logger.info("      database        : %s", meta.get("database"))
                logger.info("      validated       : %s", meta.get("validated"))
                logger.info("      executed        : %s", meta.get("executed"))
                logger.info("      sql_repairs     : %s", sql_repairs_by_step.get(step_no, 0))
                logger.info("      rows_returned   : %s", meta.get("rows_returned"))
                logger.info("      status          : %s", meta.get("status"))
                logger.info("      sql             : %s", meta.get("sql"))
                logger.info("      step_retries    : %s", step_retries_by_step.get(step_no, 0))

                    

        if r.tokens:
            logger.info(
                "  tokens             : in=%s out=%s total=%s",
                r.tokens.prompt_tokens,
                r.tokens.completion_tokens,
                r.tokens.total_tokens,
            )

        if r.all_errors:
            logger.info("\n  ERRORS:")
            for err in _dedup(r.all_errors):
                logger.info("    - %s", err)

        logger.info("\n  TRACE:")
        logger.info("    databases explored : %s", trace.databases_explored)
        logger.info("    databases in result: %s", trace.databases_in_result)
        logger.info("    tables explored    : %s", trace.tables_explored)
        logger.info("    tables in result   : %s", trace.tables_in_result)
        logger.info("    validated          : %s", trace.validated_before_exec)
        logger.info("    executed ok        : %s", trace.executed_successfully)
        logger.info("    total tool calls   : %s", trace.total_tool_calls)
        logger.info("    tool call counts   : %s", trace.tool_call_counts)

        if trace.executions:
            logger.info("\n  EXECUTIONS:")
            for i, ex in enumerate(trace.executions):
                logger.info("    Execution %s:", i)
                logger.info("      iteration       : %s", ex["iteration"])
                logger.info("      success         : %s", ex["success"])
                logger.info("      database        : %s", ex["database"])
                logger.info("      sql             : %s", ex["sql"])
                if not ex["success"]:
                    logger.info("      error           : %s", ex["error"])

        if r.run_logs:
            logger.info("\n  RUN LOGS:")
            for run in r.run_logs:
                logger.info("    Run %s:", run.get("run_index", 0))
                logger.info("      attempt         : %s", run.get("attempt_number", 0))
                logger.info("      sql_repairs     : %s", run.get("sql_repairs_used", 0))
                logger.info("      final_sql       : %s", run.get("final_sql") or "(none)")

                atk = run.get("attempt_tokens")
                if atk:
                    logger.info(
                        "      tokens          : in=%s out=%s total=%s",
                        atk.get("prompt_tokens", 0),
                        atk.get("completion_tokens", 0),
                        atk.get("total_tokens", 0),
                    )

                if run.get("validation_error"):
                    logger.info("      validation_error: %s", run["validation_error"])
                if run.get("execution_error"):
                    logger.info("      execution_error : %s", run["execution_error"])

                for entry in run.get("iterations", []):
                    tok = entry.get("tokens", {})
                    logger.info("      Iteration %s:", entry["iteration"])
                    logger.info(
                        "        tokens              : in=%s out=%s total=%s",
                        tok.get("prompt_tokens", 0),
                        tok.get("completion_tokens", 0),
                        tok.get("total_tokens", 0),
                    )
                    logger.info("        ai_content          : %s", entry.get("ai_content", ""))
                    logger.info(
                        "        tool_calls_declared : %s",
                        len(entry.get("tool_calls", [])),
                    )


                    for t in entry.get("tools", []):
                        logger.info("          TOOL   : %s", t["tool"])
                        logger.info("          ARGS   : %s", t["args"])
                        if t["tool"] == "tool_validate_sql":
                            raw = str(t["result"]).strip()
                            parsed = None

                            try:
                                parsed = json.loads(raw)
                                ok = isinstance(parsed, dict) and parsed.get("validated") is True
                            except (json.JSONDecodeError, TypeError):
                                ok = raw.lower() == "ok"

                            logger.info(
                                "          RESULT : %s",
                                f"ok (step={parsed.get('step_number')})" if ok and parsed else f"ok" if ok else f"fail: {raw[:200]}",
                            )
                        else:
                            logger.info("          RESULT : %s", t["result"])










        logger.info("\n  FINAL:")
        logger.info("    answer            : %s", r.final_answer or "(none)")
        logger.info("    sql               : %s", r.final_sql or "(none)")
        logger.info(
            "    result_data       : %s",
            r.result_data if r.result_data is not None else "(none)",
        )

    logger.info("\n" + "=" * 110)






async def store_run(run: EvalRun, results: list) -> str:
    q_hash = query_hash(run.query)
    run_id = uuid.uuid4().hex
    base = f"agent_benchmark_runstest/{q_hash}/runs/{run_id}"

    result_data_by_model = {
        r.model_id: r.result_data
        for r in results
        if not isinstance(r, Exception)
    }

    loop = asyncio.get_running_loop()

    async def _upload(path: str, data) -> None:
        try:
            await loop.run_in_executor(None, upload, path, data)
        except Exception as e:
            logger.warning("upload failed [%s]: %s", path, e)

    try:
        _update_queries_index(q_hash, run.query)
        if not query_meta_exists(q_hash):
            await _upload(
                f"agent_benchmark_runstest/{q_hash}/meta.json",
                {"query": run.query, "query_hash": q_hash},
            )
    except Exception as e:
        logger.warning("Failed to update query index: %s", e)

    await _upload(f"{base}/system_prompt.txt", load_prompt(run.prompt_name))

    summary = []
    normalized_docs = []

    for r in results:
        if isinstance(r, AgentRunError):
            snap = r.snapshot or {}
            tokens = snap.get("tokens", _empty_token_dict())
            doc = {
                "run_id": run_id,
                "query": run.query,
                "model_id": snap.get("model_id"),
                "status": "error",
                "error": str(r),
                "valid_sql": False,
                "executed": False,
                "iterations": snap.get("iterations", 0),
                "sql_repairs_used": snap.get("sql_repairs_used", 0),
                "restarts_used": snap.get("restart_count", 0),
                "total_api_calls": snap.get("total_api_calls", 0),
                "total_tool_calls": snap.get("total_tool_calls", 0),
                "latency_s": None,
                "tokens": tokens,
                "errors": snap.get("all_errors", []),
                "final_answer": None,
                "final_sql": snap.get("final_sql"),
                "trace": None,
                "run_logs": snap.get("run_logs", []),
                "task_completed": False,
                "answer_based_on_result": False,
            }
            summary.append({
                "model_id": doc["model_id"],
                "status": "error",
                "valid_sql": False,
                "executed": False,
                "iterations": doc["iterations"],
                "sql_repairs_used": doc["sql_repairs_used"],
                "restarts_used": doc["restarts_used"],
                "total_api_calls": doc["total_api_calls"],
                "total_tool_calls": doc["total_tool_calls"],
                "latency_s": None,
                "total_tokens": tokens.get("total_tokens"),
            })

        else:
            trace = extract_agent_trace(r)
            tokens = asdict(r.tokens) if r.tokens else _empty_token_dict()
            status = "completed" if r.task_completed else "incomplete"
            doc = {
                "run_id": run_id,
                "query": run.query,
                "model_id": r.model_id,
                "status": status,
                "error": None,
                "valid_sql": r.valid_sql,
                "executed": r.executed,
                "iterations": r.iterations,
                "sql_repairs_used": r.sql_repairs_used,
                "restarts_used": r.restarts_used,
                "total_api_calls": r.total_api_calls,
                "total_tool_calls": trace.total_tool_calls,
                "latency_s": r.latency_s,
                "tokens": tokens,
                "errors": _dedup(r.all_errors),
                "final_answer": r.final_answer,
                "final_sql": r.final_sql,
                "trace": asdict(trace),
                "run_logs": r.run_logs or [],
                "task_completed": r.task_completed,
                "answer_based_on_result": r.answer_based_on_result,
            }
            summary.append({
                "model_id": r.model_id,
                "status": status,
                "valid_sql": r.valid_sql,
                "executed": r.executed,
                "iterations": r.iterations,
                "sql_repairs_used": r.sql_repairs_used,
                "restarts_used": r.restarts_used,
                "total_api_calls": r.total_api_calls,
                "total_tool_calls": trace.total_tool_calls,
                "latency_s": r.latency_s,
                "total_tokens": tokens.get("total_tokens"),
            })

        normalized_docs.append(doc)

    await _upload(
        f"{base}/meta.json",
        {
            "run_id": run_id,
            "query": run.query,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt_name": run.prompt_name,
            "params": asdict(run.params),
            "model_ids": run.model_ids,
            "summary": summary,
        },
    )

    try:
        _update_runs_index(q_hash, run_id, summary, run.params)
    except Exception as e:
        logger.warning("Failed to update runs index: %s", e)

    seen_ids: set[str] = set()
    for doc in normalized_docs:
        model_id = doc["model_id"] or "unknown_model"
        key = model_id
        while key in seen_ids:
            key = f"{model_id}_{uuid.uuid4().hex[:6]}"
        seen_ids.add(key)

        result_data = result_data_by_model.get(model_id)
        await _upload(f"{base}/{key}.json", doc)
        await _upload(
            f"{base}/{key}_result_data.json",
            {
                "run_id": run_id,
                "query": run.query,
                "model_id": model_id,
                "status": doc["status"],
                "row_count": (
                    len(result_data)
                    if doc["status"] == "completed" and isinstance(result_data, list)
                    else None
                ),
                "result_data": result_data if doc["status"] == "completed" else None,
            },
        )

    logger.info("SAVED: %s/", base)
    return f"{q_hash}/{run_id}"