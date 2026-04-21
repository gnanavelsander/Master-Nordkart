import asyncio
import logging
from pathlib import Path
import sys
import time
import uuid
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_models import EvalRun, PipelineResult, AgentTrace, AgentRunError
from agent_helpers import (
    _fallback_answer_from_logs,
    _build_step_outcomes,
    _collect_step_stats,
    _dedup,
    _empty_token_dict,
    _executed_from_steps,
    _step_execution_map,
    _format_step_execution,
    _parse_tool_result,
    _short,
    _task_completed_from_steps,
    _valid_sql_from_steps,
    load_prompt,
)
from backend.storage.AzureBlobStorage import (
    upload,
    query_hash,
    query_meta_exists,
    prompt_meta_exists,
    _update_queries_index,
    _update_runs_index,
    src as STORAGE_VERSION,
)

logger = logging.getLogger(__name__)


def _extract_result_contract_info(result_data: dict | None) -> dict:
    if not isinstance(result_data, dict):
        return {
            "model_summary_result": None,
            "full_result_available": False,
            "full_result_row_count": None,
        }

    model_summary = {
        "columns": list(result_data.get("columns") or []),
        "row_count": result_data.get("row_count", result_data.get("rows_returned")),
        "sample_rows": list(result_data.get("sample_rows") or []),
        "result_truncated_for_model": bool(result_data.get("result_truncated_for_model", False)),
    }
    full_rows = result_data.get("rows")
    full_result_available = isinstance(full_rows, list)
    full_result_row_count = len(full_rows) if full_result_available else model_summary["row_count"]

    return {
        "model_summary_result": model_summary,
        "full_result_available": full_result_available,
        "full_result_row_count": full_result_row_count,
    }


def _extract_retrieval_info(run_logs: list | None, fallback: dict | None = None) -> dict:
    retrieval = dict(fallback or {})
    logs = run_logs or []
    if logs:
        latest = logs[-1] or {}
        latest_retrieval = latest.get("retrieval")
        if isinstance(latest_retrieval, dict):
            retrieval = dict(latest_retrieval)

    return {
        "scope_active": retrieval.get("scope_active", False),
        "top_score": retrieval.get("top_score"),
        "candidates": list(retrieval.get("candidates", [])),
    }


def _extract_plan_info(run_logs: list | None, fallback: dict | None = None) -> dict:
    plan = dict(fallback or {})
    logs = run_logs or []
    if logs:
        latest = logs[-1] or {}
        if "plan_finalized" in latest:
            plan = {
                "plan_finalized": latest.get("plan_finalized", False),
                "planned_steps": latest.get("planned_steps", {}),
                "required_step_ids": latest.get("required_step_ids", []),
            }

    return {
        "plan_finalized": bool(plan.get("plan_finalized", False)),
        "planned_steps": dict(plan.get("planned_steps", {})),
        "required_step_ids": list(plan.get("required_step_ids", [])),
    }


def _sorted_step_items(step_registry: dict) -> list[tuple]:
    return sorted(
        step_registry.items(),
        key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]),
    )

import json
def _is_pre_plan_guard_error(error) -> bool:
    """True if this error is a pre-plan guard block, not a real step-level failure."""
    try:
        parsed = json.loads(error) if isinstance(error, str) else (error or {})
        reason = str(parsed.get("reason", ""))
    except Exception:
        reason = str(error or "")
    return "Planning is required" in reason


def _failed_step_summary(run_logs: list | None) -> dict | None:
    active, _ = _step_issue_summaries(run_logs)
    return active[-1] if active else None


def _step_issue_summaries(run_logs: list | None) -> tuple[list[dict], list[dict]]:
    step_registry, sql_repairs_by_step, step_retries_by_step = _collect_step_stats(run_logs)
    issues_by_step: dict = {}

    for run in (run_logs or []):
        for entry in (run.get("iterations", []) or []):
            failed_step = entry.get("failed_step")
            iteration_error = entry.get("iteration_error")
            if failed_step is None or not iteration_error:
                continue
            if _is_pre_plan_guard_error(iteration_error):
                continue

            step_key = str(failed_step)
            meta = dict(step_registry.get(step_key, {}))
            issues_by_step.setdefault(step_key, []).append(
                {
                    "step": failed_step,
                    "action": meta.get("action"),
                    "status": meta.get("status"),
                    "database": meta.get("database"),
                    "sql": meta.get("sql"),
                    "error": iteration_error,
                    "sql_repairs": sql_repairs_by_step.get(step_key, 0),
                    "step_retries": step_retries_by_step.get(step_key, 0),
                    "recovered": bool(meta.get("executed")) and meta.get("status") == "ok",
                }
            )

    active, recovered = [], []
    for step_no in sorted(issues_by_step):
        latest = issues_by_step[step_no][-1]
        if latest.get("recovered"):
            recovered.append(latest)
        else:
            active.append(latest)

    return active, recovered


def _format_error_summary(run_logs: list | None) -> str:
    """Return compact error summary: '3 errors (step 1: 1, step 3: 2)' or '-'."""
    step_registry, sql_repairs_by_step, step_retries_by_step = _collect_step_stats(run_logs)
    errors_by_step: dict[str, int] = {}

    for run in (run_logs or []):
        for entry in (run.get("iterations", []) or []):
            failed_step = entry.get("failed_step")
            if failed_step is not None and entry.get("iteration_error"):
                key = str(failed_step)
                errors_by_step[key] = errors_by_step.get(key, 0) + 1

    # Also count steps that ended in a failed state (not executed)
    for step_no, meta in step_registry.items():
        key = str(step_no)
        if not meta.get("executed") and meta.get("status") not in (None, "ok"):
            if key not in errors_by_step:
                errors_by_step[key] = 1

    if not errors_by_step:
        return "-"

    total = sum(errors_by_step.values())
    parts = [
        f"step {k}: {v}"
        for k, v in sorted(errors_by_step.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    ]
    return f"{total} error{'s' if total != 1 else ''} ({', '.join(parts)})"


def _combine_active_errors(step_issues: list[dict], all_errors: list[str] | None) -> list[str]:
    step_errors = [issue.get("error") for issue in step_issues if issue.get("error")]
    other_errors = [err for err in (all_errors or []) if err]
    return _dedup(step_errors + other_errors)

def _step_status_errors(run_logs: list | None) -> list[str]:
    issues: list[str] = []

    for step in _build_step_outcomes(run_logs):
        if step.get("executed", False) and step.get("status") == "ok":
            continue
        issues.append(
            f"Step {step.get('step')} ({step.get('action')}) "
            f"validated={step.get('validated')} executed={step.get('executed')} "
            f"rows_returned={step.get('rows_returned')} status={step.get('status')}"
        )

    return issues


def _build_terminal_context(run_logs: list | None, all_errors: list[str] | None) -> dict | None:
    runs = run_logs or []
    if not runs:
        return None

    last_run = runs[-1]
    iterations = list(last_run.get("iterations") or [])
    if not iterations and not all_errors:
        return None

    recent_iterations: list[dict] = []
    for entry in iterations[-3:]:
        tools = list(entry.get("tools") or [])
        last_tool = tools[-1] if tools else None
        recent_iterations.append(
            {
                "iteration": entry.get("iteration"),
                "ai_content": entry.get("ai_content"),
                "tool_calls_declared": len(entry.get("tool_calls") or []),
                "tool_names": [tc.get("name") for tc in (entry.get("tool_calls") or []) if tc.get("name")],
                "iteration_error": entry.get("iteration_error"),
                "failed_step": entry.get("failed_step"),
                "had_error": entry.get("had_error", False),
                "had_validation_error": entry.get("had_validation_error", False),
                "had_exec_success": entry.get("had_exec_success", False),
                "last_tool": (
                    {
                        "tool": last_tool.get("tool"),
                        "args": last_tool.get("args"),
                        "result": last_tool.get("result"),
                    }
                    if last_tool else None
                ),
            }
        )

    last_iteration = iterations[-1] if iterations else {}
    last_tools = list(last_iteration.get("tools") or [])
    last_tool = last_tools[-1] if last_tools else None

    return {
        "has_errors": bool(all_errors),
        "terminal_error": (all_errors or [None])[-1],
        "last_completed_iteration": last_iteration.get("iteration"),
        "last_ai_content": last_iteration.get("ai_content"),
        "last_tool_call_count": len(last_iteration.get("tool_calls") or []),
        "last_tool": (
            {
                "tool": last_tool.get("tool"),
                "args": last_tool.get("args"),
                "result": last_tool.get("result"),
            }
            if last_tool else None
        ),
        "recent_iterations": recent_iterations,
    }


def _active_errors_for_result(result: PipelineResult) -> list[str]:
    active_step_issues, _ = _step_issue_summaries(result.run_logs)
    status_errors = _step_status_errors(result.run_logs)
    active_errors = _dedup(
        _combine_active_errors(active_step_issues, result.all_errors) + status_errors
    )
    if active_errors:
        return active_errors
    if not result.task_completed:
        return _dedup(result.all_errors)
    return []


def _summary_metrics(run_logs: list | None, trace: AgentTrace | None = None) -> dict:
    run_logs = run_logs or []
    step_registry, sql_repairs_by_step, step_retries_by_step = _collect_step_stats(run_logs)

    total_step_errors = 0
    blocked_calls = 0
    pre_plan_guard_blocks = 0
    discovery_calls = 0

    discovery_tools = {
        "tool_get_available_databases",
        "tool_get_database_catalog",
        "tool_get_available_tables",
        "tool_get_available_schemas",
        "tool_get_table_schema",
    }

    for run in run_logs:
        for entry in run.get("iterations", []) or []:
            if entry.get("iteration_error"):
                total_step_errors += 1

                if _is_pre_plan_guard_error(entry.get("iteration_error")):
                    pre_plan_guard_blocks += 1

            for t in entry.get("tools", []) or []:
                tool_name = t.get("tool")
                if tool_name in discovery_tools:
                    discovery_calls += 1

                parsed = _parse_tool_result(t.get("result"))
                if isinstance(parsed, dict) and str(parsed.get("error", "")).endswith("blocked."):
                    blocked_calls += 1

    seen_query_keys = set()
    redundant_steps = 0
    dbs_used = set()

    for _, meta in _sorted_step_items(step_registry):
        if meta.get("action") == "query" and meta.get("executed"):
            db = (meta.get("database") or "").strip()
            sql = " ".join((meta.get("sql") or "").split())
            if db:
                dbs_used.add(db)
            key = (db, sql)
            if key in seen_query_keys:
                redundant_steps += 1
            else:
                seen_query_keys.add(key)

    sql_repairs_by_step_clean = {
        str(k): v for k, v in sorted(
            sql_repairs_by_step.items(),
            key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]),
        ) if v > 0
    }
    step_retries_by_step_clean = {str(k): v for k, v in step_retries_by_step.items() if v > 0}

    if trace is not None and trace.databases_in_result:
        db_count_used = len(trace.databases_in_result)
    else:
        db_count_used = len(dbs_used)

    replan_count = max((run.get("replan_count", 0) for run in run_logs), default=0)

    replan_events: list[dict] = []
    for run in run_logs:
        trigger = run.get("replan_trigger")
        if trigger:
            replan_events.append({
                "trigger": trigger.get("reason") if isinstance(trigger, dict) else trigger,
                "step": run.get("replan_trigger_step"),
                "action": trigger.get("action") if isinstance(trigger, dict) else None,
                "blocked_count": trigger.get("blocked_count") if isinstance(trigger, dict) else None,
                "detail": trigger.get("error") if isinstance(trigger, dict) else None,
                "replan_index": run.get("replan_count", 0),
            })

    # Aggregate guard block counters across all runs
    tool_shape_errors: dict = {}
    unplanned_blocks: dict = {}
    locked_blocks: dict = {}
    missing_source: dict = {}
    for run in run_logs:
        for k, v in (run.get("tool_shape_error_counts") or {}).items():
            tool_shape_errors[k] = tool_shape_errors.get(k, 0) + v
        for k, v in (run.get("unplanned_step_block_counts") or {}).items():
            unplanned_blocks[k] = unplanned_blocks.get(k, 0) + v
        for k, v in (run.get("locked_step_block_counts") or {}).items():
            locked_blocks[k] = locked_blocks.get(k, 0) + v
        for k, v in (run.get("missing_source_step_counts") or {}).items():
            missing_source[k] = missing_source.get(k, 0) + v

    return {
        "guard_stats": {
            "sql_repairs": {
                "total": sum(sql_repairs_by_step_clean.values()),
                "by_step": sql_repairs_by_step_clean,
            },
            "step_retries": {
                "total": sum(step_retries_by_step_clean.values()),
                "by_step": step_retries_by_step_clean,
            },
            "replans": replan_count,
            "blocked_calls": blocked_calls,
            "pre_plan_guard_blocks": pre_plan_guard_blocks,
            "replan_events": replan_events,
            "tool_shape_errors": tool_shape_errors,
            "unplanned_step_blocks": unplanned_blocks,
            "locked_step_blocks": locked_blocks,
            "missing_source_step": missing_source,
        },
        "total_step_errors": total_step_errors,
        "discovery_calls": discovery_calls,
        "redundant_steps": redundant_steps,
        "db_count_used": db_count_used,
    }


def _extract_agent_trace_from_run_logs(
    run_logs: list | None,
    valid_sql: bool,
    executed: bool,
) -> AgentTrace:
    tables, databases = [], []
    total_tool_calls = 0
    tool_call_counts: dict[str, int] = {}
    executions: list[dict] = []
    merged_step_registry: dict = {}
    step_to_iteration: dict[str, int | None] = {}

    if not run_logs:
        return AgentTrace(
            tables_explored=[],
            tables_in_result=[],
            databases_explored=[],
            databases_in_result=[],
            validated_before_exec=valid_sql,
            executed_successfully=executed,
            total_tool_calls=0,
            executions=[],
            tool_call_counts={},
        )

    for run in run_logs:
        for db in run.get("discovered_databases", []):
            if db not in databases:
                databases.append(db)

        for tbl in run.get("discovered_tables", []):
            if tbl not in tables:
                tables.append(tbl)

        for step_no, meta in (run.get("step_registry") or {}).items():
            merged_step_registry[step_no] = meta

        for entry in (run.get("iterations") or []):
            iter_no = entry.get("iteration")
            for t in entry.get("tools", []):
                total_tool_calls += 1
                tool_name = t.get("tool")
                if tool_name:
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            for evt in (entry.get("step_events") or []):
                if evt.get("event_type") == "execute" and evt.get("step") is not None:
                    step_to_iteration[str(evt.get("step"))] = iter_no

    for step_no, meta in _sorted_step_items(merged_step_registry):
        executions.append(
            {
                "iteration": step_to_iteration.get(str(step_no)),
                "step": step_no,
                "action": meta.get("action"),
                "database": meta.get("database"),
                "sql": meta.get("sql"),
                "success": meta.get("executed", False),
                "error": None if meta.get("executed", False) else meta.get("status"),
                "rows_returned": meta.get("rows_returned"),
                "status": meta.get("status"),
            }
        )

    successful = [e for e in executions if e["success"]]
    all_sql_text = " ".join((e.get("sql") or "").lower() for e in successful)

    tables_in_result = [t for t in tables if t.lower() in all_sql_text]
    databases_in_result = list(
        dict.fromkeys(e["database"] for e in successful if e.get("database"))
    )

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
        validated = valid_sql
        executed_successfully = executed

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


def extract_agent_trace(result: PipelineResult) -> AgentTrace:
    return _extract_agent_trace_from_run_logs(
        result.run_logs,
        result.valid_sql,
        result.executed,
    )


DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"

RESET = "\033[0m"

BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"


def _print_summary(query: str, results: list):
    logger.info("\n" + "#" * 130)
    logger.info("QUERY")
    logger.info("  %s", query)
    logger.info("#" * 130)

    logger.info(
        "%-20s %-8s %-20s %-8s %-6s %-6s %-6s %-8s %-10s %-8s %-10s %s",
        "MODEL",
        "SQL_OK",
        "STEPS_EXEC",
        "TASK_OK",
        "ITER",
        "REP",
        "RETRY",
        "REPLAN",
        "LATENCY",
        "TOOLS",
        "TOKENS",
        "ERROR",
    )
    logger.info("-" * 140)

    for r in results:
        if isinstance(r, Exception):
            logger.info(
                "%-20s %-8s %-20s %-8s %-6s %-6s %-6s %-8s %-10s %-8s %-10s %s",
                "ERROR", "F", "-", "F",
                "-", "-", "-", "-", "-", "-", "-",
                _short(str(r), 300),
            )
            continue

        metrics = _summary_metrics(r.run_logs or [])
        g = metrics["guard_stats"]
        trace = extract_agent_trace(r)
        err = _format_error_summary(r.run_logs or [])
        total_tokens = r.tokens.total_tokens if r.tokens else "-"
        steps_exec = _format_step_execution(_step_execution_map(r.run_logs or []))

        logger.info(
            "%-20s %-8s %-20s %-8s %-6s %-6s %-6s %-8s %-10s %-8s %-10s %s",
            r.model_id,
            "T" if r.valid_sql else "F",
            steps_exec,
            "T" if r.task_completed else "F",
            r.iterations,
            g["sql_repairs"]["total"],
            g["step_retries"]["total"],
            g["replans"],
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

        metrics = _summary_metrics(r.run_logs or [])
        g = metrics["guard_stats"]
        trace = extract_agent_trace(r)
        retrieval = {}
        if r.run_logs:
            retrieval = dict((r.run_logs[-1] or {}).get("retrieval", {}))
        logger.info("[%s]", r.model_id)
        logger.info("  valid_sql          : %s", r.valid_sql)
        logger.info("  executed           : %s", r.executed)
        logger.info("  task_completed     : %s", r.task_completed)
        logger.info("  iterations         : %s", r.iterations)
        logger.info("  latency_s          : %.3f", r.latency_s)
        logger.info("  sql_repairs        : %s  %s", g["sql_repairs"]["total"], g["sql_repairs"]["by_step"] or "")
        logger.info("  step_retries       : %s  %s", g["step_retries"]["total"], g["step_retries"]["by_step"] or "")
        logger.info("  replans            : %s", g["replans"])
        logger.info("  blocked_calls      : %s", g["blocked_calls"])
        if g.get("pre_plan_guard_blocks"):
            logger.info("  pre_plan_blocks    : %s", g["pre_plan_guard_blocks"])
        if g.get("tool_shape_errors"):
            logger.info("  shape_errors       : %s", g["tool_shape_errors"])
        if g.get("unplanned_step_blocks"):
            logger.info("  unplanned_blocks   : %s", g["unplanned_step_blocks"])
        if g.get("locked_step_blocks"):
            logger.info("  locked_blocks      : %s", g["locked_step_blocks"])
        if g.get("missing_source_step"):
            logger.info("  missing_source     : %s", g["missing_source_step"])
        for evt in g.get("replan_events", []):
            logger.info(
                "  replan_trigger     : %s  step=%s  detail=%s",
                evt.get("trigger"),
                evt.get("step"),
                _short(evt.get("detail") or "", 80),
            )
        if retrieval:
            logger.info("  retrieval_score    : %s  scoped=%s", retrieval.get("top_score"), retrieval.get("scope_active"))

        step_registry, sql_repairs_by_step, step_retries_by_step = _collect_step_stats(r.run_logs)
        failed_summary = _failed_step_summary(r.run_logs)
        active_step_issues, recovered_step_issues = _step_issue_summaries(r.run_logs)

        if step_registry:
            logger.info(f"\n{GREEN}  STEP REGISTRY:{RESET}")
            for step_no, meta in _sorted_step_items(step_registry):
                logger.info("    Step %s:", step_no)
                logger.info("      action          : %s", meta.get("action"))
                logger.info("      database        : %s", meta.get("database"))
                logger.info("      validated       : %s", meta.get("validated"))
                logger.info("      executed        : %s", meta.get("executed"))
                logger.info("      sql_repairs     : %s", sql_repairs_by_step.get(step_no, 0))
                logger.info("      step_retries    : %s", step_retries_by_step.get(step_no, 0))
                logger.info("      rows_returned   : %s", meta.get("rows_returned"))
                logger.info("      status          : %s", meta.get("status"))
                logger.info("      sql             : %s", meta.get("sql"))

        if r.tokens:
            logger.info(
                "  tokens             : in=%s out=%s total=%s",
                r.tokens.prompt_tokens,
                r.tokens.completion_tokens,
                r.tokens.total_tokens,
            )

        active_errors = [issue.get("error") for issue in active_step_issues if issue.get("error")]
        if not active_errors and not r.task_completed:
            active_errors = _dedup(r.all_errors)

        if active_errors:
            logger.info(f"\n {BLUE} ERRORS:{RESET}")
            for err in _dedup(active_errors):
                logger.info("    - %s", err)

        logger.info(f"\n{YELLOW}  TRACE:{RESET}")
        logger.info("    databases explored : %s", trace.databases_explored)
        logger.info("    tables explored    : %s", trace.tables_explored)
        logger.info("    validated          : %s", trace.validated_before_exec)
        logger.info("    executed ok        : %s", trace.executed_successfully)
        logger.info("    total tool calls   : %s", trace.total_tool_calls)
        logger.info("    tool call counts   : %s", trace.tool_call_counts)

        if failed_summary and failed_summary.get("status") != "ok":
            logger.info(f"\n{RED}  FAILED STEP SUMMARY:{RESET}")
            logger.info("    step              : %s", failed_summary.get("step"))
            logger.info("    action            : %s", failed_summary.get("action"))
            logger.info("    database          : %s", failed_summary.get("database"))
            logger.info("    status            : %s", failed_summary.get("status"))
            logger.info("    sql_repairs       : %s", failed_summary.get("sql_repairs"))
            logger.info("    step_retries      : %s", failed_summary.get("step_retries"))
            logger.info("    error             : %s", failed_summary.get("error"))
            logger.info("    sql               : %s", failed_summary.get("sql"))

        if recovered_step_issues:
            logger.info(f"\n{YELLOW}  RECOVERED STEPS:{RESET}")
            for item in recovered_step_issues:
                logger.info("    step              : %s", item.get("step"))
                logger.info("    action            : %s", item.get("action"))
                logger.info("    database          : %s", item.get("database"))
                logger.info("    status            : %s", item.get("status"))
                logger.info("    sql_repairs       : %s", item.get("sql_repairs"))
                logger.info("    step_retries      : %s", item.get("step_retries"))
                logger.info("    recovered_from    : %s", item.get("error"))
                logger.info("    sql               : %s", item.get("sql"))

        if trace.executions:
            logger.info(f"\n{GREEN}  EXECUTIONS:{RESET}")
            for i, ex in enumerate(trace.executions):
                logger.info("    Execution %s:", i)
                logger.info("      step            : %s", ex["step"])
                logger.info("      action          : %s", ex["action"])
                logger.info("      success         : %s", ex["success"])
                logger.info("      database        : %s", ex["database"])
                logger.info("      rows_returned   : %s", ex["rows_returned"])
                logger.info("      sql             : %s", ex["sql"])
                if not ex["success"]:
                    logger.info("      error           : %s", ex["error"])

        if r.run_logs:
            logger.info(f"\n{RED}  RUN LOGS:{RESET}")
            for run in r.run_logs:
                logger.info(f"  {CYAN}Run {run.get('run_index', 0)}:{RESET}")
                logger.info("      attempt         : %s", run.get("attempt_number", 0))
                logger.info("      sql_repairs     : %s", run.get("sql_repairs_used", 0))
                logger.info("      final_sql       : %s", _short(str(run.get("final_sql") or "(none)"), 300))

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
                if run.get("guard_error"):
                    logger.info("      guard_error     : %s", run["guard_error"])
                if run.get("execution_error"):
                    logger.info("      execution_error : %s", run["execution_error"])

                for entry in run.get("iterations", []):
                    tok = entry.get("tokens", {})
                    logger.info(f"{BLUE}      Iteration %s:{RESET}", entry["iteration"])
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
                    logger.info("        had_error           : %s", entry.get("had_error", False))
                    logger.info("        failed_step         : %s", entry.get("failed_step"))
                    if entry.get("iteration_error"):
                        logger.info("        iteration_error     : %s", entry.get("iteration_error"))

                    step_events = entry.get("step_events", [])
                    for t in entry.get("tools", []):
                        matched_event = None
                        for event in step_events:
                            event_tool = (
                                "tool_validate_sql"
                                if event.get("event_type") == "validate"
                                else f"tool_execute_{event.get('action')}_step"
                            )
                            if event_tool == t["tool"]:
                                matched_event = event
                                break

                        logger.info("          TOOL   : %s", t.get("tool"))
                        logger.info("          ARGS   : %s", t.get("args"))
                        logger.info("          RESULT : %s", t.get("result"))

                        args_payload = t.get("args") if isinstance(t.get("args"), dict) else {}
                        step_payload = (
                            args_payload.get("step", {})
                            if isinstance(args_payload.get("step"), dict)
                            else {}
                        )
                        raw_result = t.get("result")
                        result_payload = raw_result if isinstance(raw_result, dict) else {}

                        logger.info(
                            "          STEP   : %s",
                            (matched_event or {}).get(
                                "step",
                                result_payload.get("step", step_payload.get("step")),
                            ),
                        )
                        logger.info(
                            "          TYPE   : %s",
                            (matched_event or {}).get(
                                "event_type",
                                "validate" if t.get("tool") == "tool_validate_sql" else "execute",
                            ),
                        )
                        logger.info(
                            "          ACTION : %s",
                            (matched_event or {}).get(
                                "action",
                                result_payload.get("action", step_payload.get("action")),
                            ),
                        )
                        logger.info(
                            "          STATUS : %s",
                            (matched_event or {}).get("status", result_payload.get("status")),
                        )
                        logger.info(
                            "          VALID  : %s",
                            (matched_event or {}).get(
                                "validated",
                                result_payload.get("validated", step_payload.get("validated", False)),
                            ),
                        )
                        logger.info(
                            "          EXEC   : %s",
                            (matched_event or {}).get("executed", result_payload.get("executed", False)),
                        )
                        logger.info(
                            "          REPAIR : %s",
                            (matched_event or {}).get("sql_repairs", result_payload.get("sql_repairs", 0)),
                        )
                        logger.info(
                            "          RETRY  : %s",
                            (matched_event or {}).get("step_retries", result_payload.get("step_retries", 0)),
                        )

        logger.info(f"\n{BRIGHT_RED}  FINAL:{RESET}")
        logger.info("    answer            : %s", r.final_answer or "(none)")
        if not r.final_answer:
            logger.info("    answer_missing    : True")
        logger.info("    sql               : %s", r.final_sql or "(none)")
        if r.result_data is not None:
            rc = _extract_result_contract_info(r.result_data)
            ms = rc.get("model_summary_result") or {}
            logger.info(
                "    result            : rows=%s  columns=%s  truncated=%s",
                rc.get("full_result_row_count"),
                ms.get("columns"),
                ms.get("result_truncated_for_model"),
            )
        else:
            logger.info("    result            : (none)")

    logger.info("\n" + "=" * 110)


async def store_run(run: EvalRun, results: list) -> str:
    q_hash = query_hash(run.query)
    run_id = uuid.uuid4().hex


    base = f"{STORAGE_VERSION}/{q_hash}/{run.prompt_name}/runs/{run_id}"

    loop = asyncio.get_running_loop()

    async def _upload(path: str, data) -> None:
        try:
            await loop.run_in_executor(None, upload, path, data)
        except Exception as e:
            logger.warning("upload failed [%s]: %s", path, e)

    try:
        if not query_meta_exists(q_hash):
            await _upload(
                f"{STORAGE_VERSION}/{q_hash}/query.json",
                {"query_id": run.query_id, "query": run.query, "query_hash": q_hash},
            )
    except Exception as e:
        logger.warning("Failed to upload query meta: %s", e)

    if not prompt_meta_exists(q_hash, run.prompt_name):
        await _upload(
            f"{STORAGE_VERSION}/{q_hash}/{run.prompt_name}/prompt.txt",
            load_prompt(run.prompt_name),
        )

    summary = []
    normalized_docs = []

    for r in results:
        if isinstance(r, AgentRunError):
            snap = r.snapshot or {}
            tokens = snap.get("tokens", _empty_token_dict())
            retrieval = _extract_retrieval_info(
                snap.get("run_logs", []),
                fallback=snap.get("retrieval"),
            )
            plan = _extract_plan_info(
                snap.get("run_logs", []),
                fallback={
                    "plan_finalized": snap.get("plan_finalized", False),
                    "planned_steps": snap.get("planned_steps", {}),
                    "required_step_ids": snap.get("required_step_ids", []),
                },
            )
            trace = _extract_agent_trace_from_run_logs(
                snap.get("run_logs", []),
                snap.get("valid_sql", False),
                snap.get("executed", False),
            )
            failed_summary = _failed_step_summary(snap.get("run_logs", []))
            active_step_issues, recovered_step_issues = _step_issue_summaries(snap.get("run_logs", []))
            step_outcomes = _build_step_outcomes(snap.get("run_logs", []))
            valid_sql = _valid_sql_from_steps(
                snap.get("run_logs", []),
                valid_sql_fallback=snap.get("valid_sql", trace.validated_before_exec),
            )
            executed = _executed_from_steps(
                snap.get("run_logs", []),
                executed_fallback=snap.get("executed", trace.executed_successfully),
            )
            task_completed = _task_completed_from_steps(
                snap.get("run_logs", []),
                task_completed_fallback=snap.get("task_completed", False),
                final_answer=_fallback_answer_from_logs(snap.get("run_logs", [])),
                required_step_ids=set(plan["required_step_ids"]),
                plan_finalized=plan["plan_finalized"],
                planned_steps=plan["planned_steps"],
            )
            active_errors = _dedup(
                _combine_active_errors(active_step_issues, snap.get("all_errors", []))
                + _step_status_errors(snap.get("run_logs", []))
            )
            terminal_context = _build_terminal_context(
                snap.get("run_logs", []),
                active_errors,
            )
            metrics = _summary_metrics(snap.get("run_logs", []), trace)
            result_contract = _extract_result_contract_info(None)

            doc = {
                "run_id": run_id,
                "query": run.query,
                "model_id": snap.get("model_id"),
                "status": "error",
                "error": str(r),
                "valid_sql": valid_sql,
                "executed": executed,
                "task_completed": task_completed,
                "model_task_completed": snap.get("task_completed", False),
                "iterations": snap.get("iterations", 0),
                "total_tool_calls": trace.total_tool_calls,
                "latency_s": None,
                "tokens": tokens,
                "retrieval": retrieval,
                "plan_finalized": plan["plan_finalized"],
                "planned_steps": plan["planned_steps"],
                "required_step_ids": plan["required_step_ids"],
                "result": result_contract,
                "guard_stats": metrics["guard_stats"],
                "process": {
                    "total_step_errors": metrics["total_step_errors"],
                    "discovery_calls": metrics["discovery_calls"],
                    "redundant_steps": metrics["redundant_steps"],
                    "db_count_used": metrics["db_count_used"],
                },
                "errors": active_errors,
                "trace": asdict(trace),
                "run_logs": snap.get("run_logs", []),
                "failed_step_summary": failed_summary,
                "active_step_issues": active_step_issues,
                "recovered_step_issues": recovered_step_issues,
                "step_outcomes": step_outcomes,
                "terminal_context": terminal_context,
                "final": {
                    "answer": None,
                    "sql": snap.get("final_sql"),
                    "result_data": None,
                },
            }

            summary.append(
                {
                    "model_id": doc["model_id"],
                    "status": "error",
                    "valid_sql": valid_sql,
                    "executed": executed,
                    "steps_executed": _step_execution_map(snap.get("run_logs", [])),
                    "task_completed": task_completed,
                    "iterations": doc["iterations"],
                    "total_tool_calls": trace.total_tool_calls,
                    "latency_s": None,
                    "total_tokens": tokens.get("total_tokens"),
                    "retrieval_scope_active": retrieval["scope_active"],
                    "retrieval_top_score": retrieval["top_score"],
                    "full_result_row_count": result_contract["full_result_row_count"],
                    "sql_repairs": metrics["guard_stats"]["sql_repairs"]["total"],
                    "step_retries": metrics["guard_stats"]["step_retries"]["total"],
                    "replans": metrics["guard_stats"]["replans"],
                    "blocked_calls": metrics["guard_stats"]["blocked_calls"],
                    "total_step_errors": metrics["total_step_errors"],
                    "redundant_steps": metrics["redundant_steps"],
                    "error": _format_error_summary(snap.get("run_logs", [])),
                }
            )
        else:
            trace = extract_agent_trace(r)
            tokens = asdict(r.tokens) if r.tokens else _empty_token_dict()
            retrieval = _extract_retrieval_info(r.run_logs or [])
            plan = _extract_plan_info(r.run_logs or [])
            failed_summary = _failed_step_summary(r.run_logs or [])
            active_step_issues, recovered_step_issues = _step_issue_summaries(r.run_logs or [])
            metrics = _summary_metrics(r.run_logs or [], trace)
            active_errors = _active_errors_for_result(r)
            step_outcomes = _build_step_outcomes(r.run_logs or [])
            valid_sql = _valid_sql_from_steps(
                r.run_logs or [],
                valid_sql_fallback=r.valid_sql,
            )
            executed = _executed_from_steps(
                r.run_logs or [],
                executed_fallback=r.executed,
            )
            task_completed = _task_completed_from_steps(
                r.run_logs or [],
                task_completed_fallback=r.task_completed,
                final_answer=(r.final_answer or _fallback_answer_from_logs(r.run_logs or [])),
                required_step_ids=set(plan["required_step_ids"]),
                plan_finalized=plan["plan_finalized"],
                planned_steps=plan["planned_steps"],
            )
            status = "completed" if task_completed else "incomplete"
            terminal_context = _build_terminal_context(r.run_logs or [], active_errors)
            result_contract = _extract_result_contract_info(r.result_data)

            doc = {
                "run_id": run_id,
                "query": run.query,
                "model_id": r.model_id,
                "status": status,
                "valid_sql": valid_sql,
                "executed": executed,
                "task_completed": task_completed,
                "model_task_completed": r.task_completed,
                "iterations": r.iterations,
                "total_tool_calls": trace.total_tool_calls,
                "latency_s": r.latency_s,
                "tokens": tokens,
                "retrieval": retrieval,
                "plan_finalized": plan["plan_finalized"],
                "planned_steps": plan["planned_steps"],
                "required_step_ids": plan["required_step_ids"],
                "result": result_contract,
                "guard_stats": metrics["guard_stats"],
                "process": {
                    "total_step_errors": metrics["total_step_errors"],
                    "discovery_calls": metrics["discovery_calls"],
                    "redundant_steps": metrics["redundant_steps"],
                    "db_count_used": metrics["db_count_used"],
                },
                "errors": active_errors,
                "trace": asdict(trace),
                "failed_step_summary": failed_summary,
                "active_step_issues": active_step_issues,
                "recovered_step_issues": recovered_step_issues,
                "step_outcomes": step_outcomes,
                "terminal_context": terminal_context,
                "final": {
                    "answer": r.final_answer,
                    "answer_missing": not bool(r.final_answer),
                    "sql": r.final_sql,
                    "result_data": result_contract,
                },
                "run_logs": r.run_logs or [],
            }

            summary.append(
                {
                    "model_id": r.model_id,
                    "status": status,
                    "valid_sql": valid_sql,
                    "executed": executed,
                    "steps_executed": _step_execution_map(r.run_logs or []),
                    "task_completed": task_completed,
                    "iterations": r.iterations,
                    "total_tool_calls": trace.total_tool_calls,
                    "latency_s": r.latency_s,
                    "total_tokens": tokens.get("total_tokens"),
                    "retrieval_scope_active": retrieval["scope_active"],
                    "retrieval_top_score": retrieval["top_score"],
                    "full_result_row_count": result_contract["full_result_row_count"],
                    "sql_repairs": metrics["guard_stats"]["sql_repairs"]["total"],
                    "step_retries": metrics["guard_stats"]["step_retries"]["total"],
                    "replans": metrics["guard_stats"]["replans"],
                    "blocked_calls": metrics["guard_stats"]["blocked_calls"],
                    "total_step_errors": metrics["total_step_errors"],
                    "redundant_steps": metrics["redundant_steps"],
                    "error": _format_error_summary(r.run_logs or []),
                }
            )

        normalized_docs.append(doc)

    await _upload(
        f"{base}/run_summary.json",
        {
            "run_id": run_id,
            "query_id": run.query_id,
            "query": run.query,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt_name": run.prompt_name,
            "params": asdict(run.params),
            "model_ids": run.model_ids,
            "summary": summary,
        },
    )

    embedding_active = any(s.get("retrieval_scope_active") for s in summary)

    try:
        _update_queries_index(q_hash, run.query, summary, run.prompt_name, run.query_id)
    except Exception as e:
        logger.warning("Failed to update queries index: %s", e)

    try:
        _update_runs_index(q_hash, run_id, summary, run.prompt_name, embedding_active)
    except Exception as e:
        logger.warning("Failed to update runs index: %s", e)

    seen_ids: set[str] = set()
    for doc in normalized_docs:
        model_id = doc["model_id"] or "unknown_model"
        key = model_id
        while key in seen_ids:
            key = f"{model_id}_{uuid.uuid4().hex[:6]}"
        seen_ids.add(key)

        run_logs = doc.pop("run_logs", [])
        result_doc = doc

        await _upload(f"{base}/{key}/result.json", result_doc)
        await _upload(f"{base}/{key}/logs.json", run_logs)

    logger.info("SAVED: %s/", base)
    return f"{q_hash}/{run_id}"
