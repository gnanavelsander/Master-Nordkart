import json
import logging
from typing import Any

from langchain_core.messages import ToolMessage

import logging
logger = logging.getLogger(__name__)






from agent_models import AgentState
from agent_helpers import (
    _coerce_args,
    _dedup,
    _extract_databases_from_result,
    _extract_from_database_catalog,
    _extract_tables_from_available_tables_result,
    _increment_sql_repair,
    _increment_step_retry,
    _parse_tool_result,
    _validated_sql_key,
    _build_snapshot,
)
from backend.tools.langchain_tools import (
    kommune_where_clause,
    tool_get_available_databases,
    tool_get_table_schema,
    tool_get_database_catalog,
    tool_validate_sql,
    tool_execute_query_step,
    tool_execute_filter_step,
    tool_execute_select_columns_step,
    tool_execute_spatial_join_step,
    tool_execute_attribute_join_step,
    tool_execute_buffer_step,
    tool_execute_nearest_step,
    tool_execute_distance_step,
    tool_execute_aggregate_step,
    tool_execute_sort_step,
    tool_execute_limit_step,
    tool_execute_clip_step,
    tool_execute_union_step,
    tool_execute_intersection_step,
    tool_execute_difference_step,
    tool_execute_dissolve_step,
    tool_execute_centroid_step,
    tool_execute_transform_crs_step,
)












STEP_TOOLS: frozenset[str] = frozenset({
    "tool_execute_query_step",
    "tool_execute_filter_step",
    "tool_execute_select_columns_step",
    "tool_execute_spatial_join_step",
    "tool_execute_attribute_join_step",
    "tool_execute_buffer_step",
    "tool_execute_nearest_step",
    "tool_execute_distance_step",
    "tool_execute_aggregate_step",
    "tool_execute_sort_step",
    "tool_execute_limit_step",
    "tool_execute_clip_step",
    "tool_execute_union_step",
    "tool_execute_intersection_step",
    "tool_execute_difference_step",
    "tool_execute_dissolve_step",
    "tool_execute_centroid_step",
    "tool_execute_transform_crs_step",
})




STEP_DICT_FIELDS: frozenset[str] = frozenset({
    "step", "action", "database", "sql", "validSql",
    "source_step", "left_source", "right_source",
    "join_type", "left_geom", "right_geom",
    "distance_meters", "buffer_distance",
    "filters", "select_columns", "output_columns",
    "group_by", "aggregation_type", "metric_column", "metric_as",
    "order_by", "order_direction", "limit",
    "target_crs", "join_keys",
    "schema_name", "table_name",
})





# Actions that require source_step
SOURCE_STEP_ACTIONS: frozenset[str] = frozenset({
    "filter", "select_columns", "sort", "limit",
    "buffer", "centroid", "transform_crs", "dissolve", "union", "aggregate",
})

# Actions that require left_source + right_source
DUAL_SOURCE_ACTIONS: frozenset[str] = frozenset({
    "spatial_join", "intersection", "difference",
    "clip", "nearest", "distance", "attribute_join",
})









def _normalize_tool_args(name: str, raw_args: dict, run_id: str) -> dict:
    """
    Inject run_id and recover flat-arg mistakes:
    if the model sent step fields at the top level instead of nested under 'step',
    reconstruct the correct shape automatically.
    """
    args = dict(raw_args or {})

    if name not in STEP_TOOLS:
        return args

    args["run_id"] = run_id

    step_obj = args.get("step")
    if isinstance(step_obj, dict):
        return args  

    # Reconstruct nested step dict from flat fields
    normalized_step: dict = {}

    if step_obj is not None:

         # preserve scalar step number
        normalized_step["step"] = step_obj 

    for key in list(args.keys()):
        if key in STEP_DICT_FIELDS and key != "step":
            normalized_step[key] = args.pop(key)

    if normalized_step:
        args["step"] = normalized_step
        logger.debug(
            "      [normalize] flat args reconstructed into step dict: %s",
            json.dumps(normalized_step, default=str)[:200],
        )

    return args






def _guard_step_tool(
    name: str,
    raw_args: dict,
    args: dict,
    validated_sqls: dict,
) -> tuple[str | None, dict | None]:
    action_name = name.replace("tool_execute_", "").replace("_step", "")
    step_obj = args.get("step")
    run_id = args.get("run_id")

    def _flat_hint() -> str:
        flat_keys = STEP_DICT_FIELDS - {"step"}
        if any(k in raw_args for k in flat_keys):
            nested = {k: v for k, v in raw_args.items() if k != "run_id"}
            return (
                "You sent step fields as flat top-level arguments — "
                "they must be nested inside a single 'step' dict. "
                f"You sent: {json.dumps(raw_args, default=str)[:300]}. "
                f"Correct shape: {name}(run_id='<run_id>', "
                f"step={json.dumps(nested, default=str)[:200]})"
            )
        return (
            f"The 'step' argument must be a dict, not "
            f"{type(step_obj).__name__} = {repr(step_obj)[:100]}. "
            f"Correct shape: {name}(run_id='<run_id>', "
            f"step={{'step': <int>, 'action': '{action_name}', ...}})"
        )

    if not isinstance(step_obj, dict):
        return json.dumps({
            "executed": False,
            "error": f"{name} blocked.",
            "reason": _flat_hint(),
            "step": None,
            "action": action_name,
        }), None

    step_obj = dict(step_obj)
    step_number = step_obj.get("step")
    action = step_obj.get("action") or action_name
    sql = (step_obj.get("sql") or "").strip()
    database = (step_obj.get("database") or "").strip()
    blocked_reason: str | None = None

    def _norm(s: str) -> str:
        return " ".join((s or "").split())

    if name == "tool_execute_query_step":
        valid_sql_flag = step_obj.get("validSql")
        validated_entry = validated_sqls.get(_validated_sql_key(run_id, step_number))

        if action != "query":
            blocked_reason = "step.action must be 'query'."
        elif valid_sql_flag != "ok":
            blocked_reason = "step.validSql must be 'ok'."
        elif step_number is None:
            blocked_reason = "step.step is required."
        elif not database:
            blocked_reason = "step.database is required."
        elif not sql:
            blocked_reason = "step.sql is required."
        elif validated_entry is None:
            blocked_reason = (
                f"No validated SQL found for step {step_number}. "
                "Call tool_validate_sql first with this step_number, "
                "then call tool_execute_query_step with the identical SQL."
            )
        elif _norm(validated_entry.get("sql", "")) != _norm(sql):
            blocked_reason = (
                "SQL does not match the validated entry — execute the exact same SQL. "
                f"Validated SQL starts with: {validated_entry.get('sql', '')[:120]!r}"
            )
        elif validated_entry.get("database", "") != database:
            blocked_reason = (
                f"Database mismatch: validated for '{validated_entry.get('database')}' "
                f"but received '{database}'."
            )

    else:
        if step_number is None:
            blocked_reason = "step.step is required."
        elif action == "aggregate":
            if step_obj.get("source_step") is None:
                blocked_reason = "aggregate requires source_step."
            elif not step_obj.get("group_by") and not step_obj.get("aggregation_type"):
                blocked_reason = "aggregate requires group_by or aggregation_type."
        elif action in SOURCE_STEP_ACTIONS and step_obj.get("source_step") is None:
            blocked_reason = f"{action} requires source_step."
        elif action in DUAL_SOURCE_ACTIONS:
            if step_obj.get("left_source") is None or step_obj.get("right_source") is None:
                blocked_reason = f"{action} requires left_source and right_source."

    if blocked_reason:
        return json.dumps({
            "executed": False,
            "error": f"{name} blocked.",
            "reason": blocked_reason,
            "step": step_number,
            "action": action,
        }), None

    return None, step_obj





















async def _execute_tool(
    tool: Any,
    name: str,
    args: dict,
    blocked_json: str | None,
) -> tuple[str, Any, bool, int]:
    """
    Either return the pre-built blocked result or invoke the tool.

    Returns:
        (result_str, parsed_result, exec_ok, rows_returned)
    """
    if blocked_json is not None:
        parsed = _parse_tool_result(blocked_json)
        return blocked_json, parsed, False, 0

    result_str = await tool.ainvoke(args)
    parsed = _parse_tool_result(result_str)
    exec_ok = isinstance(parsed, dict) and parsed.get("executed", False)
    rows_returned = parsed.get("rows_returned", 0) if isinstance(parsed, dict) else 0
    return result_str, parsed, exec_ok, rows_returned














def _update_step_registry(
    state: "AgentState",
    name: str,
    step_obj: dict | None,
    exec_ok: bool,
    rows_returned: int,
) -> None:
    """Write step execution metadata into state['step_registry']."""
    if step_obj is None:
        return

    step_number = step_obj.get("step")
    if step_number is None:
        return

    action = step_obj.get("action") or name.replace("tool_execute_", "").replace("_step", "")
    is_query = name == "tool_execute_query_step"

    step_registry = dict(state.get("step_registry", {}))
    prev = dict(step_registry.get(step_number, {}))
    prev.update({
        "step": step_number,
        "action": action,
        "sql": (step_obj.get("sql") or "").strip() if is_query else prev.get("sql", ""),
        "database": (step_obj.get("database") or "").strip() if is_query else prev.get("database", ""),
        "validated": prev.get("validated", False),
        "executed": exec_ok,
        "rows_returned": rows_returned,
        "status": "ok" if exec_ok else "error",
    })
    step_registry[step_number] = prev
    state["step_registry"] = step_registry


























def _update_iteration_flags(
    state: "AgentState",
    name: str,
    step_obj: dict | None,
    parsed_result: Any,
    exec_ok: bool,
    iter_flags: dict,
) -> None:
    step_number = step_obj.get("step") if step_obj else None
    run_id = state.get("run_id")

    if exec_ok:
        iter_flags["iter_any_exec_ok"] = True
        state["last_failed_step"] = None

        if name == "tool_execute_query_step":
            validated_sqls = dict(state.get("validated_sqls", {}))
            key = _validated_sql_key(run_id, step_number)
            if key:
                validated_sqls.pop(key, None)
            state["validated_sqls"] = validated_sqls
    else:
        iter_flags["iter_had_error"] = True
        state["last_failed_step"] = step_number
        state["last_execute_error"] = json.dumps(parsed_result, ensure_ascii=False)

        is_blocked = isinstance(parsed_result, dict) and str(
            parsed_result.get("error", "")
        ).endswith("blocked.")

        if not is_blocked:
            if name == "tool_execute_query_step":
                _increment_sql_repair(state, step_number)
                state["sql_repair_count"] = state.get("sql_repair_count", 0) + 1
                state["total_sql_repairs"] = state.get("total_sql_repairs", 0) + 1
            else:
                _increment_step_retry(state, step_number)
                state["step_retry_count"] = state.get("step_retry_count", 0) + 1
                state["total_step_retries"] = state.get("total_step_retries", 0) + 1

        state["all_errors"] = _dedup(
            list(state.get("all_errors", [])) + [json.dumps(parsed_result, ensure_ascii=False)]
        )

    state["last_executed_result"] = parsed_result
    if exec_ok:
        state["last_execute_error"] = None
















def _update_iteration_flags(
    state: "AgentState",
    name: str,
    step_obj: dict | None,
    parsed_result: Any,
    exec_ok: bool,
    iter_flags: dict,
) -> None:
    step_number = step_obj.get("step") if step_obj else None
    run_id = state.get("run_id")

    if exec_ok:
        iter_flags["iter_any_exec_ok"] = True
        state["last_failed_step"] = None

        if name == "tool_execute_query_step":
            validated_sqls = dict(state.get("validated_sqls", {}))
            key = _validated_sql_key(run_id, step_number)
            if key:
                validated_sqls.pop(key, None)
            state["validated_sqls"] = validated_sqls
    else:
        iter_flags["iter_had_error"] = True
        state["last_failed_step"] = step_number
        state["last_execute_error"] = json.dumps(parsed_result, ensure_ascii=False)

        is_blocked = isinstance(parsed_result, dict) and str(
            parsed_result.get("error", "")
        ).endswith("blocked.")

        if not is_blocked:
            if name == "tool_execute_query_step":
                _increment_sql_repair(state, step_number)
                state["sql_repair_count"] = state.get("sql_repair_count", 0) + 1
                state["total_sql_repairs"] = state.get("total_sql_repairs", 0) + 1
            else:
                _increment_step_retry(state, step_number)
                state["step_retry_count"] = state.get("step_retry_count", 0) + 1
                state["total_step_retries"] = state.get("total_step_retries", 0) + 1

        state["all_errors"] = _dedup(
            list(state.get("all_errors", [])) + [json.dumps(parsed_result, ensure_ascii=False)]
        )

    state["last_executed_result"] = parsed_result
    if exec_ok:
        state["last_execute_error"] = None


















def _handle_validate_sql_tool(
    state: "AgentState",
    args: dict,
    result: str,
    iter_flags: dict,
) -> None:
    sql = (args.get("sql") or "").strip()
    step_number = args.get("step_number")
    database = (args.get("databaseName") or "").strip()
    run_id = state.get("run_id")

    parsed_result = _parse_tool_result(result)
    ok = isinstance(parsed_result, dict) and parsed_result.get("validated") is True

    state["last_sql"] = sql
    state["last_validation_ok"] = ok
    state["last_validation_error"] = None if ok else json.dumps(parsed_result, ensure_ascii=False)

    if step_number is not None:
        step_registry = dict(state.get("step_registry", {}))
        prev = dict(step_registry.get(step_number, {}))
        prev.update({
            "step": step_number,
            "action": "query",
            "sql": sql,
            "database": database,
            "validated": ok,
            "executed": prev.get("executed", False),
            "rows_returned": prev.get("rows_returned", 0),
            "status": prev.get("status", "pending"),
        })
        step_registry[step_number] = prev
        state["step_registry"] = step_registry

    if ok:
        state["last_validated_sql"] = sql
        validated_sqls = dict(state.get("validated_sqls", {}))
        key = _validated_sql_key(run_id, step_number)
        if key:
            validated_sqls[key] = {"sql": sql, "database": database}
        state["validated_sqls"] = validated_sqls
    else:
        iter_flags["iter_had_validation_error"] = True
        iter_flags["iter_had_error"] = True
        state["last_failed_step"] = step_number
        state["last_validated_sql"] = None
        
        _increment_sql_repair(state, step_number)
    
        state["sql_repair_count"] = state.get("sql_repair_count", 0) + 1
        state["total_sql_repairs"] = state.get("total_sql_repairs", 0) + 1
        state["all_errors"] = _dedup(
            list(state.get("all_errors", [])) + [json.dumps(parsed_result, ensure_ascii=False)]
        )

















def _handle_discovery_tool(
    state: "AgentState",
    name: str,
    args: dict,
    result: str,
) -> None:
    parsed = _parse_tool_result(result)

    if name == "tool_get_database_catalog":
        dbs, tables = _extract_from_database_catalog(parsed)
        state["discovered_databases"] = _dedup(list(state.get("discovered_databases", [])) + dbs)
        state["discovered_tables"] = _dedup(list(state.get("discovered_tables", [])) + tables)

    elif name == "tool_get_available_databases":
        dbs = _extract_databases_from_result(parsed)
        state["discovered_databases"] = _dedup(list(state.get("discovered_databases", [])) + dbs)

    elif name == "tool_get_available_tables":
        tables = _extract_tables_from_available_tables_result(parsed)
        state["discovered_tables"] = _dedup(list(state.get("discovered_tables", [])) + tables)

    elif name == "tool_get_table_schema":
        table = args.get("table_name")
        if table:
            state["discovered_tables"] = _dedup(list(state.get("discovered_tables", [])) + [table])














def _build_tool_message_and_log(
    tc: dict,
    i: int,
    name: str,
    raw_args: dict,
    args: dict,
    result: Any,
    msgs: list,
    iteration_tools: list,
) -> None:
    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)

    iteration_tools.append({
        "tool": name,
        "raw_args": raw_args,
        "args": args,
        "result": content,
    })

    tool_call_id = tc.get("id") or f"fallback_{i}"
    msgs.append(ToolMessage(content=content, tool_call_id=tool_call_id))

















def _get_tools():
    return [
        #kommune_where_clause,
        tool_get_available_databases,
        #tool_get_available_schemas,
        #tool_get_available_tables,




        tool_get_table_schema,
        tool_get_database_catalog,
        tool_validate_sql,
        

        tool_execute_query_step,
        tool_execute_filter_step,
        tool_execute_select_columns_step,
        tool_execute_spatial_join_step,
        tool_execute_attribute_join_step,
        tool_execute_buffer_step,
        tool_execute_nearest_step,
        tool_execute_distance_step,
        tool_execute_aggregate_step,



        tool_execute_sort_step,
        tool_execute_limit_step,
        tool_execute_clip_step,
        tool_execute_union_step,
        tool_execute_intersection_step,
        tool_execute_difference_step,


        tool_execute_dissolve_step,
        tool_execute_centroid_step,
        tool_execute_transform_crs_step,
    ]




async def _tools_node(state: AgentState) -> AgentState:
    tools = _get_tools()
    tool_map = {t.name: t for t in tools}

    ai = state["current_ai"]
    msgs = list(state["msgs"])
    iteration_tools: list[dict] = []

    iter_flags: dict = {
        "iter_had_error": False,
        "iter_had_validation_error": False,
        "iter_any_exec_ok": False,
    }

    for i, tc in enumerate(ai.tool_calls, start=1):
        name = tc["name"]
        raw_args = _coerce_args(tc.get("args"))
        args = _normalize_tool_args(name, raw_args, state["run_id"])
        tool = tool_map.get(name)
        state["total_tool_calls"] = state.get("total_tool_calls", 0) + 1

        if tool is None:
            err = f"Unknown tool: {name}"
            iter_flags["iter_had_error"] = True
            state["all_errors"] = _dedup(list(state.get("all_errors", [])) + [err])
            _build_tool_message_and_log(tc, i, name, raw_args, args, err, msgs, iteration_tools)
            continue

        result_str = ""

        try:
            if name in STEP_TOOLS:
                validated_sqls = state.get("validated_sqls", {})
                blocked_json, step_obj = _guard_step_tool(name, raw_args, args, validated_sqls)

                result_str, parsed_result, exec_ok, rows_returned = await _execute_tool(
                    tool, name, args, blocked_json
                )

                _update_step_registry(state, name, step_obj, exec_ok, rows_returned)
                _update_iteration_flags(state, name, step_obj, parsed_result, exec_ok, iter_flags)

            elif name == "tool_validate_sql":
                result_str = await tool.ainvoke(args)
                _handle_validate_sql_tool(state, args, result_str, iter_flags)

            else:
                result_str = await tool.ainvoke(args)
                _handle_discovery_tool(state, name, args, result_str)

            _build_tool_message_and_log(
                tc, i, name, raw_args, args, result_str, msgs, iteration_tools
            )

        except Exception as e:
            err = f"Error: {e}"
            iter_flags["iter_had_error"] = True
            state["all_errors"] = _dedup(list(state.get("all_errors", [])) + [err])
            _build_tool_message_and_log(tc, i, name, raw_args, args, err, msgs, iteration_tools)

    state["iter_had_error"] = iter_flags["iter_had_error"]
    state["iter_had_validation_error"] = iter_flags["iter_had_validation_error"]
    state["iter_any_exec_ok"] = iter_flags["iter_any_exec_ok"]
    state["last_execute_ok"] = iter_flags["iter_any_exec_ok"]

    run_log = list(state.get("current_run_log", []))
    if run_log:
        run_log[-1]["tools"] = iteration_tools

    state["msgs"] = msgs
    state["current_run_log"] = run_log
    state["crash_snapshot"] = _build_snapshot(state)
    return state





