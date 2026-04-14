import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

# Hard timeout per step execute call (seconds). Queries that exceed this are returned
# as a blocked error so the model can rewrite and retry with reduced data.
EXECUTE_TIMEOUT_S = 90






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
    tool_get_available_schemas,
    tool_get_available_tables,
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
from backend.embeddings.retrieve_layers import build_filtered_catalog, is_candidate_in_scope












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


def _block_step_result(name: str, reason: str, step_number: int | None, action: str | None) -> str:
    return json.dumps({
        "executed": False,
        "error": f"{name} blocked.",
        "reason": reason,
        "step": step_number,
        "action": action,
    })


def _lock_step(state: "AgentState", step_number: int | None) -> None:
    if step_number is None:
        return

    locked_step_ids = set(state.get("locked_step_ids", set()))
    locked_step_ids.add(step_number)
    state["locked_step_ids"] = locked_step_ids

    step_registry = dict(state.get("step_registry", {}))
    prev = dict(step_registry.get(step_number, {}))
    if prev:
        prev["locked"] = True
        step_registry[step_number] = prev
        state["step_registry"] = step_registry


def _check_expensive_sql(sql: str) -> str | None:
    """
    Return a human-readable risk reason if the SQL contains a geometry aggregate
    (ST_Union) without a meaningful spatial or attribute pre-filter.

    Returns None if the SQL looks safe or does not match known expensive patterns.
    """
    if not re.search(r"\bST_UNION\s*\(", sql, re.IGNORECASE):
        return None

    # Meaningful filters that make aggregation safe
    has_spatial_pred = bool(re.search(
        r"\bST_(INTERSECTS|DWITHIN|WITHIN|CONTAINS)\s*\(", sql, re.IGNORECASE
    ))
    has_bbox_op = "&&" in sql
    has_kommune_filter = bool(re.search(
        r"\bKOMMUNE(NUMMER|NAVN)?\s*(=|IN\b)", sql, re.IGNORECASE
    ))
    if has_spatial_pred or has_bbox_op or has_kommune_filter:
        return None

    # Check whether a non-trivial WHERE clause is present
    has_real_filter = False
    where_match = re.search(
        r"\bWHERE\b(.*?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|;|$)",
        sql, re.IGNORECASE | re.DOTALL,
    )
    if where_match:
        where_content = where_match.group(1).strip()
        # Strip IS [NOT] NULL — these are not a meaningful reduction
        cleaned = re.sub(r"\w+\s+IS\s+(?:NOT\s+)?NULL", "", where_content, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bAND\b|\bOR\b", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned:
            has_real_filter = True

    if has_real_filter:
        return None

    if re.search(r"\bST_BUFFER\s*\(\s*ST_UNION", sql, re.IGNORECASE):
        return (
            "ST_Buffer(ST_Union(...)) without a spatial or attribute pre-filter will aggregate "
            "the entire table and is likely to time out. "
            "Add a filter (ST_Intersects, kommunenummer=..., bounding box &&) before aggregating."
        )
    return (
        "ST_Union() without a spatial or attribute pre-filter will likely be slow. "
        "Add a WHERE filter to reduce rows before aggregating."
    )


def _guard_validate_sql(state: "AgentState", args: dict) -> str | None:
    step_number = args.get("step_number")
    if step_number is None:
        return None

    step_registry = dict(state.get("step_registry", {}))
    prev = dict(step_registry.get(step_number, {}))

    if step_number in set(state.get("locked_step_ids", set())):
        return _block_step_result(
            "tool_validate_sql",
            f"Step {step_number} is locked after a successful validation/execution and cannot be changed.",
            step_number,
            prev.get("action", "query"),
        )

    prev_payload = prev.get("step_payload") if isinstance(prev.get("step_payload"), dict) else {}
    prev_action = prev_payload.get("action") or prev.get("action")
    if prev_action and prev_action != "query":
        return _block_step_result(
            "tool_validate_sql",
            f"Step {step_number} is already registered as action '{prev_action}' and cannot be changed to 'query'.",
            step_number,
            prev_action,
        )

    return None


def _guard_step_tool(
    state: "AgentState",
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
        return _block_step_result(name, _flat_hint(), None, action_name), None

    step_obj = dict(step_obj)
    step_number = step_obj.get("step")
    action = step_obj.get("action") or action_name
    sql = (step_obj.get("sql") or "").strip()
    database = (step_obj.get("database") or "").strip()
    blocked_reason: str | None = None
    prev = dict(state.get("step_registry", {}).get(step_number, {})) if step_number is not None else {}
    prev_payload = prev.get("step_payload") if isinstance(prev.get("step_payload"), dict) else {}
    prev_action = prev_payload.get("action") or prev.get("action")

    if step_number in set(state.get("locked_step_ids", set())):
        blocked_reason = (
            f"Step {step_number} is locked after a successful validation/execution and cannot be changed."
        )
    elif prev_action and prev_action != action:
        blocked_reason = (
            f"Step {step_number} is already registered as action '{prev_action}' "
            f"and cannot be changed to '{action}'."
        )

    def _norm(s: str) -> str:
        return " ".join((s or "").split())

    if blocked_reason is None and name == "tool_execute_query_step":
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
            expensive_reason = _check_expensive_sql(sql)
            if expensive_reason:
                blocked_reason = f"Expensive query blocked: {expensive_reason}"

    elif blocked_reason is None:
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
            left = step_obj.get("left_source")
            right = step_obj.get("right_source")
            if left is None or right is None:
                blocked_reason = f"{action} requires left_source and right_source as plain integers (step numbers)."
            elif not isinstance(left, int) or not isinstance(right, int):
                blocked_reason = (
                    f"{action}: left_source and right_source must be plain integers (step numbers), "
                    f"not {type(left).__name__}/{type(right).__name__}. "
                    f"Got left_source={left!r}, right_source={right!r}."
                )
            elif action == "spatial_join":
                valid_join_types = {"intersects", "within", "contains", "dwithin", "touches", "overlaps"}
                jt = step_obj.get("join_type")
                if jt not in valid_join_types:
                    blocked_reason = (
                        f"spatial_join: join_type must be a spatial predicate — one of "
                        f"{sorted(valid_join_types)}. Got {jt!r}. "
                        f"Do NOT use SQL join types like 'inner', 'left', 'outer'."
                    )

    if blocked_reason:
        return _block_step_result(name, blocked_reason, step_number, action), None

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

    step_obj = args.get("step") if isinstance(args.get("step"), dict) else {}
    step_number = step_obj.get("step") if isinstance(step_obj, dict) else None
    logger.info("      [execute] STARTING %s step=%s", name, step_number)

    try:
        result_str = await asyncio.wait_for(tool.ainvoke(args), timeout=EXECUTE_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning(
            "      [execute] TIMEOUT %s step=%s — exceeded %ss",
            name, step_number, EXECUTE_TIMEOUT_S,
        )
        timeout_result = json.dumps({
            "executed": False,
            "error": (
                f"{name} timed out after {EXECUTE_TIMEOUT_S}s. "
                "The query is too expensive to run as-is. "
                "Rewrite with a spatial filter (ST_Intersects, kommunenummer=..., bounding box) "
                "to reduce the data before executing."
            ),
            "reason": "execute_timeout",
            "step": step_number,
            "action": step_obj.get("action", "query") if isinstance(step_obj, dict) else "query",
        })
        parsed = _parse_tool_result(timeout_result)
        return timeout_result, parsed, False, 0

    parsed = _parse_tool_result(result_str)
    exec_ok = isinstance(parsed, dict) and parsed.get("executed", False)
    rows_returned = parsed.get("rows_returned", 0) if isinstance(parsed, dict) else 0
    logger.info(
        "      [execute] FINISHED %s step=%s ok=%s rows=%s",
        name, step_number, exec_ok, rows_returned,
    )
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
    if prev.get("locked"):
        return

    prev.update({
        "step": step_number,
        "action": action,
        "sql": (step_obj.get("sql") or "").strip() if is_query else prev.get("sql", ""),
        "database": (step_obj.get("database") or "").strip() if is_query else prev.get("database", ""),
        "validated": prev.get("validated", False),
        "executed": exec_ok,
        "rows_returned": rows_returned,
        "status": "ok" if exec_ok else "error",
        "locked": prev.get("locked", False),
        "step_payload": dict(step_obj),
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
        state["last_guard_error"] = None
        _lock_step(state, step_number)

        if name == "tool_execute_query_step":
            validated_sqls = dict(state.get("validated_sqls", {}))
            key = _validated_sql_key(run_id, step_number)
            if key:
                validated_sqls.pop(key, None)
            state["validated_sqls"] = validated_sqls
    else:
        iter_flags["iter_had_error"] = True
        state["last_failed_step"] = step_number
        state["last_guard_error"] = None
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

    if exec_ok:
        state["last_executed_result"] = parsed_result
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
    is_blocked = isinstance(parsed_result, dict) and str(
        parsed_result.get("error", "")
    ).endswith("blocked.")
    step_registry = dict(state.get("step_registry", {}))
    prev = dict(step_registry.get(step_number, {})) if step_number is not None else {}
    prev_payload = prev.get("step_payload") if isinstance(prev.get("step_payload"), dict) else {}
    prev_action = prev_payload.get("action") or prev.get("action")
    is_locked = bool(prev.get("locked")) or step_number in set(state.get("locked_step_ids", set()))

    state["last_sql"] = sql
    state["last_validation_ok"] = ok
    state["last_validation_error"] = None if (ok or is_blocked) else json.dumps(parsed_result, ensure_ascii=False)

    if is_blocked:
        iter_flags["iter_had_error"] = True
        state["last_failed_step"] = step_number
        state["last_validated_sql"] = None
        state["last_guard_error"] = json.dumps(parsed_result, ensure_ascii=False)
        state["all_errors"] = _dedup(
            list(state.get("all_errors", [])) + [json.dumps(parsed_result, ensure_ascii=False)]
        )
        return

    if step_number is not None and not is_locked:
        prev.update({
            "step": step_number,
            "action": prev_action or "query",
            "sql": sql,
            "database": database,
            "validated": ok,
            "executed": prev.get("executed", False),
            "rows_returned": prev.get("rows_returned", 0),
            "status": prev.get("status", "pending"),
            "locked": prev.get("locked", False),
            "step_payload": {
                "step": step_number,
                "action": prev_action or "query",
                "sql": sql,
                "database": database,
            },
        })
        step_registry[step_number] = prev
        state["step_registry"] = step_registry

    if ok:
        state["last_validated_sql"] = sql
        state["last_guard_error"] = None
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
        state["last_guard_error"] = None
        
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


def _run_retrieval_scoped_discovery_tool(
    state: "AgentState",
    name: str,
    args: dict,
) -> str | None:
    if not state.get("retrieval_scope_active", False):
        return None

    retrieval_candidates = state.get("retrieval_candidates") or []
    if not retrieval_candidates:
        return None

    if name == "tool_get_available_databases":
        databases = sorted(
            {
                str(item.get("database") or "").strip()
                for item in retrieval_candidates
                if item.get("database")
            }
        )
        return json.dumps(databases, ensure_ascii=False)

    if name == "tool_get_available_schemas":
        database_name = str(args.get("databaseName") or "").strip()
        schemas = sorted(
            {
                str(item.get("schema") or "").strip()
                for item in retrieval_candidates
                if item.get("database") == database_name and item.get("schema")
            }
        )
        return json.dumps(
            {"databaseName": database_name, "schemas": schemas},
            ensure_ascii=False,
        )

    if name == "tool_get_available_tables":
        database_name = str(args.get("databaseName") or "").strip()
        schema_name = str(args.get("schemaName") or "").strip()
        tables = sorted(
            {
                str(item.get("table") or "").strip()
                for item in retrieval_candidates
                if item.get("database") == database_name
                and item.get("schema") == schema_name
                and item.get("table")
            }
        )
        return json.dumps(
            {
                "databaseName": database_name,
                "schemaName": schema_name,
                "tables": tables,
            },
            ensure_ascii=False,
        )

    if name == "tool_get_database_catalog":
        return json.dumps(
            build_filtered_catalog(
                retrieval_candidates,
                database_name=args.get("databaseName"),
                schema_name=args.get("schemaName"),
            ),
            ensure_ascii=False,
        )

    if name == "tool_get_table_schema":
        database_name = str(args.get("databaseName") or "").strip()
        schema_name = str(args.get("schemaName") or "").strip()
        table_name = str(args.get("table_name") or "").strip()

        if not is_candidate_in_scope(
            retrieval_candidates,
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name,
        ):
            return json.dumps(
                {
                    "error": (
                        f"Table '{database_name}.{schema_name}.{table_name}' is outside the "
                        "retrieval candidate scope. Use one of the pre-retrieved candidates first."
                    ),
                    "allowed_candidates": [
                        item.get("full_name")
                        for item in retrieval_candidates
                        if item.get("full_name")
                    ],
                },
                ensure_ascii=False,
            )

    return None














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

    logger.info("      [tool] TOOL   : %s", name)
    logger.info("      [tool] ARGS   : %s", args)
    logger.info("      [tool] RESULT : %s", content)

    tool_call_id = tc.get("id") or f"fallback_{i}"
    msgs.append(ToolMessage(content=content, tool_call_id=tool_call_id))


def _build_step_event(state: "AgentState", name: str, raw_args: dict, args: dict, result: Any) -> dict | None:
    parsed_result = _parse_tool_result(result)

    if name == "tool_validate_sql":
        step_number = args.get("step_number")
        if step_number is None:
            return None

        step_meta = dict(state.get("step_registry", {}).get(step_number, {}))
        return {
            "event_type": "validate",
            "step": step_number,
            "action": step_meta.get("action", "query"),
            "database": (args.get("databaseName") or "").strip() or step_meta.get("database"),
            "sql": (args.get("sql") or "").strip() or step_meta.get("sql"),
            "validated": step_meta.get("validated", False),
            "executed": step_meta.get("executed", False),
            "status": step_meta.get("status"),
            "rows_returned": step_meta.get("rows_returned"),
            "sql_repairs": state.get("sql_repair_counts", {}).get(step_number, 0),
            "step_retries": state.get("step_retry_counts", {}).get(step_number, 0),
            "raw_args": raw_args,
            "args": args,
            "result": parsed_result,
        }

    if name not in STEP_TOOLS:
        return None

    step_obj = args.get("step")
    step_number = step_obj.get("step") if isinstance(step_obj, dict) else None
    if step_number is None:
        return None

    step_meta = dict(state.get("step_registry", {}).get(step_number, {}))
    return {
        "event_type": "execute",
        "step": step_number,
        "action": step_meta.get("action") or name.replace("tool_execute_", "").replace("_step", ""),
        "database": step_meta.get("database"),
        "sql": step_meta.get("sql"),
        "validated": step_meta.get("validated", False),
        "executed": step_meta.get("executed", False),
        "status": step_meta.get("status"),
        "rows_returned": step_meta.get("rows_returned"),
        "sql_repairs": state.get("sql_repair_counts", {}).get(step_number, 0),
        "step_retries": state.get("step_retry_counts", {}).get(step_number, 0),
        "raw_args": raw_args,
        "args": args,
        "result": parsed_result,
    }

















def _get_tools():
    return [
        #kommune_where_clause,
        tool_get_available_databases,
        tool_get_available_schemas,
        tool_get_available_tables,




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
    iteration_step_events: list[dict] = []

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
                blocked_json, step_obj = _guard_step_tool(state, name, raw_args, args, validated_sqls)

                result_str, parsed_result, exec_ok, rows_returned = await _execute_tool(
                    tool, name, args, blocked_json
                )

                _update_step_registry(state, name, step_obj, exec_ok, rows_returned)
                _update_iteration_flags(state, name, step_obj, parsed_result, exec_ok, iter_flags)

            elif name == "tool_validate_sql":
                blocked_json = _guard_validate_sql(state, args)
                if blocked_json is not None:
                    result_str = blocked_json
                else:
                    result_str = await tool.ainvoke(args)
                    # Flag expensive SQL patterns early so the model can self-correct
                    # before reaching the execute guard.
                    parsed_v = _parse_tool_result(result_str)
                    if isinstance(parsed_v, dict) and parsed_v.get("validated") is True:
                        expensive_reason = _check_expensive_sql((args.get("sql") or "").strip())
                        if expensive_reason:
                            parsed_v["expensive_query_risk"] = True
                            parsed_v["warning"] = (
                                f"Expensive query risk: {expensive_reason} "
                                "Rewrite to add a pre-filter, or this step will be blocked at execution."
                            )
                            result_str = json.dumps(parsed_v, ensure_ascii=False)
                            logger.warning(
                                "      [validate] EXPENSIVE SQL step=%s: %s",
                                args.get("step_number"), expensive_reason,
                            )
                _handle_validate_sql_tool(state, args, result_str, iter_flags)

            else:
                scoped_result = _run_retrieval_scoped_discovery_tool(state, name, args)
                if scoped_result is not None:
                    result_str = scoped_result
                else:
                    result_str = await tool.ainvoke(args)
                _handle_discovery_tool(state, name, args, result_str)

            _build_tool_message_and_log(
                tc, i, name, raw_args, args, result_str, msgs, iteration_tools
            )
            step_event = _build_step_event(state, name, raw_args, args, result_str)
            if step_event is not None:
                iteration_step_events.append(step_event)

        except Exception as e:
            err = f"Error: {e}"
            iter_flags["iter_had_error"] = True
            state["all_errors"] = _dedup(list(state.get("all_errors", [])) + [err])
            _build_tool_message_and_log(tc, i, name, raw_args, args, err, msgs, iteration_tools)
            step_event = _build_step_event(state, name, raw_args, args, err)
            if step_event is not None:
                iteration_step_events.append(step_event)

    state["iter_had_error"] = iter_flags["iter_had_error"]
    state["iter_had_validation_error"] = iter_flags["iter_had_validation_error"]
    state["iter_any_exec_ok"] = iter_flags["iter_any_exec_ok"]
    state["last_execute_ok"] = iter_flags["iter_any_exec_ok"]

    run_log = list(state.get("current_run_log", []))
    if run_log:
        run_log[-1]["tools"] = iteration_tools
        run_log[-1]["step_events"] = iteration_step_events
        run_log[-1]["iteration_error"] = (
            state.get("last_guard_error")
            or state.get("last_execute_error")
            or state.get("last_validation_error")
        )
        run_log[-1]["failed_step"] = state.get("last_failed_step")
        run_log[-1]["had_error"] = iter_flags["iter_had_error"]
        run_log[-1]["had_validation_error"] = iter_flags["iter_had_validation_error"]
        run_log[-1]["had_exec_success"] = iter_flags["iter_any_exec_ok"]

    state["msgs"] = msgs
    state["current_run_log"] = run_log
    state["crash_snapshot"] = _build_snapshot(state)
    return state
