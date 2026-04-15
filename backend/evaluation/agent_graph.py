import json
import logging

from dataclasses import asdict

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_ollama import ChatOllama

from config import (
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_FOUNDRY_ENDPOINT,
    AZURE_MODEL_DEPLOYMENT,
    AZURE_OPENAI_MODEL_IDS,
    AZURE_FOUNDRY_MODEL_IDS,
    OLLAMA_MODEL_IDS,
)

from agent_models import AgentState, ModelParams, AgentRunError
from agent_helpers import (
    _add_tokens,
    _answer_indicates_incomplete_task,
    _build_snapshot,
    _dedup,
    _empty_token_dict,
    _executed_from_steps,
    _extract_plan_from_text,
    _extract_tokens,
    _fallback_answer_from_logs,
    _format_restart_summary,
    _log_entry_key,
    _parse_tool_result,
    _run_log_entry_from_state,
    _sql_repair_ok,
    _step_retry_ok,
    _short,
    _task_completed_from_steps,
    _valid_sql_from_steps,
)
from agent_tools_node import _tools_node, _get_tools


logger = logging.getLogger(__name__)






_AGENT_GRAPH = None












def get_model(model_id: str, tools, params: ModelParams):
    dep = AZURE_MODEL_DEPLOYMENT.get(model_id)

    model_kwargs = {}
    if params.top_p is not None:
        model_kwargs["top_p"] = params.top_p
    if params.reasoning_effort is not None:
        model_kwargs["reasoning_effort"] = params.reasoning_effort

    if model_id in AZURE_OPENAI_MODEL_IDS:
        if not dep:
            raise ValueError(f"Unknown model_id: {model_id}")
        return AzureChatOpenAI(
            api_key=AZURE_OPENAI_KEY,
            azure_deployment=dep,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version="2024-12-01-preview",
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            timeout=params.timeout,
            max_retries=params.max_retries,
            model_kwargs=model_kwargs,
        ).bind_tools(tools, tool_choice=params.tool_choice)

    if model_id in AZURE_FOUNDRY_MODEL_IDS:
        if not dep:
            raise ValueError(f"Unknown model_id: {model_id}")
        return ChatOpenAI(
            model=dep,
            api_key=AZURE_OPENAI_KEY,
            base_url=AZURE_FOUNDRY_ENDPOINT,
            temperature=params.temperature,
            timeout=params.timeout,
            max_retries=params.max_retries,
            model_kwargs=model_kwargs,
        ).bind_tools(tools, tool_choice=params.tool_choice)

    if model_id in OLLAMA_MODEL_IDS:
        return ChatOllama(
            model=model_id,
            temperature=params.temperature,
        ).bind_tools(tools)

    raise ValueError(f"Unknown model_id: {model_id}")






async def _model_node(state: AgentState) -> AgentState:
    tools = _get_tools()
    model = get_model(state["model_id"], tools, state["params"])

    # logger.debug(
    #     "\n"
    #     + "=" * 90 + "\n"
    #     + "MODEL CALL\n"
    #     + "=" * 90 + "\n"
    #     + "  model              : %s\n"
    #     + "  attempt            : %s\n"
    #     + "  iter               : %s\n"
    #     + "  restart            : %s\n"
    #     + "  repair             : %s\n"
    #     + "  total_model_calls  : %s\n"
    #     + "  total_tools        : %s\n"
    #     + "  msg_count          : %s\n"
    #     + "  run_tokens         : %s\n"
    #     + "  grand_tokens       : %s\n"
    #     + "  last_validation_ok : %s\n"
    #     + "  last_execute_ok    : %s\n"
    #     + "  last_sql           : %s\n"
    #     + "  last_val_error     : %s\n"
    #     + "  last_exec_error    : %s\n"
    #     + "  discovered_dbs     : %s\n"
    #     + "  discovered_tables  : %s\n"
    #     + "=" * 90,
    #     state["model_id"],
    #     state.get("attempt_number", 0),
    #     state.get("iterations", 0),
    #     state.get("restart_count", 0),
    #     state.get("sql_repair_count", 0),
    #     state.get("total_iterations", 0),
    #     state.get("total_tool_calls", 0),
    #     len(state.get("msgs", [])),
    #     state.get("attempt_tokens", {}).get("total_tokens", 0),
    #     state.get("grand_total_tokens", {}).get("total_tokens", 0),
    #     state.get("last_validation_ok", False),
    #     state.get("last_execute_ok", False),
    #     _short(state.get("last_sql") or "", 120),
    #     _short(state.get("last_validation_error") or "", 120),
    #     _short(state.get("last_execute_error") or "", 120),
    #     state.get("discovered_databases", []),
    #     state.get("discovered_tables", []),
    # )
    state["crash_snapshot"] = _build_snapshot(state)

    try:
        ai = await model.ainvoke(state["msgs"])
    except Exception as e:
        state["all_errors"] = _dedup(list(state.get("all_errors", [])) + [str(e)])
        state["crash_snapshot"] = _build_snapshot(state, extra_error=str(e))
        raise AgentRunError(str(e), snapshot=state["crash_snapshot"])

    tool_names = [tc["name"] for tc in (ai.tool_calls or [])]
    # logger.debug(
    #     "\n"
    #     + "-" * 90 + "\n"
    #     + "MODEL RESULT\n"
    #     + "-" * 90 + "\n"
    #     + "  model              : %s\n"
    #     + "  attempt            : %s\n"
    #     + "  iteration          : %s\n"
    #     + "  tool_calls         : %s\n"
    #     + "  tool_names         : %s\n"
    #     + "  ai_content         : %s\n"
    #     + "-" * 90,
    #     state["model_id"],
    #     state.get("attempt_number", 0),
    #     state.get("iterations", 0) + 1,
    #     len(ai.tool_calls or []),
    #     tool_names if tool_names else "none",
    #     _short((ai.content or "").strip(), 200) if not ai.tool_calls else "(tool call response)",
    # )

    iter_tokens = _extract_tokens(ai)
    attempt_tokens = dict(state["attempt_tokens"])
    grand_total = dict(state["grand_total_tokens"])
    _add_tokens(attempt_tokens, iter_tokens)
    _add_tokens(grand_total, iter_tokens)

    msgs = list(state["msgs"])
    msgs.append(ai)

    state["iterations"] += 1
    state["msgs"] = msgs
    state["attempt_tokens"] = attempt_tokens
    state["grand_total_tokens"] = grand_total
    state["current_ai"] = ai
    state["current_iter_tokens"] = asdict(iter_tokens)
    state["total_iterations"] = state.get("total_iterations", 0) + 1
    state["current_ai_is_final_answer"] = not bool(ai.tool_calls)

    if not state.get("plan_finalized", False):
        planned_steps, required_step_ids = _extract_plan_from_text((ai.content or "").strip())
        if planned_steps:
            state["planned_steps"] = planned_steps
            state["required_step_ids"] = required_step_ids
            state["plan_finalized"] = True
            logger.info(
                "      [plan] finalized required_step_ids=%s planned_steps=%s",
                sorted(str(step_id) for step_id in required_step_ids),
                sorted(str(step_id) for step_id in planned_steps.keys()),
            )

    run_log = list(state.get("current_run_log", []))
    run_log.append({
        "iteration": state["iterations"],
        "tokens": dict(state["current_iter_tokens"]),
        "ai_content": (ai.content or "").strip(),



        "tool_calls": [
            {"name": tc.get("name"), "args": tc.get("args"), "type": tc.get("type")}
            for tc in (ai.tool_calls or [])
        ],
        "tools": [],
    })
    state["current_run_log"] = run_log
    state["crash_snapshot"] = _build_snapshot(state)
    return state






def _sql_repair_node(state: AgentState) -> AgentState:
    repair_msg = f"""
The SQL failed validation.

Original user query:
{state.get("user_input")}

Last SQL:
{state.get("last_sql")}

Validation error:
{state.get("last_validation_error")}

Discovered databases:
{state.get("discovered_databases", [])}

Discovered tables:
{state.get("discovered_tables", [])}

Fix only this SQL step.
Use already discovered metadata when possible.
Validate again before execution.
Do not guess final results.
Do not restart the full plan.
""".strip()

    state["msgs"] = [
        SystemMessage(content=state["prompt"]),
        HumanMessage(content=repair_msg),
    ]
    state["current_ai_is_final_answer"] = False
    state["crash_snapshot"] = _build_snapshot(state)
    return state







def _restart_or_finish_node(state: AgentState) -> AgentState:
    steps = dict(state.get("step_registry", {}))
    all_steps = list(steps.values())
    any_executed = any(s.get("executed", False) for s in all_steps)

    if state.get("restart_count", 0) >= state.get("max_restarts", 0):
        state["done"] = True
        state["crash_snapshot"] = _build_snapshot(state)
        return state

    all_run_logs = list(state.get("all_run_logs", []))
    if state.get("current_run_log"):
        new_entry = _run_log_entry_from_state(state)
        new_key = _log_entry_key(new_entry)
        if not any(_log_entry_key(r) == new_key for r in all_run_logs):
            all_run_logs.append(new_entry)

    new_state = {
        **state,
        "restart_count": state.get("restart_count", 0) + 1,
        "all_run_logs": all_run_logs,
        "msgs": [
            SystemMessage(content=state["prompt"]),
            HumanMessage(content=_format_restart_summary(state)),
        ],
        "iterations": 0,
        "sql_repair_count": 0,
        "step_retry_count": 0,
        "step_retry_counts": {},
        "last_failed_step": None,
        "current_run_log": [],
        "attempt_tokens": _empty_token_dict(),
        "last_sql": None if not any_executed else state.get("last_sql"),
        "last_validated_sql": None,
        "last_validation_ok": False,
        "last_validation_error": None,
        "last_guard_error": None,
        "last_execute_ok": False,
        "last_execute_error": None,
        "last_executed_result": state.get("last_executed_result") if any_executed else None,
        "valid_sql": False,
        "current_ai_is_final_answer": False,
        "done": False,
        "task_completed": False,
        "answer_based_on_result": False,
        "validated_sqls": {},
        "step_registry": {},
        "planned_steps": {},
        "required_step_ids": set(),
        "plan_finalized": False,
        "iter_had_error": False,
        "iter_had_validation_error": False,
        "iter_any_exec_ok": False,
        "plan_locked": False,
        "locked_step_ids": set(),

        "sql_repair_counts": {},




    }

    new_state["crash_snapshot"] = _build_snapshot(new_state)
    return new_state






def _finalize_node(state: AgentState) -> AgentState:
    current_answer = ""
    ai = state.get("current_ai")
    if ai and state.get("current_ai_is_final_answer", False):
        current_answer = (ai.content or "").strip()

    if state.get("current_run_log"):
        all_run_logs = list(state.get("all_run_logs", []))
        new_entry = _run_log_entry_from_state(state)
        new_key = _log_entry_key(new_entry)
        if not any(_log_entry_key(r) == new_key for r in all_run_logs):
            all_run_logs.append(new_entry)
        state["all_run_logs"] = all_run_logs

    if not current_answer:
        current_answer = _fallback_answer_from_logs(
            state.get("all_run_logs", []),
            state.get("current_run_log", []),
        )

    run_logs = list(state.get("all_run_logs", []))
    steps = dict(state.get("step_registry", {}))
    query_steps = [s for s in steps.values() if s.get("action") == "query"]
    all_steps = list(steps.values())

    any_executed = any(s.get("executed", False) for s in all_steps)
    all_validated = (
        all(s.get("validated", False) for s in query_steps)
        if query_steps else state.get("last_validation_ok", False)
    )
    all_executed = (
        all(s.get("executed", False) for s in all_steps)
        if all_steps else state.get("last_execute_ok", False)
    )
    has_result = state.get("last_executed_result") is not None

    has_answer = bool(current_answer)
    answer_complete = not _answer_indicates_incomplete_task(current_answer)

    step_valid_sql = _valid_sql_from_steps(run_logs, valid_sql_fallback=all_validated)
    step_executed = _executed_from_steps(run_logs, executed_fallback=any_executed or all_executed)
    step_task_completed = _task_completed_from_steps(
        run_logs,
        task_completed_fallback=all_executed,
        final_answer=current_answer,
        required_step_ids=set(state.get("required_step_ids", set())),
        plan_finalized=state.get("plan_finalized", False),
    )

    if not step_task_completed:
        current_answer = ""
        state["all_errors"] = _dedup(
            list(state.get("all_errors", [])) + ["Not all planned steps executed successfully."]
        )

    if step_task_completed:
        state["last_validation_error"] = None
        state["last_guard_error"] = None
        state["last_execute_error"] = None

    state["valid_sql"] = step_valid_sql
    state["last_execute_ok"] = step_executed
    state["answer"] = current_answer
    state["answer_based_on_result"] = step_task_completed and has_result and has_answer and answer_complete
    state["task_completed"] = step_task_completed
    state["done"] = True
    state["crash_snapshot"] = _build_snapshot(state)
    return state




def _route_after_model(state: AgentState) -> str:
    ai = state["current_ai"]

    if not ai.tool_calls:
        return "finalize"

    if state["iterations"] >= state["max_iterations"]:
        return "restart_or_finish"

    return "tools"





def _replan_node(state: AgentState) -> AgentState:
    zero_rows_step = state.get("zero_rows_step")
    replan_reason = state.get("replan_reason")
    replan_error = state.get("replan_error")
    blocked_step = state.get("last_failed_step")
    step_registry = dict(state.get("step_registry", {}))

    completed_lines = []
    for k, v in sorted(
        step_registry.items(),
        key=lambda x: int(str(x[0])) if str(x[0]).isdigit() else 0,
    ):
        rows = v.get("rows_returned")
        if v.get("executed"):
            row_note = f"executed — {rows} rows" if rows is not None else "executed"
        elif v.get("validated"):
            row_note = "validated (not yet executed)"
        else:
            row_note = "pending"
        completed_lines.append(f"  Step {k}: {v.get('action', '?')} — {row_note}")

    if replan_reason == "timeout_step":
        replan_msg = f"""Step {blocked_step} in the finalized plan timed out during execution.

Original query:
{state.get("user_input", "")}

Progress so far:
{chr(10).join(completed_lines) if completed_lines else "  (none)"}

Timeout reason:
{replan_error or "(no reason provided)"}

The current finalized plan can no longer continue as written.

Produce a new numbered plan before calling any tools. Requirements:
1. Keep the same overall task goal
2. Replace the timed-out step with a cheaper or more targeted approach
3. Renumber the new plan from Step 1 upward
4. Do not repeat the exact same timed-out query
5. Do not continue using the old finalized step numbers
""".strip()
    elif replan_reason == "blocked_step":
        replan_msg = f"""Step {blocked_step} in the finalized plan was blocked before execution.

Original query:
{state.get("user_input", "")}

Progress so far:
{chr(10).join(completed_lines) if completed_lines else "  (none)"}

Blocked reason:
{replan_error or "(no reason provided)"}

The current finalized plan can no longer continue as written.

Produce a new numbered plan before calling any tools. Requirements:
1. Keep the same overall task goal
2. Replace the blocked step with a cheaper or more targeted approach
3. Renumber the new plan from Step 1 upward
4. Do not repeat the exact same blocked query
5. Do not continue using the old finalized step numbers
""".strip()
    else:
        replan_msg = f"""Step {zero_rows_step} executed successfully but returned 0 rows.

Original query:
{state.get("user_input", "")}

Progress so far:
{chr(10).join(completed_lines) if completed_lines else "  (none)"}

The remaining planned steps cannot proceed as originally planned because they depend on step {zero_rows_step}'s result.

Produce a new numbered plan before calling any tools. Options:
1. Revise the step {zero_rows_step} query — fix the filter, check spelling variants, try kommunenummer instead of kommunenavn
2. Accept that no results were found and provide a final answer explaining why
3. Use a different approach entirely

Do not repeat the exact same queries that already returned 0 rows.""".strip()

    all_run_logs = list(state.get("all_run_logs", []))
    if state.get("current_run_log"):
        new_entry = _run_log_entry_from_state(state)
        new_key = _log_entry_key(new_entry)
        if not any(_log_entry_key(r) == new_key for r in all_run_logs):
            all_run_logs.append(new_entry)

    new_state = {
        **state,
        "msgs": [
            SystemMessage(content=state["prompt"]),
            HumanMessage(content=replan_msg),
        ],
        "plan_finalized": False,
        "planned_steps": {},
        "required_step_ids": set(),
        "step_registry": {},
        "locked_step_ids": set(),
        "validated_sqls": {},
        "needs_replan": False,
        "zero_rows_step": None,
        "replan_reason": None,
        "replan_error": None,
        "replan_count": state.get("replan_count", 0) + 1,
        "current_ai_is_final_answer": False,
        "iter_had_error": False,
        "iter_had_validation_error": False,
        "iter_any_exec_ok": False,
        "last_validation_ok": False,
        "last_validation_error": None,
        "last_guard_error": None,
        "last_execute_error": None,
        "last_execute_ok": False,
        "last_sql": None,
        "last_validated_sql": None,
        "current_run_log": [],
        "all_run_logs": all_run_logs,
    }
    new_state["crash_snapshot"] = _build_snapshot(new_state)
    return new_state


def _route_after_tools(state: AgentState) -> str:
    # Replan takes priority — a step returned 0 rows with required steps remaining
    if state.get("needs_replan"):
        max_replans = getattr(state.get("params"), "max_replans", 1)
        if state.get("replan_count", 0) < max_replans:
            return "replan"
        return "restart_or_finish"

    iter_had_error = state.get("iter_had_error", False)
    iter_had_validation_error = state.get("iter_had_validation_error", False)
    iter_any_exec_ok = state.get("iter_any_exec_ok", False)

    if iter_any_exec_ok and not iter_had_error:
        return "model"

    if iter_had_validation_error:
        failing_step = state.get("last_failed_step")
        if failing_step is not None and _sql_repair_ok(
            state, failing_step, state["params"].max_sql_repairs
        ):
            return "sql_repair"
        return "restart_or_finish"

    if iter_had_error:
        parsed_err = _parse_tool_result(state.get("last_guard_error") or state.get("last_execute_error"))

        if isinstance(parsed_err, dict):
            if str(parsed_err.get("error", "")).endswith("blocked."):
                return "model"

        failing_step = state.get("last_failed_step")
        step_registry = state.get("step_registry", {})
        failed_meta = step_registry.get(failing_step, {}) if failing_step is not None else {}
        failed_action = failed_meta.get("action")

        if failing_step is not None and failed_action == "query":
            if _sql_repair_ok(state, failing_step, state["params"].max_sql_repairs):
                return "sql_repair"
            return "restart_or_finish"

        if failing_step is not None and _step_retry_ok(
            state, failing_step, state["params"].max_step_retries
        ):
            return "model"

        return "restart_or_finish"

    if state.get("iterations", 0) >= state.get("max_iterations", 0):
        return "restart_or_finish"

    return "model"





def _route_after_restart_or_finish(state: AgentState) -> str:
    return "finalize" if state.get("done") else "model"







def get_agent_graph():
    global _AGENT_GRAPH
    if _AGENT_GRAPH is None:
        _AGENT_GRAPH = _build_agent_graph()
    return _AGENT_GRAPH


def _build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("model", _model_node)
    graph.add_node("tools", _tools_node)
    graph.add_node("sql_repair", _sql_repair_node)
    graph.add_node("restart_or_finish", _restart_or_finish_node)
    graph.add_node("replan", _replan_node)
    graph.add_node("finalize", _finalize_node)

    graph.set_entry_point("model")


    graph.add_conditional_edges(
        "model", _route_after_model,
        {"tools": "tools", "restart_or_finish": "restart_or_finish", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "tools", _route_after_tools,
        {"model": "model", "sql_repair": "sql_repair",
         "restart_or_finish": "restart_or_finish", "replan": "replan", "finalize": "finalize"},
    )
    graph.add_edge("replan", "model")
    graph.add_edge("sql_repair", "model")
    graph.add_conditional_edges(
        "restart_or_finish", _route_after_restart_or_finish,
        {"model": "model", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()
