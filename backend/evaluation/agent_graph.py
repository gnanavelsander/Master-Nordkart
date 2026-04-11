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
    _build_snapshot,
    _dedup,
    _empty_token_dict,
    _extract_tokens,
    _format_restart_summary,
    _log_entry_key,
    _parse_tool_result,
    _run_log_entry_from_state,
    _sql_repair_ok,
    _step_retry_ok,
    _short,
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

    logger.debug(
        "\n"
        + "=" * 90 + "\n"
        + "MODEL CALL\n"
        + "=" * 90 + "\n"
        + "  model              : %s\n"
        + "  attempt            : %s\n"
        + "  iter               : %s\n"
        + "  restart            : %s\n"
        + "  repair             : %s\n"
        + "  total_model_calls  : %s\n"
        + "  total_tools        : %s\n"
        + "  msg_count          : %s\n"
        + "  run_tokens         : %s\n"
        + "  grand_tokens       : %s\n"
        + "  last_validation_ok : %s\n"
        + "  last_execute_ok    : %s\n"
        + "  last_sql           : %s\n"
        + "  last_val_error     : %s\n"
        + "  last_exec_error    : %s\n"
        + "  discovered_dbs     : %s\n"
        + "  discovered_tables  : %s\n"
        + "  total_api_calls    : %s\n"
        + "=" * 90,
        state["model_id"],
        state.get("attempt_number", 0),
        state.get("iterations", 0),
        state.get("restart_count", 0),
        state.get("sql_repair_count", 0),
        state.get("total_iterations", 0),
        state.get("total_tool_calls", 0),
        len(state.get("msgs", [])),
        state.get("attempt_tokens", {}).get("total_tokens", 0),
        state.get("grand_total_tokens", {}).get("total_tokens", 0),
        state.get("last_validation_ok", False),
        state.get("last_execute_ok", False),
        _short(state.get("last_sql") or "", 120),
        _short(state.get("last_validation_error") or "", 120),
        _short(state.get("last_execute_error") or "", 120),
        state.get("discovered_databases", []),
        state.get("discovered_tables", []),
        state.get("total_api_calls", 0),
    )
    state["total_api_calls"] = state.get("total_api_calls", 0) + 1
    state["crash_snapshot"] = _build_snapshot(state)

    try:
        ai = await model.ainvoke(state["msgs"])
    except Exception as e:
        state["all_errors"] = _dedup(list(state.get("all_errors", [])) + [str(e)])
        state["crash_snapshot"] = _build_snapshot(state, extra_error=str(e))
        raise AgentRunError(str(e), snapshot=state["crash_snapshot"])

    tool_names = [tc["name"] for tc in (ai.tool_calls or [])]
    logger.debug(
        "\n"
        + "-" * 90 + "\n"
        + "MODEL RESULT\n"
        + "-" * 90 + "\n"
        + "  model              : %s\n"
        + "  attempt            : %s\n"
        + "  iteration          : %s\n"
        + "  tool_calls         : %s\n"
        + "  tool_names         : %s\n"
        + "  ai_content         : %s\n"
        + "-" * 90,
        state["model_id"],
        state.get("attempt_number", 0),
        state.get("iterations", 0) + 1,
        len(ai.tool_calls or []),
        tool_names if tool_names else "none",
        _short((ai.content or "").strip(), 200) if not ai.tool_calls else "(tool call response)",
    )

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

Last SQL:
{state.get("last_sql")}

Validation error:
{state.get("last_validation_error")}

Fix the SQL.
Use already discovered metadata when possible.
Validate again before execution.
Do not guess final results.
""".strip()

    msgs = list(state["msgs"])
    msgs.append(HumanMessage(content=repair_msg))
    state["msgs"] = msgs
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
        "iter_had_error": False,
        "iter_had_validation_error": False,
        "iter_any_exec_ok": False,
        "plan_locked": False,
        "locked_step_ids": set(),

        "sql_repair_counts": {},

        #"step_retry_counts": {},



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

    steps = dict(state.get("step_registry", {}))
    query_steps = [s for s in steps.values() if s.get("action") == "query"]
    all_steps = list(steps.values())

    any_executed = any(s.get("executed", False) for s in all_steps)

    if query_steps:
        all_validated = all(s.get("validated", False) for s in query_steps)
        all_executed = all(s.get("executed", False) for s in all_steps)
        last_step = max(all_steps, key=lambda s: s.get("step", 0), default=None)
        has_result = bool(last_step and last_step.get("executed", False))
    else:
        all_validated = state.get("last_validation_ok", False)
        all_executed = any_executed or state.get("last_execute_ok", False)
        has_result = state.get("last_executed_result") is not None

    has_answer = bool(current_answer)

    if not all_executed:
        current_answer = ""
        state["all_errors"] = _dedup(
            list(state.get("all_errors", [])) + ["Not all planned steps executed successfully."]
        )

    state["valid_sql"] = all_validated
    state["last_execute_ok"] = all_executed
    state["answer"] = current_answer
    state["answer_based_on_result"] = all_executed and has_result and has_answer
    state["task_completed"] = all_executed and has_result and has_answer
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





def _route_after_tools(state: AgentState) -> str:
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
        parsed_err = _parse_tool_result(state.get("last_execute_error"))

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
    graph.add_node("finalize", _finalize_node)

    graph.set_entry_point("model")


    graph.add_conditional_edges(
        "model", _route_after_model,
        {"tools": "tools", "restart_or_finish": "restart_or_finish", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "tools", _route_after_tools,
        {"model": "model", "sql_repair": "sql_repair",
         "restart_or_finish": "restart_or_finish", "finalize": "finalize"},
    )
    graph.add_edge("sql_repair", "model")
    graph.add_conditional_edges(
        "restart_or_finish", _route_after_restart_or_finish,
        {"model": "model", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()

