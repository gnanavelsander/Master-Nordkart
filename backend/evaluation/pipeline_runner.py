import asyncio
import logging
from pathlib import Path
import sys
import time
import uuid

from langchain_core.messages import SystemMessage, HumanMessage


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))






from agent_models import (
    AgentState,
    AgentResult,
    PipelineResult,
    EvalRun,
    ModelParams,
    TokenUsage,
    AgentRunError,
)
from agent_helpers import (
    _accumulate_retry_snapshots,
    _build_snapshot,
    _dedup,
    _empty_token_dict,
    _is_rate_limit_error,
)
from agent_graph import get_agent_graph
from reporting import extract_agent_trace, _print_summary, _print_details, store_run
from backend.logging_utils import setup_logging
from backend.mcp.client.client import mcp_client



from agent_helpers import load_prompt


_MAX_RETRY_DELAY_S = 120




logger = logging.getLogger(__name__)


async def agent(
    user_input: str,
    model_id: str,
    params: ModelParams,
    prompt: str,
    attempt_number: int = 0,
) -> AgentResult:
    init_state: AgentState = {
        "user_input": user_input,
        "model_id": model_id,
        "prompt": prompt,
        "params": params,
        "msgs": [SystemMessage(content=prompt), HumanMessage(content=user_input)],
        "iterations": 0,
        "max_iterations": params.max_iterations,
        "sql_repair_count": 0,
        "max_sql_repairs": params.max_sql_repairs,
        "restart_count": 0,
        "max_restarts": params.max_restarts,
        "current_run_log": [],
        "all_run_logs": [],
        "attempt_tokens": _empty_token_dict(),
        "grand_total_tokens": _empty_token_dict(),
        "last_sql": None,
        "last_validated_sql": None,
        "last_validation_ok": False,
        "last_validation_error": None,
        "last_executed_result": None,
        "last_execute_ok": False,
        "last_execute_error": None,
        "discovered_tables": [],
        "discovered_databases": [],
        "valid_sql": False,
        "answer": "",
        "all_errors": [],
        "done": False,
        "answer_based_on_result": False,
        "task_completed": False,
        "current_ai_is_final_answer": False,
        "total_sql_repairs": 0,
        "total_iterations": 0,
        "total_tool_calls": 0,
        "total_api_calls": 0,
        "crash_snapshot": {},
        "attempt_number": attempt_number,
        "run_id": uuid.uuid4().hex,
        "validated_sqls": {},
        "query_mode": None,
        "step_registry": {},
        "iter_had_error": False,
        "iter_had_validation_error": False,
        "iter_any_exec_ok": False,
        "step_retry_count": 0,
        "total_step_retries": 0,
        "step_retry_counts": {},
        "last_failed_step": None,
        "plan_locked": False,
        "locked_step_ids": set(),


        "sql_repair_counts": {},
        "step_retry_counts": {},

    }

    state = init_state
    try:
        state = await get_agent_graph().ainvoke(init_state)
    except Exception as e:
        if isinstance(e, AgentRunError):
            raise
        snap = _build_snapshot(state, extra_error=str(e))
        raise AgentRunError(str(e), snapshot=snap)
    finally:
        try:
            await mcp_client.call("cleanup_run", {"run_id": init_state["run_id"]})
        except Exception as cleanup_err:
            logger.warning(
                "cleanup_run failed for run_id=%s: %s",
                init_state["run_id"],
                cleanup_err,
            )


    total_tokens = TokenUsage(
        prompt_tokens=state["grand_total_tokens"]["prompt_tokens"],
        completion_tokens=state["grand_total_tokens"]["completion_tokens"],
        total_tokens=state["grand_total_tokens"]["total_tokens"],
    )

    steps = dict(state.get("step_registry", {}))
    query_steps = [s for s in steps.values() if s.get("action") == "query"]
    all_steps = list(steps.values())
    executed = any(s.get("executed", False) for s in all_steps)

    if len(query_steps) == 1:
        final_sql = query_steps[0].get("sql") or state.get("last_validated_sql") or state.get("last_sql")
    elif not query_steps:
        final_sql = state.get("last_validated_sql") or state.get("last_sql")
    else:
        final_sql = "\n\n-- STEP BOUNDARY --\n\n".join(
            s.get("sql", "") for s in sorted(query_steps, key=lambda x: x.get("step", 0))
            if s.get("sql")
        ) or None

    return AgentResult(
        answer=state["answer"],
        sql=final_sql,
        result_data=state.get("last_executed_result"),
        executed=executed,
        valid_sql=state.get("valid_sql", False),
        sql_repairs_used=state.get("total_sql_repairs", 0),
        restarts_used=state.get("restart_count", 0),
        logs=state.get("all_run_logs", []),
        all_errors=state.get("all_errors", []),
        tokens=total_tokens,
        task_completed=state.get("task_completed", False),
        answer_based_on_result=state.get("answer_based_on_result", False),
        iterations=state.get("total_iterations", 0),
        total_api_calls=state.get("total_api_calls", 0),
    )








async def pipeline(
    question: str,
    model_id: str,
    params: ModelParams,
    prompt: str,
    attempt_number: int = 0,
) -> PipelineResult:
    t0 = time.time()

    result = await agent(
        user_input=question,
        model_id=model_id,
        params=params,
        prompt=prompt,
        attempt_number=attempt_number,
    )

    return PipelineResult(
        model_id=model_id,
        final_answer=result.answer,
        final_sql=result.sql,
        result_data=result.result_data,
        valid_sql=result.valid_sql,
        executed=result.executed,
        iterations=result.iterations,
        sql_repairs_used=result.sql_repairs_used,
        restarts_used=result.restarts_used,
        latency_s=round(time.time() - t0, 3),
        all_errors=_dedup(result.all_errors),
        params=params,
        tokens=result.tokens,
        run_logs=result.logs,
        task_completed=result.task_completed,
        answer_based_on_result=result.answer_based_on_result,
        total_api_calls=result.total_api_calls,
    )





async def _run_one_with_retry(
    user_input: str,
    model_id: str,
    params: ModelParams,
    prompt: str,
    sem: asyncio.Semaphore,
    retries: int = 0,
) -> PipelineResult | AgentRunError:
    async with sem:
        delay = 30
        failed_429_calls = 0
        accumulated_snapshot: dict = {
            "model_id": model_id,
            "iterations": 0,
            "restart_count": 0,
            "sql_repairs_used": 0,
            "total_tool_calls": 0,
            "tokens": _empty_token_dict(),
            "all_errors": [],
            "final_sql": None,
            "run_logs": [],
            "discovered_databases": [],
            "discovered_tables": [],
            "last_validation_error": None,
            "last_execute_error": None,
            "total_api_calls": 0,
        }

        for attempt in range(retries + 1):
            try:
                result = await pipeline(
                    user_input, model_id,
                    params=params, prompt=prompt,
                    attempt_number=attempt,
                )
                result.total_api_calls += failed_429_calls
                return result

            except Exception as e:
                current_snapshot = getattr(e, "snapshot", None) or {
                    "model_id": model_id,
                    "all_errors": [str(e)],
                    "total_api_calls": 0,
                }
                accumulated_snapshot = _accumulate_retry_snapshots(
                    accumulated_snapshot, current_snapshot
                )

                if _is_rate_limit_error(e) and attempt < retries:
                    failed_429_calls += 1
                    accumulated_snapshot["total_api_calls"] = max(
                        accumulated_snapshot.get("total_api_calls", 0),
                        current_snapshot.get("total_api_calls", 0) + failed_429_calls,
                    )
                    accumulated_snapshot["all_errors"] = _dedup(
                        list(accumulated_snapshot.get("all_errors", [])) + [str(e)]
                    )
                    logger.warning(
                        "[RETRY][%s] 429 detected — sleeping %ss (attempt %s/%s)...",
                        model_id, delay, attempt + 1, retries + 1,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, _MAX_RETRY_DELAY_S)
                    continue

                accumulated_snapshot["total_api_calls"] = max(
                    accumulated_snapshot.get("total_api_calls", 0),
                    current_snapshot.get("total_api_calls", 0) + failed_429_calls,
                )
                accumulated_snapshot["all_errors"] = _dedup(
                    list(accumulated_snapshot.get("all_errors", [])) + [str(e)]
                )
                return AgentRunError(str(e), snapshot=accumulated_snapshot)








async def run_all(
    user_input: str,
    model_ids: list[str],
    params: ModelParams,
    prompt: str,
) -> list:
    results = []
    sem = asyncio.Semaphore(1)

    for i, model_id in enumerate(model_ids, start=1):
        logger.info("\n" + "=" * 120)
        logger.info("RUNNING MODEL %s/%s: %s", i, len(model_ids), model_id)
        logger.info("=" * 120)

        result = await _run_one_with_retry(
            user_input=user_input,
            model_id=model_id,
            params=params,
            prompt=prompt,
            sem=sem,
        )
        results.append(result)

        logger.info("\n" + "#" * 120)
        logger.info("FINAL MODEL REPORT: %s", model_id)
        logger.info("#" * 120)

        if isinstance(result, Exception):
            logger.error("[ERROR][%s] %s", model_id, result)
        else:
            _print_details([result])

    return results




# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

async def main():
    
    
    setup_logging()
    logger.setLevel(logging.DEBUG)

    queries = [
        "how many buildings are in grimstad",
    ]

    model_ids = [

      "gpt-5-mini"
    ]

    params = ModelParams(
        temperature=0.0,
        max_tokens=2000,
        timeout=300,
        max_retries=0,
        max_iterations=50,
        max_sql_repairs=3,
        max_restarts=2,
    )

    prompt_name = "prompt_v1"
    prompt = load_prompt(prompt_name)

    try:
        for q_idx, query in enumerate(queries, start=1):
            logger.info("=" * 120)
            logger.info("QUERY %s/%s: %s", q_idx, len(queries), query)
            logger.info("=" * 120)

            run = EvalRun(
                query=query,
                model_ids=model_ids,
                prompt_name=prompt_name,
                params=params,
            )

            results = await run_all(
                user_input=query,
                model_ids=model_ids,
                params=params,
                prompt=prompt,
            )

            await store_run(run, results)
            _print_summary(query, results)

    finally:
        await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())