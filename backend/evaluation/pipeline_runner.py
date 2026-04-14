import asyncio
import logging
from pathlib import Path
import random
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
    _executed_from_steps,
    _is_rate_limit_error,
    _render_final_sql,
    _task_completed_from_steps,
    _valid_sql_from_steps,
)
from agent_graph import get_agent_graph
from reporting import extract_agent_trace, _print_summary, _print_details, store_run
from backend.logging_utils import setup_logging
from backend.mcp.client.client import mcp_client
from backend.mcp.client.tools_discovery import prefetch_databases
from backend.embeddings.retrieve_layers import (
    build_retrieval_context,
    list_candidate_table_names,
    retrieve_catalog_candidates,
)
from config import EMBEDDINGS_MIN_SCORE, EMBEDDINGS_TOP_K, EMBEDDINGS_TABLE_INDEX_PATH



from agent_helpers import load_prompt


_MAX_RETRY_DELAY_S = 120




logger = logging.getLogger(__name__)


def _candidate_database_order(candidates: list[dict]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in candidates:
        database = str(item.get("database") or "").strip()
        if not database or database in seen:
            continue
        seen.add(database)
        ordered.append(database)
    return ordered


async def agent(
    user_input: str,
    model_id: str,
    params: ModelParams,
    prompt: str,
    attempt_number: int = 0,
) -> AgentResult:
    retrieval_candidates: list[dict] = []
    retrieval_context = ""
    retrieval_scope_active = False
    retrieval_top_score: float | None = None

    try:
        retrieval_candidates = retrieve_catalog_candidates(
            user_input,
            k=EMBEDDINGS_TOP_K,
            index_path=EMBEDDINGS_TABLE_INDEX_PATH,
        )
        top_score = retrieval_candidates[0]["score"] if retrieval_candidates else -1.0
        retrieval_top_score = top_score if retrieval_candidates else None
        retrieval_scope_active = bool(retrieval_candidates) and top_score >= EMBEDDINGS_MIN_SCORE
        retrieval_context = build_retrieval_context(retrieval_candidates)
        if retrieval_context and not retrieval_scope_active:
            retrieval_context += (
                "\nFallback note: retrieval confidence is low, so broader discovery tools remain available."
            )
        logger.info(
            "[RETRIEVAL][%s] candidates=%s scope_active=%s top_score=%.3f",
            model_id,
            [c.get("full_name") for c in retrieval_candidates],
            retrieval_scope_active,
            top_score,
        )
    except Exception as e:
        logger.warning("[RETRIEVAL][%s] failed: %s", model_id, e)

    seeded_databases = sorted(
        {str(item.get("database") or "").strip() for item in retrieval_candidates if item.get("database")}
    )
    seeded_tables = list_candidate_table_names(retrieval_candidates)
    preferred_databases = _candidate_database_order(retrieval_candidates)

    if preferred_databases:
        try:
            prefetch_result = await prefetch_databases(preferred_databases)
            logger.info(
                "[RETRIEVAL][%s] prefetched_databases=%s",
                model_id,
                prefetch_result.get("opened", preferred_databases) if isinstance(prefetch_result, dict) else preferred_databases,
            )
        except Exception as e:
            logger.warning("[RETRIEVAL][%s] database prefetch failed: %s", model_id, e)

    msgs = [SystemMessage(content=prompt)]
    if retrieval_context:
        msgs.append(SystemMessage(content=retrieval_context))
    msgs.append(HumanMessage(content=user_input))

    init_state: AgentState = {
        "user_input": user_input,
        "model_id": model_id,
        "prompt": prompt,
        "params": params,
        "msgs": msgs,
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
        "last_guard_error": None,
        "last_executed_result": None,
        "last_execute_ok": False,
        "last_execute_error": None,
        "discovered_tables": seeded_tables,
        "discovered_databases": seeded_databases,
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
        "retrieval_candidates": retrieval_candidates,
        "retrieval_scope_active": retrieval_scope_active,
        "retrieval_top_score": retrieval_top_score,


        "sql_repair_counts": {},

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
    run_logs = state.get("all_run_logs", [])
    executed = _executed_from_steps(
        run_logs,
        executed_fallback=any(s.get("executed", False) for s in steps.values()),
    )
    valid_sql = _valid_sql_from_steps(
        run_logs,
        valid_sql_fallback=state.get("valid_sql", False),
    )
    task_completed = _task_completed_from_steps(
        run_logs,
        task_completed_fallback=state.get("task_completed", False),
    )
    final_sql = _render_final_sql(
        steps,
        state.get("last_validated_sql"),
        state.get("last_sql"),
    )

    return AgentResult(
        answer=state["answer"],
        sql=final_sql,
        result_data=state.get("last_executed_result"),
        executed=executed,
        valid_sql=valid_sql,
        sql_repairs_used=state.get("total_sql_repairs", 0),
        restarts_used=state.get("restart_count", 0),
        logs=run_logs,
        all_errors=state.get("all_errors", []),
        tokens=total_tokens,
        task_completed=task_completed,
        answer_based_on_result=state.get("answer_based_on_result", False),
        iterations=state.get("total_iterations", 0),
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
            "retrieval": {
                "scope_active": False,
                "top_score": None,
                "candidates": [],
            },
            "last_validation_error": None,
            "last_guard_error": None,
            "last_execute_error": None,
        }

        for attempt in range(retries + 1):
            try:
                result = await pipeline(
                    model_id=model_id,
                    question= user_input,
                    params=params, prompt=prompt,
                    attempt_number=attempt,
                )
                return result

            except Exception as e:
                current_snapshot = getattr(e, "snapshot", None) or {
                    "model_id": model_id,
                    "all_errors": [str(e)],
                }
                accumulated_snapshot = _accumulate_retry_snapshots(
                    accumulated_snapshot, current_snapshot
                )

                if _is_rate_limit_error(e) and attempt < retries:
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
  
        "Count buildings located in flood hazard areas in Kristiansand.",
    ]

    model_ids = ["gpt-5-mini", "Mistral-Large-3"]

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

            if q_idx < len(queries):
                wait = random.randint(30, 40)
                logger.info("Waiting %ss before next query...", wait)
                await asyncio.sleep(wait)

    finally:
        await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())
