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
    _dedup_logs,
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
from backend.mcp.client.tools_steps import get_step_result
from backend.embeddings.retrieve_layers import (
    build_retrieval_context,
    retrieve_database_candidates,
    load_index,
)
from config import EMBEDDINGS_MIN_SCORE, EMBEDDINGS_TOP_K, EMBEDDINGS_TABLE_INDEX_PATH



from agent_helpers import load_prompt


_MAX_RETRY_DELAY_S = 120




logger = logging.getLogger(__name__)


async def _expand_result_for_frontend(result_data: dict | None) -> dict | None:
    if not isinstance(result_data, dict):
        return result_data

    run_id = result_data.get("run_id")
    step_no = result_data.get("step")
    if not run_id or step_no is None:
        return result_data

    try:
        full_result = await get_step_result(str(run_id), int(step_no))
    except Exception as e:
        logger.warning("Failed to fetch full step result run_id=%s step=%s: %s", run_id, step_no, e)
        return result_data

    if isinstance(full_result, dict) and "error" not in full_result:
        merged = dict(result_data)
        merged.update(full_result)
        return merged

    return result_data


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
    aoi_context: str | None = None,
    history_context: str | None = None,
) -> AgentResult:
    retrieval_candidates: list[dict] = []
    retrieval_context = ""
    retrieval_scope_active = False
    retrieval_top_score: float | None = None

    # --- EMBEDDING RETRIEVAL DISABLED ---
    # Uncomment to re-enable embedding-based database retrieval and prefetch.
    #
    # try:
    #     retrieval_candidates = retrieve_database_candidates(
    #         user_input,
    #         k=EMBEDDINGS_TOP_K,
    #         index_path=EMBEDDINGS_TABLE_INDEX_PATH,
    #     )
    #     top_score = retrieval_candidates[0]["score"] if retrieval_candidates else -1.0
    #     retrieval_top_score = top_score if retrieval_candidates else None
    #     retrieval_scope_active = bool(retrieval_candidates) and top_score >= EMBEDDINGS_MIN_SCORE
    #     retrieval_context = build_retrieval_context(retrieval_candidates)
    #     if retrieval_context and not retrieval_scope_active:
    #         retrieval_context += (
    #             "\nFallback note: retrieval confidence is low, so broader discovery tools remain available."
    #         )
    #     logger.info(
    #         "[RETRIEVAL][%s] candidates=%s scope_active=%s top_score=%.3f",
    #         model_id,
    #         [f"{c.get('database')}={c.get('score', -1):.3f}" for c in retrieval_candidates],
    #         retrieval_scope_active,
    #         top_score,
    #     )
    # except Exception as e:
    #     logger.warning("[RETRIEVAL][%s] failed: %s", model_id, e)
    #
    # preferred_databases = _candidate_database_order(retrieval_candidates)
    # seeded_databases = list(preferred_databases)
    # seeded_tables: list[str] = []
    #
    # if preferred_databases:
    #     try:
    #         prefetch_result = await prefetch_databases(preferred_databases)
    #         logger.info(
    #             "[RETRIEVAL][%s] prefetched_databases=%s",
    #             model_id,
    #             prefetch_result.get("opened", preferred_databases) if isinstance(prefetch_result, dict) else preferred_databases,
    #         )
    #     except Exception as e:
    #         logger.warning("[RETRIEVAL][%s] database prefetch failed: %s", model_id, e)
    # --- END EMBEDDING RETRIEVAL ---

    seeded_databases: list[str] = []
    seeded_tables: list[str] = []

    msgs = [SystemMessage(content=prompt)]
    if retrieval_context:
        msgs.append(SystemMessage(content=retrieval_context))
    if history_context:
        msgs.append(SystemMessage(content=history_context))
    if aoi_context:
        msgs.append(SystemMessage(content=aoi_context))
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
        "planned_steps": {},
        "required_step_ids": set(),
        "plan_finalized": False,
        "iter_had_error": False,
        "iter_had_validation_error": False,
        "iter_any_exec_ok": False,
        "step_retry_count": 0,
        "total_step_retries": 0,
        "step_retry_counts": {},
        "last_failed_step": None,
        "locked_step_ids": set(),
        "retrieval_candidates": retrieval_candidates,
        "retrieval_scope_active": retrieval_scope_active,
        "retrieval_top_score": retrieval_top_score,


        "sql_repair_counts": {},
        "replan_trigger": None,
        "replan_count": 0,
        "unplanned_step_block_counts": {},
        "tool_shape_error_counts": {},
        "locked_step_block_counts": {},
        "missing_source_step_counts": {},
        "original_planned_steps": {},
        "original_required_step_ids": set(),
        "zero_rows_replan_counts": {},
        "clarification_block_count": 0,
    }

    state = init_state
    expanded_result_data: dict | None = None
    task_completed = False
    try:
        state = await get_agent_graph().ainvoke(init_state)
        task_completed = _task_completed_from_steps(
            state.get("all_run_logs", []),
            task_completed_fallback=state.get("task_completed", False),
            final_answer=state.get("answer"),
            required_step_ids=set(state.get("required_step_ids", set())),
            plan_finalized=state.get("plan_finalized", False),
            planned_steps=dict(state.get("planned_steps", {})),
        )
        # Expand the last executed result BEFORE cleanup_run deletes the step data.
        # _expand_result_for_frontend fetches full rows via get_step_result, which
        # would return an error after cleanup_run runs — so this must happen here.
        if task_completed:
            expanded_result_data = await _expand_result_for_frontend(
                state.get("last_executed_result")
            )
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
        task_completed_fallback=task_completed or state.get("task_completed", False),
        final_answer=state.get("answer"),
        required_step_ids=set(state.get("required_step_ids", set())),
        plan_finalized=state.get("plan_finalized", False),
        planned_steps=dict(state.get("planned_steps", {})),
    )
    final_sql = _render_final_sql(
        steps,
        state.get("last_validated_sql"),
        state.get("last_sql"),
    )

    return AgentResult(
        answer=state["answer"],
        sql=final_sql,
        result_data=expanded_result_data,
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
    aoi_context: str | None = None,
    history_context: str | None = None,
) -> PipelineResult:
    t0 = time.time()

    result = await agent(
        user_input=question,
        model_id=model_id,
        params=params,
        prompt=prompt,
        attempt_number=attempt_number,
        aoi_context=aoi_context,
        history_context=history_context,
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
    aoi_context: str | None = None,
    history_context: str | None = None,
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
            "plan_finalized": False,
            "plan_block_count": 0,
            "planned_steps": {},
            "required_step_ids": [],
            "retrieval": {
                "scope_active": False,
                "top_score": None,
                "candidates": [],
            },
            "last_validation_error": None,
            "last_guard_error": None,
            "last_execute_error": None,
        }

        prior_tokens = _empty_token_dict()
        prior_iterations = 0
        prior_run_logs: list = []

        for attempt in range(retries + 1):
            try:
                result = await pipeline(
                    model_id=model_id,
                    question=user_input,
                    params=params,
                    prompt=prompt,
                    attempt_number=attempt,
                    aoi_context=aoi_context,
                    history_context=history_context,
                )
                # Carry forward cost data from failed 429 attempts
                if attempt > 0:
                    if result.tokens:
                        result.tokens = TokenUsage(
                            prompt_tokens=result.tokens.prompt_tokens + prior_tokens["prompt_tokens"],
                            completion_tokens=result.tokens.completion_tokens + prior_tokens["completion_tokens"],
                            total_tokens=result.tokens.total_tokens + prior_tokens["total_tokens"],
                        )
                    result.iterations += prior_iterations
                    if prior_run_logs and result.run_logs is not None:
                        result.run_logs = prior_run_logs + result.run_logs
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
                    # Save cost data from this failed attempt before retrying
                    snap_tokens = current_snapshot.get("tokens") or _empty_token_dict()
                    prior_tokens = {
                        k: prior_tokens[k] + snap_tokens.get(k, 0)
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                    }
                    prior_iterations += current_snapshot.get("iterations", 0)
                    prior_run_logs = _dedup_logs(
                        prior_run_logs + list(current_snapshot.get("run_logs", []))
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

                accumulated_snapshot["all_errors"] = _dedup(
                    list(accumulated_snapshot.get("all_errors", [])) + [str(e)]
                )
                return AgentRunError(str(e), snapshot=accumulated_snapshot)








async def run_all(
    user_input: str,
    model_ids: list[str],
    params: ModelParams,
    prompt: str,
    aoi_context: str | None = None,
    history_context: str | None = None,
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
            retries=params.max_retries,
            aoi_context=aoi_context,
            history_context=history_context,
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

 
    queries = {

       # "B9": "Find 10 buildings located in built-up and transport areas in Oslo municipality, sorted by bygningid ascending.",
        #"B10": "Find 10 parcels located in forest areas in Kristiansand municipality, sorted by teigid ascending.",

        # Level 2: single-hazard overlays, touch/contain predicates, and proximity
        #"B11": "Find 10 buildings located in flood hazard areas in Bergen municipality, sorted by bygningid ascending.",
        #"B12": "Find 10 buildings located in soil- and flood-landslide hazard areas in Stavanger municipality, sorted by bygningid ascending.",
        #"B13": "Find 10 buildings located in quick-clay hazard areas in Trondheim municipality, sorted by bygningid ascending.",
        #"B14": "Find 10 buildings located in snow-avalanche hazard areas in Narvik municipality, sorted by bygningid ascending.",
        #"B15": "Find 10 parcels that touch flood zones in Drammen municipality, sorted by teigid ascending.", problemer med ram
        # "B16": "Find 10 parcels within 100 metres of flood hazard areas in Kristiansand municipality, sorted by shortest distance first.",
        # "B17": "Find 10 buildings within 100 metres of soil- and flood-landslide hazard areas in Oslo municipality, sorted by shortest distance first.",
        # "B18": "Find 10 parcels within 100 metres of quick-clay hazard areas in Trondheim municipality, sorted by shortest distance first.",
        # "B19": "Find 10 buildings that intersect a 100 metre buffer around flood hazard areas in Oslo municipality, sorted by bygningid ascending.",
        # "B20": "Find 10 parcels in Bergen municipality that contain storm-surge polygons, sorted by teigid ascending.",

        # # Level 3: counts, set operations, and moderate multi-step spatial filtering
        # "B21": "Count all buildings in flood hazard areas in Kristiansand municipality.",
        # "B22": "Count all parcels in flood hazard areas in Oslo municipality.",
        # "B23": "Count all buildings in quick-clay hazard areas in Trondheim municipality.",
        # "B24": "Count all parcels located in agricultural land in Stavanger municipality.",
        # "B25": "Count all buildings located in forest areas in Bergen municipality.",
        "B26": "Count all buildings located in built-up and transport areas in Oslo municipality.",
        "B27": "Count all buildings in snow-avalanche hazard areas in Narvik municipality.",
        "B28": "Find 10 parcels located in quick-clay hazard areas with high risk class in Trondheim municipality, sorted by teigid ascending.",
        "B29": "Find 10 buildings in Drammen municipality that intersect the intersection of flood hazard areas and the 200-year flood zone, sorted by bygningid ascending.",
        #"B30": "Find 10 parcels in Bergen municipality that intersect the union of flood hazard areas and storm-surge areas, sorted by teigid ascending.",

        # Level 4: dual-condition queries and county/municipality aggregations
        #"B31": "Find 10 buildings in Kristiansand municipality that are located both in flood hazard areas and on agricultural land, sorted by bygningid ascending.",
        # "B32": "Find 10 buildings in Oslo municipality that are exposed to both flood hazard areas and soil- and flood-landslide hazard areas, sorted by bygningid ascending.",
        # "B33": "Find 10 buildings in Stavanger municipality that are located on agricultural land and within soil- and flood-landslide hazard areas, sorted by bygningid ascending.",
        #"B34": "Find 10 parcels in Trondheim municipality that are located in both flood hazard areas and quick-clay hazard areas, sorted by teigid ascending.",
        "B35": "Find 10 buildings in Bergen municipality that are exposed to both storm-surge areas and flood hazard areas, sorted by bygningid ascending.",
        "B36": "Find 10 parcels in Kristiansand municipality that are located on agricultural land and within flood hazard areas, sorted by teigid ascending.",
        "B37": "Find 10 buildings in Bergen municipality that are located in storm-surge areas, sorted by bygningid ascending, and report both municipality and county.",
        #"B38": "List the five municipalities in Vestland county with the highest number of buildings in flood hazard areas, sorted by count descending.",                                               
        "B39": "Provide a municipality-level overview of buildings located in quick-clay hazard areas in Trøndelag county.",
        #"B40": "Provide a municipality-level summary of parcels in Agder county that are located in agricultural land and also overlap flood hazard areas.",

        # Level 5: advanced overlay logic, grid/clipping/dissolve work, and multi-hazard summaries
        "B41": "Provide a municipality-level comparison of buildings exposed to flood hazard areas, quick-clay hazard areas, and soil- and flood-landslide hazard areas in Agder county.",
        "B42": "Find 10 buildings in Drammen municipality that are within the intersection of flood hazard areas and the 200-year flood zone, sorted by bygningid ascending.",
        "B43": "Find 10 parcels in Stavanger municipality that are located on agricultural land and overlap flood hazard areas but not soil- and flood-landslide hazard areas, sorted by teigid ascending.",
        #"B44": "Provide a municipality-level summary of parcels intersecting dissolved storm-surge areas in Nordland county.",
        "B45": "Provide a 1000 m grid-based summary of buildings exposed to flood hazard areas in Oslo municipality, using a common CRS for all layers, and highlight the grid cells with the highest counts.",
        "B46": "Classify municipalities in Vestland county based on the number of buildings exposed to flood hazard areas, quick-clay hazard areas, and storm-surge areas.",
        "B47": "Find 10 buildings in Trondheim municipality that are exposed to flood hazard areas, quick-clay hazard areas, and soil- and flood-landslide hazard areas at the same time, sorted by bygningid ascending.",
        #"B48": "Find 10 buildings in Bergen municipality that are contained in both storm-surge areas and the 200-year flood zone, sorted by bygningid ascending.",
        "B49": "Summarize buildings in Innlandet county that are located in forest areas and in flood hazard areas clipped to the county boundary.",
        "B50": "Summarize which municipalities in Rogaland county have the greatest concentration of parcels exposed to both flood hazard areas and storm-surge areas.",
    }

    
    model_ids = ["gpt-5-mini"]

    params = ModelParams(
        temperature=0.0,
        max_tokens=2000,
        timeout=300,
        max_retries=0,
        max_iterations=30,
        max_sql_repairs=2,
        max_step_retries=1,
        max_replans=1,
    )

    prompt_name = "evaluation_v1"
    prompt = load_prompt(prompt_name)

    try:
        for q_idx, (q_id, query) in enumerate(queries.items(), start=1):
            logger.info("=" * 120)
            logger.info("QUERY %s/%s [%s]: %s", q_idx, len(queries), q_id, query)
            logger.info("=" * 120)

            run = EvalRun(
                query=query,
                model_ids=model_ids,
                prompt_name=prompt_name,
                params=params,
                query_id=q_id,
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
        load_index.cache_clear()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
