import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from agent_models import TokenUsage, AgentState



import logging

logger = logging.getLogger(__name__)


_PLAN_ACTIONS = (
    "spatial_join",
    "attribute_join",
    "select_columns",
    "transform_crs",
    "intersection",
    "difference",
    "aggregate",
    "dissolve",
    "centroid",
    "distance",
    "nearest",
    "buffer",
    "filter",
    "query",
    "sort",
    "limit",
    "clip",
    "union",
)


def load_prompt(name: str) -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.txt"
    if not path.exists():
        raise ValueError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")







def _short(s: str, n: int = 120) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _dedup(xs: list[str]) -> list[str]:
    seen, out = set(), []
    for x in xs or []:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _empty_token_dict() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _add_tokens(total: dict, inc: TokenUsage) -> dict:
    total["prompt_tokens"] += inc.prompt_tokens
    total["completion_tokens"] += inc.completion_tokens
    total["total_tokens"] += inc.total_tokens
    return total


def _extract_tokens(ai) -> TokenUsage:
    usage = getattr(ai, "usage_metadata", None) or getattr(
        ai, "response_metadata", {}
    ).get("token_usage", {})
    if isinstance(usage, dict):
        pt = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        ct = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        tt = usage.get("total_tokens", 0) or (pt + ct)
        return TokenUsage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)
    pt = getattr(usage, "input_tokens", 0)
    ct = getattr(usage, "output_tokens", 0)
    tt = getattr(usage, "total_tokens", 0) or (pt + ct)
    return TokenUsage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)


def _parse_tool_result(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {"executed": False, "error": raw}
    return raw


def _extract_tables_from_available_tables_result(parsed: Any) -> list[str]:
    if isinstance(parsed, dict):
        if isinstance(parsed.get("tables"), list):
            return [t for t in parsed["tables"] if isinstance(t, str)]
        if isinstance(parsed.get("data"), list):
            return [t for t in parsed["data"] if isinstance(t, str)]
    if isinstance(parsed, list):
        return [t for t in parsed if isinstance(t, str)]
    return []


def _extract_databases_from_result(parsed: Any) -> list[str]:
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, str)]
    if isinstance(parsed, dict):
        if isinstance(parsed.get("databases"), list):
            return [x for x in parsed["databases"] if isinstance(x, str)]
        if isinstance(parsed.get("data"), list):
            return [x for x in parsed["data"] if isinstance(x, str)]
    return []


def _format_restart_summary(state: "AgentState") -> str:
    return f"""
Original query:
{state["user_input"]}

Discovered databases:
{state.get("discovered_databases", [])}

Discovered tables:
{state.get("discovered_tables", [])}

Last failed SQL:
{state.get("last_sql")}

Last validation error:
{state.get("last_validation_error")}

Last guard error:
{state.get("last_guard_error")}

Last execution error:
{state.get("last_execute_error")}

Do not repeat the same failed SQL.
Avoid redundant metadata calls if the needed metadata has already been discovered.
""".strip()


def _sorted_step_entries(steps: dict | None) -> list[tuple]:
    return sorted(
        (steps or {}).items(),
        key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]),
    )


def _normalize_step_key(step_key: Any) -> str:
    return str(step_key)


def _normalize_step_mapping(mapping: dict | None) -> dict:
    return {
        _normalize_step_key(k): v
        for k, v in (mapping or {}).items()
    }


def _extract_plan_from_text(text: str | None) -> tuple[dict, set]:
    raw = (text or "").strip()
    if not raw:
        return {}, set()

    lowered_raw = raw.lower()
    plan_headers = (
        "step list",
        "plan:",
        "execution plan",
        "query mode:",
        "my plan",
        "here is my plan",
        "steps:",
        "approach:",
        "following steps",
        "task plan",
        "query plan",
    )
    # Allow implicit plan: multiple "step N" lines even without an explicit header
    has_header = any(header in lowered_raw for header in plan_headers)
    has_implicit_plan = len(re.findall(r"\bstep\s*\d+\b", lowered_raw)) >= 2
    if not has_header and not has_implicit_plan:
        return {}, set()

    _SECTION_HEADERS = (
        "step list", "execution plan", "plan:",
        "steps:", "my plan", "approach:", "following steps", "task plan", "query plan",
    )

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    planned_steps: dict = {}
    in_step_section = False

    for line in lines:
        lower = line.lower()
        if any(header in lower for header in _SECTION_HEADERS):
            in_step_section = True
            continue
        if in_step_section and (
            "limit placement" in lower
            or "data volume estimate" in lower
            or "query mode" in lower
        ):
            continue
        if not in_step_section and not lower.startswith("step "):
            continue

        step_match = re.search(r"\bstep\s*(\d+)\b", lower)
        if not step_match:
            numbered_match = re.match(r"^(\d+)\.\s+", lower)
            if numbered_match:
                step_no = int(numbered_match.group(1))
            else:
                continue
        else:
            step_no = int(step_match.group(1))

        action = None
        for candidate in _PLAN_ACTIONS:
            if candidate in lower:
                action = candidate
                break

        if action is None:
            continue

        planned_steps[step_no] = {
            "step": step_no,
            "action": action,
            "description": line,
        }

    required_step_ids = set(planned_steps.keys())
    return planned_steps, required_step_ids


def _collect_step_stats(run_logs: list | None) -> tuple[dict, dict, dict]:
    step_registry = {}
    sql_repairs_by_step = {}
    step_retries_by_step = {}

    for run in (run_logs or []):
        step_registry.update(_normalize_step_mapping(run.get("step_registry", {})))
        for step_no, count in (run.get("sql_repair_counts") or {}).items():
            sql_repairs_by_step[_normalize_step_key(step_no)] = count
        for step_no, count in (run.get("step_retry_counts") or {}).items():
            step_retries_by_step[_normalize_step_key(step_no)] = count

    return step_registry, sql_repairs_by_step, step_retries_by_step


def _build_step_outcomes(run_logs: list | None) -> list[dict]:
    step_registry, _, _ = _collect_step_stats(run_logs)
    outcomes: list[dict] = []

    for step_no, meta in _sorted_step_entries(step_registry):
        outcomes.append(
            {
                "step": step_no,
                "action": meta.get("action"),
                "validated": meta.get("validated", False),
                "executed": meta.get("executed", False),
                "rows_returned": meta.get("rows_returned"),
                "status": meta.get("status"),
            }
        )

    return outcomes


def _valid_sql_from_steps(
    run_logs: list | None,
    valid_sql_fallback: bool = False,
) -> bool:
    outcomes = _build_step_outcomes(run_logs)
    if not outcomes:
        return valid_sql_fallback

    query_steps = [step for step in outcomes if step.get("action") == "query"]
    if query_steps:
        return all(step.get("validated", False) for step in query_steps)

    return valid_sql_fallback


def _executed_from_steps(
    run_logs: list | None,
    executed_fallback: bool = False,
) -> bool:
    outcomes = _build_step_outcomes(run_logs)
    if not outcomes:
        return executed_fallback
    return any(step.get("executed", False) for step in outcomes)


def _answer_indicates_incomplete_task(answer: str | None) -> bool:
    text = (answer or "").strip().lower()
    if not text:
        return False

    incomplete_markers = (
        "can proceed to the next step",
        "can proceed to next step",
        "can proceed to the next steps",
        "can proceed to next steps",
        "next step would be",
        "next steps would be",
        "next i would",
        "i can now proceed",
        "i can proceed",
        "we can proceed",
        "the next step is",
        "the next steps are",
        "still need to",
        "need to run",
        "need to query",
        "need to perform",
        "then run",
        "then perform the spatial join",
        "hazard query still",
        "have not yet",
        "haven't yet",
        "not yet run",
        "not yet performed",
    )
    return any(marker in text for marker in incomplete_markers)


def _task_completed_from_steps(
    run_logs: list | None,
    task_completed_fallback: bool = False,
    final_answer: str | None = None,
    required_step_ids: set | list | None = None,
    plan_finalized: bool = False,
) -> bool:
    required_ids = {
        _normalize_step_key(step_id)
        for step_id in (required_step_ids or [])
    }
    outcomes = _build_step_outcomes(run_logs)
    if not outcomes:
        return (
            task_completed_fallback
            and not required_ids
            and not plan_finalized
            and not _answer_indicates_incomplete_task(final_answer)
        )

    if plan_finalized and not required_ids:
        return False

    if not required_ids:
        return False

    outcome_by_step = {
        _normalize_step_key(step.get("step")): step
        for step in outcomes
        if step.get("step") is not None
    }

    if required_ids:
        for step_id in required_ids:
            outcome = outcome_by_step.get(step_id)
            if not outcome:
                return False
            if not outcome.get("executed", False) or outcome.get("status") != "ok":
                return False

    steps_ok = all(step.get("executed", False) and step.get("status") == "ok" for step in outcomes)
    if not steps_ok:
        return False

    if _answer_indicates_incomplete_task(final_answer):
        return False

    return True


def _render_final_sql(
    step_registry: dict | None,
    last_validated_sql: str | None,
    last_sql: str | None,
) -> str | None:
    rendered_steps: list[str] = []

    for _, meta in _sorted_step_entries(step_registry):
        action = meta.get("action")
        step_no = meta.get("step")
        payload = dict(meta.get("step_payload") or {})

        if action == "query" and meta.get("sql"):
            database = (meta.get("database") or "").strip()
            prefix = f"-- STEP {step_no}: query"
            if database:
                prefix += f" [{database}]"
            rendered_steps.append(f"{prefix}\n{meta.get('sql')}")
            continue

        if not action:
            continue

        payload.pop("sql", None)
        payload.pop("validSql", None)
        compact_payload = {
            k: v for k, v in payload.items()
            if v not in (None, "", [], {})
        }
        compact_payload.pop("action", None)
        compact_payload.pop("step", None)

        step_lines = [f"-- STEP {step_no}: {action}"]
        if compact_payload:
            step_lines.extend(
                f"-- {k}: {json.dumps(v, ensure_ascii=False, default=str)}"
                for k, v in compact_payload.items()
            )
        rendered_steps.append("\n".join(step_lines))

    if rendered_steps:
        return "\n\n-- STEP BOUNDARY --\n\n".join(rendered_steps)

    return last_validated_sql or last_sql


def _fallback_answer_from_logs(run_logs: list | None, current_run_log: list | None = None) -> str:
    candidates: list[str] = []

    for entry in current_run_log or []:
        text = (entry.get("ai_content") or "").strip()
        if text:
            candidates.append(text)

    for run in run_logs or []:
        for entry in run.get("iterations", []) or []:
            text = (entry.get("ai_content") or "").strip()
            if text:
                candidates.append(text)

    return candidates[-1] if candidates else ""







def _run_log_entry_from_state(state: "AgentState") -> dict:
    steps = state.get("step_registry", {})
    final_sql = _render_final_sql(
        steps,
        state.get("last_validated_sql"),
        state.get("last_sql"),
    )
    sql_repair_counts = _normalize_step_mapping(state.get("sql_repair_counts", {}))
    step_retry_counts = _normalize_step_mapping(state.get("step_retry_counts", {}))
    normalized_step_registry = _normalize_step_mapping(state.get("step_registry", {}))

    return {
        "attempt_number": state.get("attempt_number", 0),
        "run_index": state.get("restart_count", 0),
        "plan_finalized": state.get("plan_finalized", False),
        "planned_steps": _normalize_step_mapping(state.get("planned_steps", {})),
        "required_step_ids": sorted(
            _normalize_step_key(step_id)
            for step_id in set(state.get("required_step_ids", set()))
        ),
        "retrieval": {
            "scope_active": state.get("retrieval_scope_active", False),
            "top_score": state.get("retrieval_top_score"),
            "candidates": list(state.get("retrieval_candidates", [])),
        },
        "sql_repairs_used": sum(sql_repair_counts.values()),
        "sql_repair_counts": sql_repair_counts,
        "iterations": list(state.get("current_run_log", [])),
        "final_sql": final_sql,
        "validation_error": state.get("last_validation_error"),
        "guard_error": state.get("last_guard_error"),
        "execution_error": state.get("last_execute_error"),
        "discovered_tables": list(state.get("discovered_tables", [])),
        "discovered_databases": list(state.get("discovered_databases", [])),
        "attempt_tokens": dict(state.get("attempt_tokens", _empty_token_dict())),
        "step_registry": normalized_step_registry,
        "step_retry_counts": step_retry_counts,
    }





def _log_entry_key(entry: dict) -> str:
    """Stable content-based key for deduplication."""
    return json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)


def _dedup_logs(logs: list[dict]) -> list[dict]:
    """Deduplicate run logs by full content, not just run_index."""
    seen: set[str] = set()
    out: list[dict] = []
    for log in logs:
        key = _log_entry_key(log)
        if key not in seen:
            seen.add(key)
            out.append(log)
    return out


def _build_snapshot(state: "AgentState", extra_error: str | None = None) -> dict:
    errors = list(state.get("all_errors", []))
    if extra_error:
        errors.append(extra_error)

    run_logs = list(state.get("all_run_logs", []))
    if state.get("current_run_log"):
        current = _run_log_entry_from_state(state)
        if not any(_log_entry_key(r) == _log_entry_key(current) for r in run_logs):
            run_logs.append(current)

    valid_sql = _valid_sql_from_steps(
        run_logs,
        valid_sql_fallback=state.get("valid_sql", False),
    )
    executed = _executed_from_steps(
        run_logs,
        executed_fallback=state.get("last_execute_ok", False),
    )
    task_completed = _task_completed_from_steps(
        run_logs,
        task_completed_fallback=state.get("task_completed", False),
        final_answer=state.get("answer") or _fallback_answer_from_logs(run_logs),
        required_step_ids=set(state.get("required_step_ids", set())),
        plan_finalized=state.get("plan_finalized", False),
    )

    steps = state.get("step_registry", {})
    final_sql = _render_final_sql(
        steps,
        state.get("last_validated_sql"),
        state.get("last_sql"),
    )

    return {
        "model_id": state.get("model_id"),
        "plan_finalized": state.get("plan_finalized", False),
        "planned_steps": _normalize_step_mapping(state.get("planned_steps", {})),
        "required_step_ids": sorted(
            _normalize_step_key(step_id)
            for step_id in set(state.get("required_step_ids", set()))
        ),
        "retrieval": {
            "scope_active": state.get("retrieval_scope_active", False),
            "top_score": state.get("retrieval_top_score"),
            "candidates": list(state.get("retrieval_candidates", [])),
        },
        "valid_sql": valid_sql,
        "executed": executed,
        "task_completed": task_completed,
        "answer_based_on_result": state.get("answer_based_on_result", False),
        "iterations": state.get("total_iterations", 0),
        "restart_count": state.get("restart_count", 0),
        "sql_repairs_used": sum(
            _normalize_step_mapping(state.get("sql_repair_counts", {})).values()
        ),
        "total_tool_calls": state.get("total_tool_calls", 0),
        "tokens": dict(state.get("grand_total_tokens", _empty_token_dict())),
        "all_errors": _dedup(errors),
        "final_sql": final_sql,
        "run_logs": _dedup_logs(run_logs),
        "discovered_databases": list(state.get("discovered_databases", [])),
        "discovered_tables": list(state.get("discovered_tables", [])),
        "last_validation_error": state.get("last_validation_error"),
        "last_guard_error": state.get("last_guard_error"),
        "last_execute_error": state.get("last_execute_error"),
    }






def _accumulate_retry_snapshots(base: dict | None, new: dict | None) -> dict:
    """Accumulate snapshots across separate 429-retry pipeline attempts."""
    base = dict(base or {})
    new = dict(new or {})
    merged = dict(base)

    merged["model_id"] = new.get("model_id") or base.get("model_id")
    merged["iterations"] = base.get("iterations", 0) + new.get("iterations", 0)
    merged["restart_count"] = max(base.get("restart_count", 0), new.get("restart_count", 0))
    merged["sql_repairs_used"] = base.get("sql_repairs_used", 0) + new.get("sql_repairs_used", 0)
    merged["total_tool_calls"] = base.get("total_tool_calls", 0) + new.get("total_tool_calls", 0)

    bt = dict(base.get("tokens", _empty_token_dict()))
    nt = dict(new.get("tokens", _empty_token_dict()))
    merged["tokens"] = {
        "prompt_tokens":     bt["prompt_tokens"]     + nt["prompt_tokens"],
        "completion_tokens": bt["completion_tokens"] + nt["completion_tokens"],
        "total_tokens":      bt["total_tokens"]      + nt["total_tokens"],
    }

    merged["all_errors"] = _dedup(
        list(base.get("all_errors", [])) + list(new.get("all_errors", []))
    )
    merged["run_logs"] = _dedup_logs(
        list(base.get("run_logs", [])) + list(new.get("run_logs", []))
    )
    merged["discovered_databases"] = _dedup(
        list(base.get("discovered_databases", [])) + list(new.get("discovered_databases", []))
    )
    merged["discovered_tables"] = _dedup(
        list(base.get("discovered_tables", [])) + list(new.get("discovered_tables", []))
    )
    base_retrieval = dict(base.get("retrieval", {}))
    new_retrieval = dict(new.get("retrieval", {}))
    merged["retrieval"] = {
        "scope_active": bool(
            new_retrieval.get("scope_active", base_retrieval.get("scope_active", False))
        ),
        "top_score": (
            new_retrieval.get("top_score")
            if new_retrieval.get("top_score") is not None
            else base_retrieval.get("top_score")
        ),
        "candidates": list(new_retrieval.get("candidates") or base_retrieval.get("candidates") or []),
    }
    merged["plan_finalized"] = bool(new.get("plan_finalized", base.get("plan_finalized", False)))
    merged["planned_steps"] = _normalize_step_mapping(
        new.get("planned_steps") or base.get("planned_steps") or {}
    )
    merged["required_step_ids"] = sorted(
        {
            _normalize_step_key(step_id)
            for step_id in list(base.get("required_step_ids", [])) + list(new.get("required_step_ids", []))
        }
    )
    merged["final_sql"] = new.get("final_sql") or base.get("final_sql")
    merged["last_validation_error"] = (
        new.get("last_validation_error") or base.get("last_validation_error")
    )
    merged["last_guard_error"] = (
        new.get("last_guard_error") or base.get("last_guard_error")
    )
    merged["last_execute_error"] = (
        new.get("last_execute_error") or base.get("last_execute_error")
    )
    return merged


def _is_rate_limit_error(e: Exception) -> bool:
    try:
        import openai
        if isinstance(e, openai.RateLimitError):
            return True
    except ImportError:
        pass
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg






def _validated_sql_key(run_id: str | None, step_number: int | None) -> str | None:
    if not run_id or step_number is None:
        return None
    return f"{run_id}:{step_number}"


def _coerce_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return {}





def _increment_step_retry(state: AgentState, step_number: int) -> None:
    """Increment per-step retry counter. No-op if step_number is None."""
    if step_number is None:
        return
    counts = dict(state.get("step_retry_counts", {}))
    step_key = _normalize_step_key(step_number)
    counts[step_key] = counts.get(step_key, 0) + 1
    state["step_retry_counts"] = counts
 
 


def _increment_sql_repair(state: AgentState, step_number: int) -> None:
    if step_number is None:
        return
    counts = dict(state.get("sql_repair_counts", {}))
    step_key = _normalize_step_key(step_number)
    counts[step_key] = counts.get(step_key, 0) + 1
    state["sql_repair_counts"] = counts


def _sql_repair_ok(state: AgentState, step_number: int, max_repairs: int) -> bool:
    if step_number is None:
        return False
    step_key = _normalize_step_key(step_number)
    return state.get("sql_repair_counts", {}).get(step_key, 0) < max_repairs





def _step_retry_ok(state: AgentState, step_number: int, max_retries: int = 3) -> bool:
    """True if this step still has retry budget remaining."""
    if step_number is None:
        return False
    step_key = _normalize_step_key(step_number)
    return state.get("step_retry_counts", {}).get(step_key, 0) < max_retries
 










def _extract_from_database_catalog(parsed: Any) -> tuple[list[str], list[str]]:
    dbs = []
    tables = []

    if not isinstance(parsed, dict):
        return dbs, tables

    for db in parsed.get("databases", []):
        db_name = db.get("database")
        if db_name:
            dbs.append(db_name)

        for schema in db.get("schemas", []):
            for table in schema.get("tables", []):
                table_name = table.get("table")
                if table_name:
                    tables.append(table_name)

    return dbs, tables
