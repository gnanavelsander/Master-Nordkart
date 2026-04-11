import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from agent_models import TokenUsage, AgentState



import logging

logger = logging.getLogger(__name__)


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

Last execution error:
{state.get("last_execute_error")}

Do not repeat the same failed SQL.
Avoid redundant metadata calls if the needed metadata has already been discovered.
""".strip()







def _run_log_entry_from_state(state: "AgentState") -> dict:
    steps = state.get("step_registry", {})

    final_sql = [
        s.get("sql")
        for _, s in sorted(steps.items(), key=lambda x: int(x[0]))
        if s.get("action") == "query" and s.get("sql")
    ]

    if not final_sql:
        final_sql = state.get("last_validated_sql") or state.get("last_sql")

    return {
        "attempt_number": state.get("attempt_number", 0),
        "run_index": state.get("restart_count", 0),
        "sql_repairs_used": sum(state.get("sql_repair_counts", {}).values()),
        "sql_repair_counts": dict(state.get("sql_repair_counts", {})),
        "iterations": list(state.get("current_run_log", [])),
        "final_sql": final_sql,
        "validation_error": state.get("last_validation_error"),
        "execution_error": state.get("last_execute_error"),
        "discovered_tables": list(state.get("discovered_tables", [])),
        "discovered_databases": list(state.get("discovered_databases", [])),
        "attempt_tokens": dict(state.get("attempt_tokens", _empty_token_dict())),
        "step_registry": dict(state.get("step_registry", {})),
        "step_retry_counts": dict(state.get("step_retry_counts", {})),
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

    steps = state.get("step_registry", {})
    final_sql = [
        s.get("sql")
        for _, s in sorted(steps.items(), key=lambda x: int(x[0]))
        if s.get("action") == "query" and s.get("sql")
    ]
    if not final_sql:
        final_sql = state.get("last_validated_sql") or state.get("last_sql")

    return {
        "model_id": state.get("model_id"),
        "iterations": state.get("total_iterations", 0),
        "restart_count": state.get("restart_count", 0),
        "sql_repairs_used": state.get("total_sql_repairs", 0),
        "total_tool_calls": state.get("total_tool_calls", 0),
        "tokens": dict(state.get("grand_total_tokens", _empty_token_dict())),
        "all_errors": _dedup(errors),
        "final_sql": final_sql,
        "run_logs": _dedup_logs(run_logs),
        "discovered_databases": list(state.get("discovered_databases", [])),
        "discovered_tables": list(state.get("discovered_tables", [])),
        "last_validation_error": state.get("last_validation_error"),
        "last_execute_error": state.get("last_execute_error"),
        "total_api_calls": state.get("total_api_calls", 0),
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
    merged["total_api_calls"] = base.get("total_api_calls", 0) + new.get("total_api_calls", 0)

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
    merged["final_sql"] = new.get("final_sql") or base.get("final_sql")
    merged["last_validation_error"] = (
        new.get("last_validation_error") or base.get("last_validation_error")
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
    counts[step_number] = counts.get(step_number, 0) + 1
    state["step_retry_counts"] = counts
 
 


def _increment_sql_repair(state: AgentState, step_number: int) -> None:
    if step_number is None:
        return
    counts = dict(state.get("sql_repair_counts", {}))
    counts[step_number] = counts.get(step_number, 0) + 1
    state["sql_repair_counts"] = counts


def _sql_repair_ok(state: AgentState, step_number: int, max_repairs: int) -> bool:
    if step_number is None:
        return False
    return state.get("sql_repair_counts", {}).get(step_number, 0) < max_repairs





def _step_retry_ok(state: AgentState, step_number: int, max_retries: int = 3) -> bool:
    """True if this step still has retry budget remaining."""
    if step_number is None:
        return False
    return state.get("step_retry_counts", {}).get(step_number, 0) < max_retries
 










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

