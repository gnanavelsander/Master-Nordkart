import sys
import time
import json
import uuid
import asyncio
import logging

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Any, TypedDict

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

from typing import Literal
from pydantic import BaseModel















class CleanFormatter(logging.Formatter):
    def format(self, record):
        name = record.name
        msg = record.getMessage()

        if name == "__main__":
            if "Starting MCP server" in msg:
                return "      [SYSTEM] starting MCP server"
            if "Server ready, yielding app context" in msg:
                return "      [SYSTEM] server ready"
            return msg

        if name == "backend.database.database":
            if "Pool created for conn_id=" in msg:
                conn_id = msg.split("conn_id=")[-1]
                return f"      [DB] pool created : {conn_id}"
            return f"      [DB] {msg}"

        if name == "backend.mcp_c_s.mcp_client":
            return f"      [MCP CLIENT] {msg}"

        if name == "mcp.server.lowlevel.server":
            if "Processing request of type CallToolRequest" in msg:
                return "      [MCP] executing tool request"
            if "Processing request of type ListToolsRequest" in msg:
                return "      [MCP] listing tools"
            return f"      [MCP] {msg}"

        if name == "httpx":
            return "      [HTTPX] request sent ok" if "HTTP Request:" in msg else f"      [HTTPX] {msg}"

        if name == "openai._base_client":
            if "Request options:" in msg:
                return "      [OPENAI] request options prepared"
            if "Sending HTTP Request:" in msg:
                return "      [OPENAI] sending request"
            if "HTTP Response:" in msg:
                return "      [OPENAI] response received"
            if "request_id:" in msg:
                return f"      [OPENAI] {msg}"
            return f"      [OPENAI] {msg}"

        if name == "httpcore.connection":
            if "connect_tcp.started" in msg:
                return "      [HTTPCORE] tcp connect started"
            if "connect_tcp.complete" in msg:
                return "      [HTTPCORE] tcp connect complete"
            if "start_tls.started" in msg:
                return "      [HTTPCORE] tls start"
            if "start_tls.complete" in msg:
                return "      [HTTPCORE] tls ready"
            if "close.started" in msg:
                return "      [HTTPCORE] close started"
            if "close.complete" in msg:
                return "      [HTTPCORE] close complete"
            return f"      [HTTPCORE] {msg}"

        if name == "httpcore.http11":
            if "send_request_headers.started" in msg:
                return "      [HTTP11] sending headers"
            if "send_request_headers.complete" in msg:
                return "      [HTTP11] headers sent"
            if "send_request_body.started" in msg:
                return "      [HTTP11] sending body"
            if "send_request_body.complete" in msg:
                return "      [HTTP11] body sent"
            if "receive_response_headers.started" in msg:
                return "      [HTTP11] waiting for headers"
            if "receive_response_headers.complete" in msg:
                return "      [HTTP11] headers received"
            if "receive_response_body.started" in msg:
                return "      [HTTP11] receiving body"
            if "receive_response_body.complete" in msg:
                return "      [HTTP11] body received"
            if "response_closed.started" in msg:
                return "      [HTTP11] closing response"
            if "response_closed.complete" in msg:
                return "      [HTTP11] response closed"
            return f"      [HTTP11] {msg}"

        return f"      [{name}] {msg}"


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CleanFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    for name in [
        "__main__",
        "backend.database.database",
        "backend.mcp_c_s.mcp_client",
        "mcp.server.lowlevel.server",
        "httpx",
        "openai._base_client",
        "httpcore.connection",
        "httpcore.http11",
    ]:
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(logging.DEBUG)



logging.getLogger("azure").setLevel(logging.WARNING)



ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import (
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_FOUNDRY_ENDPOINT,
    AZURE_MODEL_DEPLOYMENT,
    AZURE_OPENAI_MODEL_IDS,
    AZURE_FOUNDRY_MODEL_IDS,
    OLLAMA_MODEL_IDS,
)

from backend.tools.longChainTools import (
    tool_validate_sql,
    tool_get_available_tables,
    tool_get_table_schema,
    kommune_where_clause,
    tool_get_available_databases,
    tool_get_available_schemas,

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
    tool_get_database_catalog,
)
from AzureBlobStorage import (
    upload,
    query_hash,
    query_meta_exists,
    _update_queries_index,
    _update_runs_index,
)

from backend.mcp_c_s.mcp_client import mcp_client

# ─────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class AgentTrace:
    tables_explored: list[str]
    tables_in_result: list[str]
    databases_explored: list[str]
    databases_in_result: list[str]
    validated_before_exec: bool
    executed_successfully: bool
    total_tool_calls: int
    executions: list[dict]
    tool_call_counts: dict[str, int]


@dataclass
class ModelParams:
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 60
    max_retries: int = 2
    tool_choice: str = "auto"
    max_iterations: int = 12
    max_sql_repairs: int = 3
    max_restarts: int = 1
    top_p: float | None = None
    reasoning_effort: str | None = None

    max_step_retries: int = 3


@dataclass
class AgentResult:
    answer: str
    sql: str | None
    result_data: Any | None
    executed: bool
    valid_sql: bool
    iterations: int
    sql_repairs_used: int
    restarts_used: int
    logs: list
    all_errors: list[str]
    tokens: TokenUsage | None = None
    task_completed: bool = False
    answer_based_on_result: bool = False
    total_api_calls: int = 0


@dataclass
class PipelineResult:
    model_id: str
    final_answer: str | None
    final_sql: str | None
    result_data: Any | None
    valid_sql: bool
    executed: bool
    iterations: int
    sql_repairs_used: int
    restarts_used: int
    latency_s: float
    all_errors: list[str]
    params: ModelParams
    tokens: TokenUsage | None = None
    run_logs: list | None = None
    task_completed: bool = False
    answer_based_on_result: bool = False
    total_api_calls: int = 0


@dataclass
class EvalRun:
    query: str
    model_ids: list[str]
    params: ModelParams | None = None
    prompt_name: str = "default"

    def __post_init__(self):
        if self.params is None:
            self.params = ModelParams()







# class QueryMode(BaseModel):
#     query_mode: Literal["single_db", "multi_db"]





# def get_classifier_model(model_id: str, params: ModelParams):
#     base = AzureChatOpenAI(
#         api_key=AZURE_OPENAI_KEY,
#         azure_deployment=AZURE_MODEL_DEPLOYMENT[model_id],
#         azure_endpoint=AZURE_OPENAI_ENDPOINT,
#         api_version="2024-12-01-preview",
#         temperature=0.0,
#         max_tokens=50,
#         timeout=params.timeout,
#         max_retries=params.max_retries,
#     )
#     return base.with_structured_output(QueryMode)
























# ─────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────

def load_prompt(name: str) -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.txt"
    if not path.exists():
        raise ValueError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────








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
        "sql_repairs_used": state.get("sql_repair_count", 0),
        "iterations": list(state.get("current_run_log", [])),
        "final_sql": final_sql,
        "validation_error": state.get("last_validation_error"),
        "execution_error": state.get("last_execute_error"),
        "discovered_tables": list(state.get("discovered_tables", [])),
        "discovered_databases": list(state.get("discovered_databases", [])),
        "attempt_tokens": dict(state.get("attempt_tokens", _empty_token_dict())),
        "step_registry": dict(state.get("step_registry", {})),
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


# ─────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────

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




from typing import Literal, Optional
from pydantic import BaseModel, Field




class PlanStep(BaseModel):
    step: int
    action: Literal[
        "query", "filter", "select_columns", "spatial_join", "attribute_join",
        "buffer", "nearest", "distance", "aggregate", "sort", "limit",
        "clip", "union", "intersection", "difference", "dissolve", "centroid", "transform_crs",
    ]
    database: Optional[str] = None
    sql: Optional[str] = None
    validSql: Optional[str] = None
    source_step: Optional[int] = None
    left_source: Optional[int] = None
    right_source: Optional[int] = None
    join_type: Optional[Literal["intersects", "within", "contains", "dwithin", "touches", "overlaps"]] = None
    left_geom: Optional[str] = None
    right_geom: Optional[str] = None
    distance_meters: Optional[float] = None
    buffer_distance: Optional[float] = None
    filters: list[str] = Field(default_factory=list)
    select_columns: list[str] = Field(default_factory=list)
    output_columns: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    aggregation_type: Optional[Literal["count", "sum", "avg", "min", "max"]] = None
    metric_column: Optional[str] = None
    metric_as: Optional[str] = None
    order_by: Optional[str] = None
    order_direction: Optional[Literal["asc", "desc"]] = None
    limit: Optional[int] = None
    target_crs: Optional[str] = None
    join_keys: list[str] = Field(default_factory=list)






class QueryPlan(BaseModel):
    query_mode: Literal["single_db", "multi_db"]
    steps: list[PlanStep] = Field(default_factory=list)













# ─────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    user_input: str
    model_id: str
    prompt: str
    params: ModelParams
 
    msgs: list
    iterations: int
    max_iterations: int
 
    sql_repair_count: int
    max_sql_repairs: int
 
    restart_count: int
    max_restarts: int
 
    current_run_log: list
    all_run_logs: list
 
    attempt_tokens: dict
    grand_total_tokens: dict
    current_iter_tokens: dict
 
    last_sql: Optional[str]
    last_validated_sql: Optional[str]
    last_validation_ok: bool
    last_validation_error: Optional[str]
 
    last_executed_result: Any        # LLM-facing summary from server (not raw rows)
    last_execute_ok: bool
    last_execute_error: Optional[str]
    last_failed_step: Optional[int]  # set explicitly on any validate/exec failure
 
    discovered_tables: list[str]
    discovered_databases: list[str]
 
    current_ai: Any
    current_ai_is_final_answer: bool
    valid_sql: bool
    answer: str
    all_errors: list[str]
    done: bool
 
    answer_based_on_result: bool
    task_completed: bool
 
    total_sql_repairs: int
    total_iterations: int
    total_tool_calls: int
    total_api_calls: int
    step_retry_count: int            # global counter for non-SQL step failures
    total_step_retries: int
 
    step_retry_counts: dict          # {step_number: int} — per-step retry budget
    step_registry: dict              # {step_number: {sql, database, validated, executed, ...}}
    validated_sqls: dict             # {step_number: {sql, database}} — cleared after exec
 
    iter_had_error: bool
    iter_had_validation_error: bool
    iter_any_exec_ok: bool
 
    plan_locked: bool
    locked_step_ids: set             # step numbers belonging to the locked plan
 
    crash_snapshot: dict
    attempt_number: int
    run_id: str
    query_mode: Optional[Literal["single_db", "multi_db"]]



























def _increment_step_retry(state: AgentState, step_number: int) -> None:
    """Increment per-step retry counter. No-op if step_number is None."""
    if step_number is None:
        return
    counts = dict(state.get("step_retry_counts", {}))
    counts[step_number] = counts.get(step_number, 0) + 1
    state["step_retry_counts"] = counts
 
 
def _step_retry_ok(state: AgentState, step_number: int, max_retries: int = 3) -> bool:
    """True if this step still has retry budget remaining."""
    if step_number is None:
        return False
    return state.get("step_retry_counts", {}).get(step_number, 0) < max_retries
 








# ─────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────

def _get_tools():
    return [
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
    ]

class AgentRunError(Exception):
    def __init__(self, message: str, snapshot: dict | None = None):
        super().__init__(message)
        self.snapshot = snapshot or {}


# ─────────────────────────────────────────────────────────────
# Graph nodes
# ─────────────────────────────────────────────────────────────

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













# async def _tools_node(state: AgentState) -> AgentState:
#     tools = _get_tools()
#     tool_map = {t.name: t for t in tools}

#     step_tools = {
#         "tool_execute_query_step",
#         "tool_execute_filter_step",
#         "tool_execute_select_columns_step",
#         "tool_execute_spatial_join_step",
#         "tool_execute_attribute_join_step",
#         "tool_execute_buffer_step",
#         "tool_execute_nearest_step",
#         "tool_execute_distance_step",
#         "tool_execute_aggregate_step",
#         "tool_execute_sort_step",
#         "tool_execute_limit_step",
#         "tool_execute_clip_step",
#         "tool_execute_union_step",
#         "tool_execute_intersection_step",
#         "tool_execute_difference_step",
#         "tool_execute_dissolve_step",
#         "tool_execute_centroid_step",
#         "tool_execute_transform_crs_step",
#     }

#     ai = state["current_ai"]
#     msgs = list(state["msgs"])
#     iteration_tools = []

#     iter_had_error = False
#     iter_had_validation_error = False
#     iter_any_exec_ok = False

#     def _norm(s: str) -> str:
#         return " ".join((s or "").split())

#     def _append_error(err_msg: str) -> None:
#         state["all_errors"] = _dedup(list(state.get("all_errors", [])) + [err_msg])

#     for i, tc in enumerate(ai.tool_calls, start=1):
#         name = tc["name"]
#         raw_args = dict(tc.get("args") or {})
#         args = dict(raw_args)
#         tool = tool_map[name]
#         state["total_tool_calls"] = state.get("total_tool_calls", 0) + 1

#         try:
#             if name in step_tools:
#                 args["run_id"] = state["run_id"]

#                 if name == "tool_execute_query_step":
#                     step_obj = args.get("step")

#                     if not isinstance(step_obj, dict):
#                         result = json.dumps({
#                             "executed": False,
#                             "error": "tool_execute_query_step blocked.",
#                             "reason": "Missing or invalid step object.",
#                             "step": None,
#                             "action": "query",
#                         })
#                         exec_parsed = _parse_tool_result(result)
#                         exec_ok = False
#                         rows_returned = 0
#                         step_number = None
#                         sql = ""
#                         database = ""
#                     else:
#                         step_obj = dict(step_obj)
#                         sql = (step_obj.get("sql") or "").strip()
#                         step_number = step_obj.get("step")
#                         database = (step_obj.get("database") or "").strip()
#                         action = step_obj.get("action")
#                         valid_sql_flag = step_obj.get("validSql")

#                         if sql:
#                             state["last_sql"] = sql

#                         validated_sqls = dict(state.get("validated_sqls", {}))
#                         validated_entry = validated_sqls.get(step_number)

#                         blocked_reason = None
#                         if action != "query":
#                             blocked_reason = "step.action must be 'query'."
#                         elif valid_sql_flag != "ok":
#                             blocked_reason = "step.validSql must be 'ok'."
#                         elif step_number is None:
#                             blocked_reason = "step.step is required."
#                         elif not database:
#                             blocked_reason = "step.database is required."
#                         elif not sql:
#                             blocked_reason = "step.sql is required."
#                         elif validated_entry is None:
#                             blocked_reason = "No validated SQL found for this step."
#                         elif _norm(validated_entry.get("sql", "")) != _norm(sql):
#                             blocked_reason = "SQL does not match validated entry."
#                         elif validated_entry.get("database", "") != database:
#                             blocked_reason = "Database does not match validated entry."

#                         if blocked_reason:
#                             result = json.dumps({
#                                 "executed": False,
#                                 "error": "tool_execute_query_step blocked.",
#                                 "reason": blocked_reason,
#                                 "step": step_number,
#                                 "action": "query",
#                             })
#                             exec_parsed = _parse_tool_result(result)
#                             exec_ok = False
#                             rows_returned = 0
#                         else:
#                             result = await tool.ainvoke(args)
#                             exec_parsed = _parse_tool_result(result)
#                             exec_ok = isinstance(exec_parsed, dict) and exec_parsed.get("executed", False)
#                             rows_returned = exec_parsed.get("rows_returned", 0) if isinstance(exec_parsed, dict) else 0

#                             if exec_ok:
#                                 validated_sqls.pop(step_number, None)
#                                 state["validated_sqls"] = validated_sqls

#                     if step_number is not None:
#                         step_registry = dict(state.get("step_registry", {}))
#                         prev = dict(step_registry.get(step_number, {}))
#                         prev.update({
#                             "step": step_number,
#                             "action": "query",
#                             "sql": sql,
#                             "database": database,
#                             "validated": prev.get("validated", False),
#                             "executed": exec_ok,
#                             "rows_returned": rows_returned,
#                             "status": "ok" if exec_ok else "error",
#                         })
#                         step_registry[step_number] = prev
#                         state["step_registry"] = step_registry

#                     state["last_executed_result"] = exec_parsed
#                     state["last_execute_error"] = None if exec_ok else json.dumps(exec_parsed, ensure_ascii=False)

#                     if exec_ok:
#                         iter_any_exec_ok = True
#                         state["last_failed_step"] = None
#                     else:
#                         iter_had_error = True
#                         state["last_failed_step"] = step_number
#                         if not (isinstance(exec_parsed, dict) and exec_parsed.get("error") == "tool_execute_query_step blocked."):
#                             _increment_step_retry(state, step_number)
#                             state["sql_repair_count"] = state.get("sql_repair_count", 0) + 1
#                             state["total_sql_repairs"] = state.get("total_sql_repairs", 0) + 1
#                         _append_error(json.dumps(exec_parsed, ensure_ascii=False))

#                 else:
#                     step_obj = args.get("step")

#                     if not isinstance(step_obj, dict):
#                         action_name = name.replace("tool_execute_", "").replace("_step", "")
#                         result = json.dumps({
#                             "executed": False,
#                             "error": f"{name} blocked.",
#                             "reason": "Missing or invalid step object.",
#                             "step": None,
#                             "action": action_name,
#                         })
#                         parsed_result = _parse_tool_result(result)
#                         exec_ok = False
#                         rows_returned = 0
#                         step_number = None
#                         action = None
#                     else:
#                         step_obj = dict(step_obj)
#                         step_number = step_obj.get("step")
#                         action = step_obj.get("action")

#                         blocked_reason = None
#                         if action == "aggregate" and step_obj.get("source_step") is None:
#                             blocked_reason = "aggregate requires source_step."
#                         elif action in {"filter", "select_columns", "sort", "limit", "buffer", "centroid", "transform_crs", "dissolve", "union"} and step_obj.get("source_step") is None:
#                             blocked_reason = f"{action} requires source_step."
#                         elif action in {"spatial_join", "intersection", "difference", "clip", "nearest", "distance"}:
#                             if step_obj.get("left_source") is None or step_obj.get("right_source") is None:
#                                 blocked_reason = f"{action} requires left_source and right_source."
#                         elif action == "attribute_join":
#                             if step_obj.get("left_source") is None or step_obj.get("right_source") is None:
#                                 blocked_reason = "attribute_join requires left_source and right_source."

#                         if blocked_reason:
#                             result = json.dumps({
#                                 "executed": False,
#                                 "error": f"{name} blocked.",
#                                 "reason": blocked_reason,
#                                 "step": step_number,
#                                 "action": action,
#                             })
#                             parsed_result = _parse_tool_result(result)
#                             exec_ok = False
#                             rows_returned = 0
#                         else:
#                             result = await tool.ainvoke(args)
#                             parsed_result = _parse_tool_result(result)
#                             exec_ok = isinstance(parsed_result, dict) and parsed_result.get("executed", False)
#                             rows_returned = parsed_result.get("rows_returned", 0) if isinstance(parsed_result, dict) else 0

#                     if step_number is not None:
#                         step_registry = dict(state.get("step_registry", {}))
#                         prev = dict(step_registry.get(step_number, {}))
#                         prev.update({
#                             "step": step_number,
#                             "action": action,
#                             "executed": exec_ok,
#                             "rows_returned": rows_returned,
#                             "status": "ok" if exec_ok else "error",
#                         })
#                         step_registry[step_number] = prev
#                         state["step_registry"] = step_registry

#                     state["last_executed_result"] = parsed_result
#                     state["last_execute_error"] = None if exec_ok else json.dumps(parsed_result, ensure_ascii=False)

#                     if exec_ok:
#                         iter_any_exec_ok = True
#                         state["last_failed_step"] = None
#                     else:
#                         iter_had_error = True
#                         state["last_failed_step"] = step_number
#                         if not (isinstance(parsed_result, dict) and str(parsed_result.get("error", "")).endswith("blocked.")):
#                             _increment_step_retry(state, step_number)
#                             state["step_retry_count"] = state.get("step_retry_count", 0) + 1
#                             state["total_step_retries"] = state.get("total_step_retries", 0) + 1
#                         _append_error(json.dumps(parsed_result, ensure_ascii=False))

#             else:
#                 result = await tool.ainvoke(args)
#                 parsed = _parse_tool_result(result)

#                 if name == "tool_get_database_catalog":
#                     dbs, tables = _extract_from_database_catalog(parsed)
#                     state["discovered_databases"] = _dedup(list(state.get("discovered_databases", [])) + dbs)
#                     state["discovered_tables"] = _dedup(list(state.get("discovered_tables", [])) + tables)

#                 elif name == "tool_get_available_databases":
#                     dbs = _extract_databases_from_result(parsed)
#                     state["discovered_databases"] = _dedup(list(state.get("discovered_databases", [])) + dbs)

#                 elif name == "tool_get_available_tables":
#                     tables = _extract_tables_from_available_tables_result(parsed)
#                     state["discovered_tables"] = _dedup(list(state.get("discovered_tables", [])) + tables)

#                 elif name == "tool_get_table_schema":
#                     table = args.get("table_name")
#                     if table:
#                         state["discovered_tables"] = _dedup(list(state.get("discovered_tables", [])) + [table])

#                 elif name == "tool_validate_sql":
#                     sql = (args.get("sql") or "").strip()
#                     step_number = args.get("step_number")
#                     database = (args.get("databaseName") or "").strip()

#                     parsed_result = _parse_tool_result(result)
#                     ok = isinstance(parsed_result, dict) and parsed_result.get("validated") is True

#                     state["last_sql"] = sql
#                     state["last_validation_ok"] = ok
#                     state["last_validation_error"] = None if ok else json.dumps(parsed_result, ensure_ascii=False)

#                     if step_number is not None:
#                         step_registry = dict(state.get("step_registry", {}))
#                         prev = dict(step_registry.get(step_number, {}))
#                         prev.update({
#                             "step": step_number,
#                             "action": "query",
#                             "sql": sql,
#                             "database": database,
#                             "validated": ok,
#                             "executed": prev.get("executed", False),
#                             "rows_returned": prev.get("rows_returned", 0),
#                             "status": prev.get("status", "pending"),
#                         })
#                         step_registry[step_number] = prev
#                         state["step_registry"] = step_registry

#                     if ok:
#                         state["last_validated_sql"] = sql
#                         validated_sqls = dict(state.get("validated_sqls", {}))
#                         validated_sqls[step_number] = {"sql": sql, "database": database}
#                         state["validated_sqls"] = validated_sqls
#                     else:
#                         iter_had_validation_error = True
#                         iter_had_error = True
#                         state["last_failed_step"] = step_number
#                         state["last_validated_sql"] = None
#                         _increment_step_retry(state, step_number)
#                         state["sql_repair_count"] = state.get("sql_repair_count", 0) + 1
#                         state["total_sql_repairs"] = state.get("total_sql_repairs", 0) + 1
#                         _append_error(json.dumps(parsed_result, ensure_ascii=False))

#             log_item = {
#                 "tool": name,
#                 "raw_args": raw_args,
#                 "args": args,
#                 "result": str(result),
#             }
#             iteration_tools.append(log_item)
#             tool_call_id = tc.get("id") or f"fallback_{i}"
#             msgs.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))

#         except Exception as e:
#             err = f"Error: {e}"
#             iter_had_error = True
#             _append_error(err)
#             iteration_tools.append({"tool": name, "raw_args": raw_args, "args": args, "result": err})
#             tool_call_id = tc.get("id") or f"fallback_{i}"
#             msgs.append(ToolMessage(content=err, tool_call_id=tool_call_id))

#     state["iter_had_error"] = iter_had_error
#     state["iter_had_validation_error"] = iter_had_validation_error
#     state["iter_any_exec_ok"] = iter_any_exec_ok

#     if iter_any_exec_ok or iter_had_error:
#         state["last_execute_ok"] = iter_any_exec_ok and not iter_had_error

#     run_log = list(state.get("current_run_log", []))
#     if run_log:
#         run_log[-1]["tools"] = iteration_tools

#     state["msgs"] = msgs
#     state["current_run_log"] = run_log
#     state["crash_snapshot"] = _build_snapshot(state)
#     return state
















"""
_tools_node — refactored into small, testable helper functions.

Public entry point:  _tools_node(state) -> AgentState
Helper functions (in call order):
  _normalize_tool_args          — recover flat-arg mistakes, inject run_id
  _guard_step_tool              — validate pre-conditions, return blocked JSON or None
  _execute_tool                 — call tool.ainvoke or return blocked result
  _update_step_registry         — write step metadata to state
  _update_iteration_flags       — set iter_* accumulators and error counters
  _handle_discovery_or_validation_tool — process non-execute tools (catalog, validate, etc.)
  _build_tool_message_and_log   — append ToolMessage to msgs and log_item to iteration log
"""

from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

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

# Fields that belong inside the nested step dict.
# Used to detect and recover flat-arg mistakes from the model.
STEP_DICT_FIELDS: frozenset[str] = frozenset({
    "step", "action", "database", "sql", "validSql",
    "source_step", "left_source", "right_source",
    "join_type", "left_geom", "right_geom",
    "distance_meters", "buffer_distance",
    "filters", "select_columns", "output_columns",
    "group_by", "aggregation_type", "metric_column", "metric_as",
    "order_by", "order_direction", "limit",
    "target_crs", "join_keys",
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




# ─────────────────────────────────────────────────────────────────────────────
# 1. Normalize tool args
# ─────────────────────────────────────────────────────────────────────────────

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
        return args  # already correct

    # Reconstruct nested step dict from flat fields
    normalized_step: dict = {}

    if step_obj is not None:
        normalized_step["step"] = step_obj  # preserve scalar step number

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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Guard step tool
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# 3. Execute tool
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# 4. Update step registry
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# 5. Update iteration flags
# ─────────────────────────────────────────────────────────────────────────────

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
            _increment_step_retry(state, step_number)
            if name == "tool_execute_query_step":
                state["sql_repair_count"] = state.get("sql_repair_count", 0) + 1
                state["total_sql_repairs"] = state.get("total_sql_repairs", 0) + 1
            else:
                state["step_retry_count"] = state.get("step_retry_count", 0) + 1
                state["total_step_retries"] = state.get("total_step_retries", 0) + 1

        state["all_errors"] = _dedup(
            list(state.get("all_errors", [])) + [json.dumps(parsed_result, ensure_ascii=False)]
        )

    state["last_executed_result"] = parsed_result
    if exec_ok:
        state["last_execute_error"] = None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Handle discovery / validation tools
# ─────────────────────────────────────────────────────────────────────────────

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
        _increment_step_retry(state, step_number)
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









# ─────────────────────────────────────────────────────────────────────────────
# 7. Build tool message and log item
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────



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
    }

    new_state["crash_snapshot"] = _build_snapshot(new_state)
    return new_state
# ─────────────────────────────────────────────────────────────
# finalize
# ─────────────────────────────────────────────────────────────

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




 
# ─────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────

# def _route_after_model(state: AgentState) -> str:
#     ai = state["current_ai"]

#     # If the model already gave the final text answer
#     if not ai.tool_calls:
#         return "finalize" if state.get("last_execute_ok") else "restart_or_finish"

#     # Only stop on max iterations if the model still wants to call tools
#     if state["iterations"] >= state["max_iterations"]:
#         return "restart_or_finish"

#     return "tools"


def _route_after_model(state: AgentState) -> str:
    ai = state["current_ai"]

    if not ai.tool_calls:
        return "finalize"

    if state["iterations"] >= state["max_iterations"]:
        return "restart_or_finish"

    return "tools"

# def _route_after_model(state: AgentState) -> str:
#     ai = state["current_ai"]

#     if ai.tool_calls:
#         return "tools"

#     if state["iterations"] >= state["max_iterations"]:
#         return "restart_or_finish"

#     return "finalize" if state.get("last_execute_ok") else "restart_or_finish"


def _route_after_tools(state: AgentState) -> str:
    iter_had_error = state.get("iter_had_error", False)
    iter_had_validation_error = state.get("iter_had_validation_error", False)
    iter_any_exec_ok = state.get("iter_any_exec_ok", False)

    if iter_any_exec_ok and not iter_had_error:
        return "model"

    if iter_had_validation_error:
        failing_step = state.get("last_failed_step")
        if failing_step is not None and _step_retry_ok(state, failing_step):
            return "sql_repair"
        return "restart_or_finish"

    if iter_had_error:
        parsed_err = _parse_tool_result(state.get("last_execute_error"))
        if isinstance(parsed_err, dict):
            if str(parsed_err.get("error", "")).endswith("blocked."):
                return "model"

        failing_step = state.get("last_failed_step")
        if failing_step is not None and _step_retry_ok(state, failing_step):
            return "model"
        return "restart_or_finish"

    if state.get("iterations", 0) >= state.get("max_iterations", 0):
        return "restart_or_finish"

    return "model"

def _route_after_restart_or_finish(state: AgentState) -> str:
    return "finalize" if state.get("done") else "model"


# ─────────────────────────────────────────────────────────────
# Graph (compiled lazily)
# ─────────────────────────────────────────────────────────────

_AGENT_GRAPH = None


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


# ─────────────────────────────────────────────────────────────
# Agent / pipeline
# ─────────────────────────────────────────────────────────────
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
    }

    state = init_state
    try:
        state = await get_agent_graph().ainvoke(init_state)
    except Exception as e:
        if isinstance(e, AgentRunError):
            raise
        snap = _build_snapshot(state, extra_error=str(e))
        raise AgentRunError(str(e), snapshot=snap)

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







# ─────────────────────────────────────────────────────────────
# Trace extraction
# ─────────────────────────────────────────────────────────────
# Trace extraction
# ─────────────────────────────────────────────────────────────

def extract_agent_trace(result: PipelineResult) -> AgentTrace:
    tables, databases = [], []
    total_tool_calls = 0
    tool_call_counts: dict[str, int] = {}
    executions: list[dict] = []
    merged_step_registry: dict = {}

    if not result.run_logs:
        return AgentTrace(
            [], [], [], [],
            result.valid_sql,
            result.executed,
            0,
            [],
            {},
        )

    for run in result.run_logs:
        for db in run.get("discovered_databases", []):
            if db not in databases:
                databases.append(db)

        for tbl in run.get("discovered_tables", []):
            if tbl not in tables:
                tables.append(tbl)

        for step_no, meta in (run.get("step_registry") or {}).items():
            merged_step_registry[step_no] = meta

        for entry in (run.get("iterations") or []):
            for t in entry.get("tools", []):
                total_tool_calls += 1
                tool_name = t.get("tool")
                if tool_name:
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

    for step_no, meta in sorted(
        merged_step_registry.items(),
        key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0])
    ):
        executions.append({
            "iteration": None,
            "step": step_no,
            "database": meta.get("database"),
            "sql": meta.get("sql"),
            "success": meta.get("executed", False),
            "error": None if meta.get("executed", False) else meta.get("status"),
            "rows_returned": meta.get("rows_returned"),
            "status": meta.get("status"),
        })

    successful = [e for e in executions if e["success"]]
    all_sql_text = " ".join((e.get("sql") or "").lower() for e in successful)

    tables_in_result = [t for t in tables if t.lower() in all_sql_text]
    databases_in_result = list(dict.fromkeys(
        e["database"] for e in successful if e.get("database")
    ))

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
        validated = result.valid_sql
        executed_successfully = result.executed

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
# ─────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────

def _print_summary(query: str, results: list):
    logger.info("\n" + "#" * 130)
    logger.info("QUERY")
    logger.info("  %s", query)
    logger.info("#" * 130)

    logger.info(
        "%-20s %-8s %-8s %-8s %-6s %-6s %-6s %-6s %-10s %-8s %-10s %s",
        "MODEL",
        "SQL_OK",
        "EXEC",
        "TASK",
        "ITER",
        "REP",
        "RST",
        "API",
        "LATENCY",
        "TOOLS",
        "TOKENS",
        "ERROR",
    )
    logger.info("-" * 130)

    for r in results:
        if isinstance(r, Exception):
            logger.info(
                "%-20s %-8s %-8s %-8s %-6s %-6s %-6s %-6s %-10s %-8s %-10s %s",
                "ERROR",
                "False",
                "False",
                "False",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                _short(str(r), 300),
            )
            continue

        trace = extract_agent_trace(r)
        err = "; ".join(_dedup(r.all_errors)) or "-"
        total_tokens = r.tokens.total_tokens if r.tokens else "-"

        logger.info(
            "%-20s %-8s %-8s %-8s %-6s %-6s %-6s %-6s %-10s %-8s %-10s %s",
            r.model_id,
            str(r.valid_sql),
            str(r.executed),
            str(r.task_completed),
            r.iterations,
            r.sql_repairs_used,
            r.restarts_used,
            r.total_api_calls,
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

        trace = extract_agent_trace(r)
        logger.info("[%s]", r.model_id)
        logger.info("  valid_sql          : %s", r.valid_sql)
        logger.info("  executed           : %s", r.executed)
        logger.info("  iterations         : %s", r.iterations)
        logger.info("  sql_repairs        : %s", r.sql_repairs_used)
        logger.info("  restarts           : %s", r.restarts_used)
        logger.info("  latency_s          : %.3f", r.latency_s)
        logger.info("  api_calls          : %s", r.total_api_calls)

        step_registry = {}
        for run in (r.run_logs or []):
            step_registry.update(run.get("step_registry", {}))

        if step_registry:
            logger.info("\n  STEP REGISTRY:")
            for step_no, meta in sorted(step_registry.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0])):
                logger.info("    Step %s:", step_no)
                logger.info("      database        : %s", meta.get("database"))
                logger.info("      validated       : %s", meta.get("validated"))
                logger.info("      executed        : %s", meta.get("executed"))
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

        if r.all_errors:
            logger.info("\n  ERRORS:")
            for err in _dedup(r.all_errors):
                logger.info("    - %s", err)

        logger.info("\n  TRACE:")
        logger.info("    databases explored : %s", trace.databases_explored)
        logger.info("    databases in result: %s", trace.databases_in_result)
        logger.info("    tables explored    : %s", trace.tables_explored)
        logger.info("    tables in result   : %s", trace.tables_in_result)
        logger.info("    validated          : %s", trace.validated_before_exec)
        logger.info("    executed ok        : %s", trace.executed_successfully)
        logger.info("    total tool calls   : %s", trace.total_tool_calls)
        logger.info("    tool call counts   : %s", trace.tool_call_counts)

        if trace.executions:
            logger.info("\n  EXECUTIONS:")
            for i, ex in enumerate(trace.executions):
                logger.info("    Execution %s:", i)
                logger.info("      iteration       : %s", ex["iteration"])
                logger.info("      success         : %s", ex["success"])
                logger.info("      database        : %s", ex["database"])
                logger.info("      sql             : %s", ex["sql"])
                if not ex["success"]:
                    logger.info("      error           : %s", ex["error"])

        if r.run_logs:
            logger.info("\n  RUN LOGS:")
            for run in r.run_logs:
                logger.info("    Run %s:", run.get("run_index", 0))
                logger.info("      attempt         : %s", run.get("attempt_number", 0))
                logger.info("      sql_repairs     : %s", run.get("sql_repairs_used", 0))
                logger.info("      final_sql       : %s", run.get("final_sql") or "(none)")

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
                if run.get("execution_error"):
                    logger.info("      execution_error : %s", run["execution_error"])

                for entry in run.get("iterations", []):
                    tok = entry.get("tokens", {})
                    logger.info("      Iteration %s:", entry["iteration"])
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







                    for t in entry.get("tools", []):
                        logger.info("          TOOL   : %s", t["tool"])
                        logger.info("          ARGS   : %s", t["args"])
                        if t["tool"] == "tool_validate_sql":
                            raw = str(t["result"]).strip()
                            parsed = None

                            try:
                                parsed = json.loads(raw)
                                ok = isinstance(parsed, dict) and parsed.get("validated") is True
                            except (json.JSONDecodeError, TypeError):
                                ok = raw.lower() == "ok"

                            logger.info(
                                "          RESULT : %s",
                                f"ok (step={parsed.get('step_number')})" if ok and parsed else f"ok" if ok else f"fail: {raw[:200]}",
                            )
                        else:
                            logger.info("          RESULT : %s", t["result"])










        logger.info("\n  FINAL:")
        logger.info("    answer            : %s", r.final_answer or "(none)")
        logger.info("    sql               : %s", r.final_sql or "(none)")
        logger.info(
            "    result_data       : %s",
            r.result_data if r.result_data is not None else "(none)",
        )

    logger.info("\n" + "=" * 110)


# ─────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────

async def store_run(run: EvalRun, results: list) -> str:
    q_hash = query_hash(run.query)
    run_id = uuid.uuid4().hex
    base = f"agent_benchmark_runstest/{q_hash}/runs/{run_id}"

    result_data_by_model = {
        r.model_id: r.result_data
        for r in results
        if not isinstance(r, Exception)
    }

    loop = asyncio.get_running_loop()

    async def _upload(path: str, data) -> None:
        try:
            await loop.run_in_executor(None, upload, path, data)
        except Exception as e:
            logger.warning("upload failed [%s]: %s", path, e)

    try:
        _update_queries_index(q_hash, run.query)
        if not query_meta_exists(q_hash):
            await _upload(
                f"agent_benchmark_runstest/{q_hash}/meta.json",
                {"query": run.query, "query_hash": q_hash},
            )
    except Exception as e:
        logger.warning("Failed to update query index: %s", e)

    await _upload(f"{base}/system_prompt.txt", load_prompt(run.prompt_name))

    summary = []
    normalized_docs = []

    for r in results:
        if isinstance(r, AgentRunError):
            snap = r.snapshot or {}
            tokens = snap.get("tokens", _empty_token_dict())
            doc = {
                "run_id": run_id,
                "query": run.query,
                "model_id": snap.get("model_id"),
                "status": "error",
                "error": str(r),
                "valid_sql": False,
                "executed": False,
                "iterations": snap.get("iterations", 0),
                "sql_repairs_used": snap.get("sql_repairs_used", 0),
                "restarts_used": snap.get("restart_count", 0),
                "total_api_calls": snap.get("total_api_calls", 0),
                "total_tool_calls": snap.get("total_tool_calls", 0),
                "latency_s": None,
                "tokens": tokens,
                "errors": snap.get("all_errors", []),
                "final_answer": None,
                "final_sql": snap.get("final_sql"),
                "trace": None,
                "run_logs": snap.get("run_logs", []),
                "task_completed": False,
                "answer_based_on_result": False,
            }
            summary.append({
                "model_id": doc["model_id"],
                "status": "error",
                "valid_sql": False,
                "executed": False,
                "iterations": doc["iterations"],
                "sql_repairs_used": doc["sql_repairs_used"],
                "restarts_used": doc["restarts_used"],
                "total_api_calls": doc["total_api_calls"],
                "total_tool_calls": doc["total_tool_calls"],
                "latency_s": None,
                "total_tokens": tokens.get("total_tokens"),
            })

        else:
            trace = extract_agent_trace(r)
            tokens = asdict(r.tokens) if r.tokens else _empty_token_dict()
            status = "completed" if r.task_completed else "incomplete"
            doc = {
                "run_id": run_id,
                "query": run.query,
                "model_id": r.model_id,
                "status": status,
                "error": None,
                "valid_sql": r.valid_sql,
                "executed": r.executed,
                "iterations": r.iterations,
                "sql_repairs_used": r.sql_repairs_used,
                "restarts_used": r.restarts_used,
                "total_api_calls": r.total_api_calls,
                "total_tool_calls": trace.total_tool_calls,
                "latency_s": r.latency_s,
                "tokens": tokens,
                "errors": _dedup(r.all_errors),
                "final_answer": r.final_answer,
                "final_sql": r.final_sql,
                "trace": asdict(trace),
                "run_logs": r.run_logs or [],
                "task_completed": r.task_completed,
                "answer_based_on_result": r.answer_based_on_result,
            }
            summary.append({
                "model_id": r.model_id,
                "status": status,
                "valid_sql": r.valid_sql,
                "executed": r.executed,
                "iterations": r.iterations,
                "sql_repairs_used": r.sql_repairs_used,
                "restarts_used": r.restarts_used,
                "total_api_calls": r.total_api_calls,
                "total_tool_calls": trace.total_tool_calls,
                "latency_s": r.latency_s,
                "total_tokens": tokens.get("total_tokens"),
            })

        normalized_docs.append(doc)

    await _upload(
        f"{base}/meta.json",
        {
            "run_id": run_id,
            "query": run.query,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt_name": run.prompt_name,
            "params": asdict(run.params),
            "model_ids": run.model_ids,
            "summary": summary,
        },
    )

    try:
        _update_runs_index(q_hash, run_id, summary, run.params)
    except Exception as e:
        logger.warning("Failed to update runs index: %s", e)

    seen_ids: set[str] = set()
    for doc in normalized_docs:
        model_id = doc["model_id"] or "unknown_model"
        key = model_id
        while key in seen_ids:
            key = f"{model_id}_{uuid.uuid4().hex[:6]}"
        seen_ids.add(key)

        result_data = result_data_by_model.get(model_id)
        await _upload(f"{base}/{key}.json", doc)
        await _upload(
            f"{base}/{key}_result_data.json",
            {
                "run_id": run_id,
                "query": run.query,
                "model_id": model_id,
                "status": doc["status"],
                "row_count": (
                    len(result_data)
                    if doc["status"] == "completed" and isinstance(result_data, list)
                    else None
                ),
                "result_data": result_data if doc["status"] == "completed" else None,
            },
        )

    logger.info("SAVED: %s/", base)
    return f"{q_hash}/{run_id}"


# ─────────────────────────────────────────────────────────────
# Retry / run loop
# ─────────────────────────────────────────────────────────────

_MAX_RETRY_DELAY_S = 120


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
        " Compare the number of buildings located in flood hazard zones, quick-clay hazard areas, and soil- and flood-landslide hazard areas for each municipality",
    ]

    model_ids = [

      "gpt-5-mini"
    ]

    params = ModelParams(
        temperature=0.0,
        max_tokens=2000,
        timeout=120,
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