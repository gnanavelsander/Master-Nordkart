from dataclasses import dataclass
from typing import Any, Optional, TypedDict, Literal
from pydantic import BaseModel, Field




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
    max_replans: int = 1


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


@dataclass
class EvalRun:
    query: str
    model_ids: list[str]
    params: ModelParams | None = None
    prompt_name: str = "default"

    def __post_init__(self):
        if self.params is None:
            self.params = ModelParams()















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
    last_guard_error: Optional[str]
 
    last_executed_result: Any      
    last_execute_error: Optional[str]
    last_failed_step: Optional[int]  
 
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



    step_retry_count: int            
    total_step_retries: int
 
   
   
   
    step_registry: dict           
    validated_sqls: dict           
    planned_steps: dict
    required_step_ids: set
    plan_finalized: bool
 
    iter_had_error: bool
    iter_had_validation_error: bool
    iter_any_exec_ok: bool
 
    plan_locked: bool
    locked_step_ids: set            
 
    crash_snapshot: dict
    attempt_number: int
    run_id: str
    query_mode: Optional[Literal["single_db", "multi_db"]]



    retrieval_candidates: list[dict]
    retrieval_scope_active: bool
    retrieval_top_score: float | None


    sql_repair_counts: dict
    step_retry_counts: dict

    needs_replan: bool
    zero_rows_step: Optional[int]
    replan_reason: Optional[str]
    replan_error: Optional[str]
    replan_count: int








class AgentRunError(Exception):
    def __init__(self, message: str, snapshot: dict | None = None):
        super().__init__(message)
        self.snapshot = snapshot or {}
