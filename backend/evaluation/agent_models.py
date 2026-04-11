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

    schema_name: Optional[str] = None
    table_name: Optional[str] = None





class QueryPlan(BaseModel):
    query_mode: Literal["single_db", "multi_db"]
    steps: list[PlanStep] = Field(default_factory=list)










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
    total_api_calls: int



    step_retry_count: int            
    total_step_retries: int
 
    #step_retry_counts: dict         
   
   
   
    step_registry: dict           
    validated_sqls: dict           
 
    iter_had_error: bool
    iter_had_validation_error: bool
    iter_any_exec_ok: bool
 
    plan_locked: bool
    locked_step_ids: set            
 
    crash_snapshot: dict
    attempt_number: int
    run_id: str
    query_mode: Optional[Literal["single_db", "multi_db"]]




    sql_repair_counts: dict     
    step_retry_counts: dict      








class AgentRunError(Exception):
    def __init__(self, message: str, snapshot: dict | None = None):
        super().__init__(message)
        self.snapshot = snapshot or {}

