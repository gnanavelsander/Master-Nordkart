# Architecture and code walkthrough

[Back to project overview](../README.md)

This guide describes the implementation at source commit `b051f44ecd4c8037fb8fbabbec0b3c230f5ec14a`. It explains the practical role of each component in this code, without claiming that alternative technologies were compared. The thesis describes the research system in Chapter 4; its experiments are distinguished from this checkout in the [evaluation guide](evaluation.md).

## Backend and interface: from question to visible result

The interface combines a question with spatial context: a drawn area or selected map feature can narrow what the user means. The backend turns that input into the same agent invocation used by the batch runner, then converts the result into a form the browser can display. FastAPI provides this asynchronous HTTP boundary; Next.js/React manages the input and result views, while Leaflet renders the returned geometry and captures map selections.

| Stage | Code and behavior |
| --- | --- |
| Send the question | `sendToBackend()` in [page.tsx](../frontend/src/app/page.tsx) calls `fetchAssistant()` with `POST /query`. The payload contains the question, selected models and prompt, area of interest, selected feature, and conversation history. |
| Prepare the request | `run_query()` in [app.py](../backend/app.py) builds context with `_aoi_to_context()`, `_history_to_context()`, and `_selected_feature_to_context()`, loads the prompt, constructs `ModelParams`, and calls `run_all()`. A matching cached response can bypass a new run. |
| Start the agent | In [pipeline_runner.py](../backend/evaluation/pipeline_runner.py), `run_all()` processes selected models sequentially through `_run_one_with_retry()` and `pipeline()`. `agent()` creates a run ID, messages, and initial state, then awaits `get_agent_graph().ainvoke(init_state)`. |
| Return a result | After a completed workflow, `agent()` calls `_expand_result_for_frontend()` to obtain full rows before cleaning up temporary step data. `_serialise()` in `app.py` returns the answer, SQL, data, steps, logs, and errors; `_result_to_geojson()` supplies map geometry. |
| Display it | `sendToBackend()` selects a primary response and sets the map data. `showOnMap()` selects another model's GeoJSON; `showTable()` reads `result_data.rows`. [MapClient.tsx](../frontend/src/app/components/MapClient.tsx) loads [MapView.tsx](../frontend/src/app/components/MapView.tsx) without server-side rendering so Leaflet runs in the browser. [ResultTable.tsx](../frontend/src/app/ResultTable.tsx) displays rows. |

This response includes intermediate work because a map or a fluent answer alone cannot explain which data and operations produced it. See the [frontend guide](../frontend/README.md) for prompt editing and API configuration.

## LangGraph: control flow and state

A spatial analysis may require several dependent queries and transformations. One model response cannot tell the backend whether all those operations have actually run. LangGraph connects the model, tools, recovery, and completion checks while carrying explicit state between them.

`_build_agent_graph()` in [agent_graph.py](../backend/evaluation/agent_graph.py) constructs `StateGraph(AgentState)` with four nodes:

| Node | Responsibility and next step |
| --- | --- |
| `model` / `_model_node()` | Calls the selected pretrained model with the conversation and bound tools. Extracts a numbered JSON plan from model output and records required steps. `_route_after_model()` sends tool calls to `tools`, or an answer without tool calls to `finalize`. |
| `tools` / `_tools_node()` | Defined in [agent_tools_node.py](../backend/evaluation/agent_tools_node.py). Checks and executes calls, updates step status, and adds tool results/errors to the conversation. `_route_after_tools()` returns to the model, requests replanning, or stops when a limit is reached. |
| `replan` / `_replan_node()` | Builds feedback describing the failure and previous work, preserves run logs, and resets the active conversation/plan bookkeeping for a new attempt. Then returns to `model`. |
| `finalize` / `_finalize_node()` | Derives execution and completion flags and suppresses the answer if plan-completion checks fail. Then ends the graph. |

The state contract is `AgentState` in [agent_models.py](../backend/evaluation/agent_models.py); `agent()` initializes it in [pipeline_runner.py](../backend/evaluation/pipeline_runner.py). Its fields have different jobs:

- `msgs` and `initial_msgs` carry the conversation and the starting context used for replanning.
- `planned_steps`, `required_step_ids`, and `plan_finalized` record what the model committed to doing.
- `validated_sqls`, `step_registry`, and `locked_step_ids` track validation, execution, and completed steps that cannot be changed within the active plan.
- `replan_trigger`, repair/retry counters, and iteration limits drive recovery and stopping decisions.
- `current_run_log`, `all_run_logs`, and `all_errors` retain evidence for inspection and reporting.

This is per-run workflow state, not the full geographic datasets or a persistent conversation database. Explicit plan and validation records let `_guard_validate_sql()` and `_guard_step_tool()` reject unplanned operations, missing source steps, or a query that differs from its validated SQL/database. `_handle_validate_sql_tool()` records validation failures for repair. Other failures, including timeouts and empty intermediate results, can trigger a new plan. Empty terminal results can be accepted. Budgets prevent recovery from continuing indefinitely.

These checks enforce execution discipline. They do not prove that the plan uses the correct spatial relation or answers the original question.

## MCP: the tool-call path

MCP carries structured calls between the agent process and a local tool-server process. Here it gives discovery, validation, query, and analysis operations one call/response interface, while the server owns connections and stored rows. The model works with tool arguments and compact results instead of database connection details or entire datasets.

There are two sets of definitions to follow:

- **Model-facing tools:** functions decorated with `@tool` in [langchain_tools.py](../backend/tools/langchain_tools.py). `_get_tools()` in [agent_tools_node.py](../backend/evaluation/agent_tools_node.py) selects the active subset. `get_model()` and `get_local_vllm_model()` in [agent_graph.py](../backend/evaluation/agent_graph.py) bind that list to the model using `bind_tools()`. Availability is explicit here; it is not inferred from every helper in the repository.
- **Server tools:** functions decorated with `@mcp.tool` in [discovery_tools.py](../backend/mcp/server/discovery_tools.py) and [step_tools.py](../backend/mcp/server/step_tools.py). [run.py](../backend/mcp/server/run.py) imports both modules to register them on the `FastMCP` instance from [registry.py](../backend/mcp/server/registry.py), then runs the server.

For a query step, follow this concrete chain:

1. `_tools_node()` reads the model's call, normalizes arguments, and applies `_guard_step_tool()`.
2. `tool_execute_query_step()` in `langchain_tools.py` invokes `execute_query_step()` in [client/tools_steps.py](../backend/mcp/client/tools_steps.py).
3. That wrapper calls `mcp_call("execute_query_step", ...)`. In [client.py](../backend/mcp/client/client.py), `MCPClientManager.call()` uses `ClientSession.call_tool()`. `_server_params()` and `_run()` start the server as a subprocess and connect over standard input/output.
4. The server's `execute_query_step()` validates the payload with `PlanStep` from [models.py](../backend/mcp/server/models.py), queries the selected database, saves rows, and returns result metadata or an error.
5. The client parses the response; the model-facing wrapper returns JSON. `_build_tool_message_and_log()` adds a `ToolMessage` to the conversation and captures the call for inspection.

Validation follows a similar route: `tool_validate_sql()` calls `validate_sql_via_mcp()` in [client/tools_discovery.py](../backend/mcp/client/tools_discovery.py), which invokes the server's `validate_sql_explain` tool. Routing helpers also exist in `langchain_tools.py`, but are not in the active `_get_tools()` list.

## PostgreSQL and PostGIS: source data and spatial queries

The sources are separate thematic databases. `app_lifespan()` in [registry.py](../backend/mcp/server/registry.py) registers building and parcel data, municipality/county boundaries, a 1 km statistical grid, AR50 land resources, and hazard layers including flooding, storm surge, quick clay, and avalanche/landslide susceptibility. These contain the attributes and geometries needed for questions such as which buildings intersect a hazard area. The data itself is external to this checkout.

PostgreSQL stores and filters the records; PostGIS lets generated SQL express spatial conditions and convert geometries. The query rules in [evaluation_v1.txt](../backend/evaluation/prompts/evaluation_v1.txt) instruct the agent to use operations such as `ST_Intersects` for overlap, `ST_Within` for containment, and `ST_DWithin` for distance, and `ST_AsGeoJSON(ST_Transform(..., 4326))` when returning geometry. The backend executes the model's SQL; these expressions are not hardcoded answers to the benchmark.

| Concern | Implementation |
| --- | --- |
| Connection configuration | [config.py](../config.py) reads `PGCONN_*` environment values. `app_lifespan()` registers them under logical dataset names. |
| Connection reuse and queries | `PoolManager.register()`, `ensure_connection()`, and `query()` in [database.py](../backend/database/database.py) open pools as needed and run queries inside read-only transactions. Repeated discovery and query calls can reuse connections. |
| Discover actual schemas | `get_database_catalog()` and `get_table_schema()` in [discovery_tools.py](../backend/mcp/server/discovery_tools.py) expose tables, geometry metadata, and column types so the model can use names from the source rather than guess them. |
| Validate SQL | `validate_sql_explain()` checks statement form and forbidden keywords, then executes `EXPLAIN (FORMAT JSON)` through the selected connection. The agent separately checks that execution uses the same validated SQL and database. |
| Execute and store rows | `execute_query_step()` in [step_tools.py](../backend/mcp/server/step_tools.py) applies query rules, fetches records, and stores geometry/CRS metadata with the result. |

One query step accesses one source database. This implementation combines sources after retrieval rather than issuing a SQL join across database connections. For example, `execute_spatial_join_step()` loads `left_source` and `right_source` results, creates GeoDataFrames, aligns their CRS when both are known and differ, and calls `geopandas.sjoin()`. Attribute joins use `execute_attribute_join_step()` and a pandas merge. Later aggregation, sorting, and limiting steps consume those stored outputs.

This division lets PostGIS filter within a source and lets step tools combine independently retrieved layers. SQL that passes validation can still use the wrong filter, count duplicate join rows, or apply the wrong spatial predicate; the [evaluation guide](evaluation.md) explains how those errors were investigated.

## Intermediate results: memory and DuckDB

A spatial join cannot run until both input queries have produced rows. An aggregate may then depend on that join. The system therefore retains each output under a run ID and step number, so subsequent tools can reference `source_step`, `left_source`, or `right_source` without asking the model to reproduce the data.

[step_storage.py](../backend/mcp/server/step_storage.py) implements that storage:

- `_save_rows()` stores up to **5,000 rows** in the in-memory `AppContext.step_results` dictionary. It records geometry column, CRS, row count, timestamp, and operation provenance.
- Above that threshold, `_save_rows()` writes a table in the temporary DuckDB file created by `app_lifespan()`. Geometry objects are converted to WKT for storage. The dictionary keeps the table reference and metadata instead of the full row list.
- `_get_rows()` hides the storage choice from downstream tools: it returns the memory rows or reads the DuckDB table and decodes the stored geometry. This is temporary storage, not a replacement for the PostGIS sources. Large results are still materialized when read; disk storage does not make every analysis memory-bounded.
- `_ok()` returns a compact response to the model, including counts and a sample of at most five rows with geometry omitted. `get_step_result()` in [step_tools.py](../backend/mcp/server/step_tools.py) can retrieve the full result for display.

`agent()` in [pipeline_runner.py](../backend/evaluation/pipeline_runner.py) fetches full rows for a completed workflow before calling `cleanup_run()` in its `finally` block. Cleanup removes that run's memory entries and DuckDB tables; server shutdown removes the temporary database file. This lifecycle separates the data needed during an analysis from the logs saved for evaluation.

## Metadata and embeddings: optional database discovery

The model may not know which logical database contains the data implied by a question. The optional retrieval path compares the question with table metadata and suggests databases to inspect first. It reduces the initial search space; it does not generate an answer or replace live schema checks.

| Stage | Code and behavior |
| --- | --- |
| Gather metadata | `build_metadata_catalog()` in [build_layer_index.py](../backend/embeddings/build_layer_index.py) calls MCP catalog/schema tools and collects database, schema, table, selected-column, and geometry metadata. |
| Build the index | Its `main()` uses `build_catalog_embedding_text()` from [retrieve_layers.py](../backend/embeddings/retrieve_layers.py) and `embed_text()` from [embeddings.py](../backend/embeddings/embeddings.py), then writes [table_index.json](../backend/embeddings/table_index.json). Embeddings come from the configured Azure embedding deployment. |
| Rank candidates | `load_index()` reads the JSON index. `retrieve_catalog_candidates()` embeds the question and ranks table entries by cosine similarity. `retrieve_database_candidates()` groups them by database using each database's best table score. |
| Add agent context | `agent()` in [pipeline_runner.py](../backend/evaluation/pipeline_runner.py) calls retrieval only when `params.use_embeddings` is true. It adds `build_retrieval_context()` to the messages and attempts to prefetch candidate database connections. The current context contains database names and scores, not full table schemas. |

`ModelParams.use_embeddings` defaults to **False** in [agent_models.py](../backend/evaluation/agent_models.py), and neither the current `/query` handler nor the batch `main()` overrides it. Retrieval is therefore optional and disabled in those entry points. When enabled, low-confidence results receive a broader-discovery fallback note; retrieval errors are logged so agent initialization can continue. Building the index and embedding questions require model access. The JSON file contains catalog metadata and vectors, not the geographic data rows.

## Evaluation: instrument the same workflow

The batch entry point uses the shared graph so model comparisons can inspect the same discovery, planning, validation, and tool operations as the interactive application. `main()` in [pipeline_runner.py](../backend/evaluation/pipeline_runner.py) holds task/model choices, constructs `EvalRun`, and calls `store_run()` in [reporting.py](../backend/evaluation/reporting.py). Reporting derives traces, step outcomes, failure categories, and latency/tool-use metrics before saving run artifacts through [AzureBlobStorage.py](../backend/storage/AzureBlobStorage.py).

This makes it possible to examine where a run stopped and what it actually executed. It does not make an execution flag a reference-answer correctness score. See [evaluation code and evidence](evaluation.md#evaluation-code-and-evidence) for the functions to read and the short comparison with the thesis's experiments.

## Entry points and retained prompt variants

The `backend/evaluation/` directory also contains the interactive agent. The browser defaults to [prompt_v4.txt](../backend/evaluation/prompts/prompt_v4.txt), while the API falls back to [prompt_v1.txt](../backend/evaluation/prompts/prompt_v1.txt) when no prompt is supplied. [prompt_v2.txt](../backend/evaluation/prompts/prompt_v2.txt) and [prompt_v3.txt](../backend/evaluation/prompts/prompt_v3.txt) remain selectable. The batch runner loads [evaluation_v1.txt](../backend/evaluation/prompts/evaluation_v1.txt); this checkout has no `prompt_v5.txt`, and no equivalence to the thesis's final v5 prompt is established.

[main.py](../main.py) is a placeholder, not the application entry point. The [runtime notes](runtime-context.md) describe dependencies and retained historical environment/maintenance files.
