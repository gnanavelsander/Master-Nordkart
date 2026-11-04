# Architecture and code walkthrough

[Back to project overview](../README.md)

This guide describes source commit `b051f44ecd4c8037fb8fbabbec0b3c230f5ec14a`. The interactive application and batch evaluation share an agent, but use different prompts and parameters. The thesis describes the research system in Chapter 4; it is not an exact release manifest for this checkout.

## From question to result

1. **Collect context.** [page.tsx](../frontend/src/app/page.tsx) sends the question, selected models and prompt, and optional area of interest, selected feature, and conversation history to `POST /query` in [app.py](../backend/app.py). The API builds context and calls `run_all`; it can also return cached responses.
2. **Initialize the agent.** [pipeline_runner.py](../backend/evaluation/pipeline_runner.py) creates messages, a run ID, and state, then invokes the graph. Selected models run sequentially. Optional embedding-based database retrieval is controlled by `ModelParams.use_embeddings`, which defaults to `False`.
3. **Discover and plan.** [agent_graph.py](../backend/evaluation/agent_graph.py) connects the `model`, `tools`, `replan`, and `finalize` nodes with LangGraph. The model uses discovery tools and emits a numbered JSON plan. State tracks planned actions, required steps, validation records, execution results, and errors.
4. **Call tools.** [agent_tools_node.py](../backend/evaluation/agent_tools_node.py) checks calls against the plan. [langchain_tools.py](../backend/tools/langchain_tools.py) exposes model-facing tool schemas. The [MCP client](../backend/mcp/client/client.py) sends calls over standard input/output to the local [server process](../backend/mcp/server/run.py).
5. **Retrieve and combine data.** [discovery_tools.py](../backend/mcp/server/discovery_tools.py) inspects catalogs and validates SQL. [step_tools.py](../backend/mcp/server/step_tools.py) executes queries and spatial/tabular operations. Queries access individual PostgreSQL/PostGIS databases; cross-database analyses combine returned datasets through step tools, including GeoPandas/Shapely operations.
6. **Finalize and display.** The graph and [agent_helpers.py](../backend/evaluation/prompts/agent_helpers.py) derive execution and plan-completion flags. The pipeline fetches the full stored final-step result. `app.py` serializes it and converts supported geometry representations to GeoJSON; the UI displays the answer, map or table, SQL, steps, and logs.

## Storage roles

| Component | Role |
| --- | --- |
| PostgreSQL / PostGIS | External source datasets. [registry.py](../backend/mcp/server/registry.py) maps logical database names to environment-supplied connections. [database.py](../backend/database/database.py) manages pools and runs queries in read-only transactions. |
| Memory and DuckDB | Temporary intermediate results keyed by run and step. [step_storage.py](../backend/mcp/server/step_storage.py) keeps smaller results in memory and spills results above 5,000 rows to a temporary DuckDB file. |
| Azure Blob Storage | The batch runner's [reporting.py](../backend/evaluation/reporting.py) and [AzureBlobStorage.py](../backend/storage/AzureBlobStorage.py) write configuration, logs, derived results, and indexes. Those artifacts are not bundled here. The interactive `/query` route does not call `store_run`. |
| Optional embedding index | [retrieve_layers.py](../backend/embeddings/retrieve_layers.py) ranks database candidates from catalog metadata. [build_layer_index.py](../backend/embeddings/build_layer_index.py) builds the index using embeddings. This supports discovery; it is not model training or a copy of the source datasets. |

The MCP server retains complete step results while returning compact samples and metadata to the model. The frontend retrieval path requests full rows separately.

## Validation and recovery

SQL validation checks statement form and forbidden keywords, then asks PostgreSQL to `EXPLAIN` the query. This checks whether the query can be planned, not whether it answers the question.

The tool node checks plan membership, argument shape, source-step availability, SQL/database agreement with prior validation, and attempts to change completed steps. Failures are recorded and returned to the model. Repair and retry budgets limit attempts. Certain failures, including timeouts, plan mismatches, and empty intermediate results, can trigger replanning. Iteration and replan budgets stop the loop.

An empty final result can be accepted, while an empty intermediate result can require a new plan. The finalizer suppresses the answer when its plan-completion checks fail. These are execution controls, not an independent semantic correctness assessment; see the [evaluation guide](evaluation.md).

## Current paths and retained variants

- `backend/evaluation/` contains the agent used by both the interactive application and the batch runner.
- The frontend defaults to [prompt_v4.txt](../backend/evaluation/prompts/prompt_v4.txt). The API falls back to [prompt_v1.txt](../backend/evaluation/prompts/prompt_v1.txt) if no prompt is supplied. The batch runner explicitly loads [evaluation_v1.txt](../backend/evaluation/prompts/evaluation_v1.txt).
- [prompt_v2.txt](../backend/evaluation/prompts/prompt_v2.txt) and [prompt_v3.txt](../backend/evaluation/prompts/prompt_v3.txt) are also retained and selectable. This checkout has no `prompt_v5.txt`; do not equate `evaluation_v1` with the thesis's final prompt without comparing sources.
- Some helpers in `langchain_tools.py`, including routing, are not in the active `_get_tools()` list in `agent_tools_node.py`.
- [main.py](../main.py) is a placeholder, not the application entry point. Historical environment and maintenance files are explained in the [runtime notes](runtime-context.md).
