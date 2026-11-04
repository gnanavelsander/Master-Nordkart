# Runtime context and historical configuration

[Back to project overview](../README.md)

The recorded demo and source code are the intended review route. Full execution requires external model endpoints and geographic databases from the project environment. The original access is no longer available. These notes explain the existing development setup; they are not a complete reproduction recipe.

## Dependencies and entry points

| Area | Existing setup |
| --- | --- |
| Python | [pyproject.toml](../pyproject.toml) declares Python 3.12 or later; [uv.lock](../uv.lock) records dependencies. The existing `uv sync` workflow installs dependencies, not datasets or model access. |
| API | The application is `backend.app:app` in [backend/app.py](../backend/app.py). Its development command, from the repository root, is `uv run uvicorn backend.app:app --reload --port 8000`. The earlier README's `api_server:app` target does not exist here. |
| Frontend | The [frontend guide](../frontend/README.md) describes its scripts and API configuration. Starting the UI alone does not provide GIS analysis. |
| MCP | The [client](../backend/mcp/client/client.py) launches `backend/mcp/server/run.py` with the current Python interpreter and inherited environment. No separate public MCP hosting is involved. |
| Batch evaluation | `main()` in [pipeline_runner.py](../backend/evaluation/pipeline_runner.py) selects tasks/models and calls the agent and cloud reporting. Running it can invoke models and write evaluation records. |

## External requirements

[config.py](../config.py) loads environment values through `python-dotenv`; the [database registry](../backend/mcp/server/registry.py) identifies active source connections.

- **Model access:** the adapter in [agent_graph.py](../backend/evaluation/agent_graph.py) supports Azure-hosted models and locally served Ollama/vLLM models. Each selected adapter needs its corresponding service and configuration. A registered model name does not prove availability or experimental coverage.
- **Geographic data:** `PGCONN_*` values connect to separate source databases. Data imports, permissions, and deployment configuration do not accompany this checkout.
- **Batch reporting:** the Blob Storage path uses `AZURE_STORAGE_CONNECTION_STRING`. Its stored results are external to this repository.
- **Optional retrieval:** building the embedding index can contact databases and a model service. It is not an offline prerequisite for reading the code.

Inspect environment variable names without supplying credentials. Keep actual keys, connection strings, and private endpoints outside version control.

## Retained development and maintenance files

The earlier README described a Docker / VS Code Dev Container workflow. The [.devcontainer configuration](../.devcontainer/devcontainer.json), [Dockerfile](../.devcontainer/Dockerfile), and [development Compose file](../docker-compose.devcontainer.yml) remain as development context. They do not provide the original datasets or model services.

The separate [docker-compose.yml](../docker-compose.yml) references a backend Dockerfile that is absent here. Its generic PostgreSQL environment variables differ from the `PGCONN_*` connections used by the current server. It is not a verified one-command launch path.

[migrate_run2_to_run3.py](../migrate_run2_to_run3.py) and [dedup_run3.py](../dedup_run3.py) are historical cloud-data maintenance scripts. They move or delete run records; they are not evaluation entry points and were not run during this documentation update.

The update was checked without starting the API, calling models, or connecting to data/cloud services. End-to-end behavior has not been reverified.
